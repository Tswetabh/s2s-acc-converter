"""CUDA graph capture/replay for FlowLM AR latent steps (scalar B=1 and batch B>1).

Design (reuse):
  - Capture once per (batch_size, lsd_decode_steps) into owned static buffers.
  - Before each generation unit: prepare() copies post-prompt live state into
    static state (no re-capture).
  - step() always replays against owned static state.

Requires fixed-capacity KV (end_offset in-place) so multi-step replay works.
Falls back to eager when capture fails or graphs are disabled.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.modules.stateful_module import increment_steps

logger = logging.getLogger(__name__)


def clone_flow_state(
    state: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Deep-clone FlowLM streaming state tensors."""
    return {
        module_name: {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in module_state.items()
        }
        for module_name, module_state in state.items()
    }


def reset_flow_state(
    target: dict[str, dict[str, torch.Tensor]],
    source: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Copy source state tensors into target (same structure, in-place)."""
    for module_name, module_state in source.items():
        for key, value in module_state.items():
            if isinstance(value, torch.Tensor):
                target[module_name][key].copy_(value)


def states_compatible(
    a: dict[str, dict[str, torch.Tensor]],
    b: dict[str, dict[str, torch.Tensor]],
) -> bool:
    """True when tensor keys and shapes match (graph can reuse static buffers)."""
    if set(a.keys()) != set(b.keys()):
        return False
    for module_name, module_a in a.items():
        module_b = b[module_name]
        if set(module_a.keys()) != set(module_b.keys()):
            return False
        for key, ta in module_a.items():
            tb = module_b[key]
            if isinstance(ta, torch.Tensor) != isinstance(tb, torch.Tensor):
                return False
            if isinstance(ta, torch.Tensor) and ta.shape != tb.shape:
                return False
    return True


class FlowLMStepGraph:
    """Capture once; prepare + replay AR steps on owned static FlowLM state.

    Supports batch size B>=1. For B>1, temperatures and eos thresholds are
    static CUDA tensors copied each step (graph-safe).
    """

    def __init__(self, model: Any, batch_size: int = 1):
        """Bind to a TTSModel exposing flow_lm, lsd_decode_steps, noise_clamp."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model = model
        self.batch_size = int(batch_size)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._static_state: dict[str, dict[str, torch.Tensor]] | None = None
        self._static_seq: torch.Tensor | None = None
        self._static_empty_emb: torch.Tensor | None = None
        self._static_temp: torch.Tensor | None = None
        self._static_eos_thr: torch.Tensor | None = None
        self._static_lat: torch.Tensor | None = None
        self._static_eos: torch.Tensor | None = None
        self._lsd_steps: int | None = None
        self._captured = False
        self._capture_error: str | None = None
        self.prepare_count = 0

    @property
    def available(self) -> bool:
        """True when a graph was captured successfully."""
        return self._captured and self.graph is not None and self._static_state is not None

    @property
    def static_state(self) -> dict[str, dict[str, torch.Tensor]]:
        """Owned state mutated by graph replay (use after prepare / capture)."""
        if self._static_state is None:
            raise RuntimeError("FlowLMStepGraph has no static_state; capture first")
        return self._static_state

    @property
    def lsd_decode_steps(self) -> int | None:
        """LSD steps baked into the captured graph, if any."""
        return self._lsd_steps

    def capture(
        self,
        model_state: dict[str, dict[str, torch.Tensor]],
        backbone_template: torch.Tensor,
        temperature: float | torch.Tensor | None = None,
        eos_threshold: float | torch.Tensor | None = None,
        lsd_decode_steps: int | None = None,
    ) -> bool:
        """Capture one AR step onto owned static state; leave state at post-prompt.

        Live model_state is never mutated. After success, static_state holds a
        copy of the post-prompt contents (ready for this generation or prepare()).

        Args:
            model_state: Live FlowLM state after text prompt (template for shapes/content).
            backbone_template: Shape [B, 1, ldim] matching self.batch_size.
            temperature: Scalar or [B] tensor; defaults to model.temp.
            eos_threshold: Scalar or [B] tensor; defaults to model.eos_threshold.
            lsd_decode_steps: Defaults to model.lsd_decode_steps.

        Returns:
            True on successful capture; False on failure (eager fallback).
        """
        if not torch.cuda.is_available():
            self._capture_error = "CUDA not available"
            self._captured = False
            return False

        if backbone_template.shape[0] != self.batch_size:
            self._capture_error = (
                f"backbone batch {backbone_template.shape[0]} != graph batch {self.batch_size}"
            )
            self._captured = False
            return False

        flow = self.model.flow_lm
        device = backbone_template.device
        dtype = backbone_template.dtype
        b = self.batch_size
        self._lsd_steps = int(
            lsd_decode_steps if lsd_decode_steps is not None else self.model.lsd_decode_steps
        )

        empty_tokens = torch.zeros((b, 0), dtype=torch.int64, device=device)
        empty_emb = flow.conditioner(TokenizedText(empty_tokens))
        empty_cond = torch.empty((b, 0, flow.dim), dtype=dtype, device=device)
        # AR post-prompt: empty text + empty audio conditioning → [B, 0, dim]
        empty_emb = torch.cat([empty_emb, empty_cond], dim=1)

        self._static_seq = backbone_template.detach().clone()
        self._static_empty_emb = empty_emb.detach().clone()

        # Static temp / eos on device (graph-safe; filled each step for batch)
        if temperature is None:
            temperature = self.model.temp
        if eos_threshold is None:
            eos_threshold = self.model.eos_threshold
        self._static_temp = self._as_batch_param(temperature, b, device, dtype)
        self._static_eos_thr = self._as_batch_param(eos_threshold, b, device, dtype)

        # Owned static state — graph binds permanently to these tensor addresses
        self._static_state = clone_flow_state(model_state)
        snap = clone_flow_state(self._static_state)
        try:
            warm_state = clone_flow_state(self._static_state)
            with torch.inference_mode():
                flow._sample_next_latent(
                    self._static_seq,
                    self._static_empty_emb,
                    warm_state,
                    self._lsd_steps,
                    self._static_temp,
                    self.model.noise_clamp,
                    self._static_eos_thr,
                )
            torch.cuda.synchronize()

            self.graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(self.graph):
                lat, eos = flow._sample_next_latent(
                    self._static_seq,
                    self._static_empty_emb,
                    self._static_state,
                    self._lsd_steps,
                    self._static_temp,
                    self.model.noise_clamp,
                    self._static_eos_thr,
                )
                self._static_lat = lat
                self._static_eos = eos
            torch.cuda.synchronize()

            # Restore post-prompt contents for optional immediate AR (same unit)
            reset_flow_state(self._static_state, snap)
            self._captured = True
            self._capture_error = None
            self.prepare_count = 0
            logger.info(
                "FlowLM CUDA graph captured for AR step (B=%s, lsd=%s)",
                self.batch_size,
                self._lsd_steps,
            )
            return True
        except Exception as exc:
            reset_flow_state(self._static_state, snap)
            self.graph = None
            self._captured = False
            self._capture_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "FlowLM CUDA graph capture failed (B=%s); using eager: %s",
                self.batch_size,
                self._capture_error,
            )
            return False

    def prepare(self, live_state: dict[str, dict[str, torch.Tensor]]) -> None:
        """Copy post-prompt live state into static buffers before AR (no re-capture).

        Args:
            live_state: Fresh post-prompt FlowLM state for this generation unit.

        Raises:
            RuntimeError: If graph is not available or shapes are incompatible.
        """
        if not self.available or self._static_state is None:
            raise RuntimeError("FlowLMStepGraph.prepare called without successful capture")
        if not states_compatible(self._static_state, live_state):
            raise RuntimeError(
                "Live state shapes incompatible with captured static state; need re-capture"
            )
        reset_flow_state(self._static_state, live_state)
        self.prepare_count += 1

    def step(
        self,
        backbone_input: torch.Tensor,
        temperature: float | torch.Tensor | None = None,
        eos_threshold: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay one AR step on owned static state; returns (latent [B,1,ldim], is_eos).

        Args:
            backbone_input: Current backbone input [B, 1, ldim].
            temperature: Optional per-step temp override (copied into static buffer).
            eos_threshold: Optional per-step eos override.

        Returns:
            next_latent [B, 1, ldim], is_eos.

        Raises:
            RuntimeError: If no graph is available.
        """
        if not self.available:
            raise RuntimeError("FlowLMStepGraph.step called without successful capture")

        assert self._static_state is not None
        assert self._static_seq is not None
        assert self._static_lat is not None
        assert self._static_eos is not None
        assert self._static_temp is not None
        assert self._static_eos_thr is not None
        assert self.graph is not None

        if backbone_input.shape[0] != self.batch_size:
            raise RuntimeError(
                f"backbone batch {backbone_input.shape[0]} != graph batch {self.batch_size}"
            )

        self._static_seq.copy_(backbone_input)
        if temperature is not None:
            self._static_temp.copy_(
                self._as_batch_param(
                    temperature,
                    self.batch_size,
                    self._static_temp.device,
                    self._static_temp.dtype,
                )
            )
        if eos_threshold is not None:
            self._static_eos_thr.copy_(
                self._as_batch_param(
                    eos_threshold,
                    self.batch_size,
                    self._static_eos_thr.device,
                    self._static_eos_thr.dtype,
                )
            )

        self.graph.replay()
        lat = self._static_lat.detach()
        if lat.ndim == 2:
            next_latent = lat[:, None, :].clone()
        else:
            next_latent = lat.clone()
        is_eos = self._static_eos.detach().clone()
        # Empty text + 1 backbone frame + empty audio conditioning
        increment_steps(self.model.flow_lm, self._static_state, increment=1)
        return next_latent, is_eos

    @staticmethod
    def _as_batch_param(
        value: float | torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a length-B device tensor for temp/eos (graph-safe storage)."""
        if isinstance(value, torch.Tensor):
            t = value.detach().to(device=device, dtype=dtype).reshape(-1)
            if t.numel() == 1 and batch_size > 1:
                t = t.expand(batch_size).clone()
            elif t.numel() != batch_size:
                raise ValueError(f"param length {t.numel()} != batch_size {batch_size}")
            return t.contiguous()
        t = torch.empty(batch_size, device=device, dtype=dtype)
        t.fill_(float(value))
        return t
