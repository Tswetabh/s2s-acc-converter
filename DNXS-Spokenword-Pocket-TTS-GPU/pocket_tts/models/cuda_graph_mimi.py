"""CUDA graph capture/replay for one Mimi quantize+decode frame (batch AR).

Same reuse pattern as FlowLMStepGraph: own static state, capture once per
batch_size, prepare() at generation start, step() each AR frame.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pocket_tts.models.cuda_graph_flow_lm import clone_flow_state, reset_flow_state, states_compatible
from pocket_tts.modules.stateful_module import increment_steps

logger = logging.getLogger(__name__)


class MimiDecodeGraph:
    """Capture one Mimi frame decode; replay with static latent + owned state."""

    def __init__(self, model: Any, batch_size: int = 1):
        """Bind to TTSModel exposing mimi, flow_lm.emb_mean/std."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model = model
        self.batch_size = int(batch_size)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._static_state: dict | None = None
        self._static_latent: torch.Tensor | None = None
        self._static_frame: torch.Tensor | None = None
        self._captured = False
        self._capture_error: str | None = None
        self.prepare_count = 0

    @property
    def available(self) -> bool:
        """True when graph capture succeeded."""
        return self._captured and self.graph is not None and self._static_state is not None

    @property
    def static_state(self) -> dict:
        """Owned Mimi streaming state mutated by replay."""
        if self._static_state is None:
            raise RuntimeError("MimiDecodeGraph has no static_state; capture first")
        return self._static_state

    def capture(
        self,
        mimi_state: dict,
        latent_template: torch.Tensor,
    ) -> bool:
        """Capture quantize+decode+increment on owned static buffers.

        Args:
            mimi_state: Template state (usually fresh init); not mutated.
            latent_template: Shape [B, 1, ldim] matching batch_size.

        Returns:
            True on success.
        """
        if not torch.cuda.is_available():
            self._capture_error = "CUDA not available"
            self._captured = False
            return False
        if latent_template.shape[0] != self.batch_size:
            self._capture_error = (
                f"latent batch {latent_template.shape[0]} != graph batch {self.batch_size}"
            )
            self._captured = False
            return False

        self._static_latent = latent_template.detach().clone()
        self._static_state = clone_flow_state(mimi_state)
        snap = clone_flow_state(self._static_state)
        flow = self.model.flow_lm
        mimi = self.model.mimi

        def _body(lat: torch.Tensor, state: dict) -> torch.Tensor:
            """Generates a frame from latent input using a flow model and updates state."""
            mimi_input = lat * flow.emb_std + flow.emb_mean
            quantized = mimi.quantizer(mimi_input.transpose(-1, -2))
            frame = mimi.decode_from_latent(quantized, state)
            increment_steps(mimi, state, increment=16)
            return frame

        try:
            warm_state = clone_flow_state(self._static_state)
            with torch.inference_mode():
                _body(self._static_latent, warm_state)
            torch.cuda.synchronize()

            self.graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(self.graph):
                self._static_frame = _body(self._static_latent, self._static_state)
            torch.cuda.synchronize()

            reset_flow_state(self._static_state, snap)
            self._captured = True
            self._capture_error = None
            self.prepare_count = 0
            logger.info("Mimi CUDA graph captured for decode step (B=%s)", self.batch_size)
            return True
        except Exception as exc:
            reset_flow_state(self._static_state, snap)
            self.graph = None
            self._captured = False
            self._capture_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Mimi CUDA graph capture failed (B=%s); eager decode: %s",
                self.batch_size,
                self._capture_error,
            )
            return False

    def prepare(self, live_state: dict) -> None:
        """Copy generation-start Mimi state into static buffers before AR."""
        if not self.available or self._static_state is None:
            raise RuntimeError("MimiDecodeGraph.prepare without successful capture")
        if not states_compatible(self._static_state, live_state):
            raise RuntimeError("Mimi live state incompatible with captured static state")
        reset_flow_state(self._static_state, live_state)
        self.prepare_count += 1

    def step(self, latent: torch.Tensor) -> torch.Tensor:
        """Replay one Mimi decode; returns audio frame [B, C, T]."""
        if not self.available:
            raise RuntimeError("MimiDecodeGraph.step without successful capture")
        assert self._static_latent is not None
        assert self._static_frame is not None
        assert self.graph is not None
        if latent.shape[0] != self.batch_size:
            raise RuntimeError(
                f"latent batch {latent.shape[0]} != graph batch {self.batch_size}"
            )
        self._static_latent.copy_(latent)
        self.graph.replay()
        return self._static_frame.detach().clone()
