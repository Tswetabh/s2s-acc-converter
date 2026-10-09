#!/usr/bin/env python3
"""T0 probe: CUDA graph capture for one fixed-shape FlowLM latent step.

Compares eager vs graph replay timing and correctness (temp=0).
Archives machine-readable JSON (stdout). CUDA only.

Examples:
  venv/bin/python tools/probe_cuda_graph.py
  venv/bin/python tools/probe_cuda_graph.py --iters 10 --multi-step 4
  venv/bin/python tools/probe_cuda_graph.py --output tests/results/cuda_graph_t0.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.models.tts_model import TTSModel


PROBE_TEXT = "A fixed-shape CUDA graph probe tests one latent generation step."


def _clone_state(state: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    """Clone every tensor in a model state for independent graph/eager tests."""
    return {
        module_name: {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in module_state.items()
        }
        for module_name, module_state in state.items()
    }


def _reset_state(
    target: dict[str, dict[str, torch.Tensor]],
    source: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Restore graph-owned cache tensors from a clean state snapshot."""
    for module_name, module_state in source.items():
        for key, value in module_state.items():
            if isinstance(value, torch.Tensor):
                target[module_name][key].copy_(value)


def _prepare_inputs(model: TTSModel) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare fixed batch-one sequence and text embeddings for capture."""
    prepared = model.flow_lm.conditioner.prepare(PROBE_TEXT)
    text_embeddings = model.flow_lm.conditioner(TokenizedText(prepared.tokens))
    sequence = torch.full(
        (1, 1, model.flow_lm.ldim),
        float("nan"),
        device=model.device,
        dtype=model.flow_lm.dtype,
    )
    return sequence, text_embeddings


def _one_step(
    model: TTSModel,
    sequence: torch.Tensor,
    text_embeddings: torch.Tensor,
    state: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one FlowLM latent step with current model settings."""
    return model.flow_lm._sample_next_latent(
        sequence,
        text_embeddings,
        state,
        model.lsd_decode_steps,
        model.temp,
        model.noise_clamp,
        model.eos_threshold,
    )


def _sync() -> None:
    """Wait for CUDA so timings include GPU work."""
    torch.cuda.synchronize()


def _time_eager_steps(
    model: TTSModel,
    sequence: torch.Tensor,
    text_embeddings: torch.Tensor,
    base_state: dict[str, Any],
    iters: int,
) -> list[float]:
    """Time eager single steps, each from a fresh clone of base_state."""
    times: list[float] = []
    for _ in range(iters):
        state = _clone_state(base_state)
        _sync()
        started = time.perf_counter()
        with torch.inference_mode():
            _one_step(model, sequence, text_embeddings, state)
        _sync()
        times.append(time.perf_counter() - started)
    return times


def main() -> None:
    """Capture and replay one fixed FlowLM step; compare eager timing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations per mode")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup steps before timing")
    parser.add_argument(
        "--multi-step",
        type=int,
        default=0,
        help="If >0, also try N sequential steps on same state (eager + graph stress)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write JSON result path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for graph probe")

    torch_version = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    gpu_name = torch.cuda.get_device_name(0)

    model = TTSModel.load_model(device="cuda")
    model.temp = 0.0
    base_state = model.get_state_for_audio_prompt("alba", truncate=True)
    sequence, text_embeddings = _prepare_inputs(model)

    result: dict[str, Any] = {
        "phase": "T0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": gpu_name,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "lsd_decode_steps": int(model.lsd_decode_steps),
        "temp": float(model.temp),
        "iters": args.iters,
        "warmup": args.warmup,
    }

    # --- Eager baseline: first-step from clean state ---
    with torch.inference_mode():
        for _ in range(args.warmup):
            _one_step(model, sequence, text_embeddings, _clone_state(base_state))
    eager_times = _time_eager_steps(
        model, sequence, text_embeddings, base_state, args.iters
    )
    result["eager_step_seconds"] = eager_times
    result["eager_step_mean_seconds"] = sum(eager_times) / len(eager_times)
    result["eager_step_mean_ms"] = result["eager_step_mean_seconds"] * 1000.0

    # Correctness reference: one eager step from base
    eager_state = _clone_state(base_state)
    with torch.inference_mode():
        eager_latent, eager_eos = _one_step(model, sequence, text_embeddings, eager_state)
    result["eager_output_finite"] = bool(torch.isfinite(eager_latent).all().item())
    result["eager_output_shape"] = list(eager_latent.shape)

    # --- Graph path: warmup advanced state then capture (matches prior probe pattern) ---
    graph_state = _clone_state(base_state)
    graph_initial_state = _clone_state(graph_state)
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                _one_step(model, sequence, text_embeddings, graph_state)
        _sync()

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            captured_latent, captured_eos = _one_step(
                model, sequence, text_embeddings, graph_state
            )
        _sync()

        # Replay from clean initial state (same as production reset pattern)
        _reset_state(graph_state, graph_initial_state)
        graph.replay()
        _sync()
        replay_latent = captured_latent.detach().clone()
        replay_eos = captured_eos.detach().clone()

        _reset_state(graph_state, graph_initial_state)
        graph.replay()
        _sync()
        second_replay_latent = captured_latent.detach().clone()
        second_replay_eos = captured_eos.detach().clone()

        replay_times: list[float] = []
        for _ in range(args.iters):
            _reset_state(graph_state, graph_initial_state)
            _sync()
            started = time.perf_counter()
            graph.replay()
            _sync()
            replay_times.append(time.perf_counter() - started)

        result["success"] = True
        result["capture_output_shape"] = list(captured_latent.shape)
        result["eos_shape"] = list(captured_eos.shape)
        result["replay_seconds"] = replay_times
        result["replay_mean_seconds"] = sum(replay_times) / len(replay_times)
        result["replay_mean_ms"] = result["replay_mean_seconds"] * 1000.0
        result["output_finite"] = bool(torch.isfinite(replay_latent).all().item())
        result["max_reset_replay_diff"] = float(
            (replay_latent - second_replay_latent).abs().max().item()
        )
        result["max_eager_diff"] = float((eager_latent - replay_latent).abs().max().item())
        result["eos_reset_equal"] = bool(torch.equal(replay_eos, second_replay_eos))
        result["eos_eager_equal"] = bool(torch.equal(eager_eos, replay_eos))
        if result["eager_step_mean_seconds"] > 0:
            result["speedup_eager_over_graph"] = (
                result["eager_step_mean_seconds"] / result["replay_mean_seconds"]
            )
            result["step_time_reduction_pct"] = (
                (result["eager_step_mean_seconds"] - result["replay_mean_seconds"])
                / result["eager_step_mean_seconds"]
                * 100.0
            )
        else:
            result["speedup_eager_over_graph"] = None
            result["step_time_reduction_pct"] = None

        # Optional multi-step stress: advance state without reset (exposes cache growth issues)
        if args.multi_step > 0:
            multi: dict[str, Any] = {"steps": args.multi_step}
            ms_state = _clone_state(base_state)
            eager_latents = []
            with torch.inference_mode():
                for i in range(args.multi_step):
                    lat, eos = _one_step(model, sequence, text_embeddings, ms_state)
                    eager_latents.append((lat.detach().clone(), eos.detach().clone()))
            multi["eager_completed"] = True

            # Graph was captured at a single cache depth; sequential replay without
            # re-capture is expected to be wrong or unsafe — record outcome.
            g_state = _clone_state(base_state)
            _reset_state(g_state, graph_initial_state)
            graph_latents = []
            multi_ok = True
            multi_error = None
            try:
                with torch.inference_mode():
                    for i in range(args.multi_step):
                        # Only first step matches capture depth if capture was after warmup
                        # Here we reset each time for fixed-shape single-step graph validity.
                        _reset_state(g_state, graph_initial_state)
                        graph.replay()
                        _sync()
                        graph_latents.append(
                            (captured_latent.detach().clone(), captured_eos.detach().clone())
                        )
                multi["graph_reset_each_step_completed"] = True
                multi["max_diff_step0"] = float(
                    (eager_latents[0][0] - graph_latents[0][0]).abs().max().item()
                )
            except Exception as exc:
                multi_ok = False
                multi_error = f"{type(exc).__name__}: {exc}"
                multi["graph_reset_each_step_completed"] = False
            multi["success"] = multi_ok
            if multi_error:
                multi["error"] = multi_error
            result["multi_step"] = multi

    except Exception as exc:
        result["success"] = False
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)

    text = json.dumps(result, indent=2)
    print(text)

    out = args.output
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = PROJECT_ROOT / "tests" / "results" / f"cuda_graph_t0_{stamp}.json"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nWrote {out}", file=sys.stderr)

    if not result.get("success"):
        raise SystemExit(1)

    # Soft gate: print summary for humans
    print(
        f"\nT0 SUMMARY: eager={result.get('eager_step_mean_ms', 0):.3f} ms/step  "
        f"graph={result.get('replay_mean_ms', 0):.3f} ms/step  "
        f"speedup={result.get('speedup_eager_over_graph', 0):.2f}x  "
        f"eager_diff={result.get('max_eager_diff')}  "
        f"reset_diff={result.get('max_reset_replay_diff')}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
