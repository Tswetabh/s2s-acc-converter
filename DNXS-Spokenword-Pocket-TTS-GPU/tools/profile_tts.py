"""Profile warm Pocket TTS generation in one process with CPU/CUDA traces."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pocket_tts.models.tts_model import TTSModel


PROFILE_TEXT = (
    "Profiler workload exercises text preparation, autoregressive latent "
    "generation, and Mimi audio decoding with a stable voice prompt."
)


def _synchronize() -> None:
    """Wait for pending CUDA work before taking a wall-clock measurement."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_generation(model: TTSModel, voice_state: dict[str, Any], text: str) -> float:
    """Generate one sample and return elapsed synchronized wall-clock seconds."""
    _synchronize()
    started = time.perf_counter()
    with torch.profiler.record_function("pocket_tts.generate_audio"):
        model.generate_audio(voice_state, text, frames_after_eos=2)
    _synchronize()
    return time.perf_counter() - started


def _profile_rows(profiler: torch.profiler.profile, limit: int) -> list[dict[str, Any]]:
    """Convert highest CUDA-cost profiler events into JSON-compatible rows."""
    rows = []
    events = profiler.key_averages()
    events = sorted(
        events,
        key=lambda event: getattr(
            event,
            "self_device_time_total",
            getattr(event, "self_cuda_time_total", 0.0),
        ),
        reverse=True,
    )
    for event in events[:limit]:
        self_device_time = getattr(
            event,
            "self_device_time_total",
            getattr(event, "self_cuda_time_total", 0.0),
        )
        device_time = getattr(
            event,
            "device_time_total",
            getattr(event, "cuda_time_total", 0.0),
        )
        self_device_memory = getattr(
            event,
            "self_device_memory_usage",
            getattr(event, "self_cuda_memory_usage", 0),
        )
        rows.append(
            {
                "name": event.key,
                "calls": event.count,
                "self_cpu_ms": event.self_cpu_time_total / 1000.0,
                "self_device_ms": self_device_time / 1000.0,
                "cpu_total_ms": event.cpu_time_total / 1000.0,
                "device_total_ms": device_time / 1000.0,
                "self_device_memory_mb": self_device_memory / (1024 * 1024),
            }
        )
    return rows


def run_profile(
    warmup_count: int,
    active_count: int,
    trace_dir: Path,
    summary_path: Path,
    top_events: int,
) -> dict[str, Any]:
    """Load Pocket TTS, warm it up, and capture CUDA operator traces."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler harness")

    trace_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    model = TTSModel.load_model(device="cuda")
    voice_state = model.get_state_for_audio_prompt("alba", truncate=True)

    warmup_times = [
        _run_generation(model, voice_state, PROFILE_TEXT)
        for _ in range(warmup_count)
    ]

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
    ) as profiler:
        active_times = []
        for _ in range(active_count):
            active_times.append(_run_generation(model, voice_state, PROFILE_TEXT))
            profiler.step()

    summary = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "warmup_count": warmup_count,
        "active_count": active_count,
        "warmup_wall_seconds": warmup_times,
        "active_wall_seconds": active_times,
        "active_wall_mean_seconds": sum(active_times) / len(active_times),
        "model_timing": model.last_generation_timing,
        "top_cuda_events": _profile_rows(profiler, top_events),
        "trace_dir": str(trace_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Parse profiler options, run capture, and print summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/pocketgpu_profile_trace"))
    parser.add_argument("--summary", type=Path, default=Path("/tmp/pocketgpu_profile.json"))
    parser.add_argument("--top-events", type=int, default=30)
    args = parser.parse_args()
    summary = run_profile(args.warmup, args.iterations, args.trace_dir, args.summary, args.top_events)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
