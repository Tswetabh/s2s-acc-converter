#!/usr/bin/env python3
"""T2: single-process generate_audio microbench (eager vs CUDA graphs)."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pocket_tts.models.tts_model import TTSModel

TEXTS = [
    "Hello, this is a short graph AR test.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "Doctor Who traveled through time and space with his companions.",
    "Please confirm the system is ready for the next chapter.",
    "Rain fell softly on the metal roof of the old TARDIS shell.",
]


def main() -> None:
    """Time N generate_audio calls with graphs off then on."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    model = TTSModel.load_model(device="cuda")
    model.temp = 0.7
    voice = model.get_state_for_audio_prompt("alba", truncate=True)

    def run_suite(graphs: bool) -> dict:
        """Runs a suite of audio generation tests with optional CUDA graph support and measures execution times."""
        model.cuda_graphs_enabled = graphs
        # warmup
        model.generate_audio(voice, TEXTS[0], frames_after_eos=2, copy_state=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times = []
        samples = []
        for text in TEXTS:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            audio = model.generate_audio(voice, text, frames_after_eos=2, copy_state=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            samples.append(int(audio.numel()))
            if not torch.isfinite(audio).all():
                raise RuntimeError(f"non-finite audio graphs={graphs}")
        return {
            "cuda_graphs": graphs,
            "n": len(TEXTS),
            "seconds": times,
            "sum_seconds": sum(times),
            "mean_seconds": sum(times) / len(times),
            "samples": samples,
        }

    off = run_suite(False)
    on = run_suite(True)
    speedup = off["sum_seconds"] / on["sum_seconds"] if on["sum_seconds"] else None
    result = {
        "phase": "T2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "eager": off,
        "graphs": on,
        "speedup_total": speedup,
        "time_reduction_pct": (
            (off["sum_seconds"] - on["sum_seconds"]) / off["sum_seconds"] * 100.0
            if off["sum_seconds"]
            else None
        ),
    }
    print(json.dumps(result, indent=2))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = PROJECT_ROOT / "tests" / "results" / f"cuda_graph_t2_{stamp}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"T2 SUMMARY: eager_sum={off['sum_seconds']:.3f}s graphs_sum={on['sum_seconds']:.3f}s "
        f"speedup={speedup:.2f}x reduction={result['time_reduction_pct']:.1f}%",
        file=sys.stderr,
    )
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
