#!/usr/bin/env python3
"""Measure FlowLM vs Mimi share of generate_audio_batch AR steps.

Uses POCKET_TTS_BATCH_TIMING=1 (CUDA synchronize per phase for accurate GPU wall).

Usage:
  venv/bin/python tools/probe_mimi_batch_split.py
  venv/bin/python tools/probe_mimi_batch_split.py --batch-size 4 --reps 5 --graphs
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["POCKET_TTS_BATCH_TIMING"] = "1"

from pocket_tts.models.tts_model import TTSModel

TEXT = (
    "The quick brown fox jumps over the lazy dog near the river bank at dawn."
)


def _requests(batch_size: int, lsd: int = 2) -> list[dict]:
    """Build equal-length batch requests for generate_audio_batch."""
    return [
        {
            "text": TEXT,
            "temperature": 0.0,
            "frames_after_eos": 2,
            "eos_threshold": -3.0,
            "lsd_decode_steps": lsd,
        }
        for _ in range(batch_size)
    ]


def main() -> None:
    """Run FlowLM vs Mimi batch step split and print JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--graphs", action="store_true", help="Enable CUDA graphs")
    parser.add_argument("--lsd", type=int, default=2)
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for this probe", file=sys.stderr)
        raise SystemExit(2)

    model = TTSModel.load_model(device="cuda")
    model.temp = 0.0
    model.lsd_decode_steps = args.lsd
    model.cuda_graphs_enabled = bool(args.graphs)
    voice = model.get_state_for_audio_prompt(args.voice, truncate=True)
    reqs = _requests(args.batch_size, lsd=args.lsd)

    # Warmup (also captures graph if enabled)
    for _ in range(max(0, args.warmup)):
        model.generate_audio_batch(voice, reqs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    rows = []
    for i in range(args.reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = model.generate_audio_batch(voice, reqs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        timing = dict(getattr(model, "last_batch_step_timing", {}) or {})
        timing["rep"] = i
        timing["wall_s"] = wall_s
        timing["samples"] = [int(o.numel()) for o in outs]
        rows.append(timing)
        print(
            f"rep={i} wall={wall_s:.3f}s steps={timing.get('steps')} "
            f"flow%={timing.get('flow_pct', 0):.1f} mimi%={timing.get('mimi_pct', 0):.1f} "
            f"flow/step={timing.get('flow_ms_per_step', 0):.2f}ms "
            f"mimi/step={timing.get('mimi_ms_per_step', 0):.2f}ms "
            f"graph={timing.get('used_graph')}",
            flush=True,
        )

    def _mean(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        return statistics.mean(vals) if vals else None

    summary = {
        "probe": "mimi_batch_split",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_size": args.batch_size,
        "lsd": args.lsd,
        "graphs": args.graphs,
        "reps": args.reps,
        "mean_wall_s": _mean("wall_s"),
        "mean_flow_pct": _mean("flow_pct"),
        "mean_mimi_pct": _mean("mimi_pct"),
        "mean_flow_ms_per_step": _mean("flow_ms_per_step"),
        "mean_mimi_ms_per_step": _mean("mimi_ms_per_step"),
        "mean_steps": _mean("steps"),
        "rows": rows,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    out = args.output
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = PROJECT_ROOT / "tests" / "results" / f"mimi_batch_split_{stamp}.json"
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)
    print(
        f"SUMMARY B={args.batch_size} graphs={args.graphs}: "
        f"flow%={summary['mean_flow_pct']:.1f} mimi%={summary['mean_mimi_pct']:.1f} "
        f"flow/step={summary['mean_flow_ms_per_step']:.2f}ms "
        f"mimi/step={summary['mean_mimi_ms_per_step']:.2f}ms",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
