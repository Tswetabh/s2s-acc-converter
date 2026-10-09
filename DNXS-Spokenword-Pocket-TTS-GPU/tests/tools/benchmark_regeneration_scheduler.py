#!/usr/bin/env python3
"""Assemble a regeneration benchmark report from stored benchmark evidence.

This tool does not run a new book-sized benchmark. It reads the repo's existing
benchmark/probe JSON files, ranks recovery strategies, and writes one concise
JSON report that the GUI or a human can inspect.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pocket_tts.regeneration_scheduler import (  # noqa: E402
    choose_regeneration_plan,
    format_plan_summary,
    load_benchmark_matrix,
)


def parse_args() -> argparse.Namespace:
    """Parse benchmark report inputs and output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui-model", default="base", help="GUI-selected ASR model.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "tests" / "results",
        help="Folder containing benchmark JSON evidence.",
    )
    parser.add_argument(
        "--tts-workers",
        type=int,
        default=3,
        help="Fallback TTS worker count when evidence does not specify one.",
    )
    parser.add_argument(
        "--tts-batch-size",
        type=int,
        default=4,
        help="Fallback TTS batch size when evidence does not specify one.",
    )
    parser.add_argument(
        "--asr-workers",
        type=int,
        default=1,
        help="Fallback Medium-ASR worker count when evidence does not specify one.",
    )
    parser.add_argument(
        "--asr-batch-size",
        type=int,
        default=1,
        help="Fallback Medium-ASR batch size when evidence does not specify one.",
    )
    parser.add_argument(
        "--safe-vram-mb",
        type=float,
        default=None,
        help="Optional hard VRAM ceiling for rejecting unsafe strategies.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tests" / "results" / "regeneration_scheduler",
        help="Directory for the generated report.",
    )
    return parser.parse_args()


def main() -> int:
    """Read benchmark JSON, choose plan, and emit one timestamped report."""
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    matrix = load_benchmark_matrix(results_root)
    plan = choose_regeneration_plan(
        args.gui_model,
        results_root=results_root,
        requested_tts_workers=args.tts_workers,
        requested_tts_batch_size=args.tts_batch_size,
        requested_asr_workers=args.asr_workers,
        requested_asr_batch_size=args.asr_batch_size,
        safe_vram_mb=args.safe_vram_mb,
    )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir.expanduser().resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    payload = {
        "run_id": run_id,
        "results_root": str(results_root),
        "candidate_count": len(matrix),
        "plan": plan.to_dict(),
        "plan_summary": format_plan_summary(plan),
        "matrix": [row.__dict__ for row in matrix],
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Evidence: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
