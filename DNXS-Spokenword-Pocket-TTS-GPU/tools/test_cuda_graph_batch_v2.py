#!/usr/bin/env python3
"""v2 correctness: generate_audio_batch eager vs CUDA graph (B=2 and B=4)."""

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

TEXTS_B2 = [
    "Hello world, short batch item one.",
    "Hello world, short batch item two.",
]
# Four equal-ish short lines for B=4 (token lengths need not match - we'll pad by choosing same text)
TEXTS_B4 = [
    "Batch four item alpha here now.",
    "Batch four item alpha here now.",
    "Batch four item alpha here now.",
    "Batch four item alpha here now.",
]


def _requests(texts: list[str], temp: float = 0.0) -> list[dict]:
    """Build generate_audio_batch request dicts with shared EOS/LSD settings."""
    return [
        {
            "text": t,
            "temperature": temp,
            "frames_after_eos": 2,
            "eos_threshold": -3.0,
            "lsd_decode_steps": 2,
        }
        for t in texts
    ]


def _run_pair(model: TTSModel, voice: dict, texts: list[str], label: str) -> dict:
    """Compare eager vs graph batch generation for one batch size."""
    reqs = _requests(texts, temp=0.0)
    model.cuda_graphs_enabled = False
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_e = model.generate_audio_batch(voice, reqs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_e = time.perf_counter() - t0

    model.cuda_graphs_enabled = True
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_g = model.generate_audio_batch(voice, reqs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_g = time.perf_counter() - t0
    model.cuda_graphs_enabled = False

    finite_e = all(torch.isfinite(a).all().item() for a in out_e)
    finite_g = all(torch.isfinite(a).all().item() for a in out_g)
    lens_e = [int(a.numel()) for a in out_e]
    lens_g = [int(a.numel()) for a in out_g]
    # temp=0 should be deterministic enough for equal lengths
    lengths_match = lens_e == lens_g
    max_diff = 0.0
    if lengths_match and finite_e and finite_g:
        for a, b in zip(out_e, out_g):
            max_diff = max(max_diff, float((a - b).abs().max().item()))

    ok = finite_e and finite_g and lengths_match and max_diff < 1e-3
    return {
        "label": label,
        "batch_size": len(texts),
        "passed": ok,
        "eager_seconds": t_e,
        "graph_seconds": t_g,
        "speedup": (t_e / t_g) if t_g > 0 else None,
        "eager_lengths": lens_e,
        "graph_lengths": lens_g,
        "lengths_match": lengths_match,
        "finite_eager": finite_e,
        "finite_graph": finite_g,
        "max_abs_diff": max_diff,
    }


def main() -> None:
    """Run B=2 and B=4 batch graph correctness+timing checks."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    model = TTSModel.load_model(device="cuda")
    model.temp = 0.0
    voice = model.get_state_for_audio_prompt("alba", truncate=True)

    cases = [
        _run_pair(model, voice, TEXTS_B2, "B2"),
        _run_pair(model, voice, TEXTS_B4, "B4"),
    ]
    # Second pass for timing stability (capture amortized)
    cases.append(_run_pair(model, voice, TEXTS_B4, "B4_repeat"))

    result = {
        "phase": "v2_batch_graph",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cases": cases,
        "all_passed": all(c["passed"] for c in cases),
    }
    print(json.dumps(result, indent=2))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = PROJECT_ROOT / "tests" / "results" / f"cuda_graph_v2_{stamp}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)
    print(
        "v2 SUMMARY: "
        + ", ".join(
            f"{c['label']}={'PASS' if c['passed'] else 'FAIL'} "
            f"({c['eager_seconds']:.3f}s->{c['graph_seconds']:.3f}s)"
            for c in cases
        ),
        file=sys.stderr,
    )
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
