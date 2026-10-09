#!/usr/bin/env python3
"""Regression tests for regeneration scheduler and benchmark reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pocket_tts.regeneration_scheduler import (
    choose_regeneration_plan,
    format_plan_summary,
    load_benchmark_matrix,
    model_rank,
    should_verify_failed_chunks,
)


class TestModelRanking(unittest.TestCase):
    """Model size ranking must treat Medium as the verification floor."""

    def test_medium_floor(self) -> None:
        """Base and small need verification; Medium and above do not."""
        self.assertTrue(should_verify_failed_chunks("base"))
        self.assertTrue(should_verify_failed_chunks("small"))
        self.assertFalse(should_verify_failed_chunks("medium"))
        self.assertFalse(should_verify_failed_chunks("large"))
        self.assertLess(model_rank("base"), model_rank("medium"))


class TestBenchmarkMatrix(unittest.TestCase):
    """Matrix loader must parse existing evidence into selectable candidates."""

    def test_only_comparable_safe_recovery_evidence_is_selectable(self) -> None:
        """Full-book overlap rows and unmeasured-resident rows must not drive recovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_dir = root / "asr_overlap_pipeline" / "20260101-000000"
            result_dir.mkdir(parents=True)
            payload = {
                "run_id": "20260101-000000",
                "scenarios": [
                    {
                        "scenario": "postgen_gpu",
                        "pipeline_wall_s": 12.5,
                        "tts_workers": 3,
                        "asr_workers": 4,
                        "asr_model": "base",
                    },
                    {
                        "scenario": "gpu_overlap",
                        "pipeline_wall_s": 15.0,
                        "tts_workers": 3,
                        "asr_workers": 4,
                        "asr_model": "medium",
                    },
                ],
            }
            (result_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

            matrix = load_benchmark_matrix(root)
            self.assertEqual(len(matrix), 0)

            plan = choose_regeneration_plan(
                "base",
                results_root=root,
                requested_tts_workers=2,
                requested_tts_batch_size=4,
                requested_asr_workers=1,
                requested_asr_batch_size=1,
                safe_vram_mb=2000.0,
            )
            self.assertEqual(plan.chosen_strategy, "fallback_sequential")
            self.assertEqual(plan.tts_workers, 1)
            self.assertEqual(plan.asr_workers, 1)
            self.assertTrue(plan.verification_required)
            summary = format_plan_summary(plan)
            self.assertTrue(any("Strategy: fallback_sequential" in line for line in summary))
            self.assertTrue(any("verification" in line.lower() for line in summary))

    def test_resident_plan_requires_peak_vram_measurement(self) -> None:
        """Resident TTS plus Medium ASR cannot be selected without peak VRAM evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_dir = root / "resident_asr_recovery" / "20260101-000000"
            result_dir.mkdir(parents=True)
            payload = {
                "total_s": 4.0,
                "rows": [],
                "regenerated_rows": 1,
                "keep_tts_resident": True,
            }
            (result_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

            plan = choose_regeneration_plan("base", results_root=root)

            self.assertEqual(plan.chosen_strategy, "fallback_sequential")
            self.assertIn("missing_peak_vram", " ".join(plan.rejected_strategies))


if __name__ == "__main__":
    unittest.main()
