"""Regression coverage for ASR Stage 1 and Stage 2 elapsed-time handoff."""

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_asr_pipeline


class AsrStageTimingTests(unittest.TestCase):
    """Ensure all parallel and streaming timing paths retain measured durations."""

    def test_streaming_stage_one_report_carries_materialized_wall_time(self):
        """Use the durable streaming Stage 1 duration in the New-pipeline report."""
        audit = {
            "summary": {
                "submitted": 2,
                "completed": 2,
                "failed": 1,
                "complete": True,
                "wall_s": 12.375,
            },
            "candidate_count": 1,
            "records": [
                {"backend": "faster_whisper", "model": "base"},
            ],
        }
        failure_path = ROOT / "tests" / "results" / "unused_asr_failures.json"
        stage_one, process, candidates = run_asr_pipeline._build_streaming_stage_one_report(
            audit,
            failure_path,
        )

        self.assertEqual(candidates, 1)
        self.assertEqual(stage_one["wall_s"], 12.375)
        self.assertEqual(process["wall_s"], 12.375)

    def test_parallel_asr_closure_updates_outer_stage_timers(self):
        """Keep Stage 1 and Stage 2 assignments bound to final generation result fields."""
        module = ast.parse(
            (ROOT / "pocket_tts" / "audiobook" / "generator.py").read_text(encoding="utf-8")
        )
        target = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef)
            and node.name == "prepare_parallel_regeneration_tasks"
        )
        declared = {
            name
            for statement in target.body
            if isinstance(statement, ast.Nonlocal)
            for name in statement.names
        }

        self.assertTrue({"asr_stage_one_time", "asr_stage_two_time"}.issubset(declared))
