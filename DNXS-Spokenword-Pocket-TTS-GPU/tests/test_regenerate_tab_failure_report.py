"""Regression coverage for Regenerate-tab ASR failure report parsing."""

import json
import tempfile
import unittest
from pathlib import Path

from pocket_tts.asr_failure_reports import (
    select_failure_report,
    write_failure_manifest,
)
from pocket_tts.gui.regenerate_tab import parse_asr_failures_json


class RegenerateTabFailureReportTests(unittest.TestCase):
    """Verify GUI adapter accepts the active and legacy ASR report schemas."""

    def test_parses_new_medium_confirmed_failure_record(self) -> None:
        """Map nested final Medium evidence into a GUI failure row."""
        payload = {
            "report_type": "new_medium_confirmed_failures_v2",
            "records": [
                {
                    "chunk_index": 76,
                    "reference": {"text_raw": "Expected words."},
                    "stage_one": {"transcript_raw": "Stage one words."},
                    "medium": {
                        "transcript_raw": "Medium words.",
                        "score": 0.25,
                        "classification": "FAIL",
                        "explanation": "Missing expected words.",
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "asr_new_medium_failures.json"
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            failures = parse_asr_failures_json(report_path)

        row = failures["chunk_00076"]
        self.assertEqual(row["original_text"], "Expected words.")
        self.assertEqual(row["transcribed_text"], "Medium words.")
        self.assertEqual(row["status"], "FAIL | Score: 0.250")
        self.assertEqual(row["explanation"], "Missing expected words.")

    def test_keeps_legacy_failure_list_compatible(self) -> None:
        """Continue accepting existing list-shaped ASR reports during migration."""
        payload = [{"chunk_index": 4, "original_text": "Expected.", "score": 0.5}]
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "asr_failures.json"
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            failures = parse_asr_failures_json(report_path)

        self.assertEqual(failures["chunk_00004"]["original_text"], "Expected.")
        self.assertEqual(failures["chunk_00004"]["status"], "FAIL | Score: 0.500")

    def test_parses_legacy_medium_confirmed_failure_record(self) -> None:
        """Map legacy Medium comparison evidence into the shared GUI row shape."""
        payload = {
            "report_type": "medium_asr_confirmed_failures",
            "records": [
                {
                    "chunk_index": 9,
                    "comparison": {
                        "ref_text_raw": "Expected words.",
                        "hyp_text_raw": "Different words.",
                    },
                    "original_failure": {"score": 0.5, "explanation": "Stage 1 mismatch."},
                    "medium_result": {
                        "score": 0.25,
                        "classification": "FAIL",
                        "explanation": "Medium confirmed mismatch.",
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "asr_medium_validation_failed.json"
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            failures = parse_asr_failures_json(report_path)

        row = failures["chunk_00009"]
        self.assertEqual(row["original_text"], "Expected words.")
        self.assertEqual(row["transcribed_text"], "Different words.")
        self.assertEqual(row["status"], "FAIL | Score: 0.250")
        self.assertEqual(row["explanation"], "Medium confirmed mismatch.")

    def test_discovers_final_legacy_medium_before_stage_one_reports(self) -> None:
        """Prefer a confirmed legacy result over lower-confidence candidate logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "asr_failures.json").write_text("[]", encoding="utf-8")
            expected = folder / "asr_medium_validation_failed.json"
            expected.write_text('{"records": []}', encoding="utf-8")
            selection = select_failure_report(folder)

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.path, expected)
        self.assertEqual(selection.kind, "medium_confirmed_failures")

    def test_discovers_each_historical_report_convention(self) -> None:
        """Keep every pre-manifest pipeline report selectable by its own name."""
        cases = (
            ("asr_new_medium_failures.json", "medium_confirmed_failures"),
            ("asr_medium_validation_failed.json", "medium_confirmed_failures"),
            ("asr_new_failures.json", "stage_one_candidates"),
            ("asr_failures.json", "stage_one_candidates"),
        )
        for filename, expected_kind in cases:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp_dir:
                folder = Path(temp_dir)
                expected = folder / filename
                expected.write_text("[]", encoding="utf-8")
                selection = select_failure_report(folder)

                self.assertIsNotNone(selection)
                assert selection is not None
                self.assertEqual(selection.path, expected)
                self.assertEqual(selection.kind, expected_kind)

    def test_manifest_overrides_filename_priority_for_its_run(self) -> None:
        """Use a current run manifest instead of an unrelated old New report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "asr_new_medium_failures.json").write_text(
                '{"records": []}', encoding="utf-8"
            )
            expected = folder / "asr_medium_validation_failed.json"
            expected.write_text('{"records": []}', encoding="utf-8")
            write_failure_manifest(
                folder,
                pipeline="legacy",
                authoritative_filename=expected.name,
                label="Legacy Medium-confirmed failures",
                kind="medium_confirmed_failures",
            )
            selection = select_failure_report(folder)

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.path, expected)
        self.assertEqual(selection.source, "run manifest")

    def test_manifest_with_missing_authoritative_report_fails_closed(self) -> None:
        """Do not fall back to stale report names when current metadata is broken."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "asr_new_medium_failures.json").write_text(
                '{"records": []}', encoding="utf-8"
            )
            write_failure_manifest(
                folder,
                pipeline="legacy",
                authoritative_filename="asr_medium_validation_failed.json",
                label="Legacy Medium-confirmed failures",
                kind="medium_confirmed_failures",
            )

            selection = select_failure_report(folder)

        self.assertIsNone(selection)
