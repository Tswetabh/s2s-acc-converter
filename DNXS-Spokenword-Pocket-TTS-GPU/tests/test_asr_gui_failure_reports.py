"""Regression coverage for ASR GUI failure-report loading."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ASR"))

from asr_gui import (  # noqa: E402
    FAILURE_FILE_TYPES,
    KNOWN_FAILURE_REPORTS,
    chunk_id_from_failure_entry,
    default_failure_report,
    failure_records_from_payload,
    load_failure_chunk_ids,
)


class AsrGuiFailureReportTests(unittest.TestCase):
    """Verify JSON/*. * picker filters and New-pipeline report parsing."""

    def test_file_dialog_defaults_to_json_and_allows_all_files(self) -> None:
        """Browser selector lists every JSON first and *.* as the fallback."""
        self.assertEqual(
            FAILURE_FILE_TYPES,
            (
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ),
        )

    def test_known_reports_include_new_pipeline_files(self) -> None:
        """Auto-detect prefers New Medium, then New Stage 1, then legacy."""
        self.assertEqual(
            KNOWN_FAILURE_REPORTS,
            (
                "asr_new_medium_failures.json",
                "asr_new_failures.json",
                "asr_failures.json",
            ),
        )

    def test_loads_new_failures_list(self) -> None:
        """``asr_new_failures.json`` list rows yield canonical chunk ids."""
        payload = [
            {"chunk_index": 12, "original_text": "Expected."},
            {"chunk_id": "chunk_00013", "filename": "audio_chunks/chunk_00013.wav"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "asr_new_failures.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            chunk_ids, error = load_failure_chunk_ids(path)

        self.assertIsNone(error)
        self.assertEqual(chunk_ids, ["chunk_00012", "chunk_00013"])

    def test_loads_new_medium_records_object(self) -> None:
        """``asr_new_medium_failures.json`` object reports use the records list."""
        payload = {
            "report_type": "new_medium_confirmed_failures_v2",
            "records": [
                {
                    "chunk_index": 76,
                    "reference": {"text_raw": "Expected words."},
                    "medium": {"transcript_raw": "Medium words.", "passed": False},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "asr_new_medium_failures.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            chunk_ids, error = load_failure_chunk_ids(path)

        self.assertIsNone(error)
        self.assertEqual(chunk_ids, ["chunk_00076"])

    def test_keeps_legacy_failure_list_compatible(self) -> None:
        """Legacy ``asr_failures.json`` lists still produce chunk ids."""
        payload = [{"chunk_index": 4, "original_text": "Expected.", "score": 0.5}]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "asr_failures.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            chunk_ids, error = load_failure_chunk_ids(path)

        self.assertIsNone(error)
        self.assertEqual(chunk_ids, ["chunk_00004"])

    def test_rejects_object_without_records(self) -> None:
        """A JSON object that is not a failure report returns a clear error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.json"
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            chunk_ids, error = load_failure_chunk_ids(path)

        self.assertEqual(chunk_ids, [])
        self.assertIn("records list", error or "")

    def test_default_report_prefers_new_medium(self) -> None:
        """Folder scan picks asr_new_medium_failures.json when several exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "asr_failures.json").write_text("[]", encoding="utf-8")
            (folder / "asr_new_failures.json").write_text("[]", encoding="utf-8")
            medium = folder / "asr_new_medium_failures.json"
            medium.write_text("[]", encoding="utf-8")
            self.assertEqual(default_failure_report([folder]), medium)

    def test_chunk_id_from_audio_path(self) -> None:
        """Full audio paths still yield a chunk id from the filename stem."""
        self.assertEqual(
            chunk_id_from_failure_entry(
                {"audio_path": "/book/TTS/audio_chunks/chunk_00009.wav"}
            ),
            "chunk_00009",
        )

    def test_records_helper_accepts_list_and_object(self) -> None:
        """Payload helper unwraps both supported failure-report shapes."""
        rows, error = failure_records_from_payload([{"chunk_index": 1}])
        self.assertIsNone(error)
        self.assertEqual(len(rows), 1)
        rows, error = failure_records_from_payload({"records": [{"chunk_index": 2}]})
        self.assertIsNone(error)
        self.assertEqual(len(rows), 1)
        rows, error = failure_records_from_payload({"not": "a report"})
        self.assertEqual(rows, [])
        self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
