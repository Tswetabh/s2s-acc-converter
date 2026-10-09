"""Regression tests for New-pipeline Medium-to-regeneration handoff."""

import json
import tempfile
import unittest
from pathlib import Path

from pocket_tts.audiobook.generator import _materialize_new_pipeline_regeneration_log


class NewPipelineRegenerationTests(unittest.TestCase):
    """Ensure only confirmed New-pipeline Medium failures reach regeneration."""

    def test_materializes_confirmed_rows_and_ignores_accepted_rows(self) -> None:
        """Expand compact Medium evidence into rows accepted by regen workers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tts_dir = Path(temp_dir)
            (tts_dir / "asr_new_medium_failures.json").write_text(
                json.dumps({
                    "records": [
                        {
                            "chunk_index": 7,
                            "chunk_id": "chunk_00007",
                            "decision": "confirmed_by_two_asr_models",
                            "reference": {"text_raw": "Expected text."},
                            "stage_one": {
                                "transcript_raw": "Wrong text.",
                                "score": 0.2,
                            },
                            "medium": {"passed": False, "score": 0.3},
                        },
                        {
                            "chunk_index": 8,
                            "decision": "accepted_by_medium",
                            "medium": {"passed": True},
                        },
                    ]
                }),
                encoding="utf-8",
            )

            output_path, failures = _materialize_new_pipeline_regeneration_log(tts_dir)

            self.assertEqual(Path(output_path).name, "asr_new_regeneration_failures.json")
            self.assertEqual([failure["chunk_index"] for failure in failures], [7])
            self.assertEqual(failures[0]["text"], "Expected text.")
            saved = json.loads(Path(output_path).read_text(encoding="utf-8"))
            self.assertEqual(len(saved["records"]), 1)


if __name__ == "__main__":
    unittest.main()
