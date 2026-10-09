"""Regression coverage for book-local Medium-ASR verification reports."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pocket_tts.audiobook.generator import (
    AudiobookGenerator,
    write_stage2_investigation_log,
)
from pocket_tts.asr_failure_reports import (
    FAILURE_MANIFEST_FILENAME,
    select_failure_report,
)


class MediumVerificationLogTests(unittest.TestCase):
    """Verify Medium verification reports retain accepted and failed rows."""

    def test_verifier_uses_packed_batch_results(self) -> None:
        """Run one packed Medium batch and map its results back to failures."""
        failures = [
            {
                "chunk_index": 3,
                "score": 0.61,
                "classification": "FAIL",
                "explanation": "substituted: 'a' -> 'b'",
                "original_text": "Alpha beta",
                "transcribed_text": "Alpha beta",
            },
            {
                "chunk_index": 4,
                "score": 0.52,
                "classification": "FAIL",
                "explanation": "substituted: 'x' -> 'y'",
                "original_text": "Gamma delta",
                "transcribed_text": "Gamma gamma",
            },
        ]
        batch_results = [
            {
                "chunk_num": "chunk_00003",
                "passed": True,
                "score": 0.94,
                "classification": "PASS",
            },
            {
                "chunk_num": "chunk_00004",
                "passed": False,
                "score": 0.58,
                "classification": "FAIL",
            },
        ]
        generator = AudiobookGenerator.__new__(AudiobookGenerator)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "ASR.asr_validator.load_asr_model_adaptive",
                return_value=(object(), "cuda"),
            ) as load_model, patch(
                "ASR.asr_validator.run_pipeline_batch_validation",
                return_value=batch_results,
            ) as run_pipeline, patch("ASR.asr_validator.cleanup_asr_model"):
                remaining, summary = generator._verify_failed_chunks_with_medium_asr(
                    failures, Path(temp_dir), 0.68, "en", model_name="small"
                )

        load_model.assert_called_once_with("small", force_device="cuda")
        run_pipeline.assert_called_once()
        self.assertEqual(
            run_pipeline.call_args.args[1], ["chunk_00003", "chunk_00004"]
        )
        self.assertEqual(run_pipeline.call_args.kwargs["pack_size"], 8)
        self.assertEqual(summary["verification_mode"], "packed_gpu_pipeline")
        self.assertEqual(summary["verification_model"], "small")
        self.assertEqual(summary["verified_pass"], 1)
        self.assertEqual(summary["verified_fail"], 1)
        self.assertEqual([row["chunk_index"] for row in remaining], [4])

    def test_writer_persists_full_audit_record(self) -> None:
        """Write a compact report containing only useful verification fields."""
        summary = {
            "attempted": 2,
            "verified_pass": 1,
            "verified_fail": 1,
            "verification_model": "medium",
        }
        records = [
            {
                "chunk_index": 3,
                "chunk_id": "chunk_00003",
                "original_failure": {
                    "score": 0.61,
                    "classification": "FAIL",
                    "explanation": "substituted: 'a' → 'b'",
                    "original_text": "Alpha beta",
                    "transcribed_text": "Alpha beta",
                },
                "medium_result": {
                    "hyp_text_raw": "Alpha altered",
                    "passed": True,
                    "score": 0.94,
                    "classification": "PASS",
                    "explanation": "",
                    "ref_normalized": "<ID0> beta",
                    "hyp_normalized": "<ID0> beta",
                },
                "decision": "accepted_by_medium",
            },
            {
                "chunk_index": 4,
                "chunk_id": "chunk_00004",
                "original_failure": {
                    "score": 0.52,
                    "classification": "FAIL",
                    "explanation": "substituted: 'x' → 'y'",
                    "original_text": "Gamma delta",
                    "transcribed_text": "Gamma gamma",
                },
                "medium_result": {
                    "passed": False,
                    "score": 0.58,
                    "classification": "FAIL",
                    "error": "repetition",
                    "ref_normalized": "<ID0> beta",
                    "hyp_normalized": "<ID0> gamma",
                },
                "decision": "confirmed_by_two_asr_models",
            },
        ]
        generator = AudiobookGenerator.__new__(AudiobookGenerator)

        with tempfile.TemporaryDirectory() as temp_dir:
            generator._write_medium_verification_log(
                Path(temp_dir), 0.7, "en", summary, records
            )
            report_path = Path(temp_dir) / "asr_medium_validation_all.json"
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            failures_payload = json.loads(
                (Path(temp_dir) / "asr_medium_validation_failed.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (Path(temp_dir) / FAILURE_MANIFEST_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            selection = select_failure_report(Path(temp_dir))

        self.assertEqual(payload["report_type"], "medium_asr_verification")
        self.assertEqual(payload["summary"]["verified_pass"], 1)
        self.assertEqual(
            [record["decision"] for record in payload["records"]],
            ["accepted_by_medium", "confirmed_by_two_asr_models"],
        )
        self.assertEqual(
            set(payload["records"][0]["comparison"].keys()),
            {"ref_text_raw", "hyp_text_raw", "ref_normalized", "hyp_normalized"},
        )
        self.assertEqual(
            payload["records"][0]["comparison"]["hyp_text_raw"],
            "Alpha altered",
        )
        self.assertEqual(
            set(payload["records"][0]["original_failure"].keys()),
            {"score", "classification", "explanation"},
        )
        self.assertEqual(
            set(payload["records"][0]["medium_result"].keys()),
            {"passed", "score", "classification"},
        )
        self.assertEqual(
            set(payload["records"][1]["medium_result"].keys()),
            {"passed", "score", "classification", "error"},
        )
        self.assertTrue(summary["verification_log_path"].endswith("asr_medium_validation_all.json"))
        self.assertEqual(
            failures_payload["report_type"], "medium_asr_confirmed_failures"
        )
        self.assertEqual(
            [record["chunk_index"] for record in failures_payload["records"]], [4]
        )
        self.assertEqual(
            manifest["authoritative_report"]["path"],
            "asr_medium_validation_failed.json",
        )
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.source, "run manifest")

    def test_writer_excludes_accepted_and_unproven_rows_from_failure_file(self) -> None:
        """Failure-only report must contain Medium-confirmed failures only."""
        summary = {"attempted": 3, "verified_pass": 1, "verified_fail": 1, "not_proven": 1}
        records = [
            {
                "chunk_index": 1,
                "chunk_id": "chunk_00001",
                "medium_result": {"passed": True, "classification": "PASS"},
                "decision": "accepted_by_medium",
            },
            {
                "chunk_index": 2,
                "chunk_id": "chunk_00002",
                "medium_result": {"error": "decoder failed"},
                "decision": "accepted_not_proven_failure",
            },
            {
                "chunk_index": 3,
                "chunk_id": "chunk_00003",
                "medium_result": {"passed": False, "classification": "FAIL"},
                "decision": "confirmed_by_two_asr_models",
            },
        ]
        generator = AudiobookGenerator.__new__(AudiobookGenerator)

        with tempfile.TemporaryDirectory() as temp_dir:
            generator._write_medium_verification_log(
                Path(temp_dir), 0.7, "en", summary, records
            )
            payload = json.loads(
                (Path(temp_dir) / "asr_medium_validation_failed.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual([row["chunk_index"] for row in payload["records"]], [3])
        self.assertEqual(payload["summary"]["medium_confirmed_failures"], 1)
        self.assertEqual(payload["summary"]["medium_accepted_or_unproven"], 2)

    def test_new_writer_separates_stages_and_writes_confirmed_failures(self) -> None:
        """New reports keep both transcripts and omit Medium passes from companion file."""
        summary = {"attempted": 2, "verified_pass": 1, "verified_fail": 1}
        records = [
            {
                "chunk_index": 3,
                "chunk_id": "chunk_00003",
                "original_failure": {
                    "original_text": "Alpha beta",
                    "transcribed_text": "Alpha bettor",
                    "hyp_normalized": "alpha bettor",
                    "score": 0.61,
                    "classification": "FAIL",
                    "explanation": "substituted: beta -> bettor",
                },
                "medium_result": {
                    "hyp_text_raw": "Alpha beta",
                    "ref_normalized": "alpha beta",
                    "hyp_normalized": "alpha beta",
                    "passed": True,
                    "score": 1.0,
                    "classification": "PASS",
                },
                "decision": "accepted_by_medium",
            },
            {
                "chunk_index": 4,
                "chunk_id": "chunk_00004",
                "original_failure": {
                    "original_text": "Gamma delta",
                    "transcribed_text": "Gamma gamma",
                    "hyp_normalized": "gamma gamma",
                    "score": 0.52,
                    "classification": "FAIL",
                    "explanation": "substituted: delta -> gamma",
                },
                "medium_result": {
                    "hyp_text_raw": "Gamma gamma",
                    "ref_normalized": "gamma delta",
                    "hyp_normalized": "gamma gamma",
                    "passed": False,
                    "score": 0.58,
                    "classification": "FAIL",
                    "explanation": "substituted: delta -> gamma",
                },
                "decision": "confirmed_by_two_asr_models",
            },
        ]
        generator = AudiobookGenerator.__new__(AudiobookGenerator)

        with tempfile.TemporaryDirectory() as temp_dir:
            tts_dir = Path(temp_dir)
            generator._write_medium_verification_log(
                tts_dir,
                0.7,
                "en",
                summary,
                records,
                report_filename="asr_new_medium_verification.json",
            )
            full = json.loads(
                (tts_dir / "asr_new_medium_verification.json").read_text(encoding="utf-8")
            )
            failures = json.loads(
                (tts_dir / "asr_new_medium_failures.json").read_text(encoding="utf-8")
            )
            stage2_path = tts_dir / "asr_stage2_investigation.log"
            self.assertTrue(stage2_path.exists(), "Stage 2 human log must be written")
            stage2_text = stage2_path.read_text(encoding="utf-8")

        self.assertEqual(full["report_type"], "new_medium_asr_verification_v2")
        self.assertEqual(full["records"][0]["stage_one"]["transcript_raw"], "Alpha bettor")
        self.assertEqual(full["records"][0]["medium"]["transcript_raw"], "Alpha beta")
        self.assertEqual(full["records"][1]["medium"]["normalized"], "gamma gamma")
        self.assertEqual(failures["report_type"], "new_medium_confirmed_failures_v2")
        self.assertEqual([row["chunk_index"] for row in failures["records"]], [4])
        self.assertTrue(summary["verification_failures_log_path"].endswith("asr_new_medium_failures.json"))
        self.assertIn("BOOK TEXT:", stage2_text)
        self.assertIn("STAGE 2 ASR:", stage2_text)
        self.assertIn("WHY FAIL:", stage2_text)
        self.assertIn("Gamma delta", stage2_text)
        self.assertIn("Gamma gamma", stage2_text)
        self.assertIn("substituted: delta -> gamma", stage2_text)
        # Accepted Medium pass (Alpha beta) must not appear as a fail block.
        self.assertNotIn("Alpha bettor", stage2_text)
        self.assertTrue(
            summary["stage2_investigation_log_path"].endswith(
                "asr_stage2_investigation.log"
            )
        )

    def test_stage2_investigation_log_is_human_readable(self) -> None:
        """Standalone Stage 2 log lists book text, Medium ASR, and why it failed."""
        records = [
            {
                "chunk_index": 186,
                "chunk_id": "chunk_00186",
                "decision": "confirmed_by_two_asr_models",
                "reference": {
                    "text_raw": '. "A onetime bargain?"',
                    "normalized": "a onetime bargain",
                },
                "stage_one": {
                    "transcript_raw": "A one-time bargain?",
                    "normalized": "a <NUM> time bargain",
                    "score": 0.52,
                    "explanation": "substituted: 'onetime' → '<NUM>'",
                },
                "medium": {
                    "transcript_raw": "A one-time bargain?",
                    "normalized": "a <NUM> time bargain",
                    "score": 0.52,
                    "classification": "FAIL",
                    "passed": False,
                    "explanation": "substituted: 'onetime' → '<NUM>'",
                },
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_stage2_investigation_log(
                Path(temp_dir),
                records,
                threshold=0.6,
                language="en",
                stage_one_candidates=10,
                is_new_pipeline=True,
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("Stage 2 confirmed fails (this file): 1", text)
        self.assertIn("CHUNK: chunk_00186", text)
        self.assertIn('BOOK TEXT:     . "A onetime bargain?"', text)
        self.assertIn("STAGE 2 ASR:   A one-time bargain?", text)
        self.assertIn("WHY FAIL:      substituted: 'onetime' → '<NUM>'", text)
        self.assertIn("BOOK NORM:   a onetime bargain", text)
        self.assertIn("ASR NORM:    a <NUM> time bargain", text)
        self.assertIn("STAGE 1 ASR:  A one-time bargain?", text)

    def test_verifier_error_does_not_send_unproven_audio_to_regeneration(self) -> None:
        """Keep original audio when the independent verifier cannot load."""
        failures = [{"chunk_index": 3, "original_text": "Alpha beta"}]
        generator = AudiobookGenerator.__new__(AudiobookGenerator)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "ASR.asr_validator.load_asr_model_adaptive",
                return_value=(None, "cuda"),
            ):
                remaining, summary = generator._verify_failed_chunks_with_medium_asr(
                    failures, Path(temp_dir), 0.68, "en"
                )

        self.assertEqual(remaining, [])
        self.assertTrue(summary["verification_skipped"])
        self.assertEqual(summary["not_proven"], 1)

    def test_explicit_report_filename_preserves_legacy_report(self) -> None:
        """Write opt-in output without replacing the legacy Medium audit."""
        failures = [{"chunk_index": 3, "original_text": "Alpha beta"}]
        generator = AudiobookGenerator.__new__(AudiobookGenerator)

        with tempfile.TemporaryDirectory() as temp_dir:
            tts_dir = Path(temp_dir)
            legacy_path = tts_dir / "asr_medium_verification.json"
            legacy_path.write_text('{"legacy": true}\n', encoding="utf-8")
            with patch(
                "ASR.asr_validator.load_asr_model_adaptive",
                return_value=(None, "cuda"),
            ):
                remaining, summary = generator._verify_failed_chunks_with_medium_asr(
                    failures,
                    tts_dir,
                    0.68,
                    "en",
                    report_filename="asr_new_medium_verification.json",
                )

            output = tts_dir / "asr_new_medium_verification.json"
            self.assertEqual(remaining, [])
            self.assertEqual(legacy_path.read_text(encoding="utf-8"), '{"legacy": true}\n')
            self.assertTrue(output.exists())
            self.assertEqual(summary["verification_log_path"], str(output))

    def test_require_gpu_reports_load_failure_without_cpu_fallback(self) -> None:
        """Keep isolated new-pipeline Medium verification GPU-only."""
        failures = [{"chunk_index": 3, "original_text": "Alpha beta"}]
        generator = AudiobookGenerator.__new__(AudiobookGenerator)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "ASR.asr_validator.load_asr_model_adaptive",
                return_value=(None, "cuda_load_failed"),
            ) as load_model:
                remaining, summary = generator._verify_failed_chunks_with_medium_asr(
                    failures,
                    Path(temp_dir),
                    0.68,
                    "en",
                    require_gpu=True,
                )

        self.assertEqual(remaining, [])
        load_model.assert_called_once_with(
            "medium",
            force_device="cuda",
            allow_cpu_fallback=False,
        )
        self.assertEqual(summary["verification_device"], "cuda_load_failed")
        self.assertEqual(
            summary["verification_error"],
            "gpu_second_stage_model_load_failed_no_cpu_fallback",
        )
