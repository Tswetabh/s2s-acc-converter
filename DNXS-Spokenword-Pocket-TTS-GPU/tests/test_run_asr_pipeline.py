"""Regression coverage for the explicit live ASR pipeline selector."""

import importlib.util
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_asr_pipeline", ROOT / "tools" / "run_asr_pipeline.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class LiveAsrPipelineTests(unittest.TestCase):
    """Ensure new mode isolates child processes while legacy stays unchanged."""

    def _make_tts_dir(self, temp_dir: str) -> Path:
        """Create the minimum folder structure required by the runner."""
        tts_dir = Path(temp_dir)
        (tts_dir / "audio_chunks").mkdir()
        (tts_dir / "text_chunks").mkdir()
        return tts_dir

    def test_new_pipeline_uses_one_child_process_per_stage(self) -> None:
        """Run new mode through three serial fresh-process status files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tts_dir = self._make_tts_dir(temp_dir)
            failure_path = tts_dir / "asr_new_failures.json"
            seen_stages = []

            def fake_run(command, **kwargs):
                """Write deterministic child artifacts without loading model runtimes."""
                stage = command[command.index("--internal-stage") + 1]
                status_path = Path(command[command.index("--status-file") + 1])
                seen_stages.append(stage)
                if stage == "stage_one":
                    failure_path.write_text(
                        json.dumps([{"chunk_index": 1, "original_text": "Hello"}]),
                        encoding="utf-8",
                    )
                result = {
                    "stage_one": {"engine": "parakeet"},
                    "medium": {"gpu_required": True, "active": True},
                    "alignment": {"decision_policy": "diagnostic_only_never_used_for_regeneration"},
                }[stage]
                status_path.write_text(json.dumps({
                    "stage": stage,
                    "pid": 100 + len(seen_stages),
                    "fresh_process": True,
                    "cuda_before_stage": {"free_bytes": 6_000_000_000},
                    "cuda_teardown": {"after": {"free_bytes": 7_000_000_000}},
                    "result": result,
                    "error": None,
                }), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(RUNNER.subprocess, "run", side_effect=fake_run):
                report = RUNNER.run_pipeline(
                    tts_dir, "new", 0.68, "en", second_stage_model="small"
                )

            self.assertEqual(seen_stages, ["stage_one", "medium", "alignment"])
            self.assertEqual(report["isolation"]["strategy"], "one_fresh_subprocess_per_model_stage")
            self.assertTrue(report["stage_one_process"]["process_exited_before_next_stage"])
            self.assertTrue(report["medium"]["gpu_required"])
            self.assertEqual(report["second_stage_model"], "small")
            self.assertIn("--second-stage-model", report["medium_process"]["command"])
            self.assertIn("small", report["medium_process"]["command"])
            self.assertEqual(
                report["forced_alignment"]["decision_policy"],
                "diagnostic_only_never_used_for_regeneration",
            )
            self.assertTrue((tts_dir / "asr_new_pipeline_report.json").exists())
            self.assertIn(
                "asr_new_medium_failures.json",
                [Path(output).name for output in report["outputs"]],
            )
            stage_status = json.loads(
                (tts_dir / "asr_new_stage_one_process.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stage_status["returncode"], 0)

    def test_new_pipeline_can_run_stage_two_from_streaming_evidence(self) -> None:
        """Existing strict-complete streaming evidence should skip the Stage 1 child."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tts_dir = self._make_tts_dir(temp_dir)
            (tts_dir / RUNNER.STREAMING_AUDIT_NAME).write_text(
                json.dumps(
                    {
                        "stage_one_source": "during_tts_streaming",
                        "candidate_count": 1,
                        "summary": {
                            "complete": True,
                            "submitted": 2,
                            "completed": 2,
                            "failed": 1,
                            "event_log_path": str(tts_dir / "events.jsonl"),
                        },
                        "records": [
                            {
                                "chunk_id": "chunk_00001",
                                "chunk_index": 1,
                                "status": "result",
                                "passed": True,
                                "classification": "PASS",
                                "score": 0.9,
                                "audio_path": str(tts_dir / "audio_chunks" / "chunk_00001.wav"),
                                "text_path": str(tts_dir / "text_chunks" / "chunk_00001.txt"),
                                "original_text": "hello",
                                "transcript": "hello",
                                "explanation": "ok",
                                "backend": "parakeet",
                                "model": "nvidia/parakeet-tdt-0.6b-v3",
                                "elapsed_s": 0.12,
                                "error": "",
                            },
                            {
                                "chunk_id": "chunk_00002",
                                "chunk_index": 2,
                                "status": "result",
                                "passed": False,
                                "classification": "FAIL",
                                "score": 0.2,
                                "audio_path": str(tts_dir / "audio_chunks" / "chunk_00002.wav"),
                                "text_path": str(tts_dir / "text_chunks" / "chunk_00002.txt"),
                                "original_text": "bye",
                                "transcript": "bad",
                                "explanation": "mismatch",
                                "backend": "parakeet",
                                "model": "nvidia/parakeet-tdt-0.6b-v3",
                                "elapsed_s": 0.18,
                                "error": "",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (tts_dir / "asr_new_failures.json").write_text(
                json.dumps([{"chunk_index": 2, "original_text": "bye", "transcribed_text": "bad"}]),
                encoding="utf-8",
            )
            seen_stages = []

            def fake_run(command, **kwargs):
                """Write deterministic child outputs for Medium and alignment only."""
                stage = command[command.index("--internal-stage") + 1]
                status_path = Path(command[command.index("--status-file") + 1])
                seen_stages.append(stage)
                result = {
                    "medium": {"gpu_required": True, "active": True, "confirmed_failures": 1},
                    "alignment": {"decision_policy": "diagnostic_only_never_used_for_regeneration"},
                }[stage]
                status_path.write_text(json.dumps({
                    "stage": stage,
                    "pid": 200 + len(seen_stages),
                    "fresh_process": True,
                    "cuda_before_stage": {"free_bytes": 6_000_000_000},
                    "cuda_teardown": {"after": {"free_bytes": 7_000_000_000}},
                    "result": result,
                    "error": None,
                }), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(RUNNER.subprocess, "run", side_effect=fake_run):
                report = RUNNER.run_pipeline(
                    tts_dir,
                    "new",
                    0.68,
                    "en",
                    second_stage_model="small",
                    use_existing_stage_one=True,
                )

        self.assertEqual(seen_stages, ["medium", "alignment"])
        self.assertEqual(report["stage_one_source"], "during_tts_streaming")
        self.assertEqual(report["stage_one_candidate_count"], 1)
        self.assertEqual(report["stage_one"]["source"], "during_tts_streaming")
        self.assertFalse(report["stage_one_process"]["fresh_process"])
        self.assertEqual(Path(report["stage_one_audit_path"]).name, RUNNER.STREAMING_AUDIT_NAME)
        self.assertEqual(Path(report["stage_one_failure_path"]).name, "asr_new_failures.json")

    def test_new_pipeline_existing_stage_one_requires_valid_audit(self) -> None:
        """Malformed or missing streaming audits must fail before Stage 2 starts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tts_dir = self._make_tts_dir(temp_dir)
            (tts_dir / "asr_new_failures.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing"):
                RUNNER.run_pipeline(
                    tts_dir,
                    "new",
                    0.68,
                    "en",
                    use_existing_stage_one=True,
                )

            (tts_dir / RUNNER.STREAMING_AUDIT_NAME).write_text(
                json.dumps({"stage_one_source": "wrong", "summary": {"complete": True}, "records": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "unsupported source"):
                RUNNER.run_pipeline(
                    tts_dir,
                    "new",
                    0.68,
                    "en",
                    use_existing_stage_one=True,
                )

    def test_legacy_is_default_pipeline_definition(self) -> None:
        """Keep faster-whisper and canonical filenames assigned to legacy mode."""
        settings = RUNNER.PIPELINES["legacy"]
        self.assertEqual(settings["stage_one_engine"], "faster_whisper")
        self.assertEqual(settings["failure_filename"], "asr_failures.json")
        self.assertEqual(settings["medium_filename"], "asr_medium_validation_all.json")
        self.assertIsNone(settings["alignment_filename"])

    def test_legacy_no_candidates_writes_both_medium_reports(self) -> None:
        """Legacy report-only runs must create empty all and failure reports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tts_dir = Path(temp_dir)
            result = RUNNER.run_medium_verifier(
                [],
                tts_dir,
                RUNNER.PIPELINES["legacy"],
                0.68,
                "en",
            )
            all_payload = json.loads(
                (tts_dir / "asr_medium_validation_all.json").read_text(
                    encoding="utf-8"
                )
            )
            failure_payload = json.loads(
                (tts_dir / "asr_medium_validation_failed.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertFalse(result["active"])
        self.assertEqual(all_payload["records"], [])
        self.assertEqual(failure_payload["records"], [])


class GuiAsrLaunchScopeTests(unittest.TestCase):
    """Protect GUI launch code from delayed local subprocess bindings."""

    def test_new_gui_launch_uses_module_subprocess_binding(self) -> None:
        """Ensure the launch closure cannot read an uninitialized enclosing variable."""
        source_path = ROOT / "pocket_tts" / "audiobook" / "generator.py"
        root_code = compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
        pending = [root_code]
        generate_code = None
        while pending:
            code = pending.pop()
            if code.co_name == "generate_audiobook":
                generate_code = code
                break
            pending.extend(
                value for value in code.co_consts if isinstance(value, types.CodeType)
            )

        self.assertIsNotNone(generate_code)
        assert generate_code is not None
        launch_codes = [
            value
            for value in generate_code.co_consts
            if isinstance(value, types.CodeType)
            and value.co_name == "_launch_post_gen_gpu_asr"
            and "Popen" in value.co_names
        ]
        self.assertEqual(len(launch_codes), 1)
        self.assertNotIn("subprocess", generate_code.co_cellvars)
        self.assertNotIn("subprocess", launch_codes[0].co_freevars)
        self.assertIn("subprocess", launch_codes[0].co_names)

    def test_new_gui_launch_forwards_selected_second_stage_model(self) -> None:
        """Ensure the GUI-derived model value reaches the New runner command."""
        generator_source = (
            ROOT / "pocket_tts" / "audiobook" / "generator.py"
        ).read_text(encoding="utf-8")
        window_source = (
            ROOT / "pocket_tts" / "gui" / "main_window.py"
        ).read_text(encoding="utf-8")
        self.assertIn("asr_config.get('second_stage_model', 'medium')", generator_source)
        self.assertIn("'--second-stage-model', second_stage_model", generator_source)
        self.assertIn('asr["second_stage_model"] = second_stage_model', window_source)


class RunAsrPipelineCliTests(unittest.TestCase):
    """Verify CLI-only routing for the Stage 4b evidence reuse flag."""

    def test_cli_rejects_existing_stage_one_without_new_pipeline(self) -> None:
        """The reuse flag must stay scoped to the New pipeline only."""
        with patch.object(RUNNER.argparse.ArgumentParser, "error", side_effect=RuntimeError("bad args")):
            with patch.object(RUNNER.sys, "argv", ["run_asr_pipeline.py", "/tmp/book", "--use-existing-stage-one"]):
                with self.assertRaisesRegex(RuntimeError, "bad args"):
                    RUNNER.main()

    def test_cli_forwards_existing_stage_one_flag_to_run_pipeline(self) -> None:
        """CLI should pass the reuse flag through to the public runner API."""
        with patch.object(RUNNER, "run_pipeline", return_value={"ok": True}) as run_pipeline:
            with patch.object(
                RUNNER.sys,
                "argv",
                [
                    "run_asr_pipeline.py",
                    "/tmp/book",
                    "--pipeline",
                    "new",
                    "--use-existing-stage-one",
                ],
            ):
                exit_code = RUNNER.main()

        self.assertEqual(exit_code, 0)
        self.assertTrue(run_pipeline.call_args.kwargs["use_existing_stage_one"])
