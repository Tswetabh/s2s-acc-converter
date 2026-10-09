"""Focused coverage for during-TTS Stage 1 shadow wiring helpers."""

from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path

from pocket_tts.audiobook.generator import (
    AudiobookGenerator,
    _SHADOW_STAGE_ONE_PARAKEET_MODEL,
)


class _FakeService:
    """Minimal shadow service double used by helper-focused tests."""

    def __init__(self, *args, **kwargs) -> None:
        """Initializes the service with arguments, sets up internal state."""
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.polled = 0
        self.submitted = []
        self.aborted = 0
        self.finished = []
        self._shadow_finish_called = False

    def start(self) -> None:
        """Record that the service start path ran."""
        self.started = True

    def poll(self) -> None:
        """Record poll usage during submit and finish paths."""
        self.polled += 1

    def submit(self, job) -> None:
        """Capture immutable jobs submitted by the generator helper."""
        self.submitted.append(job)

    def finish_collect(self, timeout_s: int):
        """Return a complete summary-like object for helper finish tests."""
        self.finished.append(timeout_s)
        return type(
            "Summary",
            (),
            {
                "complete": True,
                "submitted": len(self.submitted),
                "failed": 0,
                "completed": len(self.submitted),
                "missing_ids": (),
                "timed_out": False,
                "errors": (),
            },
        )()

    def abort(self) -> None:
        """Record abort calls used for shadow fallback paths."""
        self.aborted += 1


class ShadowStageOneDuringTtsTests(unittest.TestCase):
    """Verify helper/controller seams for Stage 3 shadow collection."""

    def _generator(self) -> AudiobookGenerator:
        """Create an uninitialized generator instance for helper testing."""
        return AudiobookGenerator.__new__(AudiobookGenerator)

    def test_shadow_request_inactive_for_default_or_postgen_settings(self) -> None:
        """Only explicit during-TTS Stage 1 settings should activate shadow collection."""
        generator = self._generator()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_paths = {
                "tts_dir": str(root),
                "audio_chunks_dir": str(root / "audio_chunks"),
                "text_chunks_dir": str(root / "text_chunks"),
            }

            self.assertIsNone(
                generator._resolve_shadow_stage_one_request(
                    asr_enabled=False,
                    save_dataset_chunks=True,
                    dataset_paths=dataset_paths,
                    asr_config={"stage_one": {"schedule": "during_tts", "engine": "faster_whisper"}},
                )
            )
            self.assertIsNone(
                generator._resolve_shadow_stage_one_request(
                    asr_enabled=True,
                    save_dataset_chunks=True,
                    dataset_paths=dataset_paths,
                    asr_config={"stage_one": {"schedule": "postgen", "engine": "faster_whisper"}},
                )
            )

    def test_shadow_request_resolves_faster_whisper_and_parakeet_models(self) -> None:
        """During-TTS shadow should pick GUI FW model and canonical NeMo Parakeet."""
        generator = self._generator()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_paths = {
                "tts_dir": str(root),
                "audio_chunks_dir": str(root / "audio_chunks"),
                "text_chunks_dir": str(root / "text_chunks"),
            }
            fw_request = generator._resolve_shadow_stage_one_request(
                asr_enabled=True,
                save_dataset_chunks=True,
                dataset_paths=dataset_paths,
                asr_config={
                    "model": "small",
                    "threshold": 0.72,
                    "language": "en",
                    "stage_one": {"schedule": "during_tts", "engine": "faster_whisper"},
                },
            )
            parakeet_request = generator._resolve_shadow_stage_one_request(
                asr_enabled=True,
                save_dataset_chunks=True,
                dataset_paths=dataset_paths,
                asr_config={
                    "model": "base",
                    "threshold": 0.72,
                    "language": "en",
                    "stage_one": {"schedule": "during_tts", "engine": "parakeet"},
                },
            )

        self.assertEqual(fw_request["model_name"], "small")
        self.assertEqual(parakeet_request["model_name"], _SHADOW_STAGE_ONE_PARAKEET_MODEL)

    def test_shadow_start_failure_falls_back_safely(self) -> None:
        """Shadow start failure must return None instead of interrupting generation."""
        generator = self._generator()

        class _BoomService(_FakeService):
            """Represents a service with a start method that fails."""
            def start(self) -> None:
                """Starts the service and raises a RuntimeError for failure."""
                raise RuntimeError("start failed")

        generator._streaming_stage_one_service_class = _BoomService
        with tempfile.TemporaryDirectory() as temp_dir:
            request = {
                "tts_dir": temp_dir,
                "engine": "faster_whisper",
                "model_name": "small",
                "threshold": 0.6,
                "language": "en",
            }
            self.assertIsNone(generator._start_shadow_stage_one_service(request))

    def test_shadow_publish_requires_completed_wav_and_text_pair(self) -> None:
        """Shadow publish should submit only after both durable sidecars exist."""
        generator = self._generator()
        service = _FakeService()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "audio_chunks" / "chunk_00000.wav"
            text_path = root / "text_chunks" / "chunk_00000.txt"
            audio_path.parent.mkdir(parents=True)
            text_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"RIFFfake")

            same_service = generator._publish_shadow_stage_one_chunk(service, audio_path, text_path, 0)
            self.assertIs(same_service, service)
            self.assertEqual(service.submitted, [])

            text_path.write_text("hello", encoding="utf-8")
            same_service = generator._publish_shadow_stage_one_chunk(service, audio_path, text_path, 0)

        self.assertIs(same_service, service)
        self.assertGreaterEqual(service.polled, 2)
        self.assertEqual(len(service.submitted), 1)
        self.assertEqual(service.submitted[0].chunk_id, "chunk_00000")

    def test_shadow_publish_keeps_service_active_when_submit_is_deferred(self) -> None:
        """Expected queue saturation must not disable shadow collection."""
        generator = self._generator()

        class _DeferredService(_FakeService):
            """Represents a deferred service that appends jobs to its internal list."""
            def submit(self, job) -> None:
                """Appends a submitted job to an internal list."""
                self.submitted.append(job)

        service = _DeferredService()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "audio_chunks" / "chunk_00001.wav"
            text_path = root / "text_chunks" / "chunk_00001.txt"
            audio_path.parent.mkdir(parents=True)
            text_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"RIFFfake")
            text_path.write_text("hello", encoding="utf-8")

            same_service = generator._publish_shadow_stage_one_chunk(service, audio_path, text_path, 1)

        self.assertIs(same_service, service)
        self.assertEqual(service.aborted, 0)
        self.assertEqual(len(service.submitted), 1)

    def test_parallel_shadow_callback_publishes_completed_chunk(self) -> None:
        """Parallel callback seam should publish through the same helper path."""
        generator = self._generator()
        service = _FakeService()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_paths = {"text_chunks_dir": str(root / "text_chunks")}
            service_holder = {"service": service}
            callback = generator._build_shadow_stage_one_original_chunk_callback(
                service_holder,
                dataset_paths,
            )
            audio_path = root / "audio_chunks" / "chunk_00003.wav"
            text_path = root / "text_chunks" / "chunk_00003.txt"
            audio_path.parent.mkdir(parents=True)
            text_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"RIFFfake")
            text_path.write_text("hello", encoding="utf-8")

            callback(3, audio_path)

        self.assertEqual(len(service.submitted), 1)
        self.assertEqual(service.submitted[0].chunk_id, "chunk_00003")
        self.assertIs(service_holder["service"], service)

    def test_parallel_shadow_callback_disables_shared_service_after_publish_failure(self) -> None:
        """Callback failure must clear shared shadow state so later finish skips it."""
        generator = self._generator()

        class _BoomService(_FakeService):
            """Represents a service with a submit method that fails."""
            def submit(self, job) -> None:
                """Submits a job and raises a RuntimeError for failure."""
                raise RuntimeError("submit failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = _BoomService()
            service_holder = {"service": service}
            dataset_paths = {"text_chunks_dir": str(root / "text_chunks")}
            callback = generator._build_shadow_stage_one_original_chunk_callback(
                service_holder,
                dataset_paths,
            )
            audio_path = root / "audio_chunks" / "chunk_00004.wav"
            text_path = root / "text_chunks" / "chunk_00004.txt"
            audio_path.parent.mkdir(parents=True)
            text_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"RIFFfake")
            text_path.write_text("hello", encoding="utf-8")

            callback(4, audio_path)

        self.assertIsNone(service_holder["service"])
        self.assertEqual(service.aborted, 1)

    def test_shadow_finish_drains_before_postgen_without_replacing_authority(self) -> None:
        """Helper finish should drain once and preserve summary for logging-only use."""
        generator = self._generator()
        service = _FakeService()
        service.submitted.append(type("Job", (), {"chunk_id": "chunk_00000"})())

        summary = generator._finish_shadow_stage_one_service(service, total_chunks=4)

        self.assertTrue(summary.complete)
        self.assertEqual(service.finished, [300])
        self.assertIs(getattr(service, "_shadow_finish_summary"), summary)

    def test_shadow_finish_is_idempotent_when_called_again_with_summary(self) -> None:
        """Repeated finish calls should return the same frozen summary without mutation."""
        generator = self._generator()
        service = _FakeService()
        service.submitted.append(type("Job", (), {"chunk_id": "chunk_00000"})())

        summary = generator._finish_shadow_stage_one_service(service, total_chunks=4)
        repeated = generator._finish_shadow_stage_one_service(summary, total_chunks=4)

        self.assertIs(repeated, summary)
        self.assertEqual(service.finished, [300])

    def test_shadow_promotion_switches_current_run_to_existing_stage_one_runner(self) -> None:
        """Strict complete streaming evidence should flip only this run's launch state."""
        generator = self._generator()
        launch_state = {
            "pipeline": "legacy",
            "failure_log": "/tmp/asr_failures.json",
            "run_log": "/tmp/asr_run.log",
            "use_existing_stage_one": False,
        }
        summary = SimpleNamespace(complete=True, submitted=4, failed=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_paths = {"tts_dir": temp_dir}
            with unittest.mock.patch(
                "ASR.streaming_stage_one.materialize_stage_one_summary",
                return_value={"candidate_count": 1},
            ) as materialize:
                promoted = generator._promote_shadow_stage_one_to_new_pipeline(
                    summary,
                    dataset_paths,
                    launch_state,
                )

        self.assertTrue(promoted)
        materialize.assert_called_once_with(summary, temp_dir)
        self.assertEqual(launch_state["pipeline"], "new")
        self.assertEqual(launch_state["failure_log"], str(Path(temp_dir) / "asr_new_failures.json"))
        self.assertEqual(launch_state["run_log"], str(Path(temp_dir) / "asr_new_pipeline.log"))
        self.assertTrue(launch_state["use_existing_stage_one"])

    def test_shadow_promotion_is_idempotent_after_current_run_is_switched(self) -> None:
        """Repeated promotion on the same run should not rematerialize evidence twice."""
        generator = self._generator()
        launch_state = {
            "pipeline": "new",
            "failure_log": "/tmp/asr_new_failures.json",
            "run_log": "/tmp/asr_new_pipeline.log",
            "use_existing_stage_one": True,
        }

        with unittest.mock.patch(
            "ASR.streaming_stage_one.materialize_stage_one_summary",
        ) as materialize:
            promoted = generator._promote_shadow_stage_one_to_new_pipeline(
                SimpleNamespace(complete=True, submitted=4, failed=1),
                {"tts_dir": "/tmp/book"},
                launch_state,
            )

        self.assertTrue(promoted)
        materialize.assert_not_called()

    def test_shadow_promotion_leaves_original_launch_state_on_incomplete_or_materialize_failure(self) -> None:
        """Fallback state must remain untouched when streaming promotion is not safe."""
        generator = self._generator()
        base_state = {
            "pipeline": "legacy",
            "failure_log": "/tmp/asr_failures.json",
            "run_log": "/tmp/asr_run.log",
            "use_existing_stage_one": False,
        }

        incomplete_state = dict(base_state)
        promoted = generator._promote_shadow_stage_one_to_new_pipeline(
            SimpleNamespace(complete=False),
            {"tts_dir": "/tmp/book"},
            incomplete_state,
        )
        self.assertFalse(promoted)
        self.assertEqual(incomplete_state, base_state)

        failed_state = dict(base_state)
        with unittest.mock.patch(
            "ASR.streaming_stage_one.materialize_stage_one_summary",
            side_effect=RuntimeError("boom"),
        ):
            promoted = generator._promote_shadow_stage_one_to_new_pipeline(
                SimpleNamespace(complete=True, submitted=2, failed=1),
                {"tts_dir": "/tmp/book"},
                failed_state,
            )

        self.assertFalse(promoted)
        self.assertEqual(failed_state, base_state)

    def test_shadow_abort_is_noop_for_finished_summary(self) -> None:
        """Shutdown abort path should ignore an already-finished frozen summary."""
        generator = self._generator()
        summary = SimpleNamespace(complete=True, submitted=1, failed=0)

        generator._abort_shadow_stage_one_service(summary, reason="shutdown")


if __name__ == "__main__":
    unittest.main()
