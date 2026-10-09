"""Focused coverage for the inert spawned Stage 1 worker foundation."""

from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ASR.streaming_stage_one import (
    ALL_RESULTS_AUDIT_NAME,
    AUTHORITATIVE_FAILURES_NAME,
    CHILD_CRASH_REPORT_NAME,
    StageOneJob,
    StageOneDetail,
    StageOneRunSummary,
    StreamingStageOneService,
    _FasterWhisperRunner,
    _ParakeetRunner,
    _child_main,
    _map_faster_whisper_batch_results,
    _select_faster_whisper_pack_profile,
    materialize_stage_one_summary,
)
from ASR.asr_validator import run_pipeline_batch_validation


class _ThreadProcess:
    """Thread-backed process double for seam-injected service tests."""

    def __init__(self, *, target, args, name="thread-proc") -> None:
        """Initializes a new thread wrapper with a target function and arguments."""
        self._target = target
        self._args = args
        self._name = name
        self._thread: threading.Thread | None = None
        self.exitcode = None

    def start(self) -> None:
        """Run the target in a background thread and capture an exit code."""
        def _runner() -> None:
            """Runs a target function in a separate thread and captures its exit code."""
            try:
                self._target(*self._args)
                if self.exitcode is None:
                    self.exitcode = 0
            except Exception:
                self.exitcode = 1

        self._thread = threading.Thread(target=_runner, name=self._name, daemon=True)
        self._thread.start()

    def is_alive(self) -> bool:
        """Return whether the backing thread is still running."""
        return bool(self._thread and self._thread.is_alive())

    def join(self, timeout: float | None = None) -> None:
        """Wait for the backing thread to finish."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def terminate(self) -> None:
        """Mark the fake process terminated for timeout-path assertions."""
        self.exitcode = -15


class _ThreadContext:
    """Minimal spawn-context double that keeps child seams patchable in tests."""

    def Queue(self, maxsize: int = 0):
        """Return a bounded stdlib queue with the requested capacity."""
        return queue.Queue(maxsize=maxsize)

    def Process(self, *, target, args, name):
        """Return a thread-backed fake process."""
        return _ThreadProcess(target=target, args=args, name=name)


class _FullQueue:
    """Queue double that raises ``queue.Full`` after a configured job count."""

    def __init__(self, maxsize: int = 0, fail_after: int = 0) -> None:
        """Queues items up to a limit, then simulates saturation."""
        self.maxsize = maxsize
        self.fail_after = fail_after
        self.put_count = 0
        self.items = []

    def put_nowait(self, item) -> None:
        """Append until the configured limit, then simulate queue saturation."""
        if self.put_count >= self.fail_after:
            raise queue.Full()
        self.put_count += 1
        self.items.append(item)

    def put(self, item) -> None:
        """Support the stop-sentinel path used by ``finish_collect``."""
        self.items.append(item)

    def get_nowait(self):
        """No queued events are available in this parent-only queue double."""
        raise queue.Empty()


class _NeverStartProcess(_ThreadProcess):
    """Process double that never invokes the target, used for parent-only checks."""

    def start(self) -> None:
        """Pretend the process started without executing the child target."""
        self.exitcode = None

    def is_alive(self) -> bool:
        """Stay idle so parent-side checks can close cleanly."""
        return False


class _ExitedProcess(_NeverStartProcess):
    """Process double that looks like a cleanly exited child."""

    def start(self) -> None:
        """Pretend the child already ran and exited successfully."""
        self.exitcode = 0


class _NeverStartContext(_ThreadContext):
    """Context double that suppresses child execution entirely."""

    def Process(self, *, target, args, name):
        """Return a no-op fake process."""
        return _NeverStartProcess(target=target, args=args, name=name)


class _ExitedContext(_NeverStartContext):
    """Context double whose process appears already exited with code 0."""

    def Process(self, *, target, args, name):
        """Return a clean-exit fake process for finish success-path tests."""
        return _ExitedProcess(target=target, args=args, name=name)


class _SubmitFullContext(_NeverStartContext):
    """Context double whose job queue fills after a configured submit count."""

    def __init__(self, fail_after: int) -> None:
        """Initializes a full queue with controlled failure behavior."""
        self._fail_after = fail_after
        self._queue_calls = 0

    def Queue(self, maxsize: int = 0):
        """Return the filling job queue first, then a passive event queue."""
        self._queue_calls += 1
        if self._queue_calls == 1:
            return _FullQueue(maxsize=maxsize, fail_after=self._fail_after)
        return _FullQueue(maxsize=maxsize, fail_after=0)


class _DeferredOnlyContext(_NeverStartContext):
    """Context double that accepts one job and defers every later submission."""

    def __init__(self, fail_after: int) -> None:
        """Initializes a queue that fails after a certain number of puts."""
        self._fail_after = fail_after
        self._queue_calls = 0
        self.job_queue = None

    def Queue(self, maxsize: int = 0):
        """Expose the first queue so tests can inspect deferred growth."""
        self._queue_calls += 1
        if self._queue_calls == 1:
            self.job_queue = _FullQueue(maxsize=maxsize, fail_after=self._fail_after)
            return self.job_queue
        return _FullQueue(maxsize=maxsize, fail_after=0)


class _StopFullQueue:
    """Queue double that stays full for stop puts until a configured release poll."""

    def __init__(self, payloads=None, stop_release_after: int = 999) -> None:
        """Rejects stop operations until a specified threshold is reached."""
        self.payloads = list(payloads or [])
        self.stop_release_after = stop_release_after
        self.stop_attempts = 0

    def put_nowait(self, item) -> None:
        """Reject stop puts until enough poll-driven attempts have passed."""
        if item == {"type": "stop"}:
            self.stop_attempts += 1
            if self.stop_attempts < self.stop_release_after:
                raise queue.Full()
        else:
            self.payloads.append(item)

    def get(self):
        """Return queued payloads in FIFO order."""
        if not self.payloads:
            raise queue.Empty()
        return self.payloads.pop(0)


class _SequenceEventQueue:
    """Event queue double that releases one event per poll attempt."""

    def __init__(self, events):
        """Provides a queue-like interface for event handling."""
        self._events = list(events)

    def get_nowait(self):
        """Return the next queued event or report empty."""
        if not self._events:
            raise queue.Empty()
        return self._events.pop(0)

    def get(self, timeout=None):
        """Mirror blocking get with immediate availability for deterministic tests."""
        return self.get_nowait()


class _StopDeadlockContext(_NeverStartContext):
    """Context double for deterministic stop-sentinel deadlock regression tests."""

    def __init__(self, *, stop_release_after: int, queued_payloads, events) -> None:
        """Similar to Item 9, manages job queues in tests."""
        self._stop_release_after = stop_release_after
        self._queued_payloads = queued_payloads
        self._events = events
        self._queue_calls = 0

    def Queue(self, maxsize: int = 0):
        """Return the prefilled job queue first, then the preloaded event queue."""
        self._queue_calls += 1
        if self._queue_calls == 1:
            return _StopFullQueue(
                payloads=self._queued_payloads,
                stop_release_after=self._stop_release_after,
            )
        return _SequenceEventQueue(self._events)


class _StopDeadlockExitedContext(_ExitedContext):
    """Stop-deadlock context variant with a clean exited child."""

    def __init__(self, *, stop_release_after: int, queued_payloads, events) -> None:
        """Manages queued payloads and events with controlled behavior."""
        self._stop_release_after = stop_release_after
        self._queued_payloads = queued_payloads
        self._events = events
        self._queue_calls = 0

    def Queue(self, maxsize: int = 0):
        """Return the prefilled job queue first, then the preloaded event queue."""
        self._queue_calls += 1
        if self._queue_calls == 1:
            return _StopFullQueue(
                payloads=self._queued_payloads,
                stop_release_after=self._stop_release_after,
            )
        return _SequenceEventQueue(self._events)


def _make_chunk(root: Path, chunk_id: str, text: str = "Hello world.") -> StageOneJob:
    """Create one temporary WAV/TXT pair and return its immutable Stage 1 job."""
    audio_dir = root / "audio_chunks"
    text_dir = root / "text_chunks"
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{chunk_id}.wav"
    text_path = text_dir / f"{chunk_id}.txt"
    audio_path.write_bytes(b"RIFFfake")
    text_path.write_text(text, encoding="utf-8")
    return StageOneJob(
        chunk_index=int(chunk_id.split("_")[-1]),
        chunk_id=chunk_id,
        audio_path=str(audio_path),
        text_path=str(text_path),
    )


def _make_detail(
    *,
    chunk_id: str,
    chunk_index: int | None,
    status: str = "result",
    passed: bool = True,
    classification: str = "PASS",
    score: float = 0.9,
    backend: str = "faster_whisper",
    model: str = "small",
    error: str = "",
) -> StageOneDetail:
    """Build one compact Stage 1 detail for materialization-only assertions."""
    return StageOneDetail(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        status=status,
        passed=passed,
        classification=classification,
        score=score,
        audio_path=f"/tmp/{chunk_id}.wav",
        text_path=f"/tmp/{chunk_id}.txt",
        original_text=f"source {chunk_id}",
        transcript=f"transcript {chunk_id}",
        explanation=f"explanation {chunk_id}",
        backend=backend,
        model=model,
        elapsed_s=0.25 if chunk_index is None else 0.25 + (chunk_index / 100.0),
        error=error,
    )


def _make_summary(
    details: tuple[StageOneDetail, ...],
    *,
    complete: bool = True,
    duplicate_terminal_ids: tuple[str, ...] = (),
) -> StageOneRunSummary:
    """Build one Stage 1 summary without spawning a worker process."""
    failed = sum(1 for detail in details if not detail.passed)
    return StageOneRunSummary(
        submitted=len(details),
        completed=len(details),
        failed=failed,
        missing_ids=(),
        complete=complete,
        details=details,
        errors=(),
        duplicate_terminal_ids=duplicate_terminal_ids,
        child_exitcode=0,
        timed_out=False,
        event_log_path="/tmp/events.jsonl",
        summary_path="/tmp/summary.json",
    )


class _FakeParakeetEvidence:
    """Simple evidence object matching the real backend transcript shape."""

    def __init__(self, audio_path: str, text: str, elapsed_s: float = 0.2) -> None:
        """Represents fake evidence for ASR transcription."""
        self.audio_path = audio_path
        self.text = text
        self.elapsed_s = elapsed_s
        self.backend = "parakeet_tdt"


class _CountingParakeetBackend:
    """Resident backend double that tracks instantiation count per run."""

    instances = 0

    def __init__(self, model_name="parakeet-model", device="cuda") -> None:
        """Initializes an instance with model and device info."""
        type(self).instances += 1
        self.model_name = model_name
        self.device = device
        self.closed = False

    def transcribe_paths(self, paths):
        """Return one evidence object for every requested path."""
        return [
            _FakeParakeetEvidence(str(Path(path)), f"parakeet:{Path(path).stem}")
            for path in paths
        ]

    def close(self) -> None:
        """Record that the resident backend was released."""
        self.closed = True


class StreamingStageOneTests(unittest.TestCase):
    """Verify Stage 2 worker foundation behavior without GPU/model downloads."""

    def test_faster_whisper_runner_accepts_gpu_success_label_without_cpu_fallback(self) -> None:
        """GPU-only faster-whisper loading should accept canonical GPU success labels."""
        calls = {}

        def _fake_load(model_name, force_device, engine, allow_cpu_fallback):
            """Model loading mock with side effects tracking."""
            calls.update(
                {
                    "model_name": model_name,
                    "force_device": force_device,
                    "engine": engine,
                    "allow_cpu_fallback": allow_cpu_fallback,
                }
            )
            return object(), "gpu"

        def _fake_transcribe(model, audio_path, language, vad_filter):
            """Dummy transcription function returning a fixed string."""
            return "fw transcript"

        def _fake_score(chunk_id, original_text, transcript, threshold, audio_path, book_term_evidence=None):
            """Fake score method returning a fixed pass result."""
            return {
                "passed": True,
                "classification": "PASS",
                "score": 0.91,
                "hyp_text_raw": transcript,
                "explanation": "matched",
                "error": "",
            }

        with patch(
            "ASR.streaming_stage_one._load_validator_runtime",
            return_value=(_fake_load, _fake_transcribe, lambda model: None, _fake_score),
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                job = _make_chunk(Path(temp_dir), "chunk_00001")
                runner = _FasterWhisperRunner(model_name="small", language="en")
                runner.start()
                detail = runner.run(job, threshold=0.6)
                runner.close()

        self.assertEqual(calls["force_device"], "cuda")
        self.assertEqual(calls["engine"], "faster_whisper")
        self.assertFalse(calls["allow_cpu_fallback"])
        self.assertEqual(detail.backend, "faster_whisper")
        self.assertTrue(detail.passed)

    def test_faster_whisper_runner_rejects_cpu_and_failure_device_labels(self) -> None:
        """Non-GPU device labels must still fail even when a model object exists."""
        for device_label in ("cpu", "cuda_load_failed"):
            with self.subTest(device_label=device_label):
                with patch(
                    "ASR.streaming_stage_one._load_validator_runtime",
                    return_value=(lambda *args, **kwargs: (object(), device_label), lambda *args, **kwargs: "unused", lambda model: None, lambda *args, **kwargs: {}),
                ):
                    runner = _FasterWhisperRunner(model_name="small", language="en")
                    with self.assertRaisesRegex(RuntimeError, device_label):
                        runner.start()

    def test_faster_whisper_pack_profile_scales_with_ready_backlog(self) -> None:
        """Adaptive FW packing should ramp 1 -> 2 -> 4 -> 8 as backlog grows."""
        self.assertEqual(_select_faster_whisper_pack_profile(1), (1, 20.0))
        self.assertEqual(_select_faster_whisper_pack_profile(2), (2, 45.0))
        self.assertEqual(_select_faster_whisper_pack_profile(3), (2, 45.0))
        self.assertEqual(_select_faster_whisper_pack_profile(4), (4, 90.0))
        self.assertEqual(_select_faster_whisper_pack_profile(7), (4, 90.0))
        self.assertEqual(_select_faster_whisper_pack_profile(8), (8, 180.0))

    def test_faster_whisper_batch_result_mapping_preserves_per_job_details(self) -> None:
        """Packed FW rows should map back into one terminal detail per submitted job."""
        jobs = [
            StageOneJob(1, "chunk_00001", "/tmp/chunk_00001.wav", "/tmp/chunk_00001.txt"),
            StageOneJob(2, "chunk_00002", "/tmp/chunk_00002.wav", "/tmp/chunk_00002.txt"),
        ]
        details = _map_faster_whisper_batch_results(
            jobs=jobs,
            result_rows=[
                {
                    "chunk_num": "chunk_00001",
                    "passed": True,
                    "score": 0.9,
                    "ref_text_raw": "source 1",
                    "hyp_text_raw": "transcript 1",
                    "audio_path": "/tmp/chunk_00001.wav",
                    "error": "",
                    "classification": "PASS",
                    "explanation": "ok",
                },
                {
                    "chunk_num": "chunk_00002",
                    "passed": False,
                    "score": 0.2,
                    "ref_text_raw": "source 2",
                    "hyp_text_raw": "transcript 2",
                    "audio_path": "/tmp/chunk_00002.wav",
                    "error": "",
                    "classification": "FAIL",
                    "explanation": "bad",
                },
            ],
            model_name="small",
        )

        self.assertEqual([detail.chunk_id for detail in details], ["chunk_00001", "chunk_00002"])
        self.assertTrue(details[0].passed)
        self.assertEqual(details[1].classification, "FAIL")
        self.assertEqual(details[1].transcript, "transcript 2")

    def test_faster_whisper_batch_result_mapping_treats_missing_row_as_worker_error(self) -> None:
        """Missing packed FW rows must become worker errors, not silent omissions."""
        jobs = [
            StageOneJob(1, "chunk_00001", "/tmp/chunk_00001.wav", "/tmp/chunk_00001.txt"),
            StageOneJob(2, "chunk_00002", "/tmp/chunk_00002.wav", "/tmp/chunk_00002.txt"),
        ]
        details = _map_faster_whisper_batch_results(
            jobs=jobs,
            result_rows=[
                {
                    "chunk_num": "chunk_00001",
                    "passed": True,
                    "score": 0.9,
                    "ref_text_raw": "source 1",
                    "hyp_text_raw": "transcript 1",
                    "audio_path": "/tmp/chunk_00001.wav",
                    "error": "",
                    "classification": "PASS",
                    "explanation": "ok",
                }
            ],
            model_name="small",
        )

        self.assertEqual(details[1].status, "worker_error")
        self.assertIn("no terminal result", details[1].error.lower())

    def test_faster_whisper_runner_reuses_batch_validator_for_multi_job_pack(self) -> None:
        """FW streaming runner should route multi-job work through shared batch validation."""
        captured = {}

        def _fake_load(model_name, force_device, engine, allow_cpu_fallback):
            """Mocks model loading to return dummy objects and device."""
            return object(), "gpu"

        def _fake_batch_validate(**kwargs):
            """Mimics batch validation with static results."""
            captured.update(kwargs)
            return [
                {
                    "chunk_num": "chunk_00001",
                    "passed": True,
                    "score": 0.8,
                    "ref_text_raw": "source 1",
                    "hyp_text_raw": "tx 1",
                    "audio_path": kwargs["tts_dir"] / "audio_chunks" / "chunk_00001.wav",
                    "error": "",
                    "classification": "PASS",
                    "explanation": "ok",
                },
                {
                    "chunk_num": "chunk_00002",
                    "passed": False,
                    "score": 0.3,
                    "ref_text_raw": "source 2",
                    "hyp_text_raw": "tx 2",
                    "audio_path": kwargs["tts_dir"] / "audio_chunks" / "chunk_00002.wav",
                    "error": "",
                    "classification": "FAIL",
                    "explanation": "bad",
                },
            ]

        with patch(
            "ASR.streaming_stage_one._load_validator_runtime",
            return_value=(_fake_load, lambda *args, **kwargs: "unused", lambda model: None, lambda *args, **kwargs: {}),
        ), patch(
            "ASR.streaming_stage_one._load_batch_validator",
            return_value=_fake_batch_validate,
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                jobs = [_make_chunk(root, "chunk_00001"), _make_chunk(root, "chunk_00002")]
                runner = _FasterWhisperRunner(model_name="small", language="en")
                runner.start()
                details = runner.run_batch(jobs, threshold=0.6)
                runner.close()

        self.assertEqual(captured["chunks"], ["chunk_00001", "chunk_00002"])
        self.assertEqual(captured["pack_size"], 2)
        self.assertEqual(captured["max_pack_seconds"], 45.0)
        self.assertEqual([detail.chunk_id for detail in details], ["chunk_00001", "chunk_00002"])

    def test_parakeet_runner_keeps_one_resident_backend_per_run(self) -> None:
        """One child run should instantiate one resident Parakeet backend only once."""
        _CountingParakeetBackend.instances = 0

        def _fake_score(chunk_id, original_text, transcript, threshold, audio_path, book_term_evidence=None):
            """Simulates a scoring mechanism for ASR transcripts."""
            return {
                "passed": transcript.endswith(chunk_id),
                "classification": "PASS" if transcript.endswith(chunk_id) else "FAIL",
                "score": 0.8,
                "hyp_text_raw": transcript,
                "explanation": "scored",
                "error": "",
            }

        with patch(
            "ASR.streaming_stage_one._load_parakeet_backend_class",
            return_value=_CountingParakeetBackend,
        ), patch(
            "ASR.streaming_stage_one._load_validator_runtime",
            return_value=(None, None, lambda model: None, _fake_score),
        ), patch(
            "ASR.streaming_stage_one._load_book_term_evidence_builder",
            return_value=lambda tts_dir: {"terms": ["chapter"]},
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                job_queue = queue.Queue()
                event_queue = queue.Queue()
                job_queue.put(_make_chunk(root, "chunk_00001").to_payload())
                job_queue.put(_make_chunk(root, "chunk_00002").to_payload())
                job_queue.put({"type": "stop"})
                _child_main(job_queue, event_queue, str(root), "parakeet", "nvidia/test", 0.6, "en")

                events = [event_queue.get_nowait(), event_queue.get_nowait()]

        self.assertEqual(_CountingParakeetBackend.instances, 1)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["detail"]["backend"], "parakeet_tdt")

    def test_service_reports_success_and_writes_durable_artifacts(self) -> None:
        """Successful collection should produce JSONL events and a complete summary."""
        def _worker(job_queue, event_queue, tts_dir, engine, model_name, threshold, language):
            """Continuously processes jobs from a queue."""
            while True:
                payload = job_queue.get()
                if payload == {"type": "stop"}:
                    return
                event_queue.put(
                    {
                        "type": "terminal",
                        "chunk_id": payload["chunk_id"],
                        "chunk_index": payload["chunk_index"],
                        "status": "result",
                        "error": "",
                        "detail": {
                            "chunk_id": payload["chunk_id"],
                            "chunk_index": payload["chunk_index"],
                            "status": "result",
                            "passed": payload["chunk_id"] != "chunk_00002",
                            "classification": "PASS" if payload["chunk_id"] != "chunk_00002" else "FAIL",
                            "score": 0.9 if payload["chunk_id"] != "chunk_00002" else 0.2,
                            "audio_path": payload["audio_path"],
                            "text_path": payload["text_path"],
                            "original_text": "source text",
                            "transcript": "asr text",
                            "explanation": "ok",
                            "backend": engine,
                            "model": model_name,
                            "elapsed_s": 0.12,
                            "error": "",
                        },
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = StreamingStageOneService(
                root,
                engine="faster_whisper",
                model_name="small",
                threshold=0.6,
                _context=_ThreadContext(),
                _worker_target=_worker,
            )
            service.start()
            service.submit(_make_chunk(root, "chunk_00001"))
            service.submit(_make_chunk(root, "chunk_00002"))
            summary = service.finish_collect(timeout_s=2.0)

            event_lines = service.event_log_path.read_text(encoding="utf-8").strip().splitlines()
            written_summary = json.loads(service.summary_path.read_text(encoding="utf-8"))

        self.assertTrue(summary.complete)
        self.assertEqual(summary.submitted, 2)
        self.assertEqual(summary.completed, 2)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.missing_ids, ())
        self.assertEqual(len(event_lines), 2)
        self.assertIn("details", written_summary)
        self.assertEqual(written_summary["complete"], True)
        self.assertEqual(written_summary["details"][0]["chunk_id"], "chunk_00001")

    def test_poll_drains_events_durably_before_finish_and_avoids_event_queue_backlog(self) -> None:
        """poll() should persist early events and let later submits continue draining."""
        def _worker(job_queue, event_queue, tts_dir, engine, model_name, threshold, language):
            """Continuously processes jobs from a queue."""
            while True:
                payload = job_queue.get()
                if payload == {"type": "stop"}:
                    return
                event_queue.put(
                    {
                        "type": "terminal",
                        "chunk_id": payload["chunk_id"],
                        "chunk_index": payload["chunk_index"],
                        "status": "result",
                        "error": "",
                        "detail": {
                            "chunk_id": payload["chunk_id"],
                            "chunk_index": payload["chunk_index"],
                            "status": "result",
                            "passed": True,
                            "classification": "PASS",
                            "score": 0.8,
                            "audio_path": payload["audio_path"],
                            "text_path": payload["text_path"],
                            "original_text": "source text",
                            "transcript": payload["chunk_id"],
                            "explanation": "ok",
                            "backend": engine,
                            "model": model_name,
                            "elapsed_s": 0.05,
                            "error": "",
                        },
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = StreamingStageOneService(
                root,
                engine="faster_whisper",
                model_name="small",
                threshold=0.6,
                max_queue_size=2,
                _context=_ThreadContext(),
                _worker_target=_worker,
            )
            service.start()
            for idx in range(1, 6):
                service.submit(_make_chunk(root, f"chunk_{idx:05d}"))
                time.sleep(0.02)
            drained = service.poll()
            event_lines = service.event_log_path.read_text(encoding="utf-8").strip().splitlines()
            summary = service.finish_collect(timeout_s=2.0)

        self.assertGreaterEqual(drained, 1)
        self.assertGreaterEqual(len(event_lines), 1)
        self.assertTrue(summary.complete)
        self.assertEqual(summary.completed, 5)

    def test_missing_input_is_worker_error_and_incomplete(self) -> None:
        """Missing files must become explicit worker errors, not candidate failures."""
        def _fake_load(model_name, force_device, engine, allow_cpu_fallback):
            """Loads a model with dummy outputs for validation."""
            return object(), "cuda"

        with patch(
            "ASR.streaming_stage_one._load_validator_runtime",
            return_value=(_fake_load, lambda *args, **kwargs: "unused", lambda model: None, lambda *args, **kwargs: {}),
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                missing_job = StageOneJob(
                    chunk_index=1,
                    chunk_id="chunk_00001",
                    audio_path=str(root / "audio_chunks" / "chunk_00001.wav"),
                    text_path=str(root / "text_chunks" / "chunk_00001.txt"),
                )
                service = StreamingStageOneService(
                    root,
                    engine="faster_whisper",
                    model_name="small",
                    threshold=0.6,
                    _context=_ThreadContext(),
                )
                service.start()
                service.submit(missing_job)
                summary = service.finish_collect(timeout_s=2.0)

        self.assertFalse(summary.complete)
        self.assertEqual(summary.completed, 1)
        self.assertEqual(summary.details[0].status, "worker_error")
        self.assertIn("does not exist", summary.details[0].error)

    def test_duplicate_terminal_event_marks_summary_incomplete(self) -> None:
        """Duplicate terminal results must invalidate authoritative handoff."""
        def _worker(job_queue, event_queue, tts_dir, engine, model_name, threshold, language):
            """Processes audio chunks, returning dummy results."""
            payload = job_queue.get()
            detail = {
                "chunk_id": payload["chunk_id"],
                "chunk_index": payload["chunk_index"],
                "status": "result",
                "passed": True,
                "classification": "PASS",
                "score": 1.0,
                "audio_path": payload["audio_path"],
                "text_path": payload["text_path"],
                "original_text": "source",
                "transcript": "transcript",
                "explanation": "ok",
                "backend": engine,
                "model": model_name,
                "elapsed_s": 0.1,
                "error": "",
            }
            event = {
                "type": "terminal",
                "chunk_id": payload["chunk_id"],
                "chunk_index": payload["chunk_index"],
                "status": "result",
                "error": "",
                "detail": detail,
            }
            event_queue.put(event)
            event_queue.put(event)
            job_queue.get()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = StreamingStageOneService(
                root,
                engine="faster_whisper",
                model_name="small",
                threshold=0.6,
                _context=_ThreadContext(),
                _worker_target=_worker,
            )
            service.start()
            service.submit(_make_chunk(root, "chunk_00001"))
            summary = service.finish_collect(timeout_s=2.0)

        self.assertFalse(summary.complete)
        self.assertEqual(summary.duplicate_terminal_ids, ("chunk_00001",))

    def test_child_crash_raises_immediately_on_submit(self) -> None:
        """A dead child should reject new work as a genuine service failure."""
        def _crash_worker(job_queue, event_queue, tts_dir, engine, model_name, threshold, language):
            """Starts a service with a crashing worker for testing."""
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = StreamingStageOneService(
                root,
                engine="faster_whisper",
                model_name="small",
                threshold=0.6,
                _context=_ThreadContext(),
                _worker_target=_crash_worker,
            )
            service.start()
            time.sleep(0.02)
            crash_report = root / CHILD_CRASH_REPORT_NAME
            crash_report.write_text(
                json.dumps(
                    {
                        "error_type": "RuntimeError",
                        "error": "boom",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "RuntimeError: boom"):
                service.submit(_make_chunk(root, "chunk_00001"))

        self.assertEqual(service._submitted_order, [])

    def test_child_main_writes_durable_startup_crash_report(self) -> None:
        """Runner creation/startup failures should write traceback evidence to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch(
                "ASR.streaming_stage_one._create_runner",
                side_effect=ValueError("bad runner"),
            ):
                with self.assertRaisesRegex(ValueError, "bad runner"):
                    _child_main(
                        queue.Queue(),
                        queue.Queue(),
                        str(root),
                        "faster_whisper",
                        "small",
                        0.6,
                        "en",
                    )

            payload = json.loads((root / CHILD_CRASH_REPORT_NAME).read_text(encoding="utf-8"))

        self.assertEqual(payload["report_type"], "streaming_stage_one_child_crash")
        self.assertEqual(payload["error_type"], "ValueError")
        self.assertEqual(payload["error"], "bad runner")
        self.assertIn("Traceback", payload["traceback"])

    def test_poll_records_child_crash_diagnostic_from_durable_report(self) -> None:
        """poll() should retain startup crash details even before any submit runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = StreamingStageOneService(
                root,
                engine="faster_whisper",
                model_name="small",
                threshold=0.6,
                _context=_NeverStartContext(),
            )
            service.start()
            service._process.exitcode = 1
            (root / CHILD_CRASH_REPORT_NAME).write_text(
                json.dumps(
                    {
                        "error_type": "RuntimeError",
                        "error": "CUDA init failed",
                        "traceback": "Traceback...",
                    }
                ),
                encoding="utf-8",
            )

            drained = service.poll()

        self.assertEqual(drained, 0)
        self.assertTrue(any("CUDA init failed" in err for err in service._errors))

    def test_timeout_marks_summary_incomplete(self) -> None:
        """Collection timeout must not claim a complete authoritative handoff."""
        def _hung_worker(job_queue, event_queue, tts_dir, engine, model_name, threshold, language):
            """Starts a service with a worker that hangs indefinitely."""
            time.sleep(0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = StreamingStageOneService(
                root,
                engine="faster_whisper",
                model_name="small",
                threshold=0.6,
                _context=_ThreadContext(),
                _worker_target=_hung_worker,
            )
            service.start()
            service.submit(_make_chunk(root, "chunk_00001"))
            summary = service.finish_collect(timeout_s=0.1)

        self.assertFalse(summary.complete)
        self.assertTrue(summary.timed_out)
        self.assertIn("Timed out", summary.errors[0])

    def test_parent_start_does_not_load_models(self) -> None:
        """Parent setup must not construct ASR models before the child runs."""
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ASR.streaming_stage_one._create_runner",
            side_effect=AssertionError("parent loaded child runner"),
        ):
            service = StreamingStageOneService(
                temp_dir,
                engine="parakeet",
                model_name="nvidia/test",
                threshold=0.6,
                _context=_NeverStartContext(),
            )
            service.start()
            service.close()

    def test_submit_queue_full_defers_jobs_without_disabling_submission(self) -> None:
        """Queue saturation should create deferred backlog entries instead of failing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = _DeferredOnlyContext(fail_after=1)
            service = StreamingStageOneService(
                root,
                engine="faster_whisper",
                model_name="small",
                threshold=0.6,
                _context=context,
            )
            service.start()
            service.submit(_make_chunk(root, "chunk_00001"))
            service.submit(_make_chunk(root, "chunk_00002"))
            service.submit(_make_chunk(root, "chunk_00003"))
            event_lines = service.event_log_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(service._submitted_order), 3)
        self.assertEqual(tuple(service._submitted_by_id), ("chunk_00001", "chunk_00002", "chunk_00003"))
        self.assertEqual(len(service._deferred_jobs), 2)
        self.assertEqual(context.job_queue.put_count, 1)
        self.assertTrue(any('"type": "deferred_submission"' in line for line in event_lines))

    def test_deferred_backlog_drains_to_completion_with_slow_worker(self) -> None:
        """Deferred jobs should complete in order once a slow worker frees queue slots."""
        def _slow_worker(job_queue, event_queue, tts_dir, engine, model_name, threshold, language):
            """Processes jobs in a queue with artificial delays."""
            while True:
                payload = job_queue.get()
                if payload == {"type": "stop"}:
                    return
                time.sleep(0.02)
                event_queue.put(
                    {
                        "type": "terminal",
                        "chunk_id": payload["chunk_id"],
                        "chunk_index": payload["chunk_index"],
                        "status": "result",
                        "error": "",
                        "detail": {
                            "chunk_id": payload["chunk_id"],
                            "chunk_index": payload["chunk_index"],
                            "status": "result",
                            "passed": True,
                            "classification": "PASS",
                            "score": 0.9,
                            "audio_path": payload["audio_path"],
                            "text_path": payload["text_path"],
                            "original_text": "source",
                            "transcript": payload["chunk_id"],
                            "explanation": "ok",
                            "backend": engine,
                            "model": model_name,
                            "elapsed_s": 0.05,
                            "error": "",
                        },
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = StreamingStageOneService(
                root,
                engine="faster_whisper",
                model_name="small",
                threshold=0.6,
                max_queue_size=1,
                _context=_ThreadContext(),
                _worker_target=_slow_worker,
            )
            service.start()
            for idx in range(1, 8):
                service.submit(_make_chunk(root, f"chunk_{idx:05d}"))
            summary = service.finish_collect(timeout_s=3.0)
            event_lines = service.event_log_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertTrue(summary.complete)
        self.assertEqual(summary.submitted, 7)
        self.assertEqual(summary.completed, 7)
        self.assertEqual(len(summary.details), 7)
        self.assertTrue(any('"type": "deferred_submission"' in line for line in event_lines))

    def test_thin_deferred_backlog_can_record_many_jobs_without_duplicates(self) -> None:
        """Deferred backlog should store only thin metadata while preserving order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = _DeferredOnlyContext(fail_after=1)
            service = StreamingStageOneService(
                root,
                engine="faster_whisper",
                model_name="small",
                threshold=0.6,
                _context=context,
            )
            service.start()
            for idx in range(1000):
                service.submit(_make_chunk(root, f"chunk_{idx:05d}"))

            lines = service.event_log_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(service._submitted_order[0], "chunk_00000")
        self.assertEqual(service._submitted_order[-1], "chunk_00999")
        self.assertEqual(len(service._submitted_order), 1000)
        self.assertEqual(len(service._submitted_by_id), 1000)
        self.assertEqual(len(service._deferred_jobs), 999)
        self.assertEqual(sum('"type": "deferred_submission"' in line for line in lines), 999)

    def test_start_resets_old_artifacts_before_new_run(self) -> None:
        """start() should delete prior run artifacts before a fresh run begins."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_path = root / "asr_streaming_stage_one_events.jsonl"
            summary_path = root / "asr_streaming_stage_one_summary.json"
            event_path.write_text("old event\n", encoding="utf-8")
            summary_path.write_text('{"old": true}\n', encoding="utf-8")

            service = StreamingStageOneService(
                root,
                engine="faster_whisper",
                model_name="small",
                threshold=0.6,
                _context=_NeverStartContext(),
            )
            service.start()

        self.assertFalse(event_path.exists())
        self.assertFalse(summary_path.exists())

    def test_parakeet_default_model_and_book_term_evidence_are_used(self) -> None:
        """Parakeet should keep backend default unless model starts with nvidia/."""
        captured = {}

        class _DefaultModelBackend(_CountingParakeetBackend):
            """A backend class simulating default behavior."""
            def __init__(self, model_name="parakeet-default", device="cuda") -> None:
                """Initializes the class with model and device settings."""
                super().__init__(model_name=model_name, device=device)

        def _fake_score(chunk_id, original_text, transcript, threshold, audio_path, book_term_evidence=None):
            """Scores text against a reference with a different structure."""
            captured["book_term_evidence"] = book_term_evidence
            return {
                "passed": True,
                "classification": "PASS",
                "score": 0.7,
                "hyp_text_raw": transcript,
                "explanation": "ok",
                "error": "",
            }

        with patch(
            "ASR.streaming_stage_one._load_parakeet_backend_class",
            return_value=_DefaultModelBackend,
        ), patch(
            "ASR.streaming_stage_one._load_validator_runtime",
            return_value=(None, None, lambda model: None, _fake_score),
        ), patch(
            "ASR.streaming_stage_one._load_book_term_evidence_builder",
            return_value=lambda tts_dir: {"terms": ["abbey", "school"]},
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                job = _make_chunk(root, "chunk_00001")
                runner = _ParakeetRunner(model_name="small", tts_dir=root)
                runner.start()
                detail = runner.run(job, threshold=0.6)
                runner.close()

        self.assertEqual(detail.model, "parakeet-default")
        self.assertEqual(captured["book_term_evidence"], {"terms": ["abbey", "school"]})

    def test_finish_collect_times_out_when_stop_sentinel_cannot_be_enqueued(self) -> None:
        """finish_collect() must return incomplete instead of hanging on a full stop queue."""
        event = {
            "type": "terminal",
            "chunk_id": "chunk_00001",
            "chunk_index": 1,
            "status": "result",
            "error": "",
            "detail": {
                "chunk_id": "chunk_00001",
                "chunk_index": 1,
                "status": "result",
                "passed": True,
                "classification": "PASS",
                "score": 0.9,
                "audio_path": "/tmp/chunk_00001.wav",
                "text_path": "/tmp/chunk_00001.txt",
                "original_text": "source",
                "transcript": "chunk_00001",
                "explanation": "ok",
                "backend": "faster_whisper",
                "model": "small",
                "elapsed_s": 0.05,
                "error": "",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = StreamingStageOneService(
                root,
                engine="faster_whisper",
                model_name="small",
                threshold=0.6,
                _context=_StopDeadlockContext(
                    stop_release_after=999,
                    queued_payloads=[{"chunk_id": "queued"}],
                    events=[event],
                ),
            )
            service.start()
            service._submitted_order = ["chunk_00001"]
            service._submitted_by_id = {
                "chunk_00001": StageOneJob(
                    chunk_index=1,
                    chunk_id="chunk_00001",
                    audio_path="/tmp/chunk_00001.wav",
                    text_path="/tmp/chunk_00001.txt",
                )
            }
            summary = service.finish_collect(timeout_s=0.1)

        self.assertFalse(summary.complete)
        self.assertTrue(summary.timed_out)
        self.assertTrue(any("stop sentinel" in err for err in summary.errors))

    def test_finish_collect_succeeds_when_poll_drains_enough_for_stop_enqueue(self) -> None:
        """Event draining should eventually free the stop path and allow clean completion."""
        event = {
            "type": "terminal",
            "chunk_id": "chunk_00001",
            "chunk_index": 1,
            "status": "result",
            "error": "",
            "detail": {
                "chunk_id": "chunk_00001",
                "chunk_index": 1,
                "status": "result",
                "passed": True,
                "classification": "PASS",
                "score": 0.9,
                "audio_path": "/tmp/chunk_00001.wav",
                "text_path": "/tmp/chunk_00001.txt",
                "original_text": "source",
                "transcript": "chunk_00001",
                "explanation": "ok",
                "backend": "faster_whisper",
                "model": "small",
                "elapsed_s": 0.05,
                "error": "",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = StreamingStageOneService(
                root,
                engine="faster_whisper",
                model_name="small",
                threshold=0.6,
                _context=_StopDeadlockExitedContext(
                    stop_release_after=2,
                    queued_payloads=[{"chunk_id": "queued"}],
                    events=[event],
                ),
            )
            service.start()
            service._submitted_order = ["chunk_00001"]
            service._submitted_by_id = {
                "chunk_00001": StageOneJob(
                    chunk_index=1,
                    chunk_id="chunk_00001",
                    audio_path="/tmp/chunk_00001.wav",
                    text_path="/tmp/chunk_00001.txt",
                )
            }
            summary = service.finish_collect(timeout_s=0.2)

        self.assertTrue(summary.complete)
        self.assertFalse(summary.timed_out)

    def test_materialize_incomplete_summary_rejects_and_preserves_old_candidate_file(self) -> None:
        """Incomplete summaries must fail before replacing prior authoritative candidates."""
        summary = _make_summary(
            (_make_detail(chunk_id="chunk_00001", chunk_index=1, passed=False),),
            complete=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_failure_path = root / AUTHORITATIVE_FAILURES_NAME
            old_failure_path.write_text('[{"old": true}]\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "incomplete"):
                materialize_stage_one_summary(summary, root)

            self.assertEqual(old_failure_path.read_text(encoding="utf-8"), '[{"old": true}]\n')
            self.assertFalse((root / ALL_RESULTS_AUDIT_NAME).exists())

    def test_materialize_summary_writes_audit_and_failed_candidates_only(self) -> None:
        """Materialization should keep full evidence while exporting only failed candidates."""
        summary = _make_summary(
            (
                _make_detail(chunk_id="chunk_00001", chunk_index=1, passed=True, classification="PASS"),
                _make_detail(
                    chunk_id="chunk_00002",
                    chunk_index=2,
                    passed=False,
                    classification="FAIL",
                    score=0.14,
                    backend="parakeet",
                    model="nvidia/parakeet-tdt-0.6b-v3",
                ),
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = materialize_stage_one_summary(summary, root)
            audit_payload = json.loads((root / ALL_RESULTS_AUDIT_NAME).read_text(encoding="utf-8"))
            failure_rows = json.loads((root / AUTHORITATIVE_FAILURES_NAME).read_text(encoding="utf-8"))

        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(audit_payload["stage_one_source"], "during_tts_streaming")
        self.assertEqual(audit_payload["candidate_count"], 1)
        self.assertEqual(len(audit_payload["records"]), 2)
        self.assertEqual(failure_rows[0]["chunk_index"], 2)
        self.assertEqual(failure_rows[0]["filename"], "/tmp/chunk_00002.wav")
        self.assertEqual(failure_rows[0]["audio_path"], "/tmp/chunk_00002.wav")
        self.assertEqual(failure_rows[0]["score"], 0.14)
        self.assertEqual(failure_rows[0]["transcribed_text"], "transcript chunk_00002")
        self.assertEqual(failure_rows[0]["classification"], "FAIL")
        self.assertEqual(failure_rows[0]["stage_one_backend"], "parakeet")
        self.assertEqual(failure_rows[0]["stage_one_model"], "nvidia/parakeet-tdt-0.6b-v3")
        self.assertIn("stage_one_elapsed_s", failure_rows[0])

    def test_materialize_worker_error_duplicate_and_missing_index_reject_cleanly(self) -> None:
        """Malformed summaries must be rejected before any output files are created."""
        invalid_summaries = (
            _make_summary(
                (_make_detail(chunk_id="chunk_00003", chunk_index=3, status="worker_error", error="boom"),)
            ),
            _make_summary(
                (_make_detail(chunk_id="chunk_00004", chunk_index=4),),
                duplicate_terminal_ids=("chunk_00004",),
            ),
            _make_summary(
                (_make_detail(chunk_id="chunk_00005", chunk_index=None, passed=False, classification="FAIL"),)
            ),
        )

        for summary in invalid_summaries:
            with self.subTest(summary=summary):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    with self.assertRaises(ValueError):
                        materialize_stage_one_summary(summary, root)
                    self.assertFalse((root / ALL_RESULTS_AUDIT_NAME).exists())
                    self.assertFalse((root / AUTHORITATIVE_FAILURES_NAME).exists())

    def test_run_pipeline_batch_validation_respects_max_pack_seconds(self) -> None:
        """Validator should split established packs before adding a duration-overflow clip."""
        durations = {
            "chunk_00001": 15.0,
            "chunk_00002": 15.0,
            "chunk_00003": 15.0,
        }
        packed_sizes = []

        def _fake_load_audio(audio_path):
            """Loads audio from a path, returning dummy data."""
            chunk_id = Path(audio_path).stem
            return object(), durations[chunk_id]

        def _fake_pack_audio_regions(pack, silence_s):
            """Packs audio regions into chunks for processing."""
            packed_sizes.append(len(pack))
            return object(), [
                {
                    "chunk_num": item["chunk_num"],
                    "ref_text": item["ref_text"],
                    "audio_path": item["audio_path"],
                    "audio": item["audio"],
                }
                for item in pack
            ]

        def _fake_transcribe_segments(asr_model, packed, language, vad_filter):
            """Transcribes audio segments but returns an empty list."""
            return []

        def _fake_assign_segments_to_regions(segments, regions):
            """Assigns segments to regions using a template format."""
            return {region["chunk_num"]: f"tx {region['chunk_num']}" for region in regions}

        def _fake_score(chunk_num, ref_text, hyp_text, threshold, audio_path, book_term_evidence=None):
            """Simulates a scoring process with a fixed score and classification."""
            return {
                "passed": True,
                "score": 0.9,
                "classification": "PASS",
                "hyp_text_raw": hyp_text,
                "explanation": "ok",
                "error": "",
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for chunk_id in durations:
                _make_chunk(root, chunk_id)
            with patch("ASR.asr_validator.load_audio_mono_16k", side_effect=_fake_load_audio), patch(
                "ASR.asr_validator.pack_audio_regions",
                side_effect=_fake_pack_audio_regions,
            ), patch(
                "ASR.asr_validator.transcribe_segments",
                side_effect=_fake_transcribe_segments,
            ), patch(
                "ASR.asr_validator.assign_segments_to_regions",
                side_effect=_fake_assign_segments_to_regions,
            ), patch(
                "ASR.asr_validator.score_asr_pair",
                side_effect=_fake_score,
            ):
                results = run_pipeline_batch_validation(
                    tts_dir=root,
                    chunks=list(durations),
                    asr_model=object(),
                    threshold=0.6,
                    load_workers=1,
                    score_workers=1,
                    pack_size=4,
                    max_pack_seconds=30.0,
                )

        self.assertEqual(len(results), 3)
        self.assertEqual(packed_sizes, [2])
