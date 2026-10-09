"""Spawned Stage 1 ASR worker foundation for future streaming integration.

This module is intentionally inert today: nothing imports it from the live
generation path yet.  Its job is to define a small public API for a single
resident Stage 1 child process that validates completed chunk files and emits
durable result evidence for later parent integration.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional


EVENT_LOG_NAME = "asr_streaming_stage_one_events.jsonl"
SUMMARY_NAME = "asr_streaming_stage_one_summary.json"
CHILD_CRASH_REPORT_NAME = "asr_streaming_stage_one_child_crash.json"
ALL_RESULTS_AUDIT_NAME = "asr_during_tts_stage_one.json"
AUTHORITATIVE_FAILURES_NAME = "asr_new_failures.json"
ENGINE_CHOICES = frozenset({"faster_whisper", "parakeet"})
_STOP_SENTINEL = {"type": "stop"}
_FW_MAX_READY_BATCH = 8
_FW_PACK_SECONDS_BY_SIZE = {
    1: 20.0,
    2: 45.0,
    4: 90.0,
    8: 180.0,
}


@dataclass(frozen=True)
class StageOneJob:
    """Immutable Stage 1 work item for one completed chunk.

    Args:
        chunk_index: Numeric chunk index when the caller has one.
        chunk_id: Stable chunk identifier such as ``chunk_00012``.
        audio_path: Final WAV path for Stage 1 validation.
        text_path: Source text path that produced the WAV.
    """

    chunk_index: Optional[int]
    chunk_id: str
    audio_path: str
    text_path: str

    def to_payload(self) -> Dict[str, Any]:
        """Return a multiprocessing-safe payload for the child queue."""
        return asdict(self)


@dataclass(frozen=True)
class StageOneDetail:
    """Terminal Stage 1 evidence for one chunk.

    These fields stay close to the canonical failure-report needs so Stage 3 can
    consume them without reconstructing context from transient worker logs.
    """

    chunk_id: str
    chunk_index: Optional[int]
    status: str
    passed: bool
    classification: str
    score: float
    audio_path: str
    text_path: str
    original_text: str
    transcript: str
    explanation: str
    backend: str
    model: str
    elapsed_s: float
    error: str

    def to_payload(self) -> Dict[str, Any]:
        """Return a JSON-safe representation for event and summary writes."""
        return asdict(self)


@dataclass(frozen=True)
class StageOneRunSummary:
    """Typed summary returned by ``finish_collect``.

    ``complete`` is strict: it becomes true only when every submitted chunk has
    exactly one terminal result, the child exited cleanly, and no worker-level
    errors or collection faults occurred.
    """

    submitted: int
    completed: int
    failed: int
    missing_ids: tuple[str, ...]
    complete: bool
    details: tuple[StageOneDetail, ...]
    errors: tuple[str, ...]
    duplicate_terminal_ids: tuple[str, ...]
    child_exitcode: Optional[int]
    timed_out: bool
    event_log_path: str
    summary_path: str
    wall_s: float = 0.0

    def to_payload(self) -> Dict[str, Any]:
        """Return a JSON-safe run summary."""
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "failed": self.failed,
            "missing_ids": list(self.missing_ids),
            "complete": self.complete,
            "details": [detail.to_payload() for detail in self.details],
            "errors": list(self.errors),
            "duplicate_terminal_ids": list(self.duplicate_terminal_ids),
            "child_exitcode": self.child_exitcode,
            "timed_out": self.timed_out,
            "event_log_path": self.event_log_path,
            "summary_path": self.summary_path,
            "wall_s": self.wall_s,
        }


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one durable JSON line immediately after receipt.

    Immediate appends matter because Stage 3 will rely on this log even if the
    parent dies mid-collection or a future stream handoff fails late.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_summary_json(path: Path, summary: StageOneRunSummary) -> None:
    """Write the authoritative final summary JSON for one Stage 1 run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_payload(), handle, indent=2, ensure_ascii=False, sort_keys=True)


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Replace one JSON artifact only after the full payload is ready on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _write_child_crash_report(
    tts_dir: str | Path,
    engine: str,
    model_name: str,
    language: str,
    exc: BaseException,
) -> Path:
    """Persist one child-startup crash report with traceback before the child exits."""
    report_path = Path(tts_dir) / CHILD_CRASH_REPORT_NAME
    _atomic_write_json(
        report_path,
        {
            "report_type": "streaming_stage_one_child_crash",
            "stage": "startup_or_runtime",
            "engine": engine,
            "model_name": model_name,
            "language": language,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return report_path


def _candidate_row_from_detail(detail: StageOneDetail) -> Dict[str, Any]:
    """Convert one failed Stage 1 result into canonical candidate-row fields."""
    return {
        "chunk_index": detail.chunk_index,
        "chunk_id": detail.chunk_id,
        "filename": detail.audio_path,
        "audio_path": detail.audio_path,
        "text_path": detail.text_path,
        "score": detail.score,
        "error": detail.error,
        "explanation": detail.explanation,
        "original_text": detail.original_text,
        "transcribed_text": detail.transcript,
        "classification": detail.classification,
        "stage_one_backend": detail.backend,
        "stage_one_model": detail.model,
        "stage_one_elapsed_s": detail.elapsed_s,
    }


def _validate_materializable_summary(summary: StageOneRunSummary) -> None:
    """Reject incomplete or malformed Stage 1 summaries before any file writes.

    Streaming evidence becomes authoritative only when every retained terminal
    detail is internally consistent and no worker fault or duplicate terminal
    path could have polluted the result set.
    """
    if not summary.complete:
        raise ValueError("Streaming Stage 1 summary is incomplete and cannot be materialized")
    if summary.duplicate_terminal_ids:
        raise ValueError("Streaming Stage 1 summary contains duplicate terminal chunk ids")

    seen_chunk_ids: set[str] = set()
    for detail in summary.details:
        if not detail.chunk_id:
            raise ValueError("Streaming Stage 1 detail is missing chunk_id")
        if detail.chunk_id in seen_chunk_ids:
            raise ValueError(f"Streaming Stage 1 detail repeats chunk_id {detail.chunk_id}")
        seen_chunk_ids.add(detail.chunk_id)
        if detail.chunk_index is None:
            raise ValueError(f"Streaming Stage 1 detail {detail.chunk_id} is missing chunk_index")
        if detail.status != "result":
            raise ValueError(
                f"Streaming Stage 1 detail {detail.chunk_id} has non-result status {detail.status!r}"
            )
        if not detail.audio_path or not detail.text_path:
            raise ValueError(f"Streaming Stage 1 detail {detail.chunk_id} is missing file paths")

    if len(summary.details) != summary.completed:
        raise ValueError("Streaming Stage 1 summary completed count does not match retained details")


def materialize_stage_one_summary(
    summary: StageOneRunSummary,
    tts_dir: str | Path,
    *,
    all_results_name: str = ALL_RESULTS_AUDIT_NAME,
    failure_filename: str = AUTHORITATIVE_FAILURES_NAME,
) -> Dict[str, Any]:
    """Write strict-complete during-TTS Stage 1 evidence into canonical JSON files.

    Args:
        summary: Finished Stage 1 summary returned by ``finish_collect``.
        tts_dir: TTS output directory that will hold authoritative artifacts.
        all_results_name: Filename for the all-results audit payload.
        failure_filename: Canonical Stage 1 candidate filename consumed by Stage 2.

    Returns:
        Metadata describing the written audit and candidate files.

    Raises:
        ValueError: If the supplied summary is incomplete and cannot become
            authoritative Stage 1 evidence.
    """
    _validate_materializable_summary(summary)

    tts_root = Path(tts_dir)
    tts_root.mkdir(parents=True, exist_ok=True)
    audit_path = tts_root / all_results_name
    failure_path = tts_root / failure_filename
    details_payload = [detail.to_payload() for detail in summary.details]
    candidate_rows = [
        _candidate_row_from_detail(detail)
        for detail in summary.details
        if detail.status == "result" and not detail.passed
    ]
    audit_payload = {
        "report_type": "during_tts_stage_one_all_results",
        "stage_one_source": "during_tts_streaming",
        "summary": summary.to_payload(),
        "candidate_count": len(candidate_rows),
        "records": details_payload,
    }
    _atomic_write_json(audit_path, audit_payload)
    _atomic_write_json(failure_path, candidate_rows)
    first_detail = summary.details[0] if summary.details else None
    return {
        "audit_path": str(audit_path),
        "failure_log_path": str(failure_path),
        "candidate_count": len(candidate_rows),
        "backend": first_detail.backend if first_detail else "",
        "model": first_detail.model if first_detail else "",
        "submitted": summary.submitted,
        "completed": summary.completed,
        "failed": summary.failed,
        "details": details_payload,
    }


def _grace_sleep(seconds: float) -> None:
    """Pause briefly while waiting for late queue delivery after child exit."""
    time.sleep(seconds)


def _load_validator_runtime() -> tuple[Callable[..., Any], Callable[..., str], Callable[[Any], None], Callable[..., Dict[str, Any]]]:
    """Import public ASR validator seams lazily inside the child process only."""
    from ASR.asr_validator import (
        cleanup_asr_model,
        load_asr_model_adaptive,
        score_asr_pair,
        transcribe_audio,
    )

    return load_asr_model_adaptive, transcribe_audio, cleanup_asr_model, score_asr_pair


def _load_batch_validator() -> Callable[..., list[Dict[str, Any]]]:
    """Import packed faster-whisper batch validation lazily inside the child only."""
    from ASR.asr_validator import run_pipeline_batch_validation

    return run_pipeline_batch_validation


def _load_parakeet_backend_class() -> type[Any]:
    """Import the resident Parakeet backend lazily inside the child process only."""
    from ASR.verification_backends import ParakeetTDTBackend

    return ParakeetTDTBackend


def _load_book_term_evidence_builder() -> Callable[[Path | str], Dict[str, Any]]:
    """Import the public recurring-term builder lazily inside the child only."""
    from ASR.asr_validator import build_tts_book_term_evidence

    return build_tts_book_term_evidence


class _FasterWhisperRunner:
    """Single-model Stage 1 runner that forbids silent CPU fallback."""

    def __init__(self, model_name: str, language: str) -> None:
        """Load one GPU-only faster-whisper model for the full child lifetime."""
        self._model_name = model_name
        self._language = language
        self._asr_model: Any = None
        self._cleanup: Optional[Callable[[Any], None]] = None
        self._score: Optional[Callable[..., Dict[str, Any]]] = None
        self._transcribe: Optional[Callable[..., str]] = None
        self._batch_validate: Optional[Callable[..., list[Dict[str, Any]]]] = None

    def start(self) -> None:
        """Construct the single resident faster-whisper model on GPU only.

        The shared loader currently reports successful Faster-Whisper GPU loads
        with canonical labels such as ``cuda`` and ``gpu``. Any CPU label or
        failure sentinel must still be rejected here because Stage 1 streaming
        is explicitly GPU-only and must not silently fall back.
        """
        load_model, transcribe_audio, cleanup_model, score_pair = _load_validator_runtime()
        model, device = load_model(
            self._model_name,
            force_device="cuda",
            engine="faster_whisper",
            allow_cpu_fallback=False,
        )
        normalized_device = str(device or "").strip().lower()
        if model is None or normalized_device not in {"cuda", "gpu"}:
            raise RuntimeError(f"faster_whisper GPU load failed: device={device}")
        self._asr_model = model
        self._transcribe = transcribe_audio
        self._cleanup = cleanup_model
        self._score = score_pair
        self._batch_validate = _load_batch_validator()

    def run(self, job: StageOneJob, threshold: float) -> StageOneDetail:
        """Transcribe one validated chunk and return terminal Stage 1 evidence."""
        if self._asr_model is None or self._transcribe is None or self._score is None:
            raise RuntimeError("faster_whisper runner was not started")
        return _score_job(
            job,
            threshold=threshold,
            backend="faster_whisper",
            model_name=self._model_name,
            transcribe_fn=lambda audio_path: self._transcribe(
                self._asr_model,
                audio_path,
                language=self._language or "en",
                vad_filter=False,
            ),
            score_fn=self._score,
        )

    def run_batch(self, jobs: list[StageOneJob], threshold: float) -> list[StageOneDetail]:
        """Validate one adaptive faster-whisper pack through shared packed decoding.

        This reuses the existing batch validator's packed decode, per-region
        mapping, and solo retry behavior so streaming Stage 1 does not fork a
        second transcript/scoring implementation.
        """
        if not jobs:
            return []
        if len(jobs) == 1:
            return [self.run(jobs[0], threshold=threshold)]
        if self._asr_model is None or self._batch_validate is None:
            raise RuntimeError("faster_whisper runner was not started")

        pack_size, max_pack_seconds = _select_faster_whisper_pack_profile(len(jobs))
        tts_dir = _infer_tts_dir_from_job(jobs[0])
        result_rows = self._batch_validate(
            tts_dir=tts_dir,
            chunks=[job.chunk_id for job in jobs],
            asr_model=self._asr_model,
            threshold=threshold,
            language=self._language or "en",
            load_workers=min(4, len(jobs)),
            score_workers=min(4, len(jobs)),
            vad_filter=False,
            pack_size=pack_size,
            pack_silence_s=0.75,
            max_pack_seconds=max_pack_seconds,
        )
        return _map_faster_whisper_batch_results(
            jobs=jobs,
            result_rows=result_rows,
            model_name=self._model_name,
        )

    def close(self) -> None:
        """Release the resident faster-whisper model after the child stops."""
        if self._asr_model is not None and self._cleanup is not None:
            self._cleanup(self._asr_model)
        self._asr_model = None


class _ParakeetRunner:
    """Single-backend Stage 1 runner for one resident Parakeet instance."""

    def __init__(self, model_name: str, tts_dir: str | Path) -> None:
        """Remember the requested model and TTS directory without parent loading."""
        self._model_name = model_name
        self._tts_dir = Path(tts_dir)
        self._backend: Any = None
        self._score: Optional[Callable[..., Dict[str, Any]]] = None
        self._book_term_evidence: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        """Construct exactly one resident CUDA Parakeet backend for the run."""
        backend_cls = _load_parakeet_backend_class()
        _, _, _, score_pair = _load_validator_runtime()
        build_book_terms = _load_book_term_evidence_builder()
        self._score = score_pair
        backend_kwargs: Dict[str, Any] = {"device": "cuda"}
        if self._model_name.startswith("nvidia/"):
            backend_kwargs["model_name"] = self._model_name
        self._backend = backend_cls(**backend_kwargs)
        self._book_term_evidence = build_book_terms(self._tts_dir)

    def run(self, job: StageOneJob, threshold: float) -> StageOneDetail:
        """Transcribe one validated chunk through the resident Parakeet backend."""
        if self._backend is None or self._score is None:
            raise RuntimeError("Parakeet runner was not started")
        return _score_job(
            job,
            threshold=threshold,
            backend="parakeet",
            model_name=str(getattr(self._backend, "model_name", self._model_name)),
            transcribe_fn=self._transcribe_one,
            score_fn=self._score,
            book_term_evidence=self._book_term_evidence,
        )

    def _transcribe_one(self, audio_path: str) -> tuple[str, float, str]:
        """Run the resident backend on one path while preserving backend evidence."""
        paths = [Path(audio_path)]
        evidence = self._backend.transcribe_paths(paths)
        if not evidence:
            raise RuntimeError("Parakeet returned no transcript")
        item = evidence[0]
        backend_name = str(getattr(item, "backend", "parakeet"))
        return str(getattr(item, "text", "") or ""), float(getattr(item, "elapsed_s", 0.0) or 0.0), backend_name

    def close(self) -> None:
        """Release the resident Parakeet backend after the child stops."""
        if self._backend is not None:
            self._backend.close()
        self._backend = None


def _score_job(
    job: StageOneJob,
    threshold: float,
    backend: str,
    model_name: str,
    transcribe_fn: Callable[[str], Any],
    score_fn: Callable[..., Dict[str, Any]],
    book_term_evidence: Optional[Dict[str, Any]] = None,
) -> StageOneDetail:
    """Validate one terminal chunk or return a worker error detail.

    Missing/unreadable files are treated as worker faults, not speech failures,
    because Stage 3 cannot trust a candidate report that never saw the inputs.
    """
    audio_path = Path(job.audio_path)
    text_path = Path(job.text_path)
    if not audio_path.is_file():
        return _worker_error_detail(job, backend, model_name, "Audio file does not exist")
    if not text_path.is_file():
        return _worker_error_detail(job, backend, model_name, "Text file does not exist")
    try:
        original_text = text_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return _worker_error_detail(job, backend, model_name, f"Text file unreadable: {exc}")
    if not original_text:
        return _worker_error_detail(job, backend, model_name, "Text file is empty")

    started = time.monotonic()
    try:
        transcribed = transcribe_fn(str(audio_path))
        if isinstance(transcribed, tuple):
            transcript = str(transcribed[0] or "")
            elapsed_s = float(transcribed[1] or 0.0)
            backend_name = str(transcribed[2] or backend)
        else:
            transcript = str(transcribed or "")
            elapsed_s = time.monotonic() - started
            backend_name = backend
    except Exception as exc:
        return _worker_error_detail(job, backend, model_name, f"ASR transcription error: {exc}")

    try:
        scored = score_fn(
            job.chunk_id,
            original_text,
            transcript,
            threshold,
            audio_path=str(audio_path),
            book_term_evidence=book_term_evidence,
        )
    except Exception as exc:
        return _worker_error_detail(job, backend_name, model_name, f"ASR comparison error: {exc}")
    if elapsed_s <= 0.0:
        elapsed_s = time.monotonic() - started
    return StageOneDetail(
        chunk_id=job.chunk_id,
        chunk_index=job.chunk_index,
        status="result",
        passed=bool(scored.get("passed", False)),
        classification=str(scored.get("classification", "FAIL") or "FAIL"),
        score=float(scored.get("score", 0.0) or 0.0),
        audio_path=str(audio_path),
        text_path=str(text_path),
        original_text=original_text,
        transcript=str(scored.get("hyp_text_raw", transcript) or transcript),
        explanation=str(scored.get("explanation", "") or ""),
        backend=backend_name,
        model=model_name,
        elapsed_s=elapsed_s,
        error=str(scored.get("error", "") or ""),
    )


def _worker_error_detail(job: StageOneJob, backend: str, model_name: str, error: str) -> StageOneDetail:
    """Return a terminal worker-fault detail for one chunk."""
    return StageOneDetail(
        chunk_id=job.chunk_id,
        chunk_index=job.chunk_index,
        status="worker_error",
        passed=False,
        classification="WORKER_ERROR",
        score=0.0,
        audio_path=job.audio_path,
        text_path=job.text_path,
        original_text="",
        transcript="",
        explanation="Stage 1 could not validate the requested chunk inputs.",
        backend=backend,
        model=model_name,
        elapsed_s=0.0,
        error=error,
    )


def _create_runner(engine: str, model_name: str, language: str, tts_dir: str | Path) -> Any:
    """Create one child-only engine runner from the supported Stage 1 engines."""
    normalized = str(engine or "").strip().lower().replace("-", "_")
    if normalized == "faster_whisper":
        return _FasterWhisperRunner(model_name=model_name, language=language)
    if normalized == "parakeet":
        return _ParakeetRunner(model_name=model_name, tts_dir=tts_dir)
    raise ValueError(f"Unsupported Stage 1 engine: {engine}")


def _infer_tts_dir_from_job(job: StageOneJob) -> Path:
    """Derive the shared TTS directory from one chunk job's final sidecar paths."""
    audio_root = Path(job.audio_path).resolve().parent.parent
    text_root = Path(job.text_path).resolve().parent.parent
    if audio_root != text_root:
        raise ValueError(
            f"Stage 1 job {job.chunk_id} does not share one TTS root across audio/text paths"
        )
    return audio_root


def _select_faster_whisper_pack_profile(backlog_count: int) -> tuple[int, float]:
    """Choose the fastest safe faster-whisper pack profile for current backlog."""
    ready = max(1, int(backlog_count))
    if ready <= 1:
        return 1, _FW_PACK_SECONDS_BY_SIZE[1]
    if ready <= 3:
        return 2, _FW_PACK_SECONDS_BY_SIZE[2]
    if ready <= 7:
        return 4, _FW_PACK_SECONDS_BY_SIZE[4]
    return 8, _FW_PACK_SECONDS_BY_SIZE[8]


def _map_faster_whisper_batch_results(
    *,
    jobs: list[StageOneJob],
    result_rows: Iterable[Dict[str, Any]],
    model_name: str,
) -> list[StageOneDetail]:
    """Convert shared batch-validator rows back into one terminal detail per job.

    Missing rows are treated as worker errors because authoritative streaming
    handoff requires exactly one terminal outcome for every submitted chunk id.
    """
    rows_by_id = {
        str(row.get("chunk_num") or ""): dict(row)
        for row in result_rows
        if str(row.get("chunk_num") or "")
    }
    details: list[StageOneDetail] = []
    for job in jobs:
        row = rows_by_id.get(job.chunk_id)
        if row is None:
            details.append(
                _worker_error_detail(
                    job,
                    "faster_whisper",
                    model_name,
                    "Packed faster-whisper batch returned no terminal result for chunk",
                )
            )
            continue
        details.append(
            StageOneDetail(
                chunk_id=job.chunk_id,
                chunk_index=job.chunk_index,
                status="worker_error" if row.get("error") else "result",
                passed=bool(row.get("passed", False)),
                classification=str(
                    row.get(
                        "classification",
                        "WORKER_ERROR" if row.get("error") else "FAIL",
                    )
                    or ("WORKER_ERROR" if row.get("error") else "FAIL")
                ),
                score=float(row.get("score", 0.0) or 0.0),
                audio_path=str(row.get("audio_path") or job.audio_path),
                text_path=job.text_path,
                original_text=str(row.get("ref_text_raw", "") or ""),
                transcript=str(row.get("hyp_text_raw", "") or ""),
                explanation=(
                    "Stage 1 could not validate the requested chunk inputs."
                    if row.get("error")
                    else str(row.get("explanation", "") or "")
                ),
                backend="faster_whisper",
                model=model_name,
                elapsed_s=0.0,
                error=str(row.get("error", "") or ""),
            )
        )
    return details


def _drain_ready_faster_whisper_jobs(
    job_queue: Any,
    first_payload: Dict[str, Any],
    pending_payloads: deque[Dict[str, Any]],
    stop_requested: bool,
) -> tuple[list[StageOneJob], bool]:
    """Collect one ready faster-whisper micro-batch without blocking for more work.

    The child keeps extra ready payloads in a thin local deque so it can observe
    backlog once, choose a pack profile, and still preserve exact submit order.
    """
    ready_payloads = [first_payload]
    while len(ready_payloads) + len(pending_payloads) < _FW_MAX_READY_BATCH:
        try:
            payload = job_queue.get_nowait()
        except queue.Empty:
            break
        if payload == _STOP_SENTINEL:
            stop_requested = True
            break
        pending_payloads.append(payload)

    pack_size, _ = _select_faster_whisper_pack_profile(1 + len(pending_payloads))
    while pending_payloads and len(ready_payloads) < pack_size:
        ready_payloads.append(pending_payloads.popleft())
    return [StageOneJob(**payload) for payload in ready_payloads], stop_requested


def _emit_terminal(event_queue: Any, detail: StageOneDetail) -> None:
    """Send one terminal event to the parent collector."""
    event_queue.put(
        {
            "type": "terminal",
            "chunk_id": detail.chunk_id,
            "chunk_index": detail.chunk_index,
            "detail": detail.to_payload(),
            "status": detail.status,
            "error": detail.error,
        }
    )


def _child_main(
    job_queue: Any,
    event_queue: Any,
    tts_dir: str,
    engine: str,
    model_name: str,
    threshold: float,
    language: str,
) -> None:
    """Run one resident Stage 1 worker until the parent sends the stop sentinel.

    Startup failures happen before any terminal chunk event exists, so this
    child writes a durable crash report with traceback before re-raising. The
    parent can then surface the real cause instead of only an exit code.
    """
    runner = None
    pending_payloads: deque[Dict[str, Any]] = deque()
    stop_requested = False
    try:
        runner = _create_runner(
            engine=engine,
            model_name=model_name,
            language=language,
            tts_dir=tts_dir,
        )
        runner.start()
        while True:
            if pending_payloads:
                payload = pending_payloads.popleft()
            else:
                payload = job_queue.get()
            if payload == _STOP_SENTINEL:
                break
            if engine == "faster_whisper":
                jobs, stop_requested = _drain_ready_faster_whisper_jobs(
                    job_queue,
                    payload,
                    pending_payloads,
                    stop_requested,
                )
                for detail in runner.run_batch(jobs, threshold=threshold):
                    _emit_terminal(event_queue, detail)
                if stop_requested and not pending_payloads:
                    break
                continue
            job = StageOneJob(**payload)
            detail = runner.run(job, threshold=threshold)
            _emit_terminal(event_queue, detail)
    except Exception as exc:
        _write_child_crash_report(tts_dir, engine, model_name, language, exc)
        raise
    finally:
        if runner is not None:
            runner.close()


class StreamingStageOneService:
    """Manage one spawned Stage 1 child process and collect durable results.

    The parent never loads ASR models.  It only queues immutable jobs, drains
    terminal events, and writes the durable JSONL + final summary artifacts.
    """

    def __init__(
        self,
        tts_dir: str | Path,
        engine: str,
        model_name: str,
        threshold: float,
        language: str = "en",
        max_queue_size: int = 64,
        finish_timeout_s: float = 30.0,
        *,
        _context: Any = None,
        _worker_target: Optional[Callable[..., None]] = None,
    ) -> None:
        """Initialize an inert parent-side controller for future Stage 3 use.

        Args:
            tts_dir: TTS directory where durable Stage 1 artifacts will be written.
            engine: Supported Stage 1 engine: ``faster_whisper`` or ``parakeet``.
            model_name: Model/backend name to send to the child runner.
            threshold: Comparator threshold for Stage 1 pass/fail classification.
            language: Language hint for faster-whisper decoding.
            max_queue_size: Parent-to-child queue bound.  Prevents unbounded job
                growth if a future producer outruns the worker.
            finish_timeout_s: Default wait budget for ``finish_collect``.
            _context: Optional spawn-context seam used only by tests.
            _worker_target: Optional child entrypoint seam used only by tests.
        """
        normalized_engine = str(engine or "").strip().lower().replace("-", "_")
        if normalized_engine not in ENGINE_CHOICES:
            raise ValueError(f"Unsupported Stage 1 engine: {engine}")
        self._tts_dir = Path(tts_dir)
        self._engine = normalized_engine
        self._model_name = str(model_name or "")
        self._threshold = float(threshold)
        self._language = str(language or "en")
        self._max_queue_size = max(1, int(max_queue_size))
        self._finish_timeout_s = max(0.1, float(finish_timeout_s))
        self._context = _context or mp.get_context("spawn")
        self._worker_target = _worker_target or _child_main
        self._process: Any = None
        self._job_queue: Any = None
        self._event_queue: Any = None
        self._submitted_order: list[str] = []
        self._submitted_by_id: Dict[str, StageOneJob] = {}
        self._details_by_id: Dict[str, StageOneDetail] = {}
        self._deferred_jobs: deque[StageOneJob] = deque()
        self._duplicate_ids: list[str] = []
        self._errors: list[str] = []
        self._finished = False
        self._started_at_monotonic: Optional[float] = None
        self._event_log_path = self._tts_dir / EVENT_LOG_NAME
        self._summary_path = self._tts_dir / SUMMARY_NAME
        self._child_crash_report_path = self._tts_dir / CHILD_CRASH_REPORT_NAME

    @property
    def event_log_path(self) -> Path:
        """Return the durable JSONL path reserved for this service."""
        return self._event_log_path

    @property
    def summary_path(self) -> Path:
        """Return the final summary JSON path reserved for this service."""
        return self._summary_path

    def start(self) -> None:
        """Spawn exactly one child process, reset artifacts, and prepare queues."""
        if self._process is not None:
            raise RuntimeError("StreamingStageOneService already started")
        self._tts_dir.mkdir(parents=True, exist_ok=True)
        for artifact_path in (self._event_log_path, self._summary_path):
            if artifact_path.exists():
                artifact_path.unlink()
        if self._child_crash_report_path.exists():
            self._child_crash_report_path.unlink()
        self._started_at_monotonic = time.monotonic()
        self._job_queue = self._context.Queue(maxsize=self._max_queue_size)
        self._event_queue = self._context.Queue(maxsize=max(8, self._max_queue_size))
        self._process = self._context.Process(
            target=self._worker_target,
            args=(
                self._job_queue,
                self._event_queue,
                str(self._tts_dir),
                self._engine,
                self._model_name,
                self._threshold,
                self._language,
            ),
            name="streaming-stage-one",
        )
        self._process.start()

    def _record_parent_event(self, event_type: str, job: StageOneJob, **extra: Any) -> None:
        """Append one parent-side durable event for deferred backlog diagnostics."""
        _append_jsonl(
            self._event_log_path,
            {
                "type": event_type,
                "chunk_id": job.chunk_id,
                "chunk_index": job.chunk_index,
                "audio_path": job.audio_path,
                "text_path": job.text_path,
                **extra,
            },
        )

    def _ensure_child_healthy(self) -> None:
        """Raise when the resident child already died and cannot accept more work."""
        if self._process is None:
            raise RuntimeError("StreamingStageOneService.start() must run before submit()")
        if not self._process.is_alive() and self._process.exitcode not in (None, 0):
            raise RuntimeError(self._child_crash_message())

    def _child_crash_message(self) -> str:
        """Return one detailed child-crash message, preferring the durable report."""
        exitcode = None if self._process is None else self._process.exitcode
        if self._child_crash_report_path.exists():
            try:
                payload = json.loads(self._child_crash_report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return (
                    "Streaming Stage 1 child exited unexpectedly with code "
                    f"{exitcode}; crash report unreadable: {exc}"
                )
            return (
                "Streaming Stage 1 child crashed during startup/run: "
                f"{payload.get('error_type', 'Error')}: {payload.get('error', 'unknown error')} "
                f"(report: {self._child_crash_report_path})"
            )
        return f"Streaming Stage 1 child exited unexpectedly with code {exitcode}"

    def _pump_deferred_jobs(self) -> int:
        """Move deferred jobs into the bounded child queue whenever slots appear.

        Deferred jobs stay as thin immutable path records only. This preserves
        order without retaining audio in memory and keeps the JSONL log as the
        durable source of truth whenever the parent crashes mid-run.
        """
        if self._job_queue is None:
            return 0
        pumped = 0
        while self._deferred_jobs:
            job = self._deferred_jobs[0]
            try:
                self._job_queue.put_nowait(job.to_payload())
            except queue.Full:
                break
            self._deferred_jobs.popleft()
            pumped += 1
        return pumped

    def submit(self, job: StageOneJob) -> None:
        """Queue one immutable Stage 1 job without blocking indefinitely.

        Duplicate chunk ids are rejected before they can produce ambiguous
        terminal events in the parent collector.
        """
        if self._process is None or self._job_queue is None:
            raise RuntimeError("StreamingStageOneService.start() must run before submit()")
        if self._finished:
            raise RuntimeError("Cannot submit after finish_collect()")
        self._ensure_child_healthy()
        self.poll()
        if job.chunk_id in self._submitted_by_id:
            raise ValueError(f"Duplicate submitted chunk id: {job.chunk_id}")
        self._submitted_by_id[job.chunk_id] = job
        self._submitted_order.append(job.chunk_id)
        try:
            self._job_queue.put_nowait(job.to_payload())
        except queue.Full:
            self._deferred_jobs.append(job)
            # Queue saturation is expected during long TTS runs. Persist the
            # deferred record immediately so later fallback can prove this job
            # never became authoritative if the process dies before draining.
            self._record_parent_event(
                "deferred_submission",
                job,
                deferred_backlog_size=len(self._deferred_jobs),
            )
            return
        self._pump_deferred_jobs()

    def poll(self) -> int:
        """Nonblockingly drain received events into durable log and retained state.

        Stage 3 can call this during long TTS runs to keep the bounded result
        queue moving. ``submit()`` also calls it first so ordinary per-chunk
        producers do not deadlock on an undrained child event queue.
        """
        if self._event_queue is None:
            return 0
        drained = 0
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            drained += 1
            self._record_event(event)
        self._pump_deferred_jobs()
        if (
            self._process is not None
            and not self._process.is_alive()
            and self._process.exitcode not in (None, 0)
        ):
            message = self._child_crash_message()
            if message not in self._errors:
                self._errors.append(message)
        return drained

    def _record_event(self, event: Dict[str, Any]) -> None:
        """Persist one child event and fold it into retained parent state."""
        _append_jsonl(self._event_log_path, event)
        if event.get("type") != "terminal":
            self._errors.append(f"Unknown Stage 1 event type: {event.get('type')}")
            return
        chunk_id = str(event.get("chunk_id") or "")
        if chunk_id in self._details_by_id:
            self._duplicate_ids.append(chunk_id)
            self._errors.append(f"Duplicate terminal event for chunk id {chunk_id}")
            return
        detail_payload = dict(event.get("detail") or {})
        self._details_by_id[chunk_id] = StageOneDetail(**detail_payload)

    def _enqueue_stop_before_deadline(self, deadline: float) -> bool:
        """Try to enqueue the stop sentinel without ever blocking indefinitely.

        This path polls first on each retry so a child blocked on a full event
        queue can make forward progress and eventually free a job-queue slot.
        """
        if self._job_queue is None:
            return False
        while True:
            self.poll()
            self._pump_deferred_jobs()
            if self._deferred_jobs:
                if time.monotonic() >= deadline:
                    return False
                _grace_sleep(0.01)
                continue
            try:
                self._job_queue.put_nowait(_STOP_SENTINEL)
                return True
            except queue.Full:
                if time.monotonic() >= deadline:
                    return False
                _grace_sleep(0.01)

    def finish_collect(self, timeout_s: Optional[float] = None) -> StageOneRunSummary:
        """Stop submission, drain all terminal events, and write final artifacts.

        ``complete`` stays false unless every submitted id produced one terminal
        event and the child exited with code 0 before the timeout expired.
        """
        if self._process is None or self._job_queue is None or self._event_queue is None:
            raise RuntimeError("StreamingStageOneService.start() must run before finish_collect()")
        if self._finished:
            raise RuntimeError("finish_collect() already called")
        self._finished = True
        deadline = time.monotonic() + (self._finish_timeout_s if timeout_s is None else max(0.1, float(timeout_s)))
        timed_out = False
        if not self._enqueue_stop_before_deadline(deadline):
            timed_out = True
            if self._deferred_jobs:
                self._errors.append(
                    f"Timed out while draining {len(self._deferred_jobs)} deferred Stage 1 job(s) before stop sentinel"
                )
            else:
                self._errors.append("Timed out while trying to enqueue Stage 1 stop sentinel")

        while True:
            self.poll()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                self._errors.append("Timed out while waiting for Stage 1 terminal events")
                break
            if len(self._details_by_id) == len(self._submitted_by_id) and not self._process.is_alive():
                break
            try:
                event = self._event_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if not self._process.is_alive():
                    self.poll()
                    if len(self._details_by_id) >= len(self._submitted_by_id):
                        break
                    if self._process.exitcode not in (0, None):
                        self._errors.append(
                            f"Stage 1 child exited unexpectedly with code {self._process.exitcode}"
                        )
                    else:
                        grace_deadline = min(deadline, time.monotonic() + 0.2)
                        while time.monotonic() < grace_deadline:
                            if self.poll() == 0:
                                _grace_sleep(0.01)
                            if len(self._details_by_id) >= len(self._submitted_by_id):
                                break
                    break
                continue
            self._record_event(event)

        self._process.join(timeout=0.2)
        if self._process.is_alive():
            timed_out = True
            self._errors.append("Stage 1 child did not exit after stop sentinel")
            self._process.terminate()
            self._process.join(timeout=0.2)

        child_exitcode = self._process.exitcode
        if child_exitcode not in (0, None):
            message = self._child_crash_message()
            if message not in self._errors:
                self._errors.append(message)

        missing_ids = tuple(
            chunk_id for chunk_id in self._submitted_order if chunk_id not in self._details_by_id
        )
        details = tuple(
            self._details_by_id[chunk_id]
            for chunk_id in self._submitted_order
            if chunk_id in self._details_by_id
        )
        failed = sum(1 for detail in details if not detail.passed)
        worker_fault_seen = any(detail.status != "result" or detail.error for detail in details)
        complete = (
            not timed_out
            and not self._duplicate_ids
            and not missing_ids
            and child_exitcode == 0
            and not worker_fault_seen
            and len(details) == len(self._submitted_by_id)
        )
        run_started = self._started_at_monotonic or time.monotonic()
        wall_s = max(0.0, time.monotonic() - run_started)
        summary = StageOneRunSummary(
            submitted=len(self._submitted_by_id),
            completed=len(details),
            failed=failed,
            missing_ids=missing_ids,
            complete=complete,
            details=details,
            errors=tuple(self._errors),
            duplicate_terminal_ids=tuple(self._duplicate_ids),
            child_exitcode=child_exitcode,
            timed_out=timed_out,
            event_log_path=str(self._event_log_path),
            summary_path=str(self._summary_path),
            wall_s=wall_s,
        )
        _write_summary_json(self._summary_path, summary)
        return summary

    def abort(self) -> None:
        """Terminate the child early and mark the service unusable for collection."""
        self._finished = True
        self._deferred_jobs.clear()
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.2)

    def close(self) -> None:
        """Idempotently stop the child if the caller abandons the run early."""
        if self._process is not None and self._process.is_alive():
            self.abort()
