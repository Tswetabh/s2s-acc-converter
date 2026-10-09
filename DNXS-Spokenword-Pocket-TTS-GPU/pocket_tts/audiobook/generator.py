"""
Audiobook generation engine with progress tracking and resume capability.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
import multiprocessing as mp
import dataclasses
import queue as thread_queue
import shutil
import subprocess
import threading
import traceback
import wave
from queue import Empty
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional, Sequence, Union, Tuple

try:
    from ..preprocessing.schema import ChunkMetadata, Config, TTSParams
    from ..config import ConfigManager
    from ..data.audio import audio_read
    from ..regeneration_scheduler import (
        choose_regeneration_plan,
        format_plan_summary,
    )
    import torch
    import psutil
except ImportError:
    from preprocessing.schema import ChunkMetadata, Config, TTSParams
    from config import ConfigManager
    from data.audio import audio_read
    from regeneration_scheduler import (
        choose_regeneration_plan,
        format_plan_summary,
    )
    import torch
    try:
        import psutil
    except ImportError:
        psutil = None
except ImportError:
    from preprocessing.schema import ChunkMetadata, Config, TTSParams
    from config import ConfigManager
    from data.audio import audio_read
    from regeneration_scheduler import (
        choose_regeneration_plan,
        format_plan_summary,
    )
    import torch

logger = logging.getLogger(__name__)

# Legacy relative paths (pre single-venv). Still recognized for migration.
_ASR_VENV_RELATIVE_PATHS = ("ASR/venv/bin/python", "ASR/venv/Scripts/python.exe")
_BYTES_PER_GIB = 1024 ** 3
_SHADOW_STAGE_ONE_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


def _format_elapsed_hms(seconds: float) -> str:
    """Format elapsed seconds as a zero-padded hours, minutes, and seconds string."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def _compact_verification_section(
    data: Optional[Dict[str, Any]],
    allowed_keys: Tuple[str, ...],
) -> Optional[Dict[str, Any]]:
    """Keep only report fields that carry review value and drop empty ones."""
    if not data:
        return None

    compacted = {}
    for key in allowed_keys:
        value = data.get(key)
        if value in (None, "", [], {}, ()):
            continue
        compacted[key] = value
    return compacted or None


def _write_stage_one_failure_manifest(
    tts_dir: Path,
    *,
    pipeline: str,
    failure_filename: str,
    threshold: float,
    language: str,
) -> Path:
    """Mark a complete Stage 1 report as manually loadable when needed.

    A later Medium verification overwrites this manifest with its stronger,
    confirmed-failure report. This early write keeps Medium-or-larger legacy
    configurations and empty Stage 1 runs discoverable in Regenerate.
    """
    from pocket_tts.asr_failure_reports import write_failure_manifest

    is_new = pipeline == "new"
    return write_failure_manifest(
        tts_dir,
        pipeline=pipeline,
        authoritative_filename=failure_filename,
        label=(
            "New Stage 1 candidates (not Medium-confirmed)"
            if is_new
            else "Legacy Stage 1 candidates (not Medium-confirmed)"
        ),
        kind="stage_one_candidates",
        threshold=threshold,
        language=language,
    )


def _build_medium_verification_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce Medium audit rows to decision-bearing fields only."""
    cleaned = {}
    chunk_index = record.get("chunk_index")
    chunk_id = record.get("chunk_id")
    decision = record.get("decision")

    if chunk_index is not None:
        cleaned["chunk_index"] = chunk_index
    if chunk_id is not None:
        cleaned["chunk_id"] = chunk_id
    if decision:
        cleaned["decision"] = decision

    original_failure = record.get("original_failure") or {}
    medium_result = record.get("medium_result") or {}

    comparison = _compact_verification_section(
        {
            "ref_text_raw": original_failure.get("original_text"),
            # This report explains Stage 2's decision, so the visible
            # hypothesis must be Medium's fresh transcript, not the stale
            # Stage 1 transcript that selected the candidate.
            "hyp_text_raw": medium_result.get("hyp_text_raw")
            or medium_result.get("transcribed_text")
            or original_failure.get("transcribed_text"),
            "ref_normalized": medium_result.get("ref_normalized"),
            "hyp_normalized": medium_result.get("hyp_normalized"),
        },
        (
            "ref_text_raw",
            "hyp_text_raw",
            "ref_normalized",
            "hyp_normalized",
        ),
    )
    if comparison:
        cleaned["comparison"] = comparison

    original_failure = _compact_verification_section(
        original_failure,
        (
            "score",
            "classification",
            "explanation",
        ),
    )
    if original_failure:
        cleaned["original_failure"] = original_failure

    medium_result = _compact_verification_section(
        medium_result,
        (
            "passed",
            "score",
            "classification",
            "explanation",
            "accepted_book_term_equivalences",
            "error",
        ),
    )
    if medium_result:
        cleaned["medium_result"] = medium_result

    return cleaned


def _build_new_medium_verification_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Keep Stage 1 and Medium evidence separate in New-pipeline audit rows."""
    cleaned: Dict[str, Any] = {}
    for key in ("chunk_index", "chunk_id", "decision"):
        value = record.get(key)
        if value is not None and value != "":
            cleaned[key] = value

    stage_one = record.get("original_failure") or {}
    medium = record.get("medium_result") or {}
    reference = _compact_verification_section(
        {
            "text_raw": stage_one.get("original_text") or medium.get("ref_text_raw"),
            "normalized": medium.get("ref_normalized") or stage_one.get("ref_normalized"),
        },
        ("text_raw", "normalized"),
    )
    if reference:
        cleaned["reference"] = reference

    stage_one_evidence = _compact_verification_section(
        {
            "transcript_raw": stage_one.get("transcribed_text") or stage_one.get("hyp_text_raw"),
            "normalized": stage_one.get("hyp_normalized"),
            "score": stage_one.get("score"),
            "classification": stage_one.get("classification"),
            "explanation": stage_one.get("explanation"),
        },
        ("transcript_raw", "normalized", "score", "classification", "explanation"),
    )
    if stage_one_evidence:
        cleaned["stage_one"] = stage_one_evidence

    medium_evidence = _compact_verification_section(
        {
            "transcript_raw": medium.get("hyp_text_raw") or medium.get("transcribed_text"),
            "normalized": medium.get("hyp_normalized"),
            "score": medium.get("score"),
            "classification": medium.get("classification"),
            "passed": medium.get("passed"),
            "explanation": medium.get("explanation"),
            "accepted_equivalences": medium.get("accepted_equivalences"),
            "accepted_book_term_equivalences": medium.get("accepted_book_term_equivalences"),
            "error": medium.get("error"),
        },
        (
            "transcript_raw",
            "normalized",
            "score",
            "classification",
            "passed",
            "explanation",
            "accepted_equivalences",
            "accepted_book_term_equivalences",
            "error",
        ),
    )
    if medium_evidence:
        cleaned["medium"] = medium_evidence
    return cleaned


def _is_confirmed_medium_failure(
    record: Dict[str, Any],
    is_new_pipeline_report: bool,
) -> bool:
    """Return whether one compact Medium row is a proven Stage 2 failure.

    Accepted rows and rows where Medium could not produce evidence must never
    enter the failure-only report.  The legacy and New reports use different
    names for their compact Medium section, so this helper keeps their filter
    semantics identical.
    """
    if record.get("decision") != "confirmed_by_two_asr_models":
        return False
    result_key = "medium" if is_new_pipeline_report else "medium_result"
    return record.get(result_key, {}).get("passed") is False


def _stage_two_verification_ran(summary: Dict[str, Any] | None) -> bool:
    """Return True when independent Stage 2 actually scored Stage 1 candidates."""
    if not summary:
        return False
    if summary.get("verification_skipped"):
        return False
    return summary.get("attempted") is not None


def _resolve_reported_stage_fail_counts(
    asr_stage_one_fails: int,
    asr_stage_two_fails: int,
    asr_pipeline: str,
    regen_verification_summary: Dict[str, Any] | None,
) -> tuple[int, int]:
    """Return Stage 1/2 fail counts for GUI display and regeneration.

    Copy Stage 1 into Stage 2 only when Stage 2 never ran (Disabled or skipped).
    A completed Stage 2 pass that accepted every candidate must stay at zero.
    """
    stage_one = int(asr_stage_one_fails or 0)
    stage_two = int(asr_stage_two_fails or 0)
    summary = regen_verification_summary or {}
    if asr_pipeline != "new" and stage_two <= 0 and not _stage_two_verification_ran(summary):
        stage_two = max(stage_two, stage_one)
    if stage_one <= 0 and summary:
        stage_one = int(
            summary.get("attempted")
            or summary.get("stage_one_candidates")
            or 0
        )
    if stage_two <= 0 and summary and _stage_two_verification_ran(summary):
        stage_two = int(
            summary.get("verified_fail")
            or summary.get("medium_confirmed_failures")
            or 0
        )
    return stage_one, stage_two


def _format_score(value: Any) -> str:
    """Format a numeric score for human investigation logs."""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def write_stage2_investigation_log(
    tts_dir: Path,
    failure_records: List[Dict[str, Any]],
    *,
    threshold: Optional[float] = None,
    language: str = "en",
    stage_one_candidates: Optional[int] = None,
    is_new_pipeline: bool = True,
    filename: str = "asr_stage2_investigation.log",
) -> Path:
    """Write a human-readable Stage 2 fail log (book text vs Medium ASR).

    Each confirmed Medium failure is one block showing:
    - BOOK TEXT: source line TTS was supposed to speak
    - STAGE 2 ASR: Faster-Whisper Medium transcript of the audio
    - WHY FAIL: comparator explanation for book text vs Stage 2 ASR
    - normalized forms the scorer actually compared
    - Stage 1 Parakeet line for reference only (not the fail decision)

    Args:
        tts_dir: Book-local TTS folder receiving the log.
        failure_records: Compact Medium-confirmed failure rows (new or legacy).
        threshold: Spoken-content threshold used by Stage 2.
        language: ASR language code.
        stage_one_candidates: How many Stage 1 fails Medium reviewed.
        is_new_pipeline: True for new-pipeline record shape (reference/medium).
        filename: Output basename under ``tts_dir``.

    Returns:
        Path to the written log file.
    """
    tts_dir = Path(tts_dir)
    tts_dir.mkdir(parents=True, exist_ok=True)
    log_path = tts_dir / filename
    reviewed = (
        stage_one_candidates
        if stage_one_candidates is not None
        else len(failure_records)
    )
    thr_text = _format_score(threshold) if threshold is not None else "n/a"

    lines: List[str] = [
        "ASR Stage 2 (Medium) Confirmed Failures — Human Investigation Log",
        "=" * 72,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Language: {language}",
        f"Threshold: {thr_text}",
        f"Stage 1 candidates reviewed by Medium: {reviewed}",
        f"Stage 2 confirmed fails (this file): {len(failure_records)}",
        "",
        "How to read each chunk:",
        "  BOOK TEXT   = source text the TTS was supposed to speak",
        "  STAGE 2 ASR = what Faster-Whisper Medium heard from the audio",
        "  WHY FAIL    = why BOOK TEXT vs STAGE 2 ASR failed the comparator",
        "  BOOK NORM / ASR NORM = normalized forms the scorer compared",
        "  Stage 1 lines are reference only — the fail decision is Stage 2.",
        "  Soft accepts = equivalences Medium already allowed (names, etc.).",
        "",
        "This is NOT the post-regen residual log (see asr_investigation.log).",
        "This is every Stage 2 confirmed fail *before* regeneration.",
        "",
    ]

    if not failure_records:
        lines.append("No Stage 2 confirmed failures.")
        lines.append("")
        log_path.write_text("\n".join(lines), encoding="utf-8")
        return log_path

    for record in failure_records:
        if is_new_pipeline:
            reference = record.get("reference") or {}
            stage_one = record.get("stage_one") or {}
            medium = record.get("medium") or {}
            book_text = reference.get("text_raw") or ""
            book_norm = reference.get("normalized") or medium.get("ref_normalized") or ""
            stage2_asr = medium.get("transcript_raw") or ""
            stage2_norm = medium.get("normalized") or ""
            stage2_expl = medium.get("explanation") or medium.get("error") or ""
            stage2_score = medium.get("score")
            stage2_class = medium.get("classification") or "FAIL"
            stage1_asr = stage_one.get("transcript_raw") or ""
            stage1_norm = stage_one.get("normalized") or ""
            stage1_expl = stage_one.get("explanation") or ""
            stage1_score = stage_one.get("score")
            soft = (
                medium.get("accepted_equivalences")
                or medium.get("accepted_book_term_equivalences")
                or []
            )
        else:
            comparison = record.get("comparison") or {}
            original = record.get("original_failure") or {}
            medium = record.get("medium_result") or {}
            book_text = comparison.get("ref_text_raw") or ""
            book_norm = comparison.get("ref_normalized") or ""
            stage2_asr = comparison.get("hyp_text_raw") or ""
            stage2_norm = comparison.get("hyp_normalized") or ""
            stage2_expl = medium.get("explanation") or medium.get("error") or ""
            stage2_score = medium.get("score")
            stage2_class = medium.get("classification") or "FAIL"
            stage1_asr = ""
            stage1_norm = ""
            stage1_expl = original.get("explanation") or ""
            stage1_score = original.get("score")
            soft = (
                medium.get("accepted_book_term_equivalences")
                or medium.get("accepted_equivalences")
                or []
            )

        chunk_id = record.get("chunk_id")
        chunk_index = record.get("chunk_index")
        if not chunk_id and chunk_index is not None:
            try:
                chunk_id = f"chunk_{int(chunk_index):05d}"
            except (TypeError, ValueError):
                chunk_id = str(chunk_index)
        chunk_id = chunk_id or "chunk_unknown"
        decision = record.get("decision") or "confirmed_by_two_asr_models"

        lines.append(f"CHUNK: {chunk_id}")
        if chunk_index is not None:
            lines.append(f"Index: {chunk_index}")
        lines.append(f"Decision: {decision}")
        lines.append("-" * 72)
        lines.append(f"BOOK TEXT:     {book_text}")
        lines.append(f"STAGE 2 ASR:   {stage2_asr}")
        lines.append(f"WHY FAIL:      {stage2_expl or '(no explanation)'}")
        lines.append(
            f"SCORE:         {_format_score(stage2_score)}  "
            f"classification={stage2_class}  threshold={thr_text}"
        )
        lines.append("")
        lines.append("  (what the scorer compared after normalization)")
        lines.append(f"  BOOK NORM:   {book_norm}")
        lines.append(f"  ASR NORM:    {stage2_norm}")
        if soft:
            soft_bits = []
            for item in soft:
                if isinstance(item, dict):
                    soft_bits.append(
                        f"{item.get('ref', '?')}≈{item.get('hyp', '?')}"
                    )
                else:
                    soft_bits.append(str(item))
            lines.append(f"  SOFT ACCEPT: {', '.join(soft_bits)}")
        lines.append("")
        lines.append("  Stage 1 (Parakeet) — reference only, not the fail decision:")
        lines.append(f"  STAGE 1 ASR:  {stage1_asr or 'n/a'}")
        if stage1_expl:
            lines.append(f"  STAGE 1 WHY:  {stage1_expl}")
        if stage1_score is not None:
            lines.append(f"  STAGE 1 SCORE:{_format_score(stage1_score)}")
        if stage1_norm:
            lines.append(f"  STAGE 1 NORM: {stage1_norm}")
        lines.append("")
        lines.append("=" * 72)
        lines.append("")

    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def _materialize_new_pipeline_regeneration_log(
    tts_dir: Path,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Convert New-pipeline Medium failures into the legacy regen-row shape.

    The isolated New runner writes compact, human-readable Medium records.
    Regeneration needs the original source text and Stage 1 evidence, so this
    function expands only confirmed failures into a separate machine-readable
    log without changing either stage's audit reports.

    Args:
        tts_dir: Book-local TTS directory containing the New Medium failure log.

    Returns:
        Tuple containing the materialized log path and normalized failure rows.
    """
    source_path = tts_dir / "asr_new_medium_failures.json"
    output_path = tts_dir / "asr_new_regeneration_failures.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else []
    failures: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or record.get("decision") != "confirmed_by_two_asr_models":
            continue
        medium = record.get("medium") or {}
        if medium.get("passed") is not False:
            continue
        reference = record.get("reference") or {}
        stage_one = record.get("stage_one") or {}
        chunk_index = record.get("chunk_index")
        original_text = reference.get("text_raw")
        if chunk_index is None or not original_text:
            logger.warning(
                "Skipping New-pipeline failure without chunk index/source text: %s",
                record.get("chunk_id"),
            )
            continue
        failures.append({
            "chunk_index": chunk_index,
            "chunk_id": record.get("chunk_id") or f"chunk_{int(chunk_index):05d}",
            "original_text": original_text,
            "text": original_text,
            "transcribed_text": stage_one.get("transcript_raw", ""),
            "score": stage_one.get("score", medium.get("score", 0.0)),
            "classification": stage_one.get("classification", "FAIL"),
            "explanation": stage_one.get("explanation", medium.get("explanation", "")),
            "medium_verification": medium,
        })
    output_path.write_text(
        json.dumps(
            {
                "report_type": "new_pipeline_regeneration_failures",
                "source_report": str(source_path),
                "records": failures,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(output_path), failures


def _alignment_decision(evidence: Dict[str, Any]) -> Tuple[str, str]:
    """Classify forced-alignment evidence without treating uncertainty as failure.

    Alignment may prove a spoken mismatch only when it is both conclusive and
    has strong missing-token or unexplained-extra-region evidence.  Everything
    else retains the original WAV because Stage 1 transcription is a candidate
    detector, not proof of bad speech.
    """
    if not evidence.get("available") or not evidence.get("conclusive"):
        return "accepted_not_proven_failure", "Alignment unavailable or inconclusive; speech mismatch was not proven."
    coverage = float(evidence.get("coverage") or 0.0)
    confidence = float(evidence.get("confidence") or 0.0)
    unaligned = evidence.get("unaligned_tokens") or []
    extra_regions = evidence.get("extra_regions") or []
    if coverage < 0.85 and confidence < 0.65 and (unaligned or extra_regions):
        return "confirmed_failed_by_alignment", "Alignment found unsupported missing or extra speech evidence."
    return "accepted_by_alignment", "Expected source text aligned with sufficient coverage and confidence."


def _build_alignment_verification_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build a human-readable report row with separately scoped Stage 1/2 evidence."""
    stage_one = record.get("stage_one") or {}
    alignment = record.get("alignment") or {}
    return {
        "chunk_index": record.get("chunk_index"),
        "chunk_id": record.get("chunk_id"),
        "original_text": record.get("original_text", ""),
        "stage_one": {
            "transcript": stage_one.get("transcribed_text", ""),
            "comparison": {
                "ref_normalized": stage_one.get("ref_normalized", ""),
                "hyp_normalized": stage_one.get("hyp_normalized", ""),
                "score": stage_one.get("score"),
                "explanation": stage_one.get("explanation", ""),
            },
        },
        "alignment": alignment,
        "decision": record.get("decision"),
        "accepted_reason": record.get("accepted_reason"),
        "explanation": record.get("explanation", ""),
    }


def _wait_asr_process_with_progress(
    asr_process: Any,
    asr_run_log: Optional[str],
    timeout: Any,
    progress_callback=None,
    total_chunks: int = 0,
) -> None:
    """Wait for ASR subprocess while streaming status from its log file.

    Polls ``asr_run.log`` so the main GUI can show download/load/validation
    progress and users know the app is not stuck.

    Args:
        asr_process: Popen handle for the ASR monitor.
        asr_run_log: Path to ASR stdout log (may be written concurrently).
        timeout: Max seconds to wait before TimeoutExpired (int or float).
        progress_callback: Optional GUI callback receiving status dicts.
        total_chunks: Expected chunk count (for status text only).

    Raises:
        subprocess.TimeoutExpired: If process exceeds timeout.
    """
    import subprocess as _sp

    # beartype: callers pass int from max(300, n*30); accept int or float
    timeout_s = float(timeout)
    deadline = time.time() + max(30.0, timeout_s)
    last_line = ""
    last_emit = 0.0
    log_path = Path(asr_run_log) if asr_run_log else None
    interesting = (
        "Download",
        "download",
        "Loading",
        "loaded",
        "Waiting",
        "Ensuring",
        "HF ",
        "ggml",
        "Worker model",
        "ASR completed",
        "validated",
        "FAILED",
        "PASSED",
        "stuck",
        "WARNING",
        "⚠️",
        "⏳",
        "⬇",
    )

    def _tail_interesting() -> str:
        """Return last interesting line from ASR run log, if any."""
        if not log_path or not log_path.is_file():
            return ""
        try:
            # Read last ~48KB only
            with open(log_path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 49152))
                data = fh.read().decode("utf-8", errors="ignore")
            lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
            for ln in reversed(lines):
                if any(tok in ln for tok in interesting):
                    # strip log prefix timestamps when present
                    if " - " in ln:
                        return ln.split(" - ", 2)[-1][:200]
                    return ln[:200]
            return lines[-1][:200] if lines else ""
        except Exception:
            return ""

    if progress_callback:
        try:
            progress_callback({
                "asr_status": True,
                "message": (
                    "Post-gen ASR running (model download/load may take minutes "
                    "on first use — not stuck)…"
                ),
                "total_chunks": total_chunks,
            })
        except Exception:
            pass

    while True:
        rc = asr_process.poll()
        now = time.time()
        if rc is not None:
            if progress_callback:
                try:
                    progress_callback({
                        "asr_status": True,
                        "message": f"Post-gen ASR finished (exit {rc})",
                        "total_chunks": total_chunks,
                    })
                except Exception:
                    pass
            return
        if now >= deadline:
            raise _sp.TimeoutExpired(asr_process.args, timeout_s)

        if now - last_emit >= 1.5:
            last_emit = now
            line = _tail_interesting()
            if line and line != last_line:
                last_line = line
                logger.info("ASR progress: %s", line)
                if progress_callback:
                    try:
                        progress_callback({
                            "asr_status": True,
                            "message": f"ASR: {line}",
                            "total_chunks": total_chunks,
                        })
                    except Exception:
                        pass
            elif progress_callback:
                try:
                    progress_callback({
                        "asr_status": True,
                        "message": (
                            "ASR still running… "
                            f"(see {log_path.name if log_path else 'asr_run.log'})"
                        ),
                        "total_chunks": total_chunks,
                    })
                except Exception:
                    pass
        time.sleep(0.5)


def calculate_vram_worker_capacity(
    free_bytes: int,
    total_bytes: int,
    worker_vram_gb: float = 1.0,
    reserve_percent: float = 10.0,
) -> int:
    """Calculate workers that fit below a device-wide VRAM reserve.

    Args:
        free_bytes: Current device-wide free VRAM reported by CUDA.
        total_bytes: Total VRAM reported by CUDA.
        worker_vram_gb: Estimated VRAM required by each worker model.
        reserve_percent: Percentage of total VRAM kept unused as reserve.

    Returns:
        Number of worker model estimates that fit in usable free VRAM.
    """
    if worker_vram_gb <= 0:
        raise ValueError("worker_vram_gb must be positive")
    if total_bytes <= 0 or free_bytes < 0:
        return 0

    reserve_fraction = min(max(reserve_percent, 0.0), 100.0) / 100.0
    reserve_bytes = total_bytes * reserve_fraction
    usable_bytes = max(0.0, free_bytes - reserve_bytes)
    return int(usable_bytes / (worker_vram_gb * _BYTES_PER_GIB))


def _resolve_asr_executable(configured_path: Optional[str] = None) -> str:
    """Resolve the Python used to run ASR/asr_validator.py.

    Prefer the **same interpreter as the main app** (single venv). Legacy
    ``ASR/venv/...`` config values always resolve to ``sys.executable``.

    Args:
        configured_path: Optional config ``executable_path`` (may be empty).

    Returns:
        Absolute or usable path to a Python executable with ASR deps.
    """
    # Explicit empty / None → same process env
    if not configured_path or not str(configured_path).strip():
        return sys.executable

    raw = str(configured_path).strip()
    normalized = raw.replace("\\", "/")

    # Legacy relative ASR/venv defaults → always main interpreter (single venv)
    if normalized in _ASR_VENV_RELATIVE_PATHS or not raw:
        return sys.executable

    # Absolute or custom relative path
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[2] / raw
    if candidate.is_file():
        return str(candidate)

    logger.warning(
        "ASR executable not found at %s — using sys.executable (%s)",
        raw,
        sys.executable,
    )
    return sys.executable


def _parse_asr_json_output(output: str) -> Dict[str, Any]:
    """Extract first JSON object from validator output with trailing logs."""
    json_start = output.find("{")
    if json_start < 0:
        raise ValueError("No JSON object found in ASR validator output")

    payload, _ = json.JSONDecoder().raw_decode(output[json_start:])
    if not isinstance(payload, dict):
        raise ValueError("ASR validator JSON payload is not an object")
    return payload


def _build_original_queue_tasks(
    chunks: List[ChunkMetadata], tokenizer: Any, batch_config: Dict[str, Any]
) -> List[tuple]:
    """Group compatible original chunks into bounded exact-length batches."""
    if not batch_config.get("enabled", False):
        return [("original", index, chunk) for index, chunk in enumerate(chunks)]

    batch_size = max(2, int(batch_config.get("batch_size", 2)))
    grouped: Dict[tuple, List[tuple[int, ChunkMetadata]]] = {}
    for index, chunk in enumerate(chunks):
        params = (
            dataclasses.asdict(chunk.tts_params)
            if dataclasses.is_dataclass(chunk.tts_params)
            else dict(chunk.tts_params)
        )
        token_length = int(tokenizer(chunk.text).tokens.shape[-1])
        key = (
            token_length,
            int(params.get("frames_after_eos", 2)),
            float(params.get("eos_threshold", -4.0)),
            int(params.get("lsd_decode_steps", 1)),
        )
        grouped.setdefault(key, []).append((index, chunk))

    tasks = []
    for grouped_chunks in grouped.values():
        for start in range(0, len(grouped_chunks), batch_size):
            batch = grouped_chunks[start : start + batch_size]
            tasks.append(("original", *batch[0]) if len(batch) == 1 else ("original_batch", batch))
    return tasks


def _original_task_chunk_indices(task: tuple) -> List[int]:
    """Return chunk indices carried by one original-phase queue task."""
    if not task:
        return []
    kind = task[0]
    if kind == "original":
        return [int(task[1])]
    if kind == "original_batch":
        return [int(index) for index, _chunk in task[1]]
    return []


def _requeue_unreceived_original_chunks(
    chunk_queue: Any,
    chunks: Sequence[Any],
    received_indices: set,
) -> List[int]:
    """Re-queue scalar original tasks for every chunk not yet reported to the parent.

    A dead worker can take tasks off the shared queue and die before putting
    results. Live siblings must not be terminated: re-queue the unfinished
    indices so they can finish. Duplicate completions are ignored by the parent.

    Args:
        chunk_queue: Shared worker input queue.
        chunks: Full original chunk list, indexed consistently with task indices.
        received_indices: Chunk indices already accepted from ``result_queue``.

    Returns:
        Sorted list of re-queued chunk indices.
    """
    missing = [index for index in range(len(chunks)) if index not in received_indices]
    for index in missing:
        chunk_queue.put(("original", index, chunks[index]))
    return missing


class AudiobookGenerator:
    """Generates audiobooks from processed text chunks with progress tracking."""

    def __init__(self, config: Optional[Union[ConfigManager, Config]] = None):
        """
        Initialize the audiobook generator.

        Args:
            config: Configuration object or manager, uses defaults if None
        """
        # Handle both ConfigManager class (legacy) and Config object
        if config is None:
            self.config = ConfigManager.load_config()
        elif isinstance(config, ConfigManager):
            # If passed the class itself (legacy behavior)
            self.config = ConfigManager.load_config()
        else:
            # Assumed to be Config object
            self.config = config

        self.tts_model = None
        self.is_cancelled = False

        # Progress tracking
        self.start_time = None
        self._chunk_process_start = None
        self._original_generation_end = None
        self.current_chunk = 0
        self.total_chunks = 0

        # Pause injection settings (can be set by caller before generate_audiobook)
        self._pause_injection_enabled = False
        self._pause_durations = {}

        # Device configuration
        device_config = getattr(self.config, 'device', {})
        if isinstance(device_config, dict):
            self._device = device_config.get('preferred', 'auto')
        else:
            self._device = 'auto'

        # Load parallel processing configuration
        parallel_config = getattr(self.config, 'parallel', {})
        self.parallel_enabled = parallel_config.get('enabled', True)
        self.max_workers_config = parallel_config.get('max_workers', 4)
        self.min_workers_config = parallel_config.get('min_workers', 1)
        self.ram_limit_percent = parallel_config.get('ram_limit_percent', 80)
        self.load_threshold = parallel_config.get('load_threshold', 8.0)
        self.worker_vram_gb = float(parallel_config.get('worker_vram_gb', 1.0))
        self.vram_reserve_percent = float(parallel_config.get('vram_reserve_percent', 10.0))
        self.vram_startup_reserve_percent = float(
            parallel_config.get('vram_startup_reserve_percent', 5.0)
        )
#        self.adaptive_workers = parallel_config.get('adaptive_workers', True)
        self.adaptive_workers = False
        # Calculate initial worker count using physical cores
        if self.parallel_enabled:
            physical_cores = psutil.cpu_count(logical=False) if psutil else None
            logical_cpus = psutil.cpu_count(logical=True) if psutil else None
            
            # GPU mode: limit by VRAM instead of CPU cores
            is_cuda = self._device in ('cuda', 'auto') and self._device != 'cpu'
            self._is_cuda = False
            if is_cuda:
                try:
                    import torch
                    if torch.cuda.is_available():
                        self._is_cuda = True
                        # Actual free VRAM is measured immediately before workers spawn.
                        cpu_based_workers = self.max_workers_config
                        logger.info(
                            "GPU mode: requested up to %s workers; capacity measured before spawn "
                            "using %.2f GiB per worker, %.1f%% settled reserve, and %.1f%% startup reserve",
                            self.max_workers_config,
                            self.worker_vram_gb,
                            self.vram_reserve_percent,
                            self.vram_startup_reserve_percent,
                        )
                    else:
                        cpu_based_workers = max(1, (physical_cores or 4) - 1)
                except Exception:
                    cpu_based_workers = max(1, (physical_cores or 4) - 1)
            elif physical_cores:
                cpu_based_workers = max(1, physical_cores - 1)
                logger.info(f"Physical CPU cores: {physical_cores}, Logical CPUs: {logical_cpus}")
            else:
                cpu_count = os.cpu_count()
                cpu_based_workers = max(1, cpu_count - 5) if cpu_count else 2
                logger.warning(f"psutil not available, using logical CPU count: {cpu_count}")
            
            self.num_workers = min(cpu_based_workers, self.max_workers_config)
            logger.info(f"Worker limit: {cpu_based_workers} ({'VRAM' if is_cuda else 'CPU'}), Config max: {self.max_workers_config}")
        else:
            self.num_workers = 1  # Sequential only

        logger.info(f"Parallel processing {'enabled' if self.parallel_enabled else 'disabled'} with {self.num_workers} workers")
        if self.adaptive_workers:
            logger.info(f"Resource limits: RAM {self.ram_limit_percent}%, Load {self.load_threshold}")
            logger.info(f"Worker range: {self.min_workers_config}-{self.max_workers_config}")

    def _check_system_resources(self) -> Tuple[bool, str]:
        """
        Check if system resources are within acceptable limits.

        Returns:
            Tuple of (resources_ok, reason_if_not)
        """
        if not psutil:
            return True, "psutil not available"

        try:
            # Check RAM usage
            ram_percent = psutil.virtual_memory().percent
            if ram_percent > self.ram_limit_percent:
                return False, f"RAM usage {ram_percent:.1f}% exceeds limit {self.ram_limit_percent}%"

            # Check system load
            load_avg = psutil.getloadavg()[0] / psutil.cpu_count()
            if load_avg > self.load_threshold:
                return False, f"System load {load_avg:.1f} exceeds threshold {self.load_threshold}"

            return True, ""

        except Exception as e:
            logger.warning(f"Failed to check system resources: {e}")
            return True, "Resource check failed"

    def _adjust_workers_for_resources(self) -> int:
        """
        Adjust the number of workers based on current system resources.

        Returns:
            Adjusted number of workers (respects config min/max limits)
        """
        if getattr(self, '_is_cuda', False):
            requested_workers = min(self.num_workers, self.max_workers_config)
            logger.info(
                "GPU worker startup will evaluate requested=%s workers sequentially; "
                "settled reserve=%.1f%%, startup reserve=%.1f%%",
                requested_workers,
                self.vram_reserve_percent,
                self.vram_startup_reserve_percent,
            )
            return requested_workers

        if not self.adaptive_workers:
            return min(self.num_workers, self.max_workers_config)

        resources_ok, reason = self._check_system_resources()
        if resources_ok:
            return min(self.num_workers, self.max_workers_config)

        # Reduce workers if resources are strained
        current_workers = self.num_workers
        reduced_workers = max(self.min_workers_config, current_workers - 1)

        logger.warning(f"Resource limits exceeded ({reason}), reducing workers from {current_workers} to {reduced_workers}")
        return reduced_workers

    def _regeneration_worker_count(self, effective_workers: int) -> int:
        """Return the bounded TTS worker count selected for recovery work.

        The recovery plan is stored on the generator because parallel generation
        runs in a separate method from the plan-selection closure.
        """
        requested_workers = getattr(self, "_regen_tts_workers", 1)
        return max(1, min(max(1, int(effective_workers)), int(requested_workers or 1)))

        logger.info("AudiobookGenerator initialized")

    def _resolve_shadow_stage_one_request(
        self,
        asr_enabled: bool,
        save_dataset_chunks: bool,
        dataset_paths: Optional[Dict[str, Any]],
        asr_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return shadow Stage 1 runtime settings when during-TTS collection is active.

        Shadow collection is opt-in only. It never replaces the existing
        authoritative post-generation ASR pipeline, reports, or regeneration.
        """
        if not asr_enabled or not save_dataset_chunks or not dataset_paths:
            return None
        stage_one = asr_config.get("stage_one") or {}
        if not isinstance(stage_one, dict):
            return None
        schedule = str(stage_one.get("schedule", "") or "").strip().lower()
        engine = str(stage_one.get("engine", "") or "").strip().lower()
        if schedule != "during_tts" or engine not in {"faster_whisper", "parakeet"}:
            return None
        tts_dir = dataset_paths.get("tts_dir")
        audio_dir = dataset_paths.get("audio_chunks_dir")
        text_dir = dataset_paths.get("text_chunks_dir")
        if not tts_dir or not audio_dir or not text_dir:
            return None
        if engine == "parakeet":
            model_name = _SHADOW_STAGE_ONE_PARAKEET_MODEL
        else:
            model_name = str(asr_config.get("model", "base") or "base")
        return {
            "tts_dir": str(tts_dir),
            "audio_chunks_dir": str(audio_dir),
            "text_chunks_dir": str(text_dir),
            "engine": engine,
            "model_name": model_name,
            "threshold": float(asr_config.get("threshold", 0.85) or 0.85),
            "language": str(asr_config.get("language", "en") or "en"),
        }

    def _make_shadow_stage_one_service(self, request: Dict[str, Any]) -> Any:
        """Construct one shadow Stage 1 service without loading ASR models in parent."""
        service_cls = getattr(self, "_streaming_stage_one_service_class", None)
        if service_cls is None:
            from ASR.streaming_stage_one import StreamingStageOneService

            service_cls = StreamingStageOneService
        return service_cls(
            request["tts_dir"],
            engine=request["engine"],
            model_name=request["model_name"],
            threshold=request["threshold"],
            language=request["language"],
        )

    def _start_shadow_stage_one_service(self, request: Optional[Dict[str, Any]]) -> Any:
        """Start shadow during-TTS Stage 1 or return ``None`` when inactive/failed."""
        if not request:
            return None
        try:
            service = self._make_shadow_stage_one_service(request)
            service.start()
            logger.info(
                "During-TTS Stage 1 shadow started: engine=%s model=%s tts_dir=%s. "
                "Post-gen ASR remains authoritative.",
                request["engine"],
                request["model_name"],
                request["tts_dir"],
            )
            return service
        except Exception as exc:
            logger.warning(
                "Could not start during-TTS Stage 1 shadow service: %s. "
                "Continuing with authoritative post-gen ASR only.",
                exc,
            )
            return None

    def _publish_shadow_stage_one_chunk(
        self,
        service: Any,
        audio_path: Union[str, Path],
        text_path: Union[str, Path],
        chunk_index: int,
    ) -> Any:
        """Submit one completed chunk to shadow Stage 1 after WAV and TXT both exist.

        Any publish failure disables the shadow service but never interrupts the
        original TTS path. Authoritative post-generation ASR remains unchanged.
        """
        if service is None:
            return None
        audio_path = Path(audio_path)
        text_path = Path(text_path)
        if not audio_path.is_file() or not text_path.is_file():
            return service
        try:
            service.poll()
            from ASR.streaming_stage_one import StageOneJob

            service.submit(
                StageOneJob(
                    chunk_index=int(chunk_index),
                    chunk_id=f"chunk_{int(chunk_index):05d}",
                    audio_path=str(audio_path),
                    text_path=str(text_path),
                )
            )
            service.poll()
            return service
        except Exception as exc:
            logger.warning(
                "During-TTS Stage 1 shadow publish failed for chunk_%05d: %s. "
                "Disabling shadow collection; authoritative post-gen ASR continues.",
                int(chunk_index),
                exc,
            )
            self._abort_shadow_stage_one_service(service, reason="publish failure")
            return None

    def _build_shadow_stage_one_original_chunk_callback(
        self,
        service_holder: Dict[str, Any],
        dataset_paths: Optional[Dict[str, Any]],
    ) -> Optional[Any]:
        """Return a parent-side callback that publishes finalized original chunks.

        The callback is used only by parallel original generation after worker
        postprocessing has already written both the WAV and text sidecar.
        """
        if service_holder.get("service") is None or not dataset_paths:
            return None
        text_chunks_dir = Path(dataset_paths["text_chunks_dir"])

        def _callback(chunk_index: int, saved_path: Union[str, Path]) -> None:
            """Publish one completed original chunk and update shared shadow state."""
            service_holder["service"] = self._publish_shadow_stage_one_chunk(
                service_holder.get("service"),
                saved_path,
                text_chunks_dir / f"chunk_{int(chunk_index):05d}.txt",
                int(chunk_index),
            )

        return _callback

    def _finish_shadow_stage_one_service(self, service: Any, total_chunks: int) -> Optional[Any]:
        """Drain shadow Stage 1 once before authoritative post-generation ASR starts.

        Later call sites may already hold the frozen returned summary instead of
        the live service instance. Treat that final summary as terminal state so
        repeated finish paths stay idempotent and never try to mutate it.
        """
        if service is None:
            return None
        if not hasattr(service, "finish_collect"):
            return service
        if getattr(service, "_shadow_finish_called", False):
            return getattr(service, "_shadow_finish_summary", None)
        setattr(service, "_shadow_finish_called", True)
        try:
            timeout_s = max(300, int(total_chunks) * 30)
            logger.info(
                "Finishing during-TTS Stage 1 shadow before authoritative post-gen ASR "
                "(timeout=%ss)",
                timeout_s,
            )
            summary = service.finish_collect(timeout_s=timeout_s)
            setattr(service, "_shadow_finish_summary", summary)
            if summary.complete:
                logger.info(
                    "During-TTS Stage 1 shadow complete: submitted=%s failed=%s. "
                    "Post-gen ASR remains authoritative.",
                    summary.submitted,
                    summary.failed,
                )
            else:
                logger.warning(
                    "During-TTS Stage 1 shadow incomplete: submitted=%s completed=%s "
                    "missing=%s timed_out=%s errors=%s. Post-gen ASR remains authoritative.",
                    summary.submitted,
                    summary.completed,
                    list(summary.missing_ids),
                    summary.timed_out,
                    list(summary.errors),
                )
            return summary
        except Exception as exc:
            logger.warning(
                "During-TTS Stage 1 shadow finish failed: %s. "
                "Continuing with authoritative post-gen ASR only.",
                exc,
            )
            self._abort_shadow_stage_one_service(service, reason="finish failure")
            return None

    def _promote_shadow_stage_one_to_new_pipeline(
        self,
        summary: Any,
        dataset_paths: Optional[Dict[str, Any]],
        launch_state: Dict[str, Any],
    ) -> bool:
        """Switch this run to the New existing-Stage-1 runner after strict completion.

        The promotion applies only to the current run's post-generation launch
        settings. Any incomplete summary or materialization failure leaves the
        original launch state untouched so the ordinary fallback pipeline still
        runs exactly as before.
        """
        if summary is None or not getattr(summary, "complete", False) or not dataset_paths:
            return False
        if launch_state.get("use_existing_stage_one"):
            return True
        try:
            from ASR.streaming_stage_one import materialize_stage_one_summary

            materialized = materialize_stage_one_summary(summary, dataset_paths["tts_dir"])
        except Exception as exc:
            logger.warning(
                "During-TTS Stage 1 materialization failed: %s. "
                "Continuing with original post-gen ASR path.",
                exc,
            )
            return False

        launch_state["pipeline"] = "new"
        launch_state["failure_log"] = str(Path(dataset_paths["tts_dir"]) / "asr_new_failures.json")
        launch_state["run_log"] = str(Path(dataset_paths["tts_dir"]) / "asr_new_pipeline.log")
        launch_state["use_existing_stage_one"] = True
        logger.info(
            "During-TTS Stage 1 promoted for this run: submitted=%s failed=%s candidates=%s. "
            "Launching New post-gen runner from materialized streaming evidence.",
            getattr(summary, "submitted", 0),
            getattr(summary, "failed", 0),
            materialized.get("candidate_count", 0),
        )
        return True

    def _abort_shadow_stage_one_service(self, service: Any, reason: str) -> None:
        """Abort an active shadow Stage 1 service without affecting authoritative ASR.

        Once the service has already been replaced by a frozen finish summary,
        there is nothing left to abort. That terminal summary should pass
        through shutdown paths without raising attribute errors.
        """
        if service is None:
            return
        if not hasattr(service, "abort"):
            return
        try:
            logger.warning(
                "Aborting during-TTS Stage 1 shadow service (%s). "
                "Authoritative post-gen ASR remains active.",
                reason,
            )
            service.abort()
        except Exception as exc:
            logger.warning("Shadow Stage 1 abort note: %s", exc)

    @staticmethod
    def extract_voice_name(voice_path: str) -> str:
        """Extract clean voice name for filename use."""
        from pathlib import Path
        import re
        from urllib.parse import unquote

        # Clean up URI prefix if present (e.g. from drag-and-drop or browser copy)
        if voice_path.startswith("file://"):
            voice_path = unquote(voice_path.replace("file://", ""))

        # Handle legacy "Custom: " prefix if present
        if voice_path.startswith("Custom:"):
             voice_path = voice_path.replace("Custom: ", "")

        # Check for path separators (Linux/Mac '/' or Windows '\')
        if "/" in voice_path or "\\" in voice_path:
            # It's a path - take the stem (filename without extension)
            voice_name = Path(voice_path).stem
        else:
            # Built-in voice - take first word (e.g. "alba (default)" -> "alba")
            voice_name = voice_path.split(" ")[0]

        # Remove special characters for filesystem safety
        # We allow alphanumerics, underscores, hyphens
        voice_name = re.sub(r'[^\w\-_]', '', voice_name)

        return voice_name

    @staticmethod
    def generate_output_paths(
        input_text_path: str,
        voice_path: str = None,
        output_dir_override: Union[str, Path, None] = None,
    ) -> Dict[str, Union[str, Path]]:
        """
        Generate automatic output paths based on input filename.

        Args:
            input_text_path: Path to input text file
            voice_path: Voice file path or name (optional)
            output_dir_override: Explicit book output directory. This preserves
                a queued batch job's isolated output folder.

        Returns:
            Dictionary with output directory paths
        """
        path = Path(input_text_path)
        filename = path.stem  # Remove .txt extension

        # Extract book title (everything before first " - " or "_output")
        if " - " in filename:
            book_title = filename.split(" - ")[0]
        elif "_output" in filename:
            book_title = filename.split("_output")[0]
        else:
            book_title = filename

        # Clean title for filesystem
        book_title = book_title.strip().replace('/', '_').replace('\\', '_')

        output_dir = (
            Path(output_dir_override)
            if output_dir_override is not None
            else Path("Output") / book_title
        )
        tts_dir = output_dir / "TTS"
        audio_chunks_dir = tts_dir / "audio_chunks"
        text_chunks_dir = tts_dir / "text_chunks"

        # Generate dynamic filename with voice info if provided
        if voice_path:
            voice_name = AudiobookGenerator.extract_voice_name(voice_path)
            final_audio_filename = f"{filename} [{voice_name}].wav"
        else:
            final_audio_filename = "audiobook.wav"

        final_audio_path = output_dir / final_audio_filename

        return {
            'output_dir': output_dir,
            'tts_dir': tts_dir,
            'audio_chunks_dir': audio_chunks_dir,
            'text_chunks_dir': text_chunks_dir,
            'final_audio_path': final_audio_path,
            'final_audio_filename': final_audio_filename
        }

    def _cleanup_existing_chunks(self, audio_chunks_dir: Union[str, Path], text_chunks_dir: Union[str, Path]):
        """Clear chunk dirs and stale root ASR artifacts before a fresh run.

        Prior runs leave WAVs, TXT files, JSON metadata, and ASR log files
        behind. Audio/text chunk dirs are emptied recursively, the audio
        ``Failed/`` dir is recreated empty, and TTS-root ``.json``/``.log``
        files are removed except ``run*.log`` history files.
        """
        audio_dir = Path(audio_chunks_dir)
        text_dir = Path(text_chunks_dir)
        tts_dir = audio_dir.parent

        self._purge_directory_contents(audio_dir)
        self._purge_directory_contents(text_dir)
        self._cleanup_tts_root_artifacts(tts_dir)

        # Recreate empty layout for this run
        audio_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "Failed").mkdir(parents=True, exist_ok=True)

    def _purge_directory_contents(self, directory: Path) -> None:
        """Delete every file and subdirectory inside one directory.

        Args:
            directory: Folder whose current contents should be removed.
        """
        if not directory.exists():
            return

        for child in directory.iterdir():
            try:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except OSError as exc:
                logger.warning("Could not delete %s: %s", child, exc)

    def _cleanup_tts_root_artifacts(self, tts_dir: Path) -> None:
        """Remove stale top-level TTS JSON and log artifacts from prior runs.

        Args:
            tts_dir: TTS directory that owns run-wide JSON and log files.
        """
        if not tts_dir.exists():
            return

        for artifact in tts_dir.iterdir():
            if not artifact.is_file():
                continue
            if artifact.suffix not in {".json", ".log"}:
                continue
            # Keep historical run logs so different runs stay inspectable.
            if artifact.suffix == ".log" and artifact.name.startswith("run"):
                continue
            try:
                artifact.unlink()
            except OSError as exc:
                logger.warning("Could not delete %s: %s", artifact, exc)

    def _write_run_settings_log(
        self,
        tts_dir: Path,
        run_settings: Optional[Dict[str, Any]],
        *,
        source_file: str,
        voice_path: str,
        output_path: str,
    ) -> Optional[Path]:
        """Write a human-readable GUI run log in the exact requested order.

        Log keeps only requested fields, prints nested details only when
        parent checkbox is enabled, and stays stable for Main and Batch runs.

        Args:
            tts_dir: Book-local TTS artifact directory.
            run_settings: Per-run values supplied by Main or Batch GUI paths.
            source_file: Input text file used for this run.
            voice_path: Voice file or voice alias used for this run.
            output_path: Final audio output path for this run.

        Returns:
            Created log path, or ``None`` if the log could not be written.
        """
        settings = run_settings or {}
        pause_enabled = bool(settings.get("pause_injection_enabled", False))
        speed_enabled = bool(
            settings.get(
                "speed_variation",
                (getattr(self.config, "speed_variation", {}) or {}).get("enabled", False),
            )
        )
        emotion_detection_enabled = bool(
            settings.get(
                "emotion_detection_enabled",
                (getattr(self.config, "emotion", {}) or {}).get(
                    "enabled", True
                ),
            )
        )
        m4b_config = dict(getattr(self.config, "m4b", {}) or {})
        asr_config = dict(getattr(self.config, "asr_quality_control", {}) or {})
        device_config = dict(getattr(self.config, "device", {}) or {})
        parallel_config = dict(getattr(self.config, "parallel", {}) or {})
        tts_core = dict(getattr(self.config, "tts_core", {}) or {})
        chunking = dict(getattr(self.config, "chunking", {}) or {})
        quality = dict(getattr(self.config, "quality", {}) or {})

        def _bool_text(value: bool) -> str:
            """Render GUI flags as lowercase text for the log."""
            return "true" if value else "false"

        def _format_number(value: Any, decimals: int = 2) -> str:
            """Format numeric settings without losing user-visible precision."""
            try:
                num = float(value)
            except (TypeError, ValueError):
                return str(value)
            return f"{num:.{decimals}f}"

        def _format_seconds_map(values: Dict[str, Any]) -> List[str]:
            """Render pause timings in the fixed punctuation order."""
            ordered_punct = ["!", ",", "--", ".", "...", ":", ";", "?"]
            lines: List[str] = []
            for punct in ordered_punct:
                if punct in values:
                    lines.append(f"    {punct:<3} = {_format_number(values[punct], 2)} s")
            return lines

        now = datetime.now().astimezone()
        tz = now.strftime("%z")
        tz = f"{tz[:-2]}:{tz[-2:]}" if len(tz) == 5 else tz
        lines: List[str] = [
            "Pocket TTS GUI Run Log",
            f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')} {tz}",
            "",
            f"  Input text file: {source_file}",
            f"  Voice: {voice_path}",
            f"  Output path: {output_path}",
            "",
            f"  Temperature: {_format_number(tts_core.get('temperature', settings.get('temperature', '')))}",
            f"  EOS threshold: {_format_number(tts_core.get('eos_threshold', settings.get('eos_threshold', '')))}",
            f"  Frames after EOS: {int(tts_core.get('frames_after_eos', settings.get('frames_after_eos', 0)) or 0)}",
            "",
            f"  Chunking mode: {chunking.get('mode', settings.get('chunking_mode', ''))}",
            f"  Min words: {int(chunking.get('min_words', settings.get('min_words', 0)) or 0)}",
            "",
            f"  LSD steps: {int(quality.get('lsd_steps', settings.get('lsd_steps', 0)) or 0)}",
            f"  Speed variation enabled: {_bool_text(speed_enabled)}",
            f"  Emotion detection enabled: {_bool_text(emotion_detection_enabled)}",
            "",
            "",
            f"  Pause injection enabled: {_bool_text(pause_enabled)}",
        ]

        if pause_enabled:
            lines.append("  Punctuation durations:")
            lines.extend(_format_seconds_map(dict(settings.get("pause_durations", {}) or {})))

        lines.extend([
            "",
            f"  Write M4B: {_bool_text(bool(m4b_config.get('write_m4b', False)))}",
            f"  Write MP3: {_bool_text(bool(m4b_config.get('write_mp3', False)))}",
            f"  Write WAV: {_bool_text(bool(m4b_config.get('write_wav', True)))}",
            f"  Chapterize: {_bool_text(bool(m4b_config.get('chapterize', False)))}",
            f"  Chapter strategy: {m4b_config.get('chapter_mode', 'legacy')}",
            f"  Max chapter minutes: {m4b_config.get('max_chapter_minutes', 0)}",
        ])
        if m4b_config.get("write_m4b") or m4b_config.get("write_mp3"):
            lines.append(
                f"  Normalization method: {m4b_config.get('normalization_type', 'peak')}"
            )

        lines.extend([
            "",
            f"  Device: {device_config.get('preferred', getattr(self.config, 'device', {}).get('preferred', ''))}",
            f"  Max workers: {int(parallel_config.get('max_workers', settings.get('max_workers', 0)) or 0)}",
            "",
            "",
            f"  ASR quality control enabled: {_bool_text(bool(asr_config.get('enabled', False)))}",
        ])

        if asr_config.get("enabled", False):
            threshold_value = asr_config.get("threshold", asr_config.get("asr_threshold", 0.75))
            lines.extend([
                f"  Engine: {asr_config.get('engine', 'faster_whisper')}",
                f"  Language: {asr_config.get('language', 'en')}",
                f"  Model: {asr_config.get('model', 'base')}",
                f"  GPU workers after TTS: {int((asr_config.get('parallel') or {}).get('gpu_workers_after_tts', 0) or 0)}",
                f"  Threshold: {_format_number(threshold_value)}",
                f"  Max retries: {int(asr_config.get('max_retries', 3) or 0)}",
                f"  Temp decrement: {_format_number(asr_config.get('temp_decrement', 0.1))}",
            ])

        try:
            tts_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = tts_dir / f"run_{timestamp}.log"
            log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
            logger.info("GUI run settings log: %s", log_path)
            return log_path
        except OSError as exc:
            logger.warning("Could not write GUI run settings log in %s: %s", tts_dir, exc)
            return None

    def generate_audiobook(self,
                           chunks: List[ChunkMetadata],
                           voice_path: str,
                           output_path: str,
                           progress_callback=None,
                           source_file: str = "unknown",
                           save_dataset_chunks: bool = True,
                           total_start_time: Optional[float] = None,
                           run_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate audiobook from text chunks.

        Args:
            chunks: List of processed text chunks
            voice_path: Path to voice file or voice name
            output_path: Where to save the generated audio
            progress_callback: Function to call with progress updates
            source_file: Path to the original text file
            total_start_time: Optional timestamp captured when GUI generation
                was started, used for end-to-end timing.
            run_settings: Effective per-run GUI values for the book-local log.

        Returns:
            Dict with generation results and statistics
        """
        import logging

        def _phase(phase: str, message: str, **extra) -> None:
            """Emit a GUI phase update (starting / tts / asr / regen / done)."""
            logger.info("[%s] %s", phase.upper(), message)
            if progress_callback:
                try:
                    payload = {"phase": phase, "message": message, **extra}
                    progress_callback(payload)
                except Exception:
                    pass

        _phase(
            "starting",
            "Starting generation "
            f"(chunks={len(chunks)}, device={getattr(self, '_device', 'auto')}, "
            f"workers={getattr(self, 'num_workers', 1)})",
            total_chunks=len(chunks),
            current_chunk=0,
        )

        # Fresh main app log every generation (pocket_tts.log) so checking
        # "the log" after a run only shows this run.
        try:
            from pocket_tts.utils.utils import reset_pocket_tts_app_log

            app_log = reset_pocket_tts_app_log()
        except Exception:
            app_log = None

        # Per-run book debug log (also overwrite, not append).
        debug_log_path = Path(output_path).with_suffix('.debug.log')
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)

        debug_logger = logging.getLogger('audiobook_debug')
        debug_logger.setLevel(logging.DEBUG)
        preprocessing_loggers = [
            logging.getLogger('pocket_tts.preprocessing.structure_detector'),
            logging.getLogger('pocket_tts.preprocessing.chunker'),
        ]
        # Keep FileHandlers that write the app log; drop prior book-debug handlers.
        try:
            from pocket_tts.utils.utils import APP_LOG_PATH

            app_log_resolved = APP_LOG_PATH.resolve()
        except Exception:
            app_log_resolved = None

        loggers_for_debug = [debug_logger, logger, *preprocessing_loggers]
        for lg in loggers_for_debug:
            for handler in list(lg.handlers):
                if not isinstance(handler, logging.FileHandler):
                    continue
                try:
                    hp = Path(getattr(handler, "baseFilename", "") or "").resolve()
                except Exception:
                    hp = None
                # Do not close the freshly attached app log handler
                if app_log_resolved is not None and hp == app_log_resolved:
                    continue
                lg.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass

        debug_handler = logging.FileHandler(debug_log_path, mode='w', encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        debug_handler.setFormatter(debug_formatter)
        debug_logger.addHandler(debug_handler)

        logger.addHandler(debug_handler)
        logger.setLevel(logging.DEBUG)
        for pre_logger in preprocessing_loggers:
            pre_logger.addHandler(debug_handler)
            pre_logger.setLevel(logging.INFO)

        logger.info("=== AUDIOBOOK GENERATION START ===")
        if app_log is not None:
            logger.info("App log (fresh this run): %s", app_log)
        logger.info("Book debug log (fresh this run): %s", debug_log_path)
        logger.debug("DEBUG: Debug logging initialized")
        debug_logger.info("DEBUG_LOGGER: Debug logging active")
        logger.info(f"Total chunks to process: {len(chunks)}")
        logger.info(f"Voice: {voice_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Source file: {source_file}")

        self.start_time = total_start_time if total_start_time is not None else time.time()
        self._original_generation_end = None
        self.total_chunks = len(chunks)
        self.current_chunk = 0
        investigation_log = []
        asr_check_time = 0.0
        asr_regeneration_time = 0.0
        asr_check_start = None
        asr_check_end = None
        asr_regeneration_start = None
        asr_regeneration_end = None

        # Setup dataset structure if enabled
        dataset_paths = None
        saved_chunk_paths = []
        shadow_stage_one_state = {"service": None}
        if save_dataset_chunks:
            # Use source_file to generate proper naming with voice info
            dataset_paths = self.generate_output_paths(
                source_file,
                voice_path,
                output_dir_override=Path(output_path).parent,
            )
            self._cleanup_existing_chunks(
                dataset_paths['audio_chunks_dir'],
                dataset_paths['text_chunks_dir']
            )
            self._write_run_settings_log(
                dataset_paths['tts_dir'],
                run_settings,
                source_file=source_file,
                voice_path=voice_path,
                output_path=str(dataset_paths['final_audio_path']),
            )
            logger.info(f"Dataset output directory: {dataset_paths['output_dir']}")
            # Override output_path to use generated path
            output_path = dataset_paths['final_audio_path']
            logger.info(f"Using automatic output path: {output_path}")

        logger.info(f"Output: {output_path}")
        logger.info(f"Source file: {source_file}")

        # Setup dataset structure if enabled
        dataset_paths = None
        saved_chunk_paths = []
        if save_dataset_chunks:
            # Use source_file to generate proper naming with voice info
            dataset_paths = self.generate_output_paths(
                source_file,
                voice_path,
                output_dir_override=Path(output_path).parent,
            )
            self._cleanup_existing_chunks(
                dataset_paths['audio_chunks_dir'],
                dataset_paths['text_chunks_dir']
            )
            logger.info(f"Dataset output directory: {dataset_paths['output_dir']}")
            # Override output_path to use generated path
            output_path = dataset_paths['final_audio_path']
            logger.info(f"Using automatic output path: {output_path}")

        try:
            # Initialize TTS model
            logger.info("Initializing TTS model...")
            self._init_tts_model()
            logger.info("TTS model initialized")

            # Convert custom voice files if needed
            import os
            if os.path.isfile(voice_path):
                from ..data.voice_converter import VoicePromptConverter
                converter = VoicePromptConverter()
                # Use the TTS directory for converted voices
                if save_dataset_chunks and dataset_paths:
                    tts_dir = dataset_paths['tts_dir']
                else:
                    # Fallback: use temp directory
                    import tempfile
                    tts_dir = Path(tempfile.gettempdir()) / "pocket_tts_converted_voices"
                    tts_dir.mkdir(exist_ok=True)
                converted_path = converter.convert(voice_path, tts_dir)
                logger.info(f"Converted custom voice to: {converted_path}")
                voice_path = str(converted_path)

            # Load voice
            logger.info(f"Loading voice: {voice_path}")
            voice_state = self._load_voice(voice_path)
            logger.info("Voice loaded successfully")

            # Save chunk data to JSON immediately after preprocessing
            if save_dataset_chunks and dataset_paths:
                logger.info("Saving chunk data to JSON (preprocessing complete)...")
                self._save_chunks_json(chunks, output_path, voice_path, source_file, dataset_paths,
                                      is_preliminary=True)
                logger.info("Preliminary JSON data saved successfully")

            # --- ASR Quality Control Setup ---
            asr_process = None
            asr_failure_log = None
            asr_run_log = None
            asr_log_handle = None
            asr_exe = None
            asr_script = 'ASR/asr_validator.py'
            asr_language = 'en'
            asr_engine = 'faster_whisper'
            asr_n_threads = 0
            asr_gpu_workers = 0
            monitor_folder = None
            threshold = 0.85
            asr_post_gen_done = False
            asr_pipeline_error: Optional[str] = None
            asr_stage_one_time: float = 0.0
            asr_stage_two_time: float = 0.0
            # Explicit audit counts for GUI: stage 1 → stage 2 → remaining after regen.
            asr_stage_one_fails: int = 0
            asr_stage_two_fails: int = 0
            asr_config = getattr(self.config, 'asr_quality_control', {})
            asr_config['_regen_voice_path'] = str(voice_path)
            asr_enabled = asr_config.get('enabled', False)
            asr_pipeline = str(asr_config.get('pipeline', 'legacy') or 'legacy').strip().lower()
            if asr_pipeline != 'new':
                asr_pipeline = 'legacy'
            asr_launch_state: Dict[str, Any] = {
                "pipeline": asr_pipeline,
                "failure_log": None,
                "run_log": None,
                "use_existing_stage_one": False,
            }
            run_alignment_diagnostic = bool(
                asr_config.get('alignment_diagnostic_enabled', False)
            )
            asr_model = asr_config.get('model', 'base')
            second_stage_model = str(
                asr_config.get('second_stage_model', 'medium') or 'medium'
            ).strip()
            regen_plan = None
            regen_plan_summary: list[str] = []
            regen_keep_tts_resident = False
            regen_tts_workers = 1
            regen_asr_model_name = "medium"
            regen_verification_summary: Dict[str, Any] = {}
            # A generator instance can serve more than one GUI run. Do not let
            # a prior run's scheduling decision leak into this run.
            self._regen_keep_tts_resident = False
            self._regen_tts_workers = 1
            self._regen_plan_summary = []

            def _launch_post_gen_gpu_asr() -> None:
                """No-op stub; replaced when ASR is enabled with dataset paths."""
                return

            if asr_enabled and save_dataset_chunks and dataset_paths:
                asr_exe = _resolve_asr_executable(
                    asr_config.get('executable_path') or None
                )
                asr_script = 'ASR/asr_validator.py'
                monitor_folder = str(Path(dataset_paths['audio_chunks_dir']))
                if asr_pipeline == 'new':
                    # Keep test-pipeline artifacts separate from canonical legacy reports.
                    asr_failure_log = str(Path(dataset_paths['tts_dir']) / 'asr_new_failures.json')
                    asr_run_log = str(Path(dataset_paths['tts_dir']) / 'asr_new_pipeline.log')
                else:
                    asr_failure_log = str(Path(dataset_paths['tts_dir']) / 'asr_failures.json')
                    asr_run_log = str(Path(dataset_paths['tts_dir']) / 'asr_run.log')
                asr_launch_state["failure_log"] = asr_failure_log
                asr_launch_state["run_log"] = asr_run_log
                threshold = asr_config.get('threshold', 0.85)

                # Ensure monitor folder exists before launching ASR
                monitor_path = Path(monitor_folder)
                if not monitor_path.exists():
                    logger.info(f"Creating ASR monitor folder: {monitor_folder}")
                    monitor_path.mkdir(parents=True, exist_ok=True)
                else:
                    logger.info(f"ASR monitor folder exists: {monitor_folder}")

                # Clear any previous failure log
                if Path(asr_failure_log).exists():
                    Path(asr_failure_log).unlink()

                # ASR is NOT started during TTS gen (avoids GPU fight / OOM).
                # After original generation finishes we launch the selected ASR device.
                asr_language = asr_config.get('language', 'en')
                parallel_cfg = asr_config.get('parallel') or {}
                if not isinstance(parallel_cfg, dict):
                    parallel_cfg = {}
                requested_asr_device = str(
                    asr_config.get('device', self._device) or self._device
                ).strip().lower()
                if requested_asr_device not in {'cpu', 'cuda', 'auto'}:
                    requested_asr_device = 'auto'
                if requested_asr_device == 'auto':
                    try:
                        requested_asr_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    except Exception:
                        requested_asr_device = 'cpu'
                worker_config_key = (
                    'cpu_workers' if requested_asr_device == 'cpu' else 'gpu_workers_after_tts'
                )
                # CPU mode deliberately stays single-model; GPU uses the calibrated cap.
                default_asr_workers = 1 if requested_asr_device == 'cpu' else 4
                asr_gpu_workers = int(
                    parallel_cfg.get(
                        worker_config_key,
                        asr_config.get(worker_config_key, default_asr_workers),
                    )
                    or 0
                )
                if requested_asr_device == 'cpu':
                    asr_gpu_workers = 1
                else:
                    asr_gpu_workers = max(0, min(16, asr_gpu_workers))
                # Legacy faster-whisper remains default; Parakeet is opt-in.
                raw_engine = str(asr_config.get('engine', 'faster_whisper') or 'faster_whisper')
                raw_engine = raw_engine.strip().lower().replace('-', '_')
                if raw_engine in ('whisper_cpp', 'whispercpp', 'cpp', 'pywhispercpp'):
                    asr_engine = 'whisper_cpp'
                elif raw_engine in ('parakeet', 'parakeet_tdt', 'nemo_parakeet'):
                    asr_engine = 'parakeet'
                else:
                    asr_engine = 'faster_whisper'
                asr_n_threads = int(parallel_cfg.get('n_threads', 0) or 0)
                # Clear stale flag from prior hybrid mode
                try:
                    stale_flag = Path(dataset_paths['tts_dir']) / 'tts_gen_done.flag'
                    if stale_flag.exists():
                        stale_flag.unlink()
                except Exception:
                    pass
                logger.info(
                    "ASR deferred until TTS gen done: pipeline=%s stage1_model=%s stage2_model=%s threshold=%s "
                    "device=%s workers=%s engine=%s (no ASR during generation)",
                    asr_pipeline,
                    asr_model,
                    second_stage_model,
                    threshold,
                    requested_asr_device,
                    asr_gpu_workers,
                    asr_engine,
                )

                try:
                    project_root = Path(__file__).parent.parent.parent
                    from ASR.asr_validator import calculate_available_vram_for_asr

                    batch_cfg = getattr(self.config, 'batch_generation', {}) or {}
                    if isinstance(batch_cfg, dict):
                        batch_size = int(batch_cfg.get('batch_size', 4) or 4)
                    else:
                        batch_size = 4
                    regen_plan = choose_regeneration_plan(
                        asr_model,
                        stage_two_model=second_stage_model,
                        results_root=project_root / "tests" / "results",
                        requested_tts_workers=max(1, int(self.num_workers or 1)),
                        requested_tts_batch_size=max(1, batch_size),
                        requested_asr_workers=max(1, int(asr_gpu_workers or 1)),
                        requested_asr_batch_size=8,
                        safe_vram_mb=float(calculate_available_vram_for_asr()),
                    )
                    regen_keep_tts_resident = bool(regen_plan.keep_tts_resident)
                    regen_tts_workers = max(1, int(regen_plan.tts_workers or 1))
                    regen_asr_model_name = str(regen_plan.regen_asr_model or "medium")
                    regen_plan_summary = format_plan_summary(regen_plan)
                    self._regen_keep_tts_resident = regen_keep_tts_resident
                    self._regen_tts_workers = regen_tts_workers
                    self._regen_plan_summary = regen_plan_summary
                except Exception as exc:
                    logger.warning("Could not precompute regeneration plan: %s", exc)

                def _launch_post_gen_gpu_asr() -> None:
                    """Start post-generation ASR on the finished TTS folder.

                    New mode invokes ``tools/run_asr_pipeline.py --pipeline new``
                    for isolated Parakeet, GPU Medium, and diagnostic alignment.
                    It is report-only and writes ``asr_new_*`` artifacts.

                    faster_whisper: one model on the selected device, pack + solo-retry
                    via ``--batch-tts-dir`` — same path as standalone ASR GUI.
                    whisper_cpp: multi-model CUDA backend (GPU mode only).

                    Does nothing if already launched or worker count is 0.
                    """
                    nonlocal asr_process, asr_log_handle, asr_post_gen_done, asr_pipeline_error
                    if asr_post_gen_done or asr_process is not None:
                        return
                    if not asr_launch_state.get("failure_log") or not dataset_paths:
                        return
                    active_pipeline = str(asr_launch_state.get("pipeline", asr_pipeline) or asr_pipeline)
                    active_run_log = str(asr_launch_state.get("run_log") or asr_run_log or "")
                    if active_pipeline == 'new':
                        pipeline_script = Path(__file__).resolve().parents[2] / 'tools' / 'run_asr_pipeline.py'
                        asr_cmd = [
                            sys.executable,
                            str(pipeline_script),
                            str(Path(dataset_paths['tts_dir'])),
                            '--pipeline', 'new',
                            '--threshold', str(threshold),
                            '--language', asr_language,
                            '--second-stage-model', second_stage_model,
                        ]
                        if bool(asr_launch_state.get("use_existing_stage_one")):
                            asr_cmd.append('--use-existing-stage-one')
                        if not run_alignment_diagnostic:
                            asr_cmd.append('--skip-alignment')
                        try:
                            asr_log_handle = open(active_run_log, 'w', encoding='utf-8')
                            asr_process = subprocess.Popen(
                                asr_cmd,
                                stdout=asr_log_handle,
                                stderr=subprocess.STDOUT,
                                text=True,
                            )
                            logger.info(
                                "New isolated ASR pipeline launched after TTS: command=%s log=%s",
                                asr_cmd,
                                active_run_log,
                            )
                        except Exception as exc:
                            asr_pipeline_error = f"launch failed: {exc}"
                            asr_post_gen_done = True
                            logger.error("Failed to launch new ASR pipeline: %s", exc)
                            if asr_log_handle is not None:
                                asr_log_handle.close()
                                asr_log_handle = None
                            _phase(
                                'asr',
                                f"New ASR pipeline failed to launch: {exc}",
                                asr_status=True,
                            )
                            _finish_asr_check()
                        return
                    workers = max(0, min(16, int(asr_gpu_workers)))
                    if workers <= 0:
                        logger.info("Post-gen ASR skipped (worker count is 0)")
                        asr_post_gen_done = True
                        return
                    tts_dir_str = str(Path(dataset_paths["tts_dir"]))
                    try:
                        # Free leftover TTS CUDA cache so ASR models can load.
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        asr_log_handle = open(active_run_log, 'w', encoding='utf-8')
                        # Prefer full-folder batch (all WAVs already on disk after gen)
                        asr_cmd = [
                            asr_exe,
                            asr_script,
                            '--batch-tts-dir', tts_dir_str,
                            '--log-file', str(asr_launch_state.get("failure_log") or asr_failure_log),
                            '--threshold', str(threshold),
                            '--model', asr_model,
                            '--language', asr_language,
                            '--device', requested_asr_device,
                            '--cpu-workers', str(workers),
                            '--engine', asr_engine,
                            '--n-threads', str(asr_n_threads),
                            '--pack-size', '8',
                        ]
                        asr_process = subprocess.Popen(
                            asr_cmd,
                            stdout=asr_log_handle,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        logger.info(
                            "Post-gen ASR batch launched: engine=%s model=%s "
                            "workers=%s pack=8 device=%s expected=%s log=%s",
                            asr_engine,
                            asr_model,
                            workers,
                            requested_asr_device,
                            len(chunks),
                            active_run_log,
                        )
                    except Exception as e:
                        logger.warning("Failed to launch post-gen GPU ASR: %s", e)
                        asr_process = None
                        if asr_log_handle is not None:
                            try:
                                asr_log_handle.close()
                            except Exception:
                                pass
                            asr_log_handle = None

                shadow_stage_one_state["service"] = self._start_shadow_stage_one_service(
                    self._resolve_shadow_stage_one_request(
                        asr_enabled=asr_enabled,
                        save_dataset_chunks=save_dataset_chunks,
                        dataset_paths=dataset_paths,
                        asr_config=asr_config,
                    )
                )

            def _finish_asr_check() -> None:
                """Close ASR-check timing after validation and failure reporting finish."""
                nonlocal asr_check_end, asr_check_time
                if asr_check_start is not None and asr_check_end is None:
                    asr_check_end = time.time()
                    asr_check_time = asr_check_end - asr_check_start

            def _report_new_pipeline_failure(detail: str) -> None:
                """Show one clear New-pipeline failure without entering regeneration."""
                log_hint = asr_run_log or "new ASR pipeline log"
                message = f"New ASR pipeline failed — {detail}. See {log_hint}"
                logger.error(message)
                _phase('asr', message, asr_status=True)
                _finish_asr_check()

            def _complete_new_pipeline_report() -> bool:
                """Report a valid isolated pipeline result without entering regeneration.

                Returns:
                    True only when the runner wrote a readable New-pipeline report.
                """
                nonlocal asr_stage_one_time, asr_stage_two_time
                nonlocal asr_stage_one_fails, asr_stage_two_fails, regen_verification_summary
                if asr_pipeline_error:
                    _report_new_pipeline_failure(asr_pipeline_error)
                    return False
                if not dataset_paths:
                    _report_new_pipeline_failure("TTS output directory is unavailable")
                    return False
                report_path = Path(dataset_paths['tts_dir']) / 'asr_new_pipeline_report.json'
                try:
                    report = json.loads(report_path.read_text(encoding='utf-8'))
                    if report.get('pipeline') != 'new':
                        raise ValueError("report does not identify the New pipeline")
                    candidate_count = report.get('stage_one_candidate_count', 0)
                    medium = report.get('medium') or {}
                    confirmed_count = medium.get('confirmed_failures', 0)
                    stage_one_process = report.get("stage_one_process") or {}
                    stage_two_process = report.get("medium_process") or {}
                    asr_stage_one_time = stage_one_process.get("wall_s")
                    asr_stage_two_time = stage_two_process.get("wall_s")
                    stage_one_summary = (report.get("stage_one") or {}).get("summary") or {}
                    asr_stage_one_fails = int(
                        stage_one_summary.get("failed", candidate_count) or candidate_count or 0
                    )
                    asr_stage_two_fails = int(confirmed_count or 0)
                    medium_summary = medium.get("summary") or {}
                    if medium_summary:
                        regen_verification_summary = dict(medium_summary)
                    else:
                        accepted = max(0, asr_stage_one_fails - asr_stage_two_fails)
                        regen_verification_summary = {
                            "attempted": asr_stage_one_fails,
                            "verified_pass": accepted,
                            "verified_fail": asr_stage_two_fails,
                            "stage_one_candidates": asr_stage_one_fails,
                            "medium_confirmed_failures": asr_stage_two_fails,
                            "medium_accepted_or_unproven": accepted,
                            "verification_model": medium.get("model") or second_stage_model,
                        }
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    _report_new_pipeline_failure(f"report unavailable: {exc}")
                    return False
                message = (
                    "New ASR report complete — "
                    f"{candidate_count} Parakeet candidate(s), {confirmed_count} Medium-confirmed"
                )
                logger.info(message)
                _phase('asr', message, asr_status=True)
                _finish_asr_check()
                return True

            def _finish_asr_regeneration() -> None:
                """Close ASR-regeneration timing after the final regen result arrives."""
                nonlocal asr_regeneration_end, asr_regeneration_time
                if asr_regeneration_start is not None and asr_regeneration_end is None:
                    asr_regeneration_end = time.time()
                    asr_regeneration_time = asr_regeneration_end - asr_regeneration_start

            # Generate audio for each chunk
            audio_chunks = []
            chunk_start_time = time.time()
            self._chunk_process_start = time.time()

            logger.info("=== STARTING PARALLEL CHUNK PROCESSING ===")
            _phase("tts", f"TTS generation starting ({len(chunks)} chunks)", total_chunks=len(chunks), current_chunk=0)
            original_chunks_count = len(chunks)

            # Use parallel processing if enabled and worthwhile
            use_parallel = self.parallel_enabled and self.num_workers > 1 and len(chunks) >= 4

            def prepare_parallel_regeneration_tasks():
                """Run post-gen ASR and build legacy regeneration tasks when applicable."""
                nonlocal asr_process, asr_log_handle, asr_post_gen_done, asr_pipeline_error
                nonlocal asr_failure_log
                nonlocal asr_check_start, asr_regeneration_start
                nonlocal regen_plan, regen_plan_summary
                nonlocal regen_keep_tts_resident, regen_tts_workers, regen_asr_model_name
                nonlocal regen_verification_summary
                nonlocal asr_stage_one_time, asr_stage_two_time
                nonlocal asr_stage_one_fails, asr_stage_two_fails
                shadow_summary = self._finish_shadow_stage_one_service(
                    shadow_stage_one_state["service"],
                    len(chunks),
                )
                shadow_stage_one_state["service"] = shadow_summary
                self._promote_shadow_stage_one_to_new_pipeline(
                    shadow_summary,
                    dataset_paths,
                    asr_launch_state,
                )
                if not asr_enabled or not asr_launch_state.get("failure_log") or not dataset_paths:
                    return []

                # Original TTS wave is done — start GPU ASR now (not during gen).
                asr_check_start = self._original_generation_end or time.time()
                logger.info("TTS original gen done — starting post-gen GPU ASR")
                if str(asr_launch_state.get("pipeline", asr_pipeline) or asr_pipeline) == 'new':
                    # The new runner starts Parakeet in another process; free parent CUDA first.
                    try:
                        self.release_gpu_memory()
                    except Exception as exc:
                        logger.warning("Parent GPU release before new ASR pipeline: %s", exc)
                _phase(
                    "asr",
                    "ASR check starting (post-gen GPU) — model download/load may take minutes first time",
                    asr_status=True,
                    total_chunks=len(chunks),
                )
                _launch_post_gen_gpu_asr()

                if asr_process is not None:
                    logger.info("Waiting for post-gen GPU ASR before regeneration phase...")
                    logger.warning(
                        "ASR may download models on first use — watch %s for progress "
                        "(not stuck).",
                        str(asr_launch_state.get("run_log") or asr_run_log),
                    )
                    try:
                        stage_one_wait_start = time.time()
                        _wait_asr_process_with_progress(
                            asr_process,
                            str(asr_launch_state.get("run_log") or asr_run_log),
                            timeout=max(300, len(chunks) * 30),
                            progress_callback=progress_callback,
                            total_chunks=len(chunks),
                        )
                        if str(asr_launch_state.get("pipeline", asr_pipeline) or asr_pipeline) == 'legacy':
                            asr_stage_one_time = time.time() - stage_one_wait_start
                        logger.info(f"ASR exit code: {asr_process.returncode}")
                        if str(asr_launch_state.get("pipeline", asr_pipeline) or asr_pipeline) == 'new' and asr_process.returncode != 0:
                            asr_pipeline_error = (
                                f"runner exited with code {asr_process.returncode}"
                            )
                    except subprocess.TimeoutExpired:
                        logger.warning("ASR monitor exceeded timeout; terminating it")
                        asr_process.terminate()
                        asr_process.wait(timeout=10)
                        if str(asr_launch_state.get("pipeline", asr_pipeline) or asr_pipeline) == 'new':
                            asr_pipeline_error = "runner exceeded the ASR timeout"
                    except Exception as exc:
                        logger.exception("Post-gen ASR wait failed")
                        if str(asr_launch_state.get("pipeline", asr_pipeline) or asr_pipeline) == 'new':
                            asr_pipeline_error = f"runner wait failed: {exc}"
                    finally:
                        if asr_log_handle is not None:
                            asr_log_handle.close()
                            asr_log_handle = None
                        asr_process = None
                        asr_post_gen_done = True

                if str(asr_launch_state.get("pipeline", asr_pipeline) or asr_pipeline) == 'new':
                    if not _complete_new_pipeline_report():
                        return []
                    try:
                        asr_failure_log, failures = _materialize_new_pipeline_regeneration_log(
                            Path(dataset_paths['tts_dir'])
                        )
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        _report_new_pipeline_failure(
                            f"confirmed-failure report unavailable: {exc}"
                        )
                        return []
                    logger.info(
                        "New ASR Medium confirmed %s chunk(s) for regeneration; log=%s",
                        len(failures),
                        asr_failure_log,
                    )
                    if not failures:
                        _finish_asr_check()
                        return []
                else:
                    if not asr_failure_log or not Path(asr_failure_log).exists():
                        logger.info("No ASR failures found for worker regeneration phase")
                        _finish_asr_check()
                        return []

                    with open(asr_failure_log, 'r', encoding='utf-8') as failure_file:
                        failures = json.load(failure_file)
                    _write_stage_one_failure_manifest(
                        Path(dataset_paths["tts_dir"]),
                        pipeline="legacy",
                        failure_filename="asr_failures.json",
                        threshold=threshold,
                        language=asr_config.get("language", "en"),
                    )

                if regen_plan is None:
                    project_root = Path(__file__).parent.parent.parent
                    try:
                        from ASR.asr_validator import calculate_available_vram_for_asr

                        safe_vram_mb = float(calculate_available_vram_for_asr())
                    except Exception:
                        safe_vram_mb = None

                    batch_cfg = getattr(self.config, 'batch_generation', {}) or {}
                    if isinstance(batch_cfg, dict):
                        batch_size = int(batch_cfg.get('batch_size', 4) or 4)
                    else:
                        batch_size = 4
                    regen_plan = choose_regeneration_plan(
                        asr_model,
                        stage_two_model=second_stage_model,
                        results_root=project_root / "tests" / "results",
                        requested_tts_workers=max(1, int(self.num_workers or 1)),
                        requested_tts_batch_size=batch_size,
                        requested_asr_workers=asr_gpu_workers,
                        requested_asr_batch_size=8,
                        safe_vram_mb=safe_vram_mb,
                    )
                regen_keep_tts_resident = bool(regen_plan.keep_tts_resident)
                regen_tts_workers = max(1, int(regen_plan.tts_workers or 1))
                regen_asr_model_name = str(regen_plan.regen_asr_model or "medium")
                regen_strategy = str(
                    asr_config.get("regeneration_strategy", "first_pass") or "first_pass"
                ).strip().lower()
                if regen_strategy not in ("first_pass", "best_of_n", "staged_three_candidates"):
                    regen_strategy = "first_pass"
                regen_plan_summary = format_plan_summary(regen_plan)
                asr_config['regen_asr_model'] = regen_asr_model_name
                asr_config['regen_keep_tts_resident'] = regen_keep_tts_resident
                asr_config['regen_plan'] = regen_plan.to_dict()

                logger.info("ASR regeneration plan: %s", " | ".join(regen_plan_summary))
                if progress_callback:
                    progress_callback({
                        "asr_status": True,
                        "message": f"Recovery plan selected: {regen_plan.chosen_strategy}",
                    })

                if asr_pipeline != 'new':
                    # Legacy Stage 1 report is the raw failure list before Medium.
                    asr_stage_one_fails = max(asr_stage_one_fails, len(failures))
                if regen_plan.verification_required and asr_pipeline != 'new':
                    stage_two_verify_start = time.time()
                    failures, regen_verification_summary = self._verify_failed_chunks_with_medium_asr(
                        failures,
                        Path(dataset_paths["tts_dir"]),
                        threshold,
                        asr_config.get('language', 'en'),
                        model_name=str(regen_plan.verification_model or second_stage_model or "medium"),
                    )
                    asr_stage_two_time = time.time() - stage_two_verify_start
                    asr_stage_two_fails = len(failures)
                    logger.info(
                        "Independent Medium accepted %s Stage 1 candidates (%s confirmed)",
                        regen_verification_summary.get("verified_pass", 0),
                        len(failures),
                    )
                    if progress_callback:
                        progress_callback({
                            "asr_status": True,
                            "message": (
                                f"Independent Medium confirmed {len(failures)} failed chunk(s) "
                                f"and accepted {regen_verification_summary.get('verified_pass', 0)}"
                            ),
                        })

                if int(asr_config.get('max_retries', 3) or 0) <= 0:
                    # Report-only runs still write the independent-ASR audit.
                    logger.info(
                        "ASR report-only mode enabled (max_retries=0); "
                        "skipping regeneration for %s independently confirmed failed chunks",
                        len(failures),
                    )
                    _phase(
                        "asr",
                        f"ASR report complete — {len(failures)} failed chunk(s) after independent verification, no regeneration",
                        asr_status=True,
                    )
                    _finish_asr_check()
                    return []

                if regen_strategy == "staged_three_candidates":
                    _finish_asr_check()
                    asr_regeneration_start = asr_check_end or time.time()
                    _phase(
                        "regen",
                        f"Staged regeneration starting for {len(failures)} failed chunk(s)",
                        total_chunks=len(chunks),
                    )
                    staged_investigations = self._reprocess_failed_chunks_staged(
                        failures,
                        None,
                        dataset_paths,
                        asr_config,
                        voice_path=str(voice_path),
                    )
                    investigation_log.extend(staged_investigations)
                    _finish_asr_regeneration()
                    _phase(
                        "regen",
                        f"Staged regeneration complete for {len(staged_investigations)} chunk(s)",
                        total_chunks=len(chunks),
                    )
                    return []

                chunk_by_index = {index: chunk for index, chunk in enumerate(chunks)}
                cleanup_config = getattr(self.config, 'audio_cleanup', {})
                project_root = Path(__file__).parent.parent.parent
                resolved_asr_exe = Path(_resolve_asr_executable(asr_exe or None))
                if not resolved_asr_exe.is_absolute():
                    resolved_asr_exe = project_root / resolved_asr_exe

                tasks = []
                for failure in failures:
                    chunk_index = failure.get('chunk_index')
                    chunk = chunk_by_index.get(chunk_index)
                    if chunk is None:
                        logger.warning("ASR failure references unknown chunk index %s", chunk_index)
                        continue
                    tts_params = (
                        dataclasses.asdict(chunk.tts_params)
                        if dataclasses.is_dataclass(chunk.tts_params)
                        else dict(chunk.tts_params)
                    )
                    tasks.append({
                        'chunk_index': chunk_index,
                        'text': chunk.text,
                        'tts_params': tts_params,
                        'post_process': chunk.post_process,
                        'original_score': failure.get('score', 0.0),
                        'original_asr': failure,
                        'audio_chunks_dir': str(dataset_paths['audio_chunks_dir']),
                        'tts_dir': str(dataset_paths['tts_dir']),
                        'asr_exe': str(resolved_asr_exe),
                        'asr_model': regen_asr_model_name,
                        'asr_language': asr_config.get('language', 'en'),
                        'threshold': asr_config.get('threshold', 0.85),
                        'max_retries': asr_config.get('max_retries', 3),
                        'temp_decrement': asr_config.get('temp_decrement', 0.1),
                        'regeneration_strategy': regen_strategy,
                        'cleanup_config': cleanup_config,
                    })
                logger.info("Queued %s ASR-failed chunks for worker regeneration", len(tasks))
                _finish_asr_check()
                if tasks:
                    asr_regeneration_start = asr_check_end
                    _phase(
                        "regen",
                        f"Regeneration queued for {len(tasks)} failed chunk(s)",
                        total_chunks=len(chunks),
                    )
                else:
                    _phase("asr", "ASR check complete — no failures to regenerate", asr_status=True)
                return tasks

            def record_parallel_regeneration_result(result):
                """Collect investigation data returned by regeneration workers."""
                if result.get('error'):
                    logger.error(
                        "Worker regeneration failed for chunk %s: %s",
                        result.get('chunk_index'),
                        result['error'],
                    )
                if result.get('investigation'):
                    investigation_log.append(result['investigation'])

            if use_parallel and save_dataset_chunks and dataset_paths:
                # Parallel processing mode
                logger.info(f"Using parallel processing with {self.num_workers} workers")
                saved_chunk_paths = self._generate_chunks_parallel(
                    chunks, voice_path, Path(dataset_paths['audio_chunks_dir']),
                    progress_callback, save_dataset_chunks, dataset_paths,
                    original_chunk_callback=self._build_shadow_stage_one_original_chunk_callback(
                        shadow_stage_one_state,
                        dataset_paths,
                    ),
                    regeneration_callback=prepare_parallel_regeneration_tasks,
                    regeneration_result_callback=record_parallel_regeneration_result,
                    regeneration_finished_callback=_finish_asr_regeneration,
                )
                audio_chunks = None  # Not used in parallel mode

                if investigation_log:
                    self._save_investigation_log(investigation_log, dataset_paths)
            else:
                # Sequential processing mode (fallback)
                logger.info("Using sequential processing")
                processed_chunk_texts = set()  # Track for duplicates

                for i, chunk in enumerate(chunks):
                    logger.info(f"=== LOOP ITERATION {i+1} ===")
                    logger.info(f"Enumerate index: {i}")
                    logger.info(f"Current chunks list length: {len(chunks)}")
                    logger.info(f"Original chunks count: {original_chunks_count}")

                    if len(chunks) != original_chunks_count:
                        logger.error(f"CRITICAL: Chunks list modified during iteration! {len(chunks)} != {original_chunks_count}")
                        logger.error("Original chunks:")
                        for idx, orig_chunk in enumerate(chunks[:original_chunks_count]):
                            logger.error(f"  {idx}: '{orig_chunk.text[:50]}...'")
                        logger.error("Current chunks:")
                        for idx, curr_chunk in enumerate(chunks):
                            logger.error(f"  {idx}: '{curr_chunk.text[:50]}...'")

                    if self.is_cancelled:
                        logger.info("Generation cancelled by user")
                        break

                    chunk_idx = i
                    chunk_text_preview = chunk.text[:50].replace('\n', '\\n')
                    logger.info(f"Processing chunk {chunk_idx + 1}/{self.total_chunks}")
                    logger.info(f"Chunk text: '{chunk_text_preview}...'")
                    logger.info(f"Chunk details: {len(chunk.text)} chars, {chunk.word_count} words")
                    logger.info(f"Emotion: {chunk.emotion}, Confidence: {chunk.emotion_confidence}")
                    logger.info(f"TTS params: {chunk.tts_params}")

                    # Check for duplicate processing
                    if chunk.text in processed_chunk_texts:
                        logger.error(f"DUPLICATE CHUNK DETECTED: '{chunk_text_preview}' processed before")
                        logger.error(f"Previously processed chunks: {list(processed_chunk_texts)}")
                    else:
                        processed_chunk_texts.add(chunk.text)

                    # Generate audio for this chunk
                    logger.debug("Calling _generate_chunk_audio...")
                    chunk_audio = self._generate_chunk_audio(chunk, voice_state)

                    if chunk_audio is not None:
                        # Save individual chunk files if dataset mode enabled
                        if save_dataset_chunks and dataset_paths:
                            chunk_idx_str = f"{chunk_idx:05d}"

                            # Save WAV chunk
                            wav_path = dataset_paths['audio_chunks_dir'] / f"chunk_{chunk_idx_str}.wav"
                            self._save_audio(chunk_audio, str(wav_path))

                            # Save text chunk
                            txt_path = dataset_paths['text_chunks_dir'] / f"chunk_{chunk_idx_str}.txt"
                            with open(txt_path, 'w', encoding='utf-8') as f:
                                f.write(chunk.text.strip())

                            # Track saved files for ASR dataset (not concatenation)
                            saved_chunk_paths.append(wav_path)
                            shadow_stage_one_state["service"] = self._publish_shadow_stage_one_chunk(
                                shadow_stage_one_state["service"],
                                wav_path,
                                txt_path,
                                chunk_idx,
                            )

                            logger.debug(f"Saved chunk {chunk_idx_str} to dataset files")
                        # Always keep in memory for concatenation to avoid conversion artifacts
                        audio_chunks.append(chunk_audio)
                        logger.info(f"Audio generated successfully: {len(chunk_audio)} samples")

                        # Update progress
                        self.current_chunk = chunk_idx + 1
                        logger.info(f"Updated current_chunk to: {self.current_chunk}")

                        if progress_callback:
                            elapsed = time.time() - self.start_time
                            avg_time = elapsed / self.current_chunk
                            eta = avg_time * (self.total_chunks - self.current_chunk)

                            progress_data = {
                                'current_chunk': self.current_chunk,
                                'total_chunks': self.total_chunks,
                                'elapsed_seconds': int(elapsed),
                                'eta_seconds': int(eta),
                                'chunk_text': chunk.text[:50]
                            }
                            logger.debug(f"Sending progress callback: {progress_data}")
                            progress_callback(progress_data)
                    else:
                        logger.warning(f"Audio generation failed for chunk {i}, returned None")

                    logger.info(f"=== END LOOP ITERATION {i+1} ===")

            logger.info("=== CHUNK PROCESSING LOOP COMPLETED ===")
            logger.info(f"Total loop iterations: {len(chunks)}")
            if audio_chunks is not None:
                logger.info(f"Audio chunks generated: {len(audio_chunks)}")
            else:
                logger.info("Audio chunks generated via parallel processing")
            logger.info(f"Final current_chunk: {self.current_chunk}")

            # Stop chunk process timer
            if self._original_generation_end is None:
                self._original_generation_end = time.time()

            # --- ASR Quality Control (post-gen GPU only; sequential / leftover path) ---
            # Parallel path already ran ASR inside prepare_parallel_regeneration_tasks
            # (after killing TTS workers). Sequential path still holds parent TTS model.
            shadow_summary = self._finish_shadow_stage_one_service(
                shadow_stage_one_state["service"],
                len(chunks),
            )
            shadow_stage_one_state["service"] = shadow_summary
            self._promote_shadow_stage_one_to_new_pipeline(
                shadow_summary,
                dataset_paths,
                asr_launch_state,
            )
            if asr_enabled and not asr_post_gen_done and asr_launch_state.get("failure_log"):
                asr_check_start = self._original_generation_end or time.time()
                logger.info("Starting post-gen GPU ASR after chunk loop")
                try:
                    self.release_gpu_memory()
                except Exception as exc:
                    logger.warning("Parent GPU release before post-gen ASR: %s", exc)
                _launch_post_gen_gpu_asr()

            if asr_process is not None:
                logger.info("Waiting for post-gen GPU ASR to validate all chunks...")
                logger.warning(
                    "ASR model download/load can take minutes the first time — "
                    "progress is written to %s (not stuck).",
                    str(asr_launch_state.get("run_log") or asr_run_log),
                )
                try:
                    stage_one_wait_start = time.time()
                    _wait_asr_process_with_progress(
                        asr_process,
                        str(asr_launch_state.get("run_log") or asr_run_log),
                        timeout=max(300, len(chunks) * 30),
                        progress_callback=progress_callback,
                        total_chunks=len(chunks),
                    )
                    if str(asr_launch_state.get("pipeline", asr_pipeline) or asr_pipeline) == 'legacy':
                        asr_stage_one_time = time.time() - stage_one_wait_start
                    logger.info(f"ASR exit code: {asr_process.returncode}")
                    logger.info("ASR monitor terminated")
                    if str(asr_launch_state.get("pipeline", asr_pipeline) or asr_pipeline) == 'new' and asr_process.returncode != 0:
                        asr_pipeline_error = (
                            f"runner exited with code {asr_process.returncode}"
                        )
                except subprocess.TimeoutExpired:
                    logger.warning("ASR monitor exceeded timeout; terminating it")
                    asr_process.terminate()
                    asr_process.wait(timeout=10)
                    if str(asr_launch_state.get("pipeline", asr_pipeline) or asr_pipeline) == 'new':
                        asr_pipeline_error = "runner exceeded the ASR timeout"
                except Exception as e:
                    logger.warning(f"Error terminating ASR monitor: {e}")
                    asr_process.kill()
                    if str(asr_launch_state.get("pipeline", asr_pipeline) or asr_pipeline) == 'new':
                        asr_pipeline_error = f"runner wait failed: {e}"
                finally:
                    if asr_log_handle is not None:
                        asr_log_handle.close()
                        asr_log_handle = None
                    asr_post_gen_done = True

                _finish_asr_check()

                # Check and process failure log
                if str(asr_launch_state.get("pipeline", asr_pipeline) or asr_pipeline) == 'new':
                    if not _complete_new_pipeline_report():
                        investigation_log = []
                    else:
                        try:
                            asr_failure_log, _ = _materialize_new_pipeline_regeneration_log(
                                Path(dataset_paths['tts_dir'])
                            )
                        except (OSError, ValueError, json.JSONDecodeError) as exc:
                            _report_new_pipeline_failure(
                                f"confirmed-failure report unavailable: {exc}"
                            )
                            investigation_log = []
                        else:
                            if int(asr_config.get('max_retries', 3) or 0) <= 0:
                                logger.info(
                                    "ASR report-only mode enabled (max_retries=0); "
                                    "skipping regeneration"
                                )
                                investigation_log = []
                            else:
                                asr_regeneration_start = time.time()
                                investigation_log = self._reprocess_failed_chunks(
                                    asr_failure_log,
                                    voice_state,
                                    dataset_paths,
                                    asr_config,
                                )
                                _finish_asr_regeneration()
                elif asr_failure_log and Path(asr_failure_log).exists():
                    logger.info(f"ASR failure log found: {asr_failure_log}")
                    # Log the contents
                    try:
                        with open(asr_failure_log, 'r') as f:
                            failures = json.load(f)
                            _write_stage_one_failure_manifest(
                                Path(dataset_paths["tts_dir"]),
                                pipeline="legacy",
                                failure_filename="asr_failures.json",
                                threshold=threshold,
                                language=asr_config.get("language", "en"),
                            )
                            logger.info(f"ASR found {len(failures)} failed chunks")
                            for failure in failures:
                                logger.info(f"  Failed: chunk_{failure.get('chunk_index', 'unknown'):05d} - score: {failure.get('score', 0):.2f}")
                    except Exception as log_err:
                        logger.warning(f"Could not read ASR failure log: {log_err}")

                    logger.info("Processing ASR failure log...")
                    if int(asr_config.get('max_retries', 3) or 0) <= 0:
                        # Zero means ASR-only reporting; do not touch failed audio or regenerate.
                        logger.info(
                            "ASR report-only mode enabled (max_retries=0); "
                            "skipping regeneration"
                        )
                        investigation_log = []
                    else:
                        asr_regeneration_start = time.time()
                        investigation_log = self._reprocess_failed_chunks(
                            asr_failure_log,
                            voice_state,
                            dataset_paths,
                            asr_config
                        )
                        _finish_asr_regeneration()
                else:
                    logger.info("No ASR failure log found - no chunks failed quality check")
                    investigation_log = []

            # Decide deliverables first so we can skip the full WAV when unused.
            m4b_cfg = dict(getattr(self.config, "m4b", {}) or {})
            from pocket_tts.audio.metadata import (
                export_metadata_tags,
                metadata_cover_path,
            )

            export_metadata = export_metadata_tags(
                m4b_cfg.get("metadata"),
                title_fallback=Path(source_file).stem,
            )
            cover_path = metadata_cover_path(m4b_cfg.get("metadata"))
            write_m4b = bool(m4b_cfg.get("write_m4b", False))
            write_mp3 = bool(m4b_cfg.get("write_mp3", False))
            write_wav = bool(m4b_cfg.get("write_wav", True))
            # Legacy: master enabled + output_mode when write_* not present.
            if (
                "write_m4b" not in m4b_cfg
                and "write_mp3" not in m4b_cfg
                and "write_wav" not in m4b_cfg
            ):
                legacy_on = bool(m4b_cfg.get("enabled", False))
                mode = str(m4b_cfg.get("output_mode", "m4b") or "m4b").lower()
                write_wav = True
                write_m4b = legacy_on and mode in {"m4b", "both"}
                write_mp3 = legacy_on and mode in {"mp3", "both"}
            chapterize = bool(m4b_cfg.get("chapterize", False))
            max_chapter_minutes = float(m4b_cfg.get("max_chapter_minutes", 0) or 0)
            chapter_mode = str(m4b_cfg.get("chapter_mode") or "")

            # Combine chunk audio only when a full WAV is requested.
            audio_duration = 0.0
            if write_wav:
                if save_dataset_chunks and dataset_paths and saved_chunk_paths:
                    logger.info("Concatenating saved chunk files → full WAV…")
                    self._concatenate_from_files(saved_chunk_paths, output_path)
                    logger.info(f"Final audio saved to: {output_path}")
                    audio_duration = self._probe_audio_duration(output_path)
                    logger.info(f"Calculated audio duration: {audio_duration:.2f}s")
                elif audio_chunks and not self.is_cancelled:
                    logger.info("Combining audio chunks in memory…")
                    final_audio = self._combine_audio_chunks(audio_chunks)
                    logger.info(f"Combined audio: {len(final_audio)} total samples")
                    logger.info("Saving audio file…")
                    self._save_audio(final_audio, output_path)
                    logger.info("Audio file saved successfully")
                    audio_duration = len(final_audio) / 24000
                else:
                    raise ValueError("No audio chunks were generated to save.")
            else:
                logger.info(
                    "Skipping full WAV (write_wav=false); duration from chunk files if needed"
                )
                if save_dataset_chunks and dataset_paths and saved_chunk_paths:
                    try:
                        from ..audio.chapter_export import _wav_duration_seconds
                    except ImportError:
                        from audio.chapter_export import _wav_duration_seconds
                    for path in saved_chunk_paths:
                        try:
                            audio_duration += _wav_duration_seconds(Path(path))
                        except Exception:
                            pass
                elif audio_chunks:
                    audio_duration = sum(len(a) for a in audio_chunks) / 24000
                else:
                    raise ValueError("No audio chunks were generated to save.")

            # Record chunk-only speed before ASR and final-output work are counted.
            chunk_process_time = (
                self._original_generation_end - self._chunk_process_start
                if self._original_generation_end and self._chunk_process_start
                else time.time() - self.start_time
            )
            self.chunk_processing_time = chunk_process_time
            self.realtime_factor = audio_duration / chunk_process_time if chunk_process_time > 0 else 0
            self.chunks_processed = self.current_chunk + 1
            self.audio_duration = audio_duration

            # --- Efficient export: only checked formats ---
            final_output_path = output_path if write_wav else None
            if write_m4b or write_mp3 or write_wav:
                logger.info(
                    "Export flags: m4b=%s mp3=%s wav=%s chapterize=%s mode=%s max_min=%s",
                    write_m4b,
                    write_mp3,
                    write_wav,
                    chapterize,
                    chapter_mode or "legacy",
                    max_chapter_minutes,
                )
            if write_m4b or write_mp3:
                try:
                    book_dir = Path(output_path).parent
                    try:
                        from ..audio.chapter_export import export_book_chapters
                    except ImportError:
                        from audio.chapter_export import export_book_chapters

                    export_result = export_book_chapters(
                        book_dir,
                        wav_path=Path(output_path) if write_wav else None,
                        chunks_json=(
                            Path(dataset_paths["text_chunks_dir"]) / "audiobook.chunks.json"
                            if dataset_paths
                            else None
                        ),
                        audio_chunks_dir=(
                            Path(dataset_paths["audio_chunks_dir"])
                            if dataset_paths
                            else None
                        ),
                        chapterize=chapterize,
                        max_chapter_minutes=max_chapter_minutes,
                        chapter_mode=chapter_mode,
                        write_m4b=write_m4b,
                        write_mp3=write_mp3,
                        write_wav=False,  # WAV already handled above when requested
                        m4b_config=m4b_cfg,
                        title=export_metadata.get("title", Path(source_file).stem),
                        artist=export_metadata.get("artist", ""),
                        metadata=export_metadata,
                        cover_path=cover_path,
                    )
                    if export_result.get("m4b_path"):
                        final_output_path = export_result["m4b_path"]
                    elif export_result.get("mp3_files"):
                        final_output_path = export_result["mp3_files"][0]
                    elif write_wav:
                        final_output_path = output_path
                    logger.info(
                        "Export complete: chapters=%s m4b=%s mp3=%s",
                        export_result.get("chapter_count"),
                        export_result.get("m4b_path"),
                        export_result.get("mp3_dir") or export_result.get("mp3_files"),
                    )
                    if not audio_duration and export_result.get("total_duration_s"):
                        audio_duration = float(export_result["total_duration_s"])
                        self.audio_duration = audio_duration
                except Exception as e:
                    logger.error(f"Error during M4B/MP3 export: {e}", exc_info=True)
                    if write_wav:
                        final_output_path = output_path

            if write_wav and export_metadata:
                try:
                    from pocket_tts.audio.chapter_export import apply_media_metadata

                    apply_media_metadata(
                        Path(output_path),
                        export_metadata,
                        ffmpeg_path=str(m4b_cfg.get("ffmpeg_path") or "ffmpeg"),
                    )
                except Exception as exc:
                    # Metadata failure must not discard a completed WAV deliverable.
                    logger.warning("Could not add WAV metadata: %s", exc)

            if final_output_path is None:
                # No deliverable file path — keep chunk folder as success anchor.
                final_output_path = (
                    Path(dataset_paths["audio_chunks_dir"])
                    if dataset_paths
                    else Path(output_path).parent
                )

            total_time = time.time() - self.start_time
            self.processing_time = total_time
            self.total_realtime_factor = audio_duration / total_time if total_time > 0 else 0

            stage_one_model = "parakeet" if asr_pipeline == "new" else str(asr_model)
            stage_two_model = str(
                regen_verification_summary.get("verification_model")
                or (second_stage_model if asr_pipeline == "new" else regen_asr_model_name)
            )
            # Remaining fails after regen (investigation log excludes successful regens).
            asr_remaining_fails = len(investigation_log)
            asr_stage_one_fails, asr_stage_two_fails = _resolve_reported_stage_fail_counts(
                asr_stage_one_fails,
                asr_stage_two_fails,
                asr_pipeline,
                regen_verification_summary,
            )
            regeneration_file_count = int(asr_stage_two_fails or 0)
            regeneration_worker_count = int(regen_tts_workers if regeneration_file_count else 0)
            regeneration_asr_worker_count = int(
                regen_plan.asr_workers if regen_plan and regeneration_file_count else 0
            )
            regeneration_batch_size = int(regen_plan.tts_batch_size if regen_plan else 0)
            regeneration_batch_count = (
                (regeneration_file_count + regeneration_batch_size - 1) // regeneration_batch_size
                if regeneration_file_count and regeneration_batch_size
                else 0
            )

            logger.info(
                "Timing summary: ASR Check=%s ASR Regen=%s End-to-end=%s",
                _format_elapsed_hms(asr_check_time),
                _format_elapsed_hms(asr_regeneration_time),
                _format_elapsed_hms(total_time),
            )
            logger.info("ASR Check: %s", _format_elapsed_hms(asr_check_time))
            logger.info("ASR Regen: %s", _format_elapsed_hms(asr_regeneration_time))
            logger.info(
                "ASR summary: stage1_model=%s stage2_model=%s regen_files=%s "
                "regen_tts_workers=%s regen_asr_workers=%s regen_batches=%s regen_batch_size=%s",
                stage_one_model,
                stage_two_model,
                regeneration_file_count,
                regeneration_worker_count,
                regeneration_asr_worker_count,
                regeneration_batch_count,
                regeneration_batch_size,
            )
            logger.info("End-to-end processing time: %s", _format_elapsed_hms(total_time))

            # Save final metadata after WAV/M4B output is complete.
            logger.info("Saving final chunk data to JSON...")
            self._save_chunks_json(
                chunks,
                final_output_path,
                voice_path,
                source_file,
                dataset_paths or {},
            )
            logger.info("JSON data saved successfully")

            result = {
                'success': True,
                'output_path': str(final_output_path),
                'audio_duration': audio_duration,
                'processing_time': total_time,
                'realtime_factor': self.realtime_factor,
                'chunk_processing_time': self.chunk_processing_time,
                'total_realtime_factor': self.total_realtime_factor,
                'asr_check_time': asr_check_time,
                'asr_regeneration_time': asr_regeneration_time,
                'asr_stage_one_time': asr_stage_one_time,
                'asr_stage_two_time': asr_stage_two_time,
                'asr_stage_one_model': stage_one_model,
                'asr_stage_two_model': stage_two_model,
                'asr_regeneration_model': regen_asr_model_name,
                'asr_stage_one_fails': int(asr_stage_one_fails or 0),
                'asr_stage_two_fails': int(asr_stage_two_fails or 0),
                'asr_remaining_fails': int(asr_remaining_fails or 0),
                'regeneration_file_count': regeneration_file_count,
                'regeneration_tts_workers': regeneration_worker_count,
                'regeneration_asr_workers': regeneration_asr_worker_count,
                'regeneration_batch_size': regeneration_batch_size,
                'regeneration_batch_count': regeneration_batch_count,
                'chunks_processed': self.current_chunk,
                'total_chunks': self.total_chunks,
                'asr_investigation_required': investigation_log,
                'regeneration_plan': regen_plan.to_dict() if regen_plan else None,
                'regeneration_plan_summary': regen_plan_summary,
                'medium_verification_summary': regen_verification_summary,
            }

            logger.info("=== GENERATION COMPLETED SUCCESSFULLY ===")
            logger.info(
                f"Generated {audio_duration:.2f}s audio in {total_time:.2f}s "
                f"(chunks: {result['realtime_factor']:.2f}x, "
                f"end-to-end: {result['total_realtime_factor']:.2f}x realtime)"
            )
            _phase(
                "done",
                f"Complete: {audio_duration:.1f}s audio in {total_time:.1f}s "
                f"({result['total_realtime_factor']:.1f}x e2e)",
                current_chunk=self.current_chunk,
                total_chunks=self.total_chunks,
            )
            return result

        except Exception as e:
            logger.error(f"Audiobook generation failed: {e}")
            logger.error("Exception details:", exc_info=True)
            try:
                _phase("done", f"Failed: {e}")
            except Exception:
                pass
            return {
                'success': False,
                'reason': str(e),
                'chunks_completed': self.current_chunk
            }

        except Exception as e:
            logger.error(f"Audiobook generation failed: {e}")
            logger.error("Exception details:", exc_info=True)
            return {
                'success': False,
                'reason': str(e),
                'chunks_completed': self.current_chunk
            }
        finally:
            if shadow_stage_one_state["service"] is not None and not getattr(
                shadow_stage_one_state["service"],
                "_shadow_finish_called",
                False,
            ):
                self._abort_shadow_stage_one_service(
                    shadow_stage_one_state["service"],
                    reason="generation shutdown",
                )
            self.release_gpu_memory()
            # Clean up debug logging from all loggers we attached
            for lg in (
                logger,
                logging.getLogger('audiobook_debug'),
                logging.getLogger('pocket_tts.preprocessing.structure_detector'),
                logging.getLogger('pocket_tts.preprocessing.chunker'),
            ):
                if debug_handler in lg.handlers:
                    lg.removeHandler(debug_handler)
            try:
                debug_handler.close()
            except Exception:
                pass

    def release_gpu_memory(self) -> None:
        """Release parent-process TTS CUDA allocations after generation completes."""
        model = self.tts_model
        self.tts_model = None
        if model is None:
            return

        try:
            free_before = None
            if torch.cuda.is_available():
                free_before, _ = torch.cuda.mem_get_info()
            del model
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                free_after, total_after = torch.cuda.mem_get_info()
                logger.info(
                    "Released parent TTS model and emptied CUDA cache: free VRAM %.2f -> %.2f GiB "
                    "of %.2f GiB",
                    free_before / _BYTES_PER_GIB if free_before is not None else 0.0,
                    free_after / _BYTES_PER_GIB,
                    total_after / _BYTES_PER_GIB,
                )
        except Exception as exc:
            logger.warning("Failed to fully release parent GPU memory: %s", exc)

    def cancel_generation(self):
        """Request cancel of the current generation process.

        Note: parallel workers only stop after the parent observes is_cancelled
        between result receives (or on shutdown). The log line means something
        called this method (Stop button or window close), not that the process
        has fully halted yet.
        """
        self.is_cancelled = True
        logger.info(
            "Audiobook generation cancel requested (is_cancelled=True); "
            "caller should be Stop button or window close"
        )

    def _generate_chunks_parallel(self, chunks: List[ChunkMetadata], voice_path: str,
                                   output_dir: Path, progress_callback=None,
                                   save_dataset_chunks: bool = False, dataset_paths: Dict = None,
                                   original_chunk_callback=None,
                                   regeneration_callback=None,
                                   regeneration_result_callback=None,
                                   regeneration_finished_callback=None) -> List[Path]:
        """
        Generate audio chunks in parallel using dynamic queue-based dispatching.
        Workers pull chunks from a shared queue as they become available, providing
        better load balancing than pre-divided batches.

        Args:
            chunks: List of chunk metadata
            voice_path: Path to voice file
            output_dir: Directory to save audio chunks
            progress_callback: Progress callback function
            original_chunk_callback: Optional parent-side callback invoked after
                one original chunk WAV and text sidecar are finalized.
            regeneration_callback: Callback that returns regeneration tasks
                after original chunks and ASR validation complete.
            regeneration_result_callback: Callback for each regeneration result.
            regeneration_finished_callback: Callback after the final regeneration
                result is received, before regeneration workers stop.

        Returns:
            List of paths to saved audio files
        """
        ctx = mp.get_context("spawn")

        target_workers = self._adjust_workers_for_resources()

        output_dir_str = str(output_dir)
        total_chunks = len(chunks)
        saved_chunk_paths = []
        completed_chunks = 0
        cleanup_config = getattr(self.config, 'audio_cleanup', {}) or {}
        if not isinstance(cleanup_config, dict):
            cleanup_config = {}
        batch_config = getattr(self.config, 'batch_generation', {}) or {}
        batch_size = max(2, int(batch_config.get('batch_size', 2))) if batch_config.get('enabled', False) else 1
        if cleanup_config.get('enabled', True) and cleanup_config.get('async', True):
            logger.info(
                "Async audio cleanup enabled (max_pending=%s) — gen will not wait on Silero/WAV",
                int(cleanup_config.get('max_pending', 12)),
            )
        timing_dir = output_dir.parent / "worker_timing"
        timing_dir.mkdir(parents=True, exist_ok=True)
        for timing_file in timing_dir.glob("worker_*.jsonl"):
            timing_file.unlink()

        chunk_queue = ctx.Queue()

        queue_tasks = _build_original_queue_tasks(
            chunks,
            self.tts_model.flow_lm.conditioner.tokenizer,
            batch_config,
        )
        result_queue = ctx.Queue()
        ready_queue = ctx.Queue()

        workers = []
        def abort_workers(reason: str) -> None:
            """Terminate remaining workers before propagating a pool failure."""
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in workers:
                worker.join(timeout=10)
            raise RuntimeError(reason)

        # Propagate CUDA graph opt-in to spawn workers via environment.
        cuda_graphs_cfg = getattr(self.config, "cuda_graphs", None) or {}
        if isinstance(cuda_graphs_cfg, dict) and cuda_graphs_cfg.get("enabled", False):
            os.environ["POCKET_TTS_CUDA_GRAPHS"] = "1"
            logger.info("CUDA graphs enabled for workers (POCKET_TTS_CUDA_GRAPHS=1)")

        observed_worker_cost_bytes = None
        observed_settled_cost_bytes = None
        for worker_id in range(target_workers):
            free_before = None
            if self._is_cuda:
                try:
                    free_before, total_bytes = torch.cuda.mem_get_info()
                    settled_reserve_bytes = total_bytes * self.vram_reserve_percent / 100
                    startup_reserve_bytes = total_bytes * self.vram_startup_reserve_percent / 100
                    estimated_cost = (
                        observed_worker_cost_bytes
                        if observed_worker_cost_bytes is not None
                        else self.worker_vram_gb * _BYTES_PER_GIB
                    )
                    estimated_settled_cost = (
                        observed_settled_cost_bytes
                        if observed_settled_cost_bytes is not None
                        else self.worker_vram_gb * _BYTES_PER_GIB
                    )
                    if worker_id > 0 and free_before < settled_reserve_bytes:
                        logger.warning(
                            "Stopping worker startup at %s workers: settled free VRAM %.2f GiB "
                            "is below %.1f%% reserve",
                            len(workers),
                            free_before / _BYTES_PER_GIB,
                            self.vram_reserve_percent,
                        )
                        break
                    if worker_id > 0 and free_before - estimated_settled_cost < settled_reserve_bytes:
                        logger.warning(
                            "Stopping worker startup at %s workers: next settled estimate %.2f GiB "
                            "would breach %.1f%% settled reserve",
                            len(workers),
                            estimated_settled_cost / _BYTES_PER_GIB,
                            self.vram_reserve_percent,
                        )
                        break
                    if worker_id > 0 and free_before - estimated_cost < startup_reserve_bytes:
                        logger.warning(
                            "Stopping worker startup at %s workers: next estimate %.2f GiB "
                            "would breach %.1f%% startup reserve",
                            len(workers),
                            estimated_cost / _BYTES_PER_GIB,
                            self.vram_startup_reserve_percent,
                        )
                        break
                except Exception as exc:
                    logger.warning("Could not measure VRAM before worker %s startup: %s", worker_id, exc)

            worker = ctx.Process(
                target=_queue_worker,
                args=(
                    chunk_queue,
                    voice_path,
                    output_dir_str,
                    result_queue,
                    worker_id,
                    self._pause_injection_enabled,
                    self._pause_durations,
                    self._device,
                    cleanup_config.get('enabled', True),
                    cleanup_config.get('use_silero_vad', True),
                    cleanup_config.get('speech_endpoint_threshold', 0.004),
                    cleanup_config.get('trimming_buffer_ms', 100),
                    str(timing_dir),
                    batch_size,
                    ready_queue,
                    bool(cleanup_config.get('async', True)),
                    int(cleanup_config.get('max_pending', 12)),
                )
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)

            try:
                ready = ready_queue.get(timeout=max(300, len(chunks) * 30))
            except Empty:
                abort_workers(f"Worker {worker_id} did not become ready before timeout")
            if ready.get("status") != "ready":
                abort_workers(
                    f"Worker {worker_id} failed during startup: {ready.get('error', 'unknown error')}"
                )

            if self._is_cuda:
                try:
                    free_after, _ = torch.cuda.mem_get_info()
                    peak_free_mb = ready.get("peak_device_free_mb")
                    if free_before is not None and peak_free_mb is not None:
                        peak_free_bytes = int(peak_free_mb * 1024 ** 2)
                        startup_cost = max(0, free_before - peak_free_bytes)
                    else:
                        startup_cost = 0
                    if free_before is not None:
                        startup_cost = max(startup_cost, free_before - free_after)
                        settled_free_mb = ready.get("settled_device_free_mb")
                        if settled_free_mb is not None:
                            settled_free_bytes = int(settled_free_mb * 1024 ** 2)
                            settled_cost = max(0, free_before - settled_free_bytes)
                            if settled_cost:
                                observed_settled_cost_bytes = max(
                                    observed_settled_cost_bytes or 0,
                                    settled_cost,
                                )
                        if startup_cost:
                            observed_worker_cost_bytes = max(
                                observed_worker_cost_bytes or 0,
                                startup_cost,
                            )
                            logger.info(
                                "Worker %s ready: startup peak cost %.2f GiB, settled free %.2f GiB "
                                "(settled reserve %.1f%%, startup reserve %.1f%%)",
                                worker_id,
                                startup_cost / _BYTES_PER_GIB,
                                (ready.get("settled_device_free_mb") or free_after / (1024 ** 2)) / 1024,
                                self.vram_reserve_percent,
                                self.vram_startup_reserve_percent,
                            )
                except Exception as exc:
                    logger.warning("Could not measure VRAM after worker %s startup: %s", worker_id, exc)

        effective_workers = len(workers)
        if effective_workers == 0:
            abort_workers("No workers became available")
        if effective_workers != self.num_workers:
            logger.info(
                "Starting parallel generation for %s chunks with %s workers (adjusted from %s)",
                len(chunks), effective_workers, self.num_workers,
            )
        else:
            logger.info("Starting parallel generation for %s chunks with %s workers", len(chunks), effective_workers)

        for task in queue_tasks:
            chunk_queue.put(task)

        self.current_chunk = 0
        received_indices: set = set()
        failed_chunks = []
        handled_dead_slots: set = set()

        while len(received_indices) < total_chunks:
            received_chunks = len(received_indices)
            if self.is_cancelled:
                logger.info(
                    "Parallel generation stopping early after cancel "
                    "(received=%s/%s)",
                    received_chunks,
                    total_chunks,
                )
                for _ in range(effective_workers):
                    chunk_queue.put(("shutdown",))
                for worker in workers:
                    worker.join(timeout=10)
                    if worker.is_alive():
                        worker.terminate()
                raise RuntimeError(
                    f"Generation cancelled after {received_chunks}/{total_chunks} chunks"
                )
            try:
                result = result_queue.get(timeout=1)
                if result and result.get("kind") == "original":
                    saved_path = result["saved_path"]
                    try:
                        chunk_idx = int(result["chunk_idx"])
                    except (TypeError, ValueError):
                        chunk_idx = -1
                    chunk_text = result["chunk_text"]
                    # Fatal worker payloads use chunk_idx=-1 and are not a real chunk.
                    if chunk_idx < 0:
                        logger.error(
                            "Parallel worker reported a fatal error without a chunk index: %s",
                            result.get("error", "unknown error"),
                        )
                        continue
                    if chunk_idx in received_indices:
                        continue
                    received_indices.add(chunk_idx)
                    received_chunks = len(received_indices)
                    if saved_path is not None:
                        saved_chunk_paths.append(saved_path)
                        completed_chunks += 1
                        self.current_chunk = completed_chunks
                        if original_chunk_callback:
                            try:
                                original_chunk_callback(chunk_idx, saved_path)
                            except Exception as exc:
                                logger.warning(
                                    "Original chunk callback failed for %s: %s",
                                    chunk_idx,
                                    exc,
                                )

                        if progress_callback:
                            elapsed = time.time() - self.start_time
                            avg_time = elapsed / completed_chunks if completed_chunks > 0 else 0
                            eta = avg_time * (total_chunks - completed_chunks)

                            progress_data = {
                                'current_chunk': completed_chunks,
                                'total_chunks': total_chunks,
                                'elapsed_seconds': int(elapsed),
                                'eta_seconds': int(eta),
                                'chunk_text': chunk_text[:50] if chunk_text else ''
                            }
                            progress_callback(progress_data)
                    else:
                        failed_chunks.append(chunk_idx)
                        logger.error(
                            "Parallel worker failed for chunk %s: %s",
                            chunk_idx,
                            result.get("error", "no audio path returned"),
                        )
            except Empty:
                dead_ids = [
                    index for index, worker in enumerate(workers) if not worker.is_alive()
                ]
                new_dead = [index for index in dead_ids if index not in handled_dead_slots]
                if new_dead:
                    for index in new_dead:
                        worker = workers[index]
                        logger.error(
                            "Parallel worker %s exited early: pid=%s exitcode=%s received=%s/%s",
                            index,
                            getattr(worker, "pid", None),
                            worker.exitcode,
                            len(received_indices),
                            total_chunks,
                        )
                    missing = _requeue_unreceived_original_chunks(
                        chunk_queue, chunks, received_indices
                    )
                    logger.warning(
                        "Re-queued %s unfinished chunk(s) after worker death; "
                        "live workers continue. missing=%s",
                        len(missing),
                        missing[:20],
                    )
                    handled_dead_slots.update(new_dead)
                live_workers = [worker for worker in workers if worker.is_alive()]
                if not live_workers:
                    abort_workers(
                        "All parallel worker processes exited before all chunks completed: "
                        f"workers={dead_ids}, "
                        f"exitcodes={[workers[index].exitcode for index in dead_ids]}, "
                        f"received={len(received_indices)}/{total_chunks}"
                    )
                continue
            except Exception as e:
                logger.warning(f"Error getting result from worker: {e}")

            if received_chunks and received_chunks % 10 == 0:
                logger.info(f"Completed {received_chunks}/{total_chunks} chunks")

        if failed_chunks:
            for _ in range(effective_workers):
                chunk_queue.put(("shutdown",))
            for worker in workers:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.terminate()
            raise RuntimeError(
                f"Parallel generation failed for chunks: {sorted(failed_chunks)}"
            )

        # Mark original generation before ASR validation and regeneration begin.
        self._original_generation_end = time.time()

        def _stop_pool(pool: list, q: Any, label: str) -> None:
            """Shut down spawn workers and free their GPU memory."""
            n = len(pool)
            for _ in range(n):
                try:
                    q.put(("shutdown",))
                except Exception:
                    pass
            for worker in pool:
                worker.join(timeout=15)
                if worker.is_alive():
                    logger.warning("%s worker still alive — terminate", label)
                    worker.terminate()
                    worker.join(timeout=5)
            pool.clear()
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    free_b, total_b = torch.cuda.mem_get_info()
                    logger.info(
                        "%s stopped — free VRAM %.2f / %.2f GiB",
                        label,
                        free_b / _BYTES_PER_GIB,
                        total_b / _BYTES_PER_GIB,
                    )
            except Exception as exc:
                logger.warning("%s VRAM cleanup note: %s", label, exc)

        def _start_pool(count: int, id_offset: int = 0) -> list:
            """Spawn TTS worker processes (each loads own model)."""
            started = []
            for i in range(count):
                wid = id_offset + i
                worker = ctx.Process(
                    target=_queue_worker,
                    args=(
                        chunk_queue,
                        voice_path,
                        output_dir_str,
                        result_queue,
                        wid,
                        self._pause_injection_enabled,
                        self._pause_durations,
                        self._device,
                        cleanup_config.get('enabled', True),
                        cleanup_config.get('use_silero_vad', True),
                        cleanup_config.get('speech_endpoint_threshold', 0.004),
                        cleanup_config.get('trimming_buffer_ms', 100),
                        str(timing_dir),
                        batch_size,
                        ready_queue,
                        bool(cleanup_config.get('async', True)),
                        int(cleanup_config.get('max_pending', 12)),
                    ),
                )
                worker.daemon = True
                worker.start()
                started.append(worker)
                try:
                    ready = ready_queue.get(timeout=max(300, len(chunks) * 30))
                except Empty:
                    _stop_pool(started, chunk_queue, "regen-spawn-timeout")
                    raise RuntimeError(f"Regen worker {wid} did not become ready")
                if ready.get("status") != "ready":
                    _stop_pool(started, chunk_queue, "regen-spawn-fail")
                    raise RuntimeError(
                        f"Regen worker {wid} failed: {ready.get('error', 'unknown')}"
                    )
                logger.info("Regen-phase worker %s ready", wid)
            return started

        keep_resident = bool(getattr(self, "_regen_keep_tts_resident", False))
        if keep_resident:
            logger.info(
                "Original gen done — keeping %s TTS workers resident during post-gen ASR",
                effective_workers,
            )
        else:
            # Free TTS worker VRAM before ASR when benchmark evidence says unload/reload wins.
            logger.info(
                "Original gen done — shutting down %s TTS workers before post-gen ASR",
                effective_workers,
            )
            _stop_pool(workers, chunk_queue, "post-gen-pre-ASR")
            workers = []
        # Parent process may also hold a TTS model (tokenizer / sequential path).
        try:
            self.release_gpu_memory()
        except Exception as exc:
            logger.warning("Parent GPU release before ASR: %s", exc)

        # Callback launches GPU ASR (free GPU) then returns regen task list.
        regeneration_tasks = regeneration_callback() if regeneration_callback else []

        if regeneration_tasks:
            regen_n = self._regeneration_worker_count(effective_workers)
            if keep_resident and workers:
                logger.info(
                    "Reusing resident TTS pool with %s worker(s) for %s ASR regeneration task(s)",
                    len(workers),
                    len(regeneration_tasks),
                )
            else:
                logger.info(
                    "Respawning %s TTS worker(s) for %s ASR regeneration task(s)",
                    regen_n,
                    len(regeneration_tasks),
                )
                # Fresh queues — prior shutdown sent poison pills on the old ones.
                chunk_queue = ctx.Queue()
                result_queue = ctx.Queue()
                ready_queue = ctx.Queue()
                workers = _start_pool(regen_n, id_offset=100)
            for task in regeneration_tasks:
                chunk_queue.put(("regeneration", task))

            regeneration_completed = 0
            while regeneration_completed < len(regeneration_tasks):
                result = result_queue.get(timeout=300)
                if not result:
                    continue
                if result.get("kind") == "regeneration_attempt":
                    event = result["event"]
                    logger.info(
                        "ASR regeneration chunk=%s attempt=%s/%s temp=%.2f score=%.3f passed=%s eos=%s",
                        event["chunk_num"], event["attempt"], event["max_retries"],
                        event["temperature"], event["score"], event["passed"], event["eos_detected"],
                    )
                    print(
                        f"[ASR] {event['chunk_num']} attempt {event['attempt']}/{event['max_retries']} "
                        f"temp={event['temperature']:.2f} score={event['score']:.3f} "
                        f"passed={event['passed']} eos={event['eos_detected']}",
                        flush=True,
                    )
                    logger.info("ASR regeneration original=%s", event["original_text"])
                    logger.info("ASR regeneration transcript=%s", event["transcribed_text"])
                    print(f"[ASR] original: {event['original_text']}", flush=True)
                    print(f"[ASR] transcript: {event['transcribed_text']}", flush=True)
                    if event.get("explanation"):
                        logger.info("ASR regeneration explanation=%s", event["explanation"])
                        print(f"[ASR] explanation: {event['explanation']}", flush=True)
                    if progress_callback:
                        progress_callback({"regeneration": event})
                    continue
                if result.get("kind") != "regeneration":
                    continue
                regeneration_completed += 1
                if regeneration_result_callback:
                    regeneration_result_callback(result["result"])

            if regeneration_finished_callback:
                regeneration_finished_callback()
            _stop_pool(workers, chunk_queue, "post-regen")
        else:
            logger.info("No ASR regeneration tasks — TTS workers already stopped")

        logger.info(f"Parallel generation complete. Generated {len(saved_chunk_paths)} chunks")

        # CUDA graph telemetry from worker JSONL → main app log (easy to spot)
        try:
            graph_summary = _summarize_cuda_graph_telemetry(timing_dir)
            logger.info(
                "CUDA graph telemetry: enabled_any=%s workers=%s captures_ok=%s captures_fail=%s "
                "reuses=%s prepares=%s steps_graph=%s steps_eager=%s steps_graph_pct=%.1f",
                graph_summary.get("cuda_graphs_enabled_any"),
                graph_summary.get("workers"),
                graph_summary.get("captures_ok"),
                graph_summary.get("captures_fail"),
                graph_summary.get("reuses"),
                graph_summary.get("prepares"),
                graph_summary.get("steps_graph"),
                graph_summary.get("steps_eager"),
                graph_summary.get("steps_graph_pct") or 0.0,
            )
            logger.info(
                "Mimi graph telemetry: captures_ok=%s captures_fail=%s "
                "steps_graph=%s steps_eager=%s steps_graph_pct=%.1f",
                graph_summary.get("mimi_captures_ok"),
                graph_summary.get("mimi_captures_fail"),
                graph_summary.get("mimi_steps_graph"),
                graph_summary.get("mimi_steps_eager"),
                graph_summary.get("mimi_steps_graph_pct") or 0.0,
            )
            for worker_name, wstats in (graph_summary.get("per_worker") or {}).items():
                logger.info(
                    "CUDA graph %s: enabled=%s flow_captures_ok=%s fail=%s reuses=%s "
                    "flow_steps_graph=%s eager=%s mimi_captures_ok=%s mimi_steps_graph=%s "
                    "mimi_eager=%s last_capture_ok=%s last_batch_size=%s last_error=%s",
                    worker_name,
                    wstats.get("enabled"),
                    wstats.get("captures_ok"),
                    wstats.get("captures_fail"),
                    wstats.get("reuses"),
                    wstats.get("steps_graph"),
                    wstats.get("steps_eager"),
                    wstats.get("mimi_captures_ok"),
                    wstats.get("mimi_steps_graph"),
                    wstats.get("mimi_steps_eager"),
                    wstats.get("last_capture_ok"),
                    wstats.get("last_batch_size"),
                    wstats.get("last_capture_error"),
                )
        except Exception as exc:
            logger.warning("Could not summarize CUDA graph telemetry: %s", exc)

        saved_chunk_paths.sort()
        return [Path(p) for p in saved_chunk_paths]

    def _init_tts_model(self):
        """Initialize the TTS model."""
        try:
            from ..models.tts_model import TTSModel
            # Determine device from config
            device_config = getattr(self.config, 'device', {}) if hasattr(self.config, 'device') else {}
            if isinstance(device_config, dict):
                preferred = device_config.get('preferred', 'auto')
            else:
                preferred = 'auto'
            self.tts_model = TTSModel.load_model(device=preferred)
            logger.info(f"TTS model loaded on {self.tts_model.device}")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise

    def _load_voice(self, voice_path: str):
        """Load voice conditioning."""
        try:
            voice_state = self.tts_model.get_state_for_audio_prompt(voice_path, truncate=True)
            logger.info(f"Voice loaded: {voice_path}")
            return voice_state
        except Exception as e:
            logger.error(f"Failed to load voice {voice_path}: {e}")
            raise

    def _generate_chunk_audio(self, chunk: ChunkMetadata, voice_state):
        """Generate audio for a single chunk."""
        from ..preprocessing.pause_injector import has_inline_pause_markers

        try:
            logger.debug(f"_generate_chunk_audio called for chunk: '{chunk.text[:30]}...'")

            # Get TTS parameters from chunk (handle both dict and object formats)
            tts_params = chunk.tts_params
            if hasattr(tts_params, 'temperature'):  # TTSParams object
                temperature = tts_params.temperature
                frames_after_eos = tts_params.frames_after_eos
                eos_threshold = tts_params.eos_threshold
                speed_factor = getattr(tts_params, 'speed_factor', 1.0)
            else:  # dict format from JSON/metadata
                temperature = tts_params.get('temperature', 0.7)
                frames_after_eos = tts_params.get('frames_after_eos', 2)
                eos_threshold = tts_params.get('eos_threshold', -4.0)
                speed_factor = tts_params.get('speed_factor', 1.0)

            logger.debug(f"TTS parameters: temp={temperature}, frames_after_eos={frames_after_eos}, eos_threshold={eos_threshold}, speed_factor={speed_factor}")

            # Update model parameters
            if hasattr(self.tts_model, 'temp'):
                logger.debug(f"Setting model temperature to {temperature}")
                self.tts_model.temp = float(temperature)
            if hasattr(self.tts_model, 'eos_threshold'):
                logger.debug(f"Setting model eos_threshold to {eos_threshold}")
                self.tts_model.eos_threshold = float(eos_threshold)
            lsd_steps = getattr(self.config, 'quality', {}).get('lsd_steps', 2)
            if hasattr(self, 'lsd_steps_spin'):
                lsd_steps = self.lsd_steps_spin.value()
            logger.debug(f"Using LSD steps: {lsd_steps}")
            self.tts_model.lsd_decode_steps = lsd_steps

            pause_plan_text = None
            # Store self-contained dialogue while preserving pause-tag spacing.
            from ..preprocessing.text_normalizer import (
                flatten_newlines_for_json,
                normalize_text_chunk_for_storage,
            )
            chunk_text = normalize_text_chunk_for_storage(chunk.text or "")
            chunk.text = chunk_text
            cleanup_config = getattr(self.config, 'audio_cleanup', {})
            # The GUI switch controls automatic punctuation events only.  Manual
            # [Xs] plans remain active in the branch below when it is disabled.
            if getattr(self, '_pause_injection_enabled', False):
                from ..preprocessing.pause_injector import inject_pauses_for_punctuation

                pause_plan_text = flatten_newlines_for_json(
                    inject_pauses_for_punctuation(chunk_text, self._pause_durations)
                )
            elif has_inline_pause_markers(chunk_text):
                pause_plan_text = chunk_text

            if pause_plan_text and has_inline_pause_markers(pause_plan_text):
                from ..preprocessing.pause_injector import (
                    generate_audio_with_pauses,
                    render_text_with_native_pauses,
                )

                def render_segment(text: str):
                    """Generate and postprocess one independent pause-plan segment."""
                    segment = self.tts_model.generate_audio(
                        voice_state, text, frames_after_eos=frames_after_eos
                    )
                    if speed_factor != 1.0:
                        segment = self._ffmpeg_atempo(segment, speed_factor)
                    if cleanup_config.get('enabled', True):
                        from ..audio_processing.endpoint_cleanup import cleanup_audio_endpoint

                        segment = cleanup_audio_endpoint(
                            segment,
                            getattr(self.tts_model, 'sample_rate', 24000),
                            threshold=cleanup_config.get('speech_endpoint_threshold', 0.004),
                            buffer_ms=cleanup_config.get('trimming_buffer_ms', 100),
                            use_silero=cleanup_config.get('use_silero_vad', True),
                        )
                    return segment

                audio, pause_events = generate_audio_with_pauses(
                    self.tts_model, voice_state, pause_plan_text,
                    generate_segment=render_segment,
                )
                chunk.text = pause_plan_text
                chunk.pause_events = pause_events
                chunk._spoken_text = render_text_with_native_pauses(pause_plan_text)
                chunk._pause_events = pause_events
            else:
                logger.debug("Calling self.tts_model.generate_audio...")
                audio = self.tts_model.generate_audio(
                    voice_state,
                    chunk_text,
                    frames_after_eos=frames_after_eos
                )

                # Non-pause chunks retain the existing whole-chunk processing.
                if speed_factor != 1.0:
                    logger.debug(f"Applying speed adjustment: {speed_factor:.4f}x")
                    audio = self._ffmpeg_atempo(audio, speed_factor)
                if cleanup_config.get('enabled', True):
                    from ..audio_processing.endpoint_cleanup import cleanup_audio_endpoint

                    audio = cleanup_audio_endpoint(
                        audio,
                        getattr(self.tts_model, 'sample_rate', 24000),
                        threshold=cleanup_config.get('speech_endpoint_threshold', 0.004),
                        buffer_ms=cleanup_config.get('trimming_buffer_ms', 100),
                        use_silero=cleanup_config.get('use_silero_vad', True),
                    )

            # --- POST-PROCESSING SILENCE INSERTION ---
            # Retrieve calculated digital silence duration
            silence_duration_sec = chunk.post_process.get('silence_duration', 0.0)

            if silence_duration_sec > 0:
                import torch
                # Calculate number of zero samples to append
                # sample_rate is typically 24000
                sample_rate = getattr(self.tts_model, 'sample_rate', 24000)
                num_silence_samples = int(silence_duration_sec * sample_rate)

                logger.debug(f"Appending {silence_duration_sec:.2f}s digital silence ({num_silence_samples} samples)")

                # Create zero tensor on same device as audio
                # audio shape is likely [samples] or [1, samples]
                # Check shape to match dimensions
                if len(audio.shape) == 1:
                    silence_tensor = torch.zeros(num_silence_samples, device=audio.device)
                    audio = torch.cat([audio, silence_tensor], dim=0)
                else:
                    # Assume [channels, samples] format if 2D
                    silence_tensor = torch.zeros(audio.shape[0], num_silence_samples, device=audio.device)
                    audio = torch.cat([audio, silence_tensor], dim=1)

            logger.debug(f"Audio generation successful: {len(audio) if len(audio.shape)==1 else audio.shape[1]} samples returned")
            return audio

        except Exception as e:
            logger.error(f"Failed to generate audio for chunk: {e}")
            logger.error(f"Failed chunk details: text='{chunk.text[:100]}...', params={chunk.tts_params}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _combine_audio_chunks(self, audio_chunks):
        """Combine multiple audio tensors into one."""
        import torch
        return torch.cat(audio_chunks, dim=0)

    def _save_audio(self, audio, output_path: str):
        """Save audio tensor to WAV file in PCM format."""
        try:
            import scipy.io.wavfile
            import numpy as np
            # Convert to numpy and ensure proper format for WAV
            sample_rate = getattr(self.tts_model, 'sample_rate', 24000)
            audio_np = audio.cpu().numpy().clip(-1.0, 1.0)
            audio_int16 = (np.array(audio_np) * 32767).astype(np.int16)
            scipy.io.wavfile.write(output_path, sample_rate, audio_int16)
            logger.info(f"Audio saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

    def _ffmpeg_atempo(self, audio, speed_factor: float):
        """Apply time-stretch to audio using FFmpeg's atempo filter (WSOLA).

        Args:
            audio: Torch tensor of audio samples
            speed_factor: Playback speed multiplier (1.0 = no change, <1.0 = slower, >1.0 = faster)

        Returns:
            Stretched audio tensor with same shape and device as input
        """
        import subprocess
        import tempfile
        import torch
        import numpy as np
        import scipy.io.wavfile
        from ..data.audio import audio_read

        sample_rate = getattr(self.tts_model, 'sample_rate', 24000)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_wav = os.path.join(tmpdir, "input.wav")
            output_wav = os.path.join(tmpdir, "output.wav")

            # Save audio to temp WAV (float32 format)
            audio_np = audio.cpu().numpy()
            scipy.io.wavfile.write(input_wav, sample_rate, audio_np.astype(np.float32))

            # Run ffmpeg with atempo filter. POCKET_TTS_FFMPEG_PATH is set by
            # the Windows launcher when a private (non-PATH) copy was
            # downloaded, since ffmpeg is never bundled or on PATH by default.
            cmd = [
                os.environ.get("POCKET_TTS_FFMPEG_PATH", "ffmpeg"),
                "-y", "-i", input_wav,
                "-filter:a", f"atempo={speed_factor}",
                "-ar", str(sample_rate),
                "-f", "wav", output_wav
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
            if result.returncode != 0:
                logger.warning(f"ffmpeg atempo failed: {result.stderr}")
                return audio  # Return original on failure

            # Read result back (audio_read returns torch.Tensor, not numpy)
            stretched, _ = audio_read(output_wav)
            stretched = stretched.squeeze().to(audio.device)

        return stretched



    def _get_ffmpeg_executable(self) -> str:
        """Resolve configured FFmpeg path while honoring launcher overrides."""
        m4b_config = getattr(self.config, "m4b", {}) or {}
        configured_path = m4b_config.get("ffmpeg_path") if isinstance(m4b_config, dict) else None
        return configured_path or os.environ.get("POCKET_TTS_FFMPEG_PATH", "ffmpeg")

    def _concatenate_from_files(self, chunk_paths: List[Path], output_path: Union[str, Path]):
        """Stitch saved chunk WAVs into one file without loading audio into RAM."""
        if not chunk_paths:
            raise ValueError("No chunk files found to concatenate.")

        ordered_paths = sorted(Path(path) for path in chunk_paths)
        missing_paths = [path for path in ordered_paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Missing audio chunks: {missing_paths[:5]}")

        try:
            from ..audio.chapter_export import concat_wavs
        except ImportError:
            from audio.chapter_export import concat_wavs

        logger.info("Concatenating %s chunks", len(ordered_paths))
        concat_wavs(
            [str(path) for path in ordered_paths],
            Path(output_path),
            ffmpeg_path=self._get_ffmpeg_executable(),
        )

    def _probe_audio_duration(self, audio_path: Union[str, Path]) -> float:
        """Read audio duration from metadata without decoding samples into memory."""
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode == 0:
                try:
                    return float(result.stdout.strip())
                except ValueError:
                    pass

        with wave.open(str(audio_path), "rb") as audio_file:
            return audio_file.getnframes() / audio_file.getframerate()

    def _save_chunks_json(self, chunks: List[ChunkMetadata], output_path: Union[str, Path], voice_path: str, source_file: str, dataset_paths: Dict[str, Union[str, Path]], is_preliminary: bool = False):
        """Save chunk data to JSON file in text_chunks directory."""
        try:
            import json
            from datetime import datetime

            json_path = Path(dataset_paths['text_chunks_dir']) / 'audiobook.chunks.json'

            # Build metadata
            output_path_obj = Path(output_path)
            metadata = {
                'source_file': source_file,
                'output_filename': output_path_obj.name,  # Actual dynamic filename
                'output_directory': str(output_path_obj.parent),
                'voice_used': voice_path,
                'generation_timestamp': datetime.now().isoformat(),
                'total_chunks': len(chunks)
            }

            # Add generation statistics if available (final save) or null placeholders (preliminary)
            if not is_preliminary:
                # Final save - include generation statistics
                total_time = time.time() - getattr(self, 'start_time', time.time())
                audio_duration = getattr(self, 'audio_duration', 0)
                metadata.update({
                    'processing_time': getattr(self, 'processing_time', total_time),
                    'realtime_factor': getattr(self, 'realtime_factor', 0),
                    'chunk_processing_time': getattr(self, 'chunk_processing_time', None),
                    'total_realtime_factor': getattr(self, 'total_realtime_factor', 0),
                    'chunks_processed': getattr(self, 'current_chunk', 0) + 1,
                    'audio_duration': audio_duration
                })
            else:
                # Preliminary save - null placeholders
                metadata.update({
                    'processing_time': None,
                    'realtime_factor': None,
                    'chunks_processed': None,
                    'audio_duration': None
                })

            # JSON must match the self-contained text written to sidecar files.
            from ..preprocessing.text_normalizer import normalize_text_chunk_for_storage

            chunks_data = []
            for chunk in chunks:
                chunk_dict = {
                    'index': chunk.index,
                    'text': normalize_text_chunk_for_storage(chunk.text or ""),
                    'word_count': chunk.word_count,
                    'character_count': chunk.character_count,
                    'boundary_type': chunk.boundary_type.value,
                    'punctuation': chunk.punctuation,
                    'start_position': chunk.start_position,
                    'end_position': chunk.end_position,
                    'emotion': chunk.emotion.value if chunk.emotion else None,
                    'emotion_scores': chunk.emotion_scores,
                    'emotion_confidence': chunk.emotion_confidence,
                      'tts_params': dataclasses.asdict(chunk.tts_params) if isinstance(chunk.tts_params, TTSParams) else chunk.tts_params,
                    'post_process': chunk.post_process,
                    'chapter_number': chunk.chapter_number,
                    'chapter_id': getattr(chunk, 'chapter_id', chunk.chapter_number),
                    'chapter_title': getattr(chunk, 'chapter_title', None),
                    'is_dialogue': chunk.is_dialogue,
                    'has_emphasis': chunk.has_emphasis,
                    'pause_events': chunk.pause_events
                }
                chunks_data.append(chunk_dict)

            # Create final JSON structure
            json_data = {
                '_metadata': metadata,
                'chunks': chunks_data
            }

            # Save to file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Chunk data saved to {json_path}")

        except Exception as e:
            logger.error(f"Failed to save chunk metadata: {e}")
            raise

    def _reprocess_failed_chunks_staged(
        self,
        failures: List[Dict[str, Any]],
        voice_state: Any,
        dataset_paths: Dict[str, Any],
        asr_config: Dict[str, Any],
        voice_path: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate three candidates, release TTS, then batch-score candidates.

        TTS generation and ASR validation are deliberately separate phases. The
        temporary validation tree lets one ASR model score every candidate after
        the TTS model has been fully released, avoiding GPU model overlap.
        Only chunks whose best candidate still fails threshold are retained in
        the returned investigation log and written to disk.
        """
        if not failures:
            return []

        audio_chunks_dir = Path(dataset_paths["audio_chunks_dir"])
        text_chunks_dir = Path(dataset_paths["text_chunks_dir"])
        tts_dir = Path(dataset_paths["tts_dir"])
        failed_dir = audio_chunks_dir / "Failed"
        failed_dir.mkdir(exist_ok=True)
        cleanup_config = getattr(self.config, "audio_cleanup", {}) or {}
        candidate_rows: List[Dict[str, Any]] = []
        candidate_count = 3
        temp_decrement = float(asr_config.get("temp_decrement", 0.1) or 0.1)
        threshold = float(asr_config.get("threshold", 0.85) or 0.85)
        sample_rate = 24000

        logger.info(
            "Staged regeneration: generating %s candidates for %s chunks before ASR",
            candidate_count,
            len(failures),
        )
        import gc

        for failure in failures:
            chunk_index = int(failure.get("chunk_index", -1))
            chunk_num = f"{chunk_index:05d}"
            original_path = audio_chunks_dir / f"chunk_{chunk_num}.wav"
            chunk_text = failure.get("text", "")
            if not chunk_text:
                text_path = text_chunks_dir / f"chunk_{chunk_num}.txt"
                if text_path.exists():
                    chunk_text = text_path.read_text(encoding="utf-8")
            if not chunk_text:
                logger.warning("No text data available for staged chunk %s", chunk_num)
                continue

            tts_params = failure.get("tts_params", {}) or {}
            current_temp = float(tts_params.get("temperature", 0.7) or 0.7)
            frames_after_eos = int(tts_params.get("frames_after_eos", 2) or 2)
            row = {
                "chunk_index": chunk_index,
                "chunk_num": chunk_num,
                "text": chunk_text,
                "tts_params": tts_params,
                "post_process": failure.get("post_process", {}) or {},
                "original_path": original_path,
                "original_score": float(failure.get("score", 0.0) or 0.0),
                "original_asr": failure,
                "candidates": [],
            }
            for attempt in range(1, candidate_count + 1):
                temperature = max(0.1, current_temp - ((attempt - 1) * temp_decrement))
                candidate_path = audio_chunks_dir / f"chunk_{chunk_num}_attempt_{attempt}.wav"
                row["candidates"].append({
                    "attempt": attempt,
                    "temp": temperature,
                    "audio_path": str(candidate_path),
                    "score": 0.0,
                    "asr_result": {},
                    "is_original": False,
                    "eos_detected": None,
                    "eos_step": None,
                    "max_generation_length": None,
                })
            candidate_rows.append(row)

        if not voice_path:
            voice_path = asr_config.get("_regen_voice_path")
        if not voice_path:
            raise RuntimeError("Staged regeneration requires a reloadable voice path")

        # Drop parent model references before worker TTS loading.
        voice_state = None
        self.release_gpu_memory()
        logger.info("Staged regeneration: starting parallel TTS candidate phase")

        worker_count = max(1, min(2, int(asr_config.get("regen_tts_workers", 2) or 2)))
        batch_size = max(1, int(asr_config.get("regen_tts_batch_size", 2) or 2))
        start_gap_s = max(0.0, float(asr_config.get("regen_worker_start_gap_s", 2.0) or 2.0))
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        ready_queue = ctx.Queue()
        workers = []
        try:
            for worker_id in range(worker_count):
                worker = ctx.Process(
                    target=_staged_regeneration_worker,
                    args=(
                        task_queue,
                        result_queue,
                        ready_queue,
                        str(voice_path),
                        worker_id,
                        self._device,
                        batch_size,
                        cleanup_config,
                    ),
                )
                worker.daemon = True
                worker.start()
                workers.append(worker)
                ready = ready_queue.get(timeout=600)
                if ready.get("status") != "ready":
                    raise RuntimeError(ready.get("error", "staged TTS worker failed to load"))
                logger.info(
                    "Staged regeneration TTS worker %s ready pid=%s; delaying next worker %.1fs",
                    worker_id,
                    ready.get("pid"),
                    start_gap_s if worker_id + 1 < worker_count else 0.0,
                )
                if worker_id + 1 < worker_count:
                    time.sleep(start_gap_s)

            for start in range(0, len(candidate_rows), batch_size):
                task_queue.put(candidate_rows[start : start + batch_size])
            for _ in workers:
                task_queue.put(None)

            expected_candidates = len(candidate_rows) * candidate_count
            received_candidates = 0
            while received_candidates < expected_candidates:
                event = result_queue.get(timeout=max(600, expected_candidates * 30))
                if event.get("kind") == "error":
                    raise RuntimeError(event.get("error", "staged TTS worker failed"))
                if event.get("kind") != "candidate_ready":
                    continue
                received_candidates += 1
                row = next(
                    item for item in candidate_rows
                    if item["chunk_index"] == event["chunk_index"]
                )
                candidate = next(
                    item for item in row["candidates"]
                    if item["attempt"] == event["attempt"]
                )
                candidate.update({
                    "eos_detected": event.get("eos_detected"),
                    "eos_step": event.get("eos_step"),
                    "max_generation_length": event.get("max_generation_length"),
                })
        finally:
            for worker in workers:
                worker.join(timeout=30)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
            workers.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        logger.info("Staged regeneration: TTS phase complete; loading ASR phase")

        staging_dir = tts_dir / ".asr_regeneration_candidates"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        (staging_dir / "audio_chunks").mkdir(parents=True)
        (staging_dir / "text_chunks").mkdir()
        staged_lookup: Dict[str, Dict[str, Any]] = {}
        try:
            stems = []
            for row in candidate_rows:
                for candidate in row["candidates"]:
                    staged_index = len(stems)
                    stem = f"chunk_{staged_index:05d}"
                    shutil.copyfile(candidate["audio_path"], staging_dir / "audio_chunks" / f"{stem}.wav")
                    (staging_dir / "text_chunks" / f"{stem}.txt").write_text(
                        row["text"], encoding="utf-8"
                    )
                    staged_lookup[stem] = candidate
                    stems.append(stem)

            from ASR.asr_validator import load_asr_model_adaptive, run_pipeline_batch_validation

            model_name = str(asr_config.get("regen_asr_model") or "medium")
            asr_model, device = load_asr_model_adaptive(
                model_name,
                force_device="cuda",
                engine="faster_whisper",
                n_threads=2,
            )
            if not asr_model:
                raise RuntimeError(f"Could not load staged regeneration ASR model {model_name}")
            logger.info("Staged regeneration: ASR model=%s device=%s candidates=%s", model_name, device, len(stems))
            results = run_pipeline_batch_validation(
                staging_dir,
                stems,
                asr_model,
                threshold,
                language=str(asr_config.get("language", "en") or "en"),
                load_workers=4,
                score_workers=4,
                pack_size=8,
            )
            for result in results:
                stem = str(result.get("chunk_num", ""))
                candidate = staged_lookup.get(stem)
                if candidate is not None:
                    candidate["score"] = float(result.get("score", 0.0) or 0.0)
                    candidate["asr_result"] = result
        finally:
            try:
                del asr_model
            except UnboundLocalError:
                pass
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            shutil.rmtree(staging_dir, ignore_errors=True)

        investigations = []
        for row in candidate_rows:
            original = {
                "attempt": 0,
                "temp": None,
                "score": row["original_score"],
                "audio_path": str(row["original_path"]),
                "asr_result": row["original_asr"],
                "is_original": True,
            }
            candidates = [original, *row["candidates"]]
            winner = max(
                candidates,
                key=lambda candidate: (
                    bool(candidate.get("asr_result", {}).get("passed", False)),
                    float(candidate.get("score", 0.0)),
                ),
            )
            if not winner.get("is_original"):
                if row["original_path"].exists():
                    row["original_path"].replace(failed_dir / row["original_path"].name)
                Path(winner["audio_path"]).replace(row["original_path"])
            for candidate in row["candidates"]:
                if candidate is winner:
                    continue
                candidate_path = Path(candidate["audio_path"])
                if candidate_path.exists():
                    candidate_path.replace(
                        failed_dir / f"chunk_{row['chunk_num']}_attempt_{candidate['attempt']}.wav"
                    )

            winner_result = winner.get("asr_result", {}) or {}
            winner_passed = bool(winner_result.get("passed", False))
            if winner_passed:
                # Final pass rows belong in the medium verification report, not
                # the failure-only investigation log the user reviews.
                continue

            investigations.append({
                "chunk_index": row["chunk_index"],
                "chunk_num": row["chunk_num"],
                "best_score": float(winner.get("score", 0.0)),
                "threshold": threshold,
                "best_attempt": winner,
                "all_attempts": candidates,
                "text": row["text"],
                "original_asr": row["original_asr"],
            })
        self._save_investigation_log(investigations, dataset_paths)
        return investigations

    def _reprocess_failed_chunks(self, failure_log_path: str, voice_state, dataset_paths: Dict, asr_config: Dict):
        """
        Reprocess chunks that failed ASR validation by regenerating with lower temperature.
        Uses self-contained data from failure log instead of looking up in current_chunks.
        Validates each regeneration attempt and saves best result.
        Creates investigation log for chunks where ALL attempts fail threshold.
        """

        try:
            with open(failure_log_path, 'r') as f:
                failures = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read ASR failure log: {e}")
            return []

        if not failures:
            logger.info("No ASR failures to reprocess")
            return []

        logger.info(f"Reprocessing {len(failures)} ASR-failed chunks...")

        audio_chunks_dir = Path(dataset_paths['audio_chunks_dir'])
        text_chunks_dir = Path(dataset_paths['text_chunks_dir'])
        tts_dir = Path(dataset_paths['tts_dir'])
        max_retries = asr_config.get('max_retries', 3)
        if int(max_retries or 0) <= 0:
            # Report-only mode is handled before this legacy regeneration path.
            logger.info("Skipping failed-chunk regeneration because max_retries=0")
            return []
        temp_decrement = asr_config.get('temp_decrement', 0.1)
        threshold = asr_config.get('threshold', 0.85)
        regen_strategy = str(asr_config.get('regeneration_strategy', 'first_pass') or 'first_pass').strip().lower()
        regen_asr_model = str(asr_config.get('regen_asr_model') or 'medium').strip() or 'medium'

        if regen_strategy == "staged_three_candidates":
            return self._reprocess_failed_chunks_staged(
                failures,
                voice_state,
                dataset_paths,
                asr_config,
                voice_path=asr_config.get("_regen_voice_path"),
            )

        # Create Failed subdirectory for organizing debug files
        failed_dir = audio_chunks_dir / "Failed"
        failed_dir.mkdir(exist_ok=True)
        logger.info(f"ASR debug files will be organized in: {failed_dir}")

        reprocessed_count = 0
        failed_permanently = 0
        investigation_log = []

        for failure in failures:
            chunk_index = failure.get('chunk_index')
            original_score = failure.get('score', 0)
            chunk_num = f"{chunk_index:05d}"
            chunk_filename = f"chunk_{chunk_num}.wav"
            original_chunk_path = audio_chunks_dir / chunk_filename

            chunk_text = failure.get('text', '')
            tts_params = failure.get('tts_params', {})
            post_process = failure.get('post_process', {})

            if not chunk_text:
                text_filename = f"chunk_{chunk_num}.txt"
                text_path = text_chunks_dir / text_filename
                if text_path.exists():
                    try:
                        with open(text_path, 'r', encoding='utf-8') as f:
                            chunk_text = f.read()
                    except Exception as e:
                        logger.warning(f"Could not read text file for chunk {chunk_num}: {e}")
                        continue
                else:
                    logger.warning(f"No text data available for chunk {chunk_num}, skipping")
                    continue

            current_temp = tts_params.get('temperature', 0.7)
            frames_after_eos = tts_params.get('frames_after_eos', 2)

            candidates = []
            best_score = float("-inf")
            best_asr_result = None

            # Track original as a candidate if it exists
            if original_chunk_path.exists():
                original_asr_result = {
                    "hyp_text_raw": failure.get("transcribed_text", ""),
                    "transcribed_text": failure.get("transcribed_text", ""),
                    "explanation": failure.get("explanation", ""),
                    "prose_score": failure.get("prose_score", 0.0),
                    "id_score": failure.get("id_score", 0.0),
                    "score": original_score,
                    "hallucination_warning": failure.get("hallucination_warning", ""),
                    "truncation_warning": failure.get("truncation_warning", "")
                }

                candidates.append({
                    "attempt": 0,
                    "temp": None,
                    "score": original_score,
                    "audio": None,
                    "audio_path": str(original_chunk_path),
                    "asr_result": original_asr_result,
                    "is_original": True
                })
                best_score = original_score
                best_asr_result = original_asr_result
            else:
                logger.warning(f"Original chunk {chunk_num} not found - using regeneration attempts only")

            try:
                for retry in range(max_retries):
                    retry_temp = max(0.1, current_temp - (retry * temp_decrement))
                    temp_filename = f"chunk_{chunk_num}_attempt_{retry + 1}.wav"
                    temp_path = audio_chunks_dir / temp_filename

                    logger.info(f"Regenerating chunk {chunk_num} attempt {retry + 1}/{max_retries} with temp={retry_temp:.2f}")

                    if hasattr(self.tts_model, 'temp'):
                        self.tts_model.temp = float(retry_temp)

                    from ..preprocessing.pause_injector import (
                        generate_audio_with_pauses,
                        has_inline_pause_markers,
                    )
                    from ..preprocessing.text_normalizer import flatten_newlines_for_json

                    plan_text = flatten_newlines_for_json(chunk_text)
                    cleanup_config = getattr(self.config, 'audio_cleanup', {})
                    sample_rate = getattr(self.tts_model, 'sample_rate', 24000)
                    if has_inline_pause_markers(plan_text):
                        def render_segment(text: str):
                            """Generate and clean one sequential regeneration segment."""
                            segment = self.tts_model.generate_audio(
                                voice_state, text, frames_after_eos=frames_after_eos
                            )
                            return _postprocess_inline_speech_segment(
                                segment,
                                {"speed_factor": 1.0},
                                sample_rate,
                                cleanup_config.get('enabled', True),
                                cleanup_config.get('use_silero_vad', True),
                                cleanup_config.get('speech_endpoint_threshold', 0.004),
                                cleanup_config.get('trimming_buffer_ms', 100),
                            )

                        audio, _pause_events = generate_audio_with_pauses(
                            self.tts_model, voice_state, plan_text,
                            generate_segment=render_segment,
                        )
                    else:
                        audio = self.tts_model.generate_audio(
                            voice_state, plan_text, frames_after_eos=frames_after_eos
                        )
                        audio = _postprocess_inline_speech_segment(
                            audio,
                            {"speed_factor": 1.0},
                            sample_rate,
                            cleanup_config.get('enabled', True),
                            cleanup_config.get('use_silero_vad', True),
                            cleanup_config.get('speech_endpoint_threshold', 0.004),
                            cleanup_config.get('trimming_buffer_ms', 100),
                        )

                    if audio is not None:

                        silence_duration = post_process.get('silence_duration', 0.0)
                        if silence_duration > 0:
                            import torch
                            sample_rate = getattr(self.tts_model, 'sample_rate', 24000)
                            num_silence_samples = int(silence_duration * sample_rate)
                            if len(audio.shape) == 1:
                                silence_tensor = torch.zeros(num_silence_samples, device=audio.device)
                                audio = torch.cat([audio, silence_tensor], dim=0)

                        self._save_audio(audio, str(temp_path))

                        attempt_chunk_id = f"chunk_{chunk_num}"
                        asr_result = self._validate_chunk_asr(
                            attempt_chunk_id,
                            str(tts_dir),
                            threshold,
                            audio_path=str(temp_path),
                            model_name=regen_asr_model,
                        )
                        score = asr_result.get('score', 0.0)

                        attempt_info = {
                            "attempt": retry + 1,
                            "temp": retry_temp,
                            "score": score,
                            "audio": audio,
                            "audio_path": str(temp_path),
                            "asr_result": asr_result,
                            "is_original": False
                        }
                        candidates.append(attempt_info)

                        if score > best_score:
                            best_score = score
                            best_asr_result = asr_result
                        if regen_strategy == 'first_pass' and bool(asr_result.get('passed', False)):
                            logger.info(
                                "Chunk %s accepted on first regen pass; skipping remaining attempts",
                                chunk_num,
                            )
                            break
                    else:
                        logger.warning(f"No audio generated for chunk {chunk_num} attempt {retry + 1}")

            except Exception as e:
                logger.error(f"Error reprocessing chunk {chunk_num}: {e}")
                failed_permanently += 1
                continue

            if not candidates:
                logger.warning(f"No candidates available for chunk {chunk_num}; skipping")
                failed_permanently += 1
                continue

            winner = max(candidates, key=lambda x: x['score'])
            winner_is_original = winner.get("is_original", False)

            if winner_is_original:
                # Original stays in main directory; move all attempt files to Failed/
                for candidate in candidates:
                    if not candidate.get("is_original"):
                        temp_path = Path(candidate["audio_path"])
                        if temp_path.exists():
                            failed_attempt_path = failed_dir / f"chunk_{chunk_num}_attempt_{candidate['attempt']}.wav"
                            try:
                                temp_path.rename(failed_attempt_path)
                            except Exception as e:
                                logger.warning(f"Could not move attempt file {temp_path} to Failed/: {e}")
                                try:
                                    temp_path.unlink(missing_ok=True)
                                except Exception:
                                    pass
            else:
                # Regeneration wins: move original to Failed/ before saving winner
                if original_chunk_path.exists():
                    original_failed_path = failed_dir / chunk_filename
                    try:
                        original_chunk_path.rename(original_failed_path)
                    except Exception as e:
                        logger.warning(f"Could not move original chunk {chunk_num} to Failed/: {e}")

                final_path = audio_chunks_dir / chunk_filename
                if winner.get("audio") is not None:
                    self._save_audio(winner["audio"], str(final_path))

                # Move losing attempts to Failed/
                for candidate in candidates:
                    if candidate is winner:
                        continue
                    if not candidate.get("is_original"):
                        temp_path = Path(candidate["audio_path"])
                        if temp_path.exists():
                            failed_attempt_path = failed_dir / f"chunk_{chunk_num}_attempt_{candidate['attempt']}.wav"
                            try:
                                temp_path.rename(failed_attempt_path)
                            except Exception as e:
                                logger.warning(f"Could not move attempt file {temp_path} to Failed/: {e}")
                                try:
                                    temp_path.unlink(missing_ok=True)
                                except Exception:
                                    pass

                # Clean up winner temp file if it still exists
                winner_temp_path = Path(winner["audio_path"])
                if winner_temp_path.exists():
                    try:
                        winner_temp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            logger.info(f"Chunk {chunk_num} regeneration complete - best score: {best_score:.3f}")
            reprocessed_count += 1

            if best_score < threshold:
                original_asr = {
                    "score": original_score,
                    "original_text": failure.get("original_text", ""),
                    "transcribed_text": failure.get("transcribed_text", ""),
                    "explanation": failure.get("explanation", ""),
                    "prose_score": failure.get("prose_score", 0.0),
                    "id_score": failure.get("id_score", 0.0)
                }

                investigation_log.append({
                    "chunk_index": chunk_index,
                    "chunk_num": chunk_num,
                    "best_score": best_score,
                    "threshold": threshold,
                    "best_attempt": winner,
                    "all_attempts": candidates,
                    "text": chunk_text,
                    "original_asr": original_asr
                })

        logger.info(f"ASR reprocessing complete: {reprocessed_count} reprocessed, {failed_permanently} permanently failed")
        logger.info(f"ASR debug files organized in: {failed_dir}")

        if investigation_log:
            self._save_investigation_log(investigation_log, dataset_paths)
            logger.info(f"Investigation log created for {len(investigation_log)} chunks")

        logger.info(f"ASR failure log preserved at: {failure_log_path}")
        return investigation_log

    def _validate_chunk_asr(
        self,
        chunk_num: str,
        tts_dir: str,
        threshold: float,
        audio_path: str | None = None,
        model_name: str | None = None,
    ) -> Dict[str, Any]:
        """Validate one chunk by calling ASR validator via subprocess.

        Args:
            chunk_num: Chunk stem like ``chunk_00005``.
            tts_dir: Parent TTS directory containing audio and text chunks.
            threshold: Minimum similarity score required to pass.
            audio_path: Optional explicit WAV path to score.
            model_name: Optional override for ASR model size. When omitted, the
                configured regen model is used, then the GUI model as fallback.
        """
        import subprocess
        import json
        from pathlib import Path

        try:
            project_root = Path(__file__).parent.parent.parent

            asr_exe = None
            if hasattr(self.config, 'asr_quality_control') and hasattr(self.config.asr_quality_control, 'executable_path'):
                asr_exe = self.config.asr_quality_control.executable_path
            elif isinstance(self.config, dict) and 'asr_quality_control' in self.config:
                asr_exe = self.config['asr_quality_control'].get('executable_path')
            asr_exe = _resolve_asr_executable(asr_exe)

            asr_script = project_root / 'ASR' / 'asr_validator.py'
            asr_exe = str(asr_exe)

            asr_script = str(asr_script.resolve())

            if not Path(asr_exe).exists():
                logger.warning(f"ASR executable not found: {asr_exe}")
                return {'score': 1.0, 'passed': True, 'error': 'ASR executable not found'}
            if not Path(asr_script).exists():
                logger.warning(f"ASR validator script not found: {asr_script}")
                return {'score': 1.0, 'passed': True, 'error': 'ASR validator not available'}

            working_dir = Path(tts_dir)
            asr_qc = getattr(self.config, 'asr_quality_control', {}) or {}
            if isinstance(self.config, dict):
                asr_qc = self.config.get('asr_quality_control', {}) or {}
            asr_model_name = model_name or (
                asr_qc.get('regen_asr_model')
                if hasattr(asr_qc, 'get')
                else None
            ) or (asr_qc.get('model', 'base') if hasattr(asr_qc, 'get') else 'base')
            asr_language = asr_qc.get('language', 'en') if hasattr(asr_qc, 'get') else 'en'
            command = [
                asr_exe, asr_script,
                '--single-chunk', chunk_num,
                '--tts-dir', str(working_dir),
                '--threshold', str(threshold),
                '--model', asr_model_name,
                '--language', asr_language,
                '--json'
            ]

            if audio_path:
                command.extend(['--audio-path', audio_path])

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120
            )

            try:
                return _parse_asr_json_output(result.stdout)
            except (TypeError, ValueError) as exc:
                error_msg = result.stderr.strip() if result.stderr else str(exc)
                logger.warning(
                    "ASR validation produced no usable JSON for %s (exit %s): %s",
                    chunk_num,
                    result.returncode,
                    error_msg,
                )
                return {'score': 0.0, 'passed': False, 'error': error_msg}

        except subprocess.TimeoutExpired:
            logger.warning(f"ASR validation timeout for {chunk_num}")
            return {'score': 0.0, 'passed': False, 'error': 'Validation timeout'}
        except Exception as e:
            logger.warning(f"ASR validation error for {chunk_num}: {e}")
            return {'score': 0.0, 'passed': False, 'error': str(e)}

    def _verify_failed_chunks_with_medium_asr(
        self,
        failures: List[Dict[str, Any]],
        tts_dir: Path,
        threshold: float,
        language: str,
        model_name: str = "medium",
        report_filename: str = "asr_medium_validation_all.json",
        require_gpu: bool = False,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Confirm Stage 1 candidates with an independent packed ASR model.

        Only candidates that fail both the Parakeet Stage 1 comparison and the
        Medium comparison enter regeneration. A verifier load or decode error
        does not prove bad speech and therefore preserves the original WAV.

        Args:
            failures: Stage 1 candidate rows from ``asr_failures.json``.
            tts_dir: Book-local TTS directory holding chunk WAV/text files.
            threshold: Shared spoken-content score threshold.
            language: ASR language code passed to the Stage 2 decoder.
            model_name: Faster-Whisper model selected for Stage 2 verification.
            report_filename: Book-local all-decisions Medium audit filename.
                The paired failure-only report is written beside it.
            require_gpu: Disable CPU fallback when Stage 2 is being run in an
                isolated GPU-only verification stage.

        Returns:
            Candidate rows confirmed by both ASR models and an audit summary.
        """
        from ASR.asr_validator import (
            build_tts_book_term_evidence,
            cleanup_asr_model,
            load_asr_model_adaptive,
            run_pipeline_batch_validation,
        )

        summary: Dict[str, Any] = {
            "attempted": len(failures),
            "verified_pass": 0,
            "verified_fail": 0,
            "not_proven": 0,
            "verification_model": model_name,
            "verification_device": None,
            "verification_mode": "packed_gpu_pipeline",
            "verification_skipped": False,
            "verification_error": None,
            "verification_log_path": None,
        }
        verification_records: List[Dict[str, Any]] = []
        if not failures:
            return [], summary

        asr_model = None
        try:
            # This loads once and packs candidates, avoiding per-chunk Stage 2
            # startup while keeping the verifier independent from Parakeet.
            load_options = {"force_device": "cuda"}
            if require_gpu:
                load_options["allow_cpu_fallback"] = False
            asr_model, device = load_asr_model_adaptive(model_name, **load_options)
            if not asr_model:
                summary["verification_skipped"] = True
                summary["verification_device"] = device
                summary["verification_error"] = (
                    "gpu_second_stage_model_load_failed_no_cpu_fallback"
                    if require_gpu
                    else "failed_to_load_second_stage_model"
                )
                for failure in failures:
                    verification_records.append({
                        "chunk_index": failure.get("chunk_index"),
                        "chunk_id": None,
                        "original_failure": failure,
                        "medium_result": None,
                        "decision": "accepted_not_proven_failure",
                    })
                summary["not_proven"] = len(failures)
                return [], summary
            summary["verification_device"] = device

            chunk_ids = [
                f"chunk_{int(failure['chunk_index']):05d}"
                for failure in failures
                if failure.get("chunk_index") is not None
            ]
            book_term_evidence = build_tts_book_term_evidence(tts_dir)
            summary["book_term_evidence_count"] = len(book_term_evidence.get("terms") or {})
            pipeline_results = run_pipeline_batch_validation(
                tts_dir,
                chunk_ids,
                asr_model,
                threshold,
                language=language or "en",
                load_workers=4,
                score_workers=4,
                vad_filter=False,
                pack_size=8,
                pack_silence_s=0.75,
                book_term_evidence=book_term_evidence,
            )
            results_by_chunk = {
                str(result.get("chunk_num") or ""): result
                for result in pipeline_results
            }
            confirmed: List[Dict[str, Any]] = []
            for failure in failures:
                chunk_index = failure.get("chunk_index")
                if chunk_index is None:
                    summary["not_proven"] += 1
                    verification_records.append({
                        "chunk_index": None,
                        "chunk_id": None,
                        "original_failure": failure,
                        "medium_result": None,
                        "decision": "accepted_not_proven_failure",
                    })
                    continue
                chunk_id = f"chunk_{int(chunk_index):05d}"
                result = results_by_chunk.get(chunk_id)
                if result is None or result.get("error"):
                    summary["not_proven"] += 1
                    verification_records.append({
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                        "original_failure": failure,
                        "medium_result": result or {"error": "Missing Medium verification result"},
                        "decision": "accepted_not_proven_failure",
                    })
                    continue
                if bool(result.get("passed", False)):
                    summary["verified_pass"] += 1
                    verification_records.append({
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                        "original_failure": failure,
                        "medium_result": result,
                        "decision": "accepted_by_medium",
                    })
                    continue

                summary["verified_fail"] += 1
                verification_records.append({
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                    "original_failure": failure,
                    "medium_result": result,
                    "decision": "confirmed_by_two_asr_models",
                })
                failure_copy = dict(failure)
                failure_copy["medium_verification"] = result
                confirmed.append(failure_copy)
            return confirmed, summary
        except Exception as exc:
            summary["verification_skipped"] = True
            summary["verification_error"] = str(exc)
            summary["not_proven"] = len(failures)
            logger.warning("Independent Medium verification unavailable: %s", exc)
            for failure in failures:
                verification_records.append({
                    "chunk_index": failure.get("chunk_index"),
                    "chunk_id": None,
                    "original_failure": failure,
                    "medium_result": {"error": str(exc)},
                    "decision": "accepted_not_proven_failure",
                })
            return [], summary
        finally:
            if asr_model is not None:
                try:
                    cleanup_asr_model(asr_model)
                except Exception as exc:
                    logger.warning("Could not release Medium verifier: %s", exc)
            self._write_medium_verification_log(
                tts_dir,
                threshold,
                language,
                summary,
                verification_records,
                report_filename=report_filename,
            )

    def _write_medium_verification_log(
        self,
        tts_dir: Path,
        threshold: float,
        language: str,
        summary: Dict[str, Any],
        verification_records: List[Dict[str, Any]],
        report_filename: str = "asr_medium_validation_all.json",
    ) -> None:
        """Write all Medium decisions and a companion failure-only report.

        Args:
            tts_dir: Book-local TTS folder receiving the report.
            threshold: Spoken-content threshold used by Medium verification.
            language: ASR language code passed to the verifier.
            summary: Aggregate counts updated with the saved report path.
            verification_records: Per-candidate Medium decisions.
            report_filename: Relative filename used for the all-decisions
                audit output. The matching failure-only filename is selected
                from the active legacy or New pipeline convention.
        """
        log_path = tts_dir / Path(report_filename).name
        is_new_pipeline_report = log_path.name == "asr_new_medium_verification.json"
        record_builder = (
            _build_new_medium_verification_record
            if is_new_pipeline_report
            else _build_medium_verification_record
        )
        records = [record_builder(record) for record in verification_records]
        payload = {
            "report_type": (
                "new_medium_asr_verification_v2"
                if is_new_pipeline_report
                else "medium_asr_verification"
            ),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "threshold": threshold,
            "language": language,
            "summary": summary,
            "records": records,
        }
        try:
            log_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            summary["verification_log_path"] = str(log_path)
            logger.info("Medium verification report written: %s", log_path)
            failure_records = [
                record
                for record in records
                if _is_confirmed_medium_failure(record, is_new_pipeline_report)
            ]
            failures_path = tts_dir / (
                "asr_new_medium_failures.json"
                if is_new_pipeline_report
                else "asr_medium_validation_failed.json"
            )
            failures_payload = {
                "report_type": (
                    "new_medium_confirmed_failures_v2"
                    if is_new_pipeline_report
                    else "medium_asr_confirmed_failures"
                ),
                "generated_at": payload["generated_at"],
                "source_report": str(log_path),
                "threshold": threshold,
                "language": language,
                "summary": {
                    "stage_one_candidates": len(records),
                    "medium_confirmed_failures": len(failure_records),
                    "medium_accepted_or_unproven": len(records) - len(failure_records),
                },
                "records": failure_records,
            }
            failures_path.write_text(
                json.dumps(failures_payload, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            from pocket_tts.asr_failure_reports import write_failure_manifest

            stage_one_filename = (
                "asr_new_failures.json"
                if is_new_pipeline_report
                else "asr_failures.json"
            )
            write_failure_manifest(
                tts_dir,
                pipeline="new" if is_new_pipeline_report else "legacy",
                authoritative_filename=failures_path.name,
                label=(
                    "New Medium-confirmed failures"
                    if is_new_pipeline_report
                    else "Legacy Medium-confirmed failures"
                ),
                kind="medium_confirmed_failures",
                fallback_filenames=(stage_one_filename,),
                threshold=threshold,
                language=language,
            )
            summary["verification_failures_log_path"] = str(failures_path)
            logger.info("Medium confirmed-failures report written: %s", failures_path)
            # Human-readable Stage 2 fail log: book text vs Medium ASR + why.
            try:
                stage2_log = write_stage2_investigation_log(
                    tts_dir,
                    failure_records,
                    threshold=threshold,
                    language=language,
                    stage_one_candidates=len(records),
                    is_new_pipeline=is_new_pipeline_report,
                    filename=(
                        "asr_stage2_investigation.log"
                        if is_new_pipeline_report
                        else "asr_stage2_investigation_legacy.log"
                    ),
                )
                summary["stage2_investigation_log_path"] = str(stage2_log)
                logger.info("Stage 2 human investigation log written: %s", stage2_log)
            except OSError as stage2_exc:
                summary["stage2_investigation_log_error"] = str(stage2_exc)
                logger.warning(
                    "Could not write Stage 2 investigation log under %s: %s",
                    tts_dir,
                    stage2_exc,
                )
        except OSError as exc:
            summary["verification_log_error"] = str(exc)
            logger.warning("Could not write Medium verification report %s: %s", log_path, exc)

    def _verify_failed_chunks_with_forced_alignment(
        self,
        failures: List[Dict[str, Any]],
        tts_dir: Path,
        threshold: float,
        language: str,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Confirm Stage 1 candidates through known-text forced alignment only.

        Returns only positive, conclusive alignment failures for regeneration.
        Tool or model uncertainty accepts the original audio and records why.
        """
        from ASR.verification_backends import NemoForcedAligner

        summary: Dict[str, Any] = {
            "attempted": len(failures),
            "accepted": 0,
            "confirmed_failed": 0,
            "accepted_not_proven_failure": 0,
            "verification_model": "stt_en_fastconformer_hybrid_large_pc",
            "verification_mode": "nemo_forced_alignment",
            "verification_log_path": None,
        }
        verification_records: List[Dict[str, Any]] = []
        if not failures:
            return [], summary

        try:
            aligner = NemoForcedAligner()
            jobs = []
            valid_failures = []
            for failure in failures:
                chunk_index = failure.get("chunk_index")
                if chunk_index is None:
                    continue
                chunk_id = f"chunk_{int(chunk_index):05d}"
                jobs.append({
                    "audio_path": str(tts_dir / "audio_chunks" / f"{chunk_id}.wav"),
                    "text": str(failure.get("original_text") or ""),
                })
                valid_failures.append(failure)
            evidence_by_path = {
                item.audio_path: item.to_dict()
                for item in aligner.align_batch(jobs, tts_dir / ".asr_alignment")
            }
            remaining: List[Dict[str, Any]] = []
            for failure in valid_failures:
                chunk_index = failure.get("chunk_index")
                chunk_num = f"{int(chunk_index):05d}"
                audio_path = str(tts_dir / "audio_chunks" / f"chunk_{chunk_num}.wav")
                alignment = evidence_by_path.get(audio_path, {
                    "available": False, "conclusive": False,
                    "error": "No alignment result returned",
                })
                decision, explanation = _alignment_decision(alignment)
                record = {
                    "chunk_index": chunk_index,
                    "chunk_id": f"chunk_{chunk_num}",
                    "original_text": failure.get("original_text", ""),
                    "stage_one": failure,
                    "alignment": alignment,
                    "decision": decision,
                    "accepted_reason": "not_proven_failure" if decision == "accepted_not_proven_failure" else None,
                    "explanation": explanation,
                }
                verification_records.append(record)
                if decision == "confirmed_failed_by_alignment":
                    summary["confirmed_failed"] += 1
                    failure_copy = dict(failure)
                    failure_copy["alignment_verification"] = alignment
                    remaining.append(failure_copy)
                else:
                    summary["accepted"] += 1
                    if decision == "accepted_not_proven_failure":
                        summary["accepted_not_proven_failure"] += 1
            return remaining, summary
        except Exception as exc:
            logger.warning("Forced alignment unavailable: %s", exc)
            for failure in failures:
                chunk_index = failure.get("chunk_index")
                verification_records.append({
                    "chunk_index": chunk_index,
                    "chunk_id": f"chunk_{int(chunk_index):05d}" if chunk_index is not None else None,
                    "original_text": failure.get("original_text", ""),
                    "stage_one": failure,
                    "alignment": {"available": False, "conclusive": False, "error": str(exc)},
                    "decision": "accepted_not_proven_failure",
                    "accepted_reason": "not_proven_failure",
                    "explanation": "Alignment unavailable or inconclusive; speech mismatch was not proven.",
                })
            summary["accepted"] = len(failures)
            summary["accepted_not_proven_failure"] = len(failures)
            return [], summary
        finally:
            self._write_alignment_verification_log(
                tts_dir,
                threshold,
                language,
                summary,
                verification_records,
            )

    def _write_alignment_verification_log(
        self,
        tts_dir: Path,
        threshold: float,
        language: str,
        summary: Dict[str, Any],
        verification_records: List[Dict[str, Any]],
    ) -> None:
        """Persist every Stage 2 decision with separate Stage 1 and alignment evidence."""
        log_path = tts_dir / "asr_alignment_verification.json"
        compact_records = [
            _build_alignment_verification_record(record)
            for record in verification_records
        ]
        payload = {
            "report_type": "forced_alignment_verification",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "threshold": threshold,
            "language": language,
            "summary": summary,
            "records": compact_records,
        }
        try:
            log_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            summary["verification_log_path"] = str(log_path)
            logger.info("Forced-alignment report written: %s", log_path)
        except OSError as exc:
            summary["verification_log_error"] = str(exc)
            logger.warning("Could not write forced-alignment report %s: %s", log_path, exc)

    def _save_investigation_log(self, investigation_log: List[Dict], dataset_paths: Dict):
        """Save failure-only investigation log for chunks that still failed threshold."""
        log_path = Path(dataset_paths['tts_dir']) / "asr_investigation.log"

        final_failures = []
        for entry in investigation_log:
            best_attempt = entry.get("best_attempt", {}) or {}
            best_result = best_attempt.get("asr_result", {}) or {}
            # ``passed`` includes policy failures such as protected-token or
            # content mismatches. Those can have a numeric score above threshold.
            if not bool(best_result.get("passed", False)):
                final_failures.append(entry)

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("ASR Final Failure Investigation Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Final failed chunks: {len(final_failures)}\n\n")

            for entry in final_failures:
                chunk_id = f"chunk_{entry['chunk_index']:05d}"
                best_attempt = entry['best_attempt']
                asr_result = best_attempt.get('asr_result', {})
                original_asr = entry['original_asr']
                best_attempt_id = (best_attempt.get('attempt'), best_attempt.get('is_original', False))

                f.write(f"CHUNK: {chunk_id}\n")
                best_passed = bool(best_attempt.get('asr_result', {}).get('passed', False))
                status = "PASSED" if best_passed else "FAILED"
                f.write(f"Status: {status} (Score: {entry['best_score']:.3f})\n")
                if best_attempt.get('is_original'):
                    f.write(f"Best available: Original (Score: {entry['best_score']:.3f})\n")
                else:
                    f.write(f"Best regeneration attempt: {best_attempt['attempt']} (Temp: {best_attempt['temp']:.2f})\n")
                original_text = original_asr.get('original_text') or entry.get('text', 'N/A')
                f.write(f"Original Text: {original_text}\n")
                f.write(f"Transcribed Text: {asr_result.get('hyp_text_raw', asr_result.get('transcribed_text', 'N/A'))}\n")
                f.write(f"Explanation: {asr_result.get('explanation', asr_result.get('explanation', 'N/A'))}\n")
                f.write("-" * 40 + "\n")

                f.write(f"Prose Score: {asr_result.get('prose_score', 0):.3f}, ")
                f.write(f"ID Score: {asr_result.get('id_score', 0):.3f}, ")
                f.write(f"Combined: {asr_result.get('score', 0):.3f}\n")

                if asr_result.get('hallucination_warning'):
                    f.write(f"Hallucination: {asr_result['hallucination_warning']}\n")
                else:
                    f.write("Hallucination: None detected\n")

                if asr_result.get('truncation_warning'):
                    f.write(f"Truncation: {asr_result['truncation_warning']}\n")
                else:
                    f.write("Truncation: None detected\n")

                f.write("\nRegeneration Attempts:\n")
                for attempt in entry['all_attempts']:
                    attempt_id = (attempt.get('attempt'), attempt.get('is_original', False))
                    if attempt.get('is_original'):
                        saved_marker = " (SAVED - Best available)" if attempt_id == best_attempt_id else ""
                        label = "ORIGINAL"
                    else:
                        status = "✓" if attempt.get('asr_result', {}).get('passed', False) else "✗"
                        saved_marker = " (SAVED - Best available)" if attempt_id == best_attempt_id else ""
                        label = f"{status} Attempt {attempt['attempt']} Temp={attempt['temp']:.2f}"
                    attempt_asr = attempt.get('asr_result', {})
                    f.write(f"  {label}: Score={attempt['score']:.3f}{saved_marker}\n")
                    f.write(f"    Original: {entry['text']}\n")
                    f.write(
                        "    Transcript: "
                        f"{attempt_asr.get('transcribed_text', attempt_asr.get('hyp_text_raw', 'N/A'))}\n"
                    )
                    if attempt_asr.get('explanation'):
                        f.write(f"    Explanation: {attempt_asr['explanation']}\n")
                    if attempt_asr.get('error'):
                        f.write(f"    Error: {attempt_asr['error']}\n")
                    if not attempt.get('is_original'):
                        f.write(
                            f"    EOS detected: {attempt.get('eos_detected', 'unknown')}\n"
                        )
                    if attempt.get('is_original'):
                        continue

                f.write("\n" + "=" * 50 + "\n\n")

        logger.info(f"ASR investigation log saved: {log_path}")


def _ffmpeg_atempo_standalone(tts_model_or_rate, audio, speed_factor: float):
    """Module-level atempo helper for parallel workers (no self reference).

    Args:
        tts_model_or_rate: TTSModel (sample_rate attribute) or int sample rate.
        audio: Torch tensor of audio samples
        speed_factor: Playback speed multiplier

    Returns:
        Stretched audio tensor on the same device as the input audio.
    """
    import subprocess
    import tempfile
    import torch
    import numpy as np
    import scipy.io.wavfile
    from ..data.audio import audio_read

    if isinstance(tts_model_or_rate, (int, float)):
        sample_rate = int(tts_model_or_rate)
    else:
        sample_rate = getattr(tts_model_or_rate, "sample_rate", 24000)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_wav = os.path.join(tmpdir, "input.wav")
        output_wav = os.path.join(tmpdir, "output.wav")

        # Save audio to temp WAV (float32 format)
        audio_np = audio.detach().float().cpu().numpy()
        scipy.io.wavfile.write(input_wav, sample_rate, audio_np.astype(np.float32))

        # Run ffmpeg with atempo filter. POCKET_TTS_FFMPEG_PATH is set by
        # the Windows launcher when a private (non-PATH) copy was
        # downloaded, since ffmpeg is never bundled or on PATH by default.
        cmd = [
            os.environ.get("POCKET_TTS_FFMPEG_PATH", "ffmpeg"),
            "-y", "-i", input_wav,
            "-filter:a", f"atempo={speed_factor}",
            "-ar", str(sample_rate),
            "-f", "wav", output_wav
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        if result.returncode != 0:
            logger.warning(f"ffmpeg atempo failed: {result.stderr}")
            return audio  # Return original on failure

        # Read result back (audio_read returns torch.Tensor, not numpy)
        stretched, _ = audio_read(output_wav)
        stretched = stretched.squeeze().to(device=audio.device, dtype=audio.dtype)

    return stretched


def _validate_chunk_asr_worker(
    asr_exe: str,
    chunk_num: str,
    tts_dir: str,
    threshold: float,
    audio_path: str,
    model_name: str = "base",
    language: str = "en",
) -> Dict[str, Any]:
    """Validate regeneration audio from a worker through the ASR subprocess.

    Args:
        asr_exe: Path to ASR Python executable.
        chunk_num: Chunk stem like chunk_00005.
        tts_dir: Parent TTS directory.
        threshold: Similarity pass threshold.
        audio_path: WAV path to validate.
        model_name: faster-whisper model size.
        language: Whisper language code to pin during decode.
    """
    import json
    import subprocess
    import traceback

    project_root = Path(__file__).parent.parent.parent
    asr_script = project_root / "ASR" / "asr_validator.py"
    command = [
        asr_exe,
        str(asr_script.resolve()),
        "--single-chunk",
        chunk_num,
        "--tts-dir",
        tts_dir,
        "--threshold",
        str(threshold),
        "--model",
        model_name,
        "--language",
        language or "en",
        "--audio-path",
        audio_path,
        "--json",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
        )
        try:
            return _parse_asr_json_output(result.stdout)
        except (TypeError, ValueError) as exc:
            return {
                "score": 0.0,
                "passed": False,
                "error": result.stderr.strip() or f"ASR validation failed: {exc}",
            }
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "passed": False, "error": "Validation timeout"}
    except Exception as exc:
        return {"score": 0.0, "passed": False, "error": str(exc)}


def _write_worker_audio(audio: Any, output_path: Path, sample_rate: int) -> None:
    """Write worker-generated tensor audio as a normalized 16-bit WAV file."""
    import numpy as np
    import scipy.io.wavfile

    audio_np = audio.detach().cpu().numpy()
    audio_np = np.asarray(audio_np).squeeze().clip(-1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    scipy.io.wavfile.write(str(output_path), sample_rate, audio_int16)


def _cuda_memory_snapshot(stage: str) -> Dict[str, Any]:
    """Capture process and device CUDA memory at one worker lifecycle stage."""
    snapshot: Dict[str, Any] = {"memory_stage": stage, "cuda_available": False}
    try:
        import torch

        if not torch.cuda.is_available():
            return snapshot

        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        snapshot.update(
            {
                "cuda_available": True,
                "memory_allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 3),
                "memory_reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 3),
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 3),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 3),
                "device_free_mb": round(free_bytes / (1024 ** 2), 3),
                "device_total_mb": round(total_bytes / (1024 ** 2), 3),
            }
        )
    except Exception as exc:
        snapshot["memory_error"] = str(exc)
    return snapshot


def _record_worker_event(
    timing_dir: str | None,
    worker_id: int,
    event: str,
    **details: Any,
) -> None:
    """Append one structured worker lifecycle event to its JSONL file."""
    if not timing_dir:
        return
    import json
    import os
    from datetime import datetime, timezone

    path = Path(timing_dir) / f"worker_{worker_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monotonic": time.monotonic(),
        "worker_id": worker_id,
        "pid": os.getpid(),
        "event": event,
        **details,
    }
    with path.open("a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _cuda_graph_stats_payload(tts_model: Any) -> Dict[str, Any]:
    """Flatten TTSModel CUDA graph counters for worker JSONL / parent summary."""
    if tts_model is None or not hasattr(tts_model, "get_cuda_graph_stats"):
        return {
            "cuda_graphs_enabled": False,
            "cuda_graph_captures_ok": 0,
            "cuda_graph_captures_fail": 0,
            "cuda_graph_reuses": 0,
            "cuda_graph_prepares": 0,
            "cuda_graph_registry_size": 0,
            "cuda_graph_steps_graph": 0,
            "cuda_graph_steps_eager": 0,
            "cuda_graph_steps_total": 0,
            "cuda_graph_steps_graph_pct": 0.0,
            "cuda_graph_last_capture_ok": None,
            "cuda_graph_last_capture_error": None,
            "cuda_graph_last_batch_size": None,
            "mimi_graph_captures_ok": 0,
            "mimi_graph_captures_fail": 0,
            "mimi_graph_steps_graph": 0,
            "mimi_graph_steps_eager": 0,
            "mimi_graph_steps_total": 0,
            "mimi_graph_steps_graph_pct": 0.0,
            "mimi_graph_registry_size": 0,
        }
    stats = tts_model.get_cuda_graph_stats()
    return {
        "cuda_graphs_enabled": bool(stats.get("enabled", False)),
        "cuda_graph_captures_ok": int(stats.get("captures_ok", 0)),
        "cuda_graph_captures_fail": int(stats.get("captures_fail", 0)),
        "cuda_graph_reuses": int(stats.get("reuses", 0)),
        "cuda_graph_prepares": int(stats.get("prepares", 0)),
        "cuda_graph_registry_size": int(stats.get("registry_size", 0)),
        "cuda_graph_steps_graph": int(stats.get("steps_graph", 0)),
        "cuda_graph_steps_eager": int(stats.get("steps_eager", 0)),
        "cuda_graph_steps_total": int(stats.get("steps_total", 0)),
        "cuda_graph_steps_graph_pct": float(stats.get("steps_graph_pct", 0.0)),
        "cuda_graph_last_capture_ok": stats.get("last_capture_ok"),
        "cuda_graph_last_capture_error": stats.get("last_capture_error"),
        "cuda_graph_last_batch_size": stats.get("last_batch_size"),
        "mimi_graph_captures_ok": int(stats.get("mimi_captures_ok", 0)),
        "mimi_graph_captures_fail": int(stats.get("mimi_captures_fail", 0)),
        "mimi_graph_steps_graph": int(stats.get("mimi_steps_graph", 0)),
        "mimi_graph_steps_eager": int(stats.get("mimi_steps_eager", 0)),
        "mimi_graph_steps_total": int(stats.get("mimi_steps_total", 0)),
        "mimi_graph_steps_graph_pct": float(stats.get("mimi_steps_graph_pct", 0.0)),
        "mimi_graph_registry_size": int(stats.get("mimi_registry_size", 0)),
    }


def _summarize_cuda_graph_telemetry(timing_dir: Path) -> Dict[str, Any]:
    """Aggregate CUDA graph fields from worker_exiting (or last chunk) events."""
    import json

    summary = {
        "workers": 0,
        "cuda_graphs_enabled_any": False,
        "captures_ok": 0,
        "captures_fail": 0,
        "reuses": 0,
        "prepares": 0,
        "steps_graph": 0,
        "steps_eager": 0,
        "steps_total": 0,
        "steps_graph_pct": 0.0,
        "mimi_captures_ok": 0,
        "mimi_captures_fail": 0,
        "mimi_steps_graph": 0,
        "mimi_steps_eager": 0,
        "mimi_steps_total": 0,
        "mimi_steps_graph_pct": 0.0,
        "per_worker": {},
    }
    if not timing_dir.is_dir():
        return summary

    for path in sorted(timing_dir.glob("worker_*.jsonl")):
        last_stats = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") in ("worker_exiting", "cuda_graph_snapshot", "chunk_completed"):
                if "cuda_graph_steps_total" in event or "cuda_graphs_enabled" in event:
                    last_stats = event
        if not last_stats:
            continue
        summary["workers"] += 1
        enabled = bool(last_stats.get("cuda_graphs_enabled", False))
        summary["cuda_graphs_enabled_any"] = summary["cuda_graphs_enabled_any"] or enabled
        ok = int(last_stats.get("cuda_graph_captures_ok", 0) or 0)
        fail = int(last_stats.get("cuda_graph_captures_fail", 0) or 0)
        reuses = int(last_stats.get("cuda_graph_reuses", 0) or 0)
        prepares = int(last_stats.get("cuda_graph_prepares", 0) or 0)
        sg = int(last_stats.get("cuda_graph_steps_graph", 0) or 0)
        se = int(last_stats.get("cuda_graph_steps_eager", 0) or 0)
        mimi_ok = int(last_stats.get("mimi_graph_captures_ok", 0) or 0)
        mimi_fail = int(last_stats.get("mimi_graph_captures_fail", 0) or 0)
        mimi_sg = int(last_stats.get("mimi_graph_steps_graph", 0) or 0)
        mimi_se = int(last_stats.get("mimi_graph_steps_eager", 0) or 0)
        summary["captures_ok"] += ok
        summary["captures_fail"] += fail
        summary["reuses"] += reuses
        summary["prepares"] += prepares
        summary["steps_graph"] += sg
        summary["steps_eager"] += se
        summary["mimi_captures_ok"] += mimi_ok
        summary["mimi_captures_fail"] += mimi_fail
        summary["mimi_steps_graph"] += mimi_sg
        summary["mimi_steps_eager"] += mimi_se
        summary["per_worker"][path.stem] = {
            "enabled": enabled,
            "captures_ok": ok,
            "captures_fail": fail,
            "reuses": reuses,
            "prepares": prepares,
            "steps_graph": sg,
            "steps_eager": se,
            "mimi_captures_ok": mimi_ok,
            "mimi_captures_fail": mimi_fail,
            "mimi_steps_graph": mimi_sg,
            "mimi_steps_eager": mimi_se,
            "last_capture_ok": last_stats.get("cuda_graph_last_capture_ok"),
            "last_capture_error": last_stats.get("cuda_graph_last_capture_error"),
            "last_batch_size": last_stats.get("cuda_graph_last_batch_size"),
        }
    summary["steps_total"] = summary["steps_graph"] + summary["steps_eager"]
    if summary["steps_total"] > 0:
        summary["steps_graph_pct"] = (
            100.0 * summary["steps_graph"] / summary["steps_total"]
        )
    summary["mimi_steps_total"] = summary["mimi_steps_graph"] + summary["mimi_steps_eager"]
    if summary["mimi_steps_total"] > 0:
        summary["mimi_steps_graph_pct"] = (
            100.0 * summary["mimi_steps_graph"] / summary["mimi_steps_total"]
        )
    return summary


def _regenerate_chunk_in_worker(
    tts_model: Any,
    voice_state: Any,
    task: Dict[str, Any],
    worker_logger: logging.Logger,
    attempt_callback: Any = None,
) -> Dict[str, Any]:
    """Regenerate one failed chunk and select its highest-scoring attempt."""
    import traceback
    import torch

    chunk_index = int(task["chunk_index"])
    chunk_num = f"{chunk_index:05d}"
    chunk_filename = f"chunk_{chunk_num}.wav"
    audio_chunks_dir = Path(task["audio_chunks_dir"])
    failed_dir = audio_chunks_dir / "Failed"
    failed_dir.mkdir(exist_ok=True)
    original_chunk_path = audio_chunks_dir / chunk_filename
    threshold = float(task["threshold"])
    current_temp = float(task["tts_params"].get("temperature", 0.7))
    frames_after_eos = int(task["tts_params"].get("frames_after_eos", 2))
    max_retries = int(task["max_retries"])
    temp_decrement = float(task["temp_decrement"])
    # first_pass: stop after first regen that passes threshold (speed).
    # best_of_n: always run max_retries and pick best (legacy).
    regen_strategy = str(task.get("regeneration_strategy", "first_pass")).strip().lower()
    if regen_strategy not in ("first_pass", "best_of_n"):
        regen_strategy = "first_pass"
    sample_rate = getattr(tts_model, "sample_rate", 24000)
    candidates = [{
        "attempt": 0,
        "temp": None,
        "score": float(task.get("original_score", 0.0)),
        "audio_path": str(original_chunk_path),
        "asr_result": task.get("original_asr", {}),
        "is_original": True,
    }]

    try:
        for retry in range(max_retries):
            retry_temp = max(0.1, current_temp - (retry * temp_decrement))
            temp_path = audio_chunks_dir / f"chunk_{chunk_num}_attempt_{retry + 1}.wav"
            worker_logger.info(
                "Worker %s: Regenerating chunk %s attempt %s/%s with temp=%.2f strategy=%s",
                task.get("worker_id", "?"),
                chunk_num,
                retry + 1,
                max_retries,
                retry_temp,
                regen_strategy,
            )

            if hasattr(tts_model, "temp"):
                tts_model.temp = retry_temp
            cleanup_config = task["cleanup_config"]
            from pocket_tts.preprocessing.pause_injector import (
                generate_audio_with_pauses,
                has_inline_pause_markers,
            )
            from pocket_tts.preprocessing.text_normalizer import flatten_newlines_for_json

            plan_text = flatten_newlines_for_json(task["text"])
            if has_inline_pause_markers(plan_text):
                def render_segment(text: str):
                    """Generate and clean one regeneration pause-plan segment."""
                    segment = tts_model.generate_audio(
                        voice_state, text, frames_after_eos=frames_after_eos
                    )
                    return _postprocess_inline_speech_segment(
                        segment,
                        {"speed_factor": 1.0},
                        sample_rate,
                        cleanup_config.get("enabled", True),
                        cleanup_config.get("use_silero_vad", True),
                        cleanup_config.get("speech_endpoint_threshold", 0.004),
                        cleanup_config.get("trimming_buffer_ms", 100),
                    )

                audio, _pause_events = generate_audio_with_pauses(
                    tts_model, voice_state, plan_text, generate_segment=render_segment
                )
            else:
                audio = tts_model.generate_audio(
                    voice_state, plan_text, frames_after_eos=frames_after_eos
                )
                audio = _postprocess_inline_speech_segment(
                    audio,
                    {"speed_factor": 1.0},
                    sample_rate,
                    cleanup_config.get("enabled", True),
                    cleanup_config.get("use_silero_vad", True),
                    cleanup_config.get("speech_endpoint_threshold", 0.004),
                    cleanup_config.get("trimming_buffer_ms", 100),
                )

            silence_duration = float(task["post_process"].get("silence_duration", 0.0))
            if silence_duration > 0:
                silence_samples = int(silence_duration * sample_rate)
                if audio.ndim == 1:
                    silence = torch.zeros(silence_samples, device=audio.device)
                    audio = torch.cat([audio, silence], dim=0)
                else:
                    silence = torch.zeros(audio.shape[0], silence_samples, device=audio.device)
                    audio = torch.cat([audio, silence], dim=1)

            _write_worker_audio(audio, temp_path, sample_rate)
            asr_result = _validate_chunk_asr_worker(
                task["asr_exe"],
                f"chunk_{chunk_num}",
                task["tts_dir"],
                threshold,
                str(temp_path),
                task.get("asr_model", "base"),
                task.get("asr_language", "en"),
            )
            candidates.append({
                "attempt": retry + 1,
                "temp": retry_temp,
                "score": float(asr_result.get("score", 0.0)),
                "audio_path": str(temp_path),
                "asr_result": asr_result,
                "is_original": False,
                "eos_detected": not getattr(tts_model, "last_generation_reached_limit", False),
                "eos_step": getattr(tts_model, "last_generation_eos_step", None),
                "max_generation_length": getattr(tts_model, "last_generation_max_len", None),
            })
            if attempt_callback:
                attempt_callback({
                    "chunk_num": chunk_num,
                    "attempt": retry + 1,
                    "max_retries": max_retries,
                    "temperature": retry_temp,
                    "score": float(asr_result.get("score", 0.0)),
                    "passed": bool(asr_result.get("passed", False)),
                    "original_text": task["text"],
                    "transcribed_text": asr_result.get("transcribed_text", asr_result.get("hyp_text_raw", "")),
                    "explanation": asr_result.get("explanation", ""),
                    "error": asr_result.get("error", ""),
                    "eos_detected": not getattr(tts_model, "last_generation_reached_limit", False),
                    "eos_step": getattr(tts_model, "last_generation_eos_step", None),
                    "max_generation_length": getattr(tts_model, "last_generation_max_len", None),
                    "regeneration_strategy": regen_strategy,
                })
            # first_pass: accept first regen that meets ASR threshold; skip remaining.
            if regen_strategy == "first_pass" and bool(asr_result.get("passed", False)):
                worker_logger.info(
                    "Worker %s: first_pass accept chunk %s attempt %s score=%.3f",
                    task.get("worker_id", "?"),
                    chunk_num,
                    retry + 1,
                    float(asr_result.get("score", 0.0)),
                )
                break

        winner = max(
            candidates,
            key=lambda candidate: (
                bool(candidate.get("asr_result", {}).get("passed", False)),
                candidate["score"],
            ),
        )
        if not winner.get("is_original"):
            if original_chunk_path.exists():
                original_chunk_path.rename(failed_dir / chunk_filename)
            Path(winner["audio_path"]).replace(original_chunk_path)

        for candidate in candidates:
            if candidate is winner or candidate.get("is_original"):
                continue
            attempt_path = Path(candidate["audio_path"])
            if attempt_path.exists():
                attempt_path.replace(
                    failed_dir / f"chunk_{chunk_num}_attempt_{candidate['attempt']}.wav"
                )

        best_score = float(winner["score"])
        worker_logger.info(
            "Worker %s: Regeneration complete for chunk %s, best score %.3f",
            task.get("worker_id", "?"),
            chunk_num,
            best_score,
        )
        investigation = {
            "chunk_index": chunk_index,
            "chunk_num": chunk_num,
            "best_score": best_score,
            "threshold": threshold,
            "best_attempt": winner,
            "all_attempts": candidates,
            "text": task["text"],
            "original_asr": task.get("original_asr", {}),
        }
        return {"chunk_index": chunk_index, "investigation": investigation}
    except Exception as exc:
        worker_logger.error("Worker %s: regeneration failed for chunk %s: %s", task.get("worker_id", "?"), chunk_num, exc)
        worker_logger.error(traceback.format_exc())
        return {"chunk_index": chunk_index, "error": str(exc), "investigation": None}


def _chunk_tts_params(chunk: ChunkMetadata | SimpleNamespace) -> Dict[str, Any]:
    """Return TTS settings from real or staged worker chunk metadata."""
    if dataclasses.is_dataclass(chunk.tts_params):
        return dataclasses.asdict(chunk.tts_params)
    return dict(chunk.tts_params)


def _postprocess_inline_speech_segment(
    audio: Any,
    params: Dict[str, Any],
    sample_rate: int,
    cleanup_enabled: bool,
    use_silero_vad: bool,
    endpoint_threshold: float,
    trimming_buffer_ms: int,
) -> Any:
    """Apply speed and endpoint cleanup before a pause-plan segment is joined.

    Digital pause tensors are added only after this function returns, ensuring
    neither atempo nor endpoint cleanup can rescale or remove requested silence.
    """
    speed_factor = float(params.get("speed_factor", 1.0))
    if speed_factor != 1.0:
        audio = _ffmpeg_atempo_standalone(sample_rate, audio, speed_factor)
    if cleanup_enabled:
        from pocket_tts.audio_processing.endpoint_cleanup import cleanup_audio_endpoint

        audio = cleanup_audio_endpoint(
            audio,
            sample_rate,
            threshold=endpoint_threshold,
            buffer_ms=trimming_buffer_ms,
            use_silero=use_silero_vad,
        )
    return audio


def _generate_batch_chunk_audio(
    tts_model: Any,
    voice_state: Any,
    batch_items: List[tuple[int, ChunkMetadata | SimpleNamespace]],
    batch_size: int,
    pause_injection_enabled: bool,
    pause_durations: Dict[str, float],
    worker_logger: logging.Logger,
    cleanup_enabled: bool = True,
    use_silero_vad: bool = True,
    endpoint_threshold: float = 0.004,
    trimming_buffer_ms: int = 100,
) -> Dict[int, Any]:
    """Generate chunk text events in batches and reassemble complete segments."""
    from pocket_tts.preprocessing.pause_injector import (
        has_inline_pause_markers,
        inject_pauses_for_punctuation,
        is_speech_text_segment,
        parse_text_with_pauses,
        render_text_with_native_pauses,
    )

    from pocket_tts.preprocessing.text_normalizer import (
        build_tts_prompt,
        normalize_text_chunk_for_storage,
    )

    units = []
    chunk_events = {}
    chunks_by_index = {}
    tokenizer = tts_model.flow_lm.conditioner.tokenizer
    for global_index, chunk in batch_items:
        # Shared normalizer preserves manual marker boundaries without commas.
        raw_text = normalize_text_chunk_for_storage(chunk.text or "")
        if pause_injection_enabled and pause_durations:
            raw_text = inject_pauses_for_punctuation(raw_text, pause_durations)
            raw_text = normalize_text_chunk_for_storage(raw_text)
        chunk.text = raw_text
        chunks_by_index[global_index] = chunk
        if has_inline_pause_markers(raw_text):
            events, _ = parse_text_with_pauses(raw_text)
            spoken = render_text_with_native_pauses(raw_text)
            worker_logger.info(
                "Chunk %s split pause plan: %r",
                global_index,
                spoken[:160],
            )
            chunk._inline_pause_text = raw_text
            chunk._spoken_text = spoken
            chunk._inline_pause_preprocessed = True
        else:
            chunk._inline_pause_text = None
            chunk._spoken_text = raw_text
            chunk._inline_pause_preprocessed = False
            events = [("text", raw_text)]

        chunk_events[global_index] = []
        params = _chunk_tts_params(chunk)
        for kind, value in events:
            if kind == "pause":
                chunk_events[global_index].append(("pause", float(value)))
                continue
            if not is_speech_text_segment(str(value).strip()):
                # Preserve quote-only source text in the plan/sidecar, but do
                # not turn it into a context-free TTS request.
                continue
            text = build_tts_prompt(str(value).strip())
            if not text:
                continue
            unit_id = len(units)
            request = {
                "text": text,
                "temperature": float(params.get("temperature", 0.7)),
                "frames_after_eos": int(params.get("frames_after_eos", 2)),
                "eos_threshold": float(params.get("eos_threshold", -4.0)),
                "lsd_decode_steps": int(params.get("lsd_decode_steps", 1)),
            }
            units.append({
                "id": unit_id,
                "request": request,
                "params": params,
                "inline": bool(getattr(chunk, "_inline_pause_preprocessed", False)),
                "token_length": int(tokenizer(text).tokens.shape[-1]),
            })
            chunk_events[global_index].append(("audio", {"id": unit_id, "text": text}))

    generated = {}
    grouped = {}
    for unit in units:
        request = unit["request"]
        key = (
            unit["token_length"],
            request["frames_after_eos"],
            request["eos_threshold"],
            request["lsd_decode_steps"],
        )
        grouped.setdefault(key, []).append(unit)

    for grouped_units in grouped.values():
        for start in range(0, len(grouped_units), batch_size):
            batch = grouped_units[start : start + batch_size]
            requests = [unit["request"] for unit in batch]
            try:
                outputs = tts_model.generate_audio_batch(voice_state, requests)
            except Exception as batch_error:
                worker_logger.warning("Batch generation failed; using scalar fallback: %s", batch_error)
                outputs = []
                for request in requests:
                    tts_model.temp = request["temperature"]
                    tts_model.eos_threshold = request["eos_threshold"]
                    tts_model.lsd_decode_steps = request["lsd_decode_steps"]
                    outputs.append(
                        tts_model.generate_audio(
                            voice_state,
                            request["text"],
                            frames_after_eos=request["frames_after_eos"],
                        )
                    )
            for unit, audio in zip(batch, outputs):
                if unit["inline"]:
                    audio = _postprocess_inline_speech_segment(
                        audio,
                        unit["params"],
                        getattr(tts_model, "sample_rate", 24000),
                        cleanup_enabled,
                        use_silero_vad,
                        endpoint_threshold,
                        trimming_buffer_ms,
                    )
                generated[unit["id"]] = audio

    sample_rate = getattr(tts_model, "sample_rate", 24000)
    result = {}
    for global_index, events in chunk_events.items():
        chunk = chunks_by_index[global_index]
        reference_audio = next(
            (generated[value["id"]] for kind, value in events if kind == "audio"),
            None,
        )
        pieces = []
        pause_events = []
        for kind, value in events:
            if kind == "audio":
                audio = generated[value["id"]]
                pieces.append(audio)
                if getattr(chunk, "_inline_pause_preprocessed", False):
                    pause_events.append({
                        "kind": "text",
                        "text": value["text"],
                        "samples": int(audio.shape[-1]),
                    })
            else:
                samples = max(0, int(round(float(value) * sample_rate)))
                if reference_audio is not None:
                    pieces.append(reference_audio.new_zeros((*reference_audio.shape[:-1], samples)))
                pause_events.append({"kind": "pause", "seconds": float(value), "samples": samples})
        if reference_audio is None:
            total_pause_samples = sum(
                int(record["samples"])
                for record in pause_events
                if record["kind"] == "pause"
            )
            result[global_index] = torch.zeros(total_pause_samples, device=tts_model.device)
        elif pieces:
            result[global_index] = torch.cat(pieces, dim=-1)
        else:
            result[global_index] = torch.zeros(0, device=tts_model.device)
        if getattr(chunk, "_inline_pause_preprocessed", False):
            chunk.pause_events = pause_events
            chunk._pause_events = pause_events
    return result


def _finalize_and_save_worker_audio(
    audio: Any,
    chunk: Any,
    global_index: int,
    tts_model: Any,
    output_dir: str,
    cleanup_enabled: bool,
    use_silero_vad: bool,
    endpoint_threshold: float,
    trimming_buffer_ms: int,
    sample_rate: int | None = None,
) -> tuple[str, float, float, float]:
    """Apply chunk postprocessing and persist WAV plus ASR text sidecar.

    Safe to call from a CPU postprocess thread when audio is already on CPU.
    Prefers explicit sample_rate so async workers need not touch the TTS model.
    """
    rate = int(
        sample_rate
        if sample_rate is not None
        else getattr(tts_model, "sample_rate", 24000)
    )
    params = _chunk_tts_params(chunk)
    cleanup_start = time.monotonic()
    inline_preprocessed = bool(getattr(chunk, "_inline_pause_preprocessed", False))
    if not inline_preprocessed:
        speed_factor = float(params.get("speed_factor", 1.0))
        if speed_factor != 1.0:
            # Prefer int rate so post thread does not touch the live CUDA model.
            audio = _ffmpeg_atempo_standalone(rate, audio, speed_factor)
        if cleanup_enabled:
            from pocket_tts.audio_processing.endpoint_cleanup import cleanup_audio_endpoint

            audio = cleanup_audio_endpoint(
                audio,
                rate,
                threshold=endpoint_threshold,
                buffer_ms=trimming_buffer_ms,
                use_silero=use_silero_vad,
            )
    cleanup_time = time.monotonic() - cleanup_start

    inline_pause_text = getattr(chunk, "_inline_pause_text", None)
    pause_events = list(getattr(chunk, "_pause_events", []) or [])

    silence_start = time.monotonic()
    silence_duration_sec = chunk.post_process.get("silence_duration", 0.0)
    if silence_duration_sec > 0:
        num_silence_samples = int(silence_duration_sec * rate)
        if audio.ndim == 1:
            silence_tensor = torch.zeros(num_silence_samples, device=audio.device)
            audio = torch.cat([audio, silence_tensor], dim=0)
        else:
            silence_tensor = torch.zeros(audio.shape[0], num_silence_samples, device=audio.device)
            audio = torch.cat([audio, silence_tensor], dim=1)
    silence_time = time.monotonic() - silence_start

    io_start = time.time()
    chunk_filename = f"chunk_{global_index:05d}.wav"
    chunk_path = Path(output_dir) / chunk_filename
    text_chunks_dir = Path(output_dir).parent / "text_chunks"
    text_chunks_dir.mkdir(parents=True, exist_ok=True)
    stem = chunk_filename.replace(".wav", "")
    text_file = text_chunks_dir / f"{stem}.txt"
    plan_text = str(getattr(chunk, "text", "") or "")
    spoken_text = str(getattr(chunk, "_spoken_text", "") or "")
    with open(text_file, "w", encoding="utf-8") as text_handle:
        text_handle.write(plan_text)
    # Proof sidecars record complete text segments and exact zero-sample events.
    if inline_pause_text or pause_events:
        (text_chunks_dir / f"{stem}.pause_plan.txt").write_text(
            plan_text, encoding="utf-8"
        )
        (text_chunks_dir / f"{stem}.spoken.txt").write_text(
            spoken_text, encoding="utf-8"
        )
        import json as _json

        payload = {
            "chunk_index": global_index,
            "plan_text": plan_text,
            "spoken_text": spoken_text,
            "sample_rate": rate,
            "events": pause_events,
        }
        (text_chunks_dir / f"{stem}.pauses.json").write_text(
            _json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    import numpy as _np
    import scipy.io.wavfile
    audio_np = audio.detach().float().cpu().numpy().clip(-1.0, 1.0)
    audio_int16 = (_np.array(audio_np) * 32767).astype(_np.int16)
    scipy.io.wavfile.write(str(chunk_path), rate, audio_int16)
    io_time = time.time() - io_start
    return str(chunk_path), cleanup_time, silence_time, io_time


_POSTPROCESS_SENTINEL = object()


class _WorkerPostprocessPool:
    """One CPU thread per TTS worker: cleanup + WAV while gen starts next batch.

    Backpressure: submit blocks when max_pending jobs are already queued/running
    so RAM cannot grow without bound under a slow Silero path.
    """

    def __init__(
        self,
        worker_id: int,
        result_queue: Any,
        timing_dir: str | None,
        output_dir: str,
        sample_rate: int,
        cleanup_enabled: bool,
        use_silero_vad: bool,
        endpoint_threshold: float,
        trimming_buffer_ms: int,
        max_pending: int,
        worker_logger: logging.Logger,
    ):
        """Start the dedicated postprocess thread for this worker process."""
        self.worker_id = worker_id
        self.result_queue = result_queue
        self.timing_dir = timing_dir
        self.output_dir = output_dir
        self.sample_rate = int(sample_rate)
        self.cleanup_enabled = cleanup_enabled
        self.use_silero_vad = use_silero_vad
        self.endpoint_threshold = endpoint_threshold
        self.trimming_buffer_ms = trimming_buffer_ms
        self.worker_logger = worker_logger
        self.max_pending = max(1, int(max_pending))
        self._queue: thread_queue.Queue = thread_queue.Queue(maxsize=self.max_pending)
        self.completed = 0
        self.failed = 0
        self.total_io_time = 0.0
        self.total_cleanup_time = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name=f"tts-post-{worker_id}",
            daemon=True,
        )
        self._thread.start()
        worker_logger.info(
            "Worker %s: async postprocess thread started (max_pending=%s)",
            worker_id,
            self.max_pending,
        )

    def submit(
        self,
        audio: Any,
        chunk: Any,
        global_index: int,
        generation_ms: float,
        batch_size: int | None = None,
        graph_payload: dict | None = None,
        phase: str = "original",
    ) -> None:
        """Enqueue CPU postprocess for one chunk; blocks at max_pending.

        Args:
            audio: Audio tensor (will be moved to CPU here if still on GPU).
            chunk: Chunk metadata (text + post_process); ChunkMetadata or duck-type.
            global_index: Chunk index for filenames and parent result_queue.
            generation_ms: Already-measured gen time for telemetry.
            batch_size: Optional batch size tag for timing events.
            graph_payload: Snapshot of CUDA graph stats at submit time.
            phase: Timing phase label (usually original).
        """
        # Free GPU tensor before waiting on a full queue (backpressure).
        audio_cpu = audio.detach().float().cpu().contiguous()
        job = {
            "audio": audio_cpu,
            "chunk": chunk,
            "global_index": global_index,
            "generation_ms": float(generation_ms),
            "batch_size": batch_size,
            "graph_payload": dict(graph_payload or {}),
            "phase": phase,
            "submit_mono": time.monotonic(),
        }
        self._queue.put(job)

    def shutdown(self, wait: bool = True, timeout: float = 600.0) -> None:
        """Drain remaining jobs and stop the postprocess thread."""
        self._queue.put(_POSTPROCESS_SENTINEL)
        if wait:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.worker_logger.warning(
                    "Worker %s: postprocess thread still alive after %.0fs drain",
                    self.worker_id,
                    timeout,
                )

    def _run(self) -> None:
        """Consume postprocess jobs until sentinel; never touches the TTS model."""
        while True:
            job = self._queue.get()
            if job is _POSTPROCESS_SENTINEL:
                break
            global_index = job["global_index"]
            chunk = job["chunk"]
            chunk_start = time.monotonic()
            try:
                chunk_path, cleanup_time, silence_time, io_time = _finalize_and_save_worker_audio(
                    job["audio"],
                    chunk,
                    global_index,
                    tts_model=None,
                    output_dir=self.output_dir,
                    cleanup_enabled=self.cleanup_enabled,
                    use_silero_vad=self.use_silero_vad,
                    endpoint_threshold=self.endpoint_threshold,
                    trimming_buffer_ms=self.trimming_buffer_ms,
                    sample_rate=self.sample_rate,
                )
                self.completed += 1
                self.total_io_time += io_time
                self.total_cleanup_time += cleanup_time
                queue_wait_ms = max(0.0, (chunk_start - float(job["submit_mono"])) * 1000.0)
                event_kwargs = {
                    "phase": job.get("phase") or "original",
                    "chunk_index": global_index,
                    "generation_ms": round(float(job["generation_ms"]), 3),
                    "cleanup_ms": round(cleanup_time * 1000, 3),
                    "silence_ms": round(silence_time * 1000, 3),
                    "io_ms": round(io_time * 1000, 3),
                    "post_queue_wait_ms": round(queue_wait_ms, 3),
                    "async_postprocess": True,
                    "total_ms": round((time.monotonic() - chunk_start) * 1000, 3),
                }
                if job.get("batch_size") is not None:
                    event_kwargs["batch_size"] = int(job["batch_size"])
                event_kwargs.update(job.get("graph_payload") or {})
                try:
                    _record_worker_event(
                        self.timing_dir,
                        self.worker_id,
                        "chunk_completed",
                        **event_kwargs,
                    )
                except Exception as timing_exc:
                    # Never lose a successful WAV because telemetry failed.
                    self.worker_logger.warning(
                        "Worker %s: postprocess timing log failed chunk %s: %s",
                        self.worker_id,
                        global_index,
                        timing_exc,
                    )
                self.result_queue.put({
                    "kind": "original",
                    "saved_path": chunk_path,
                    "chunk_idx": global_index,
                    "chunk_text": chunk.text[:50],
                })
            except Exception as exc:
                self.failed += 1
                self.worker_logger.error(
                    "Worker %s: async postprocess FAILED chunk %s: %s",
                    self.worker_id,
                    global_index,
                    exc,
                )
                self.worker_logger.error(traceback.format_exc())
                self.result_queue.put({
                    "kind": "original",
                    "saved_path": None,
                    "chunk_idx": global_index,
                    "chunk_text": getattr(chunk, "text", "")[:50] if chunk else "",
                    "error": str(exc),
                })
            finally:
                try:
                    del job["audio"]
                except Exception:
                    pass


def _staged_regeneration_worker(
    task_queue: Any,
    result_queue: Any,
    ready_queue: Any,
    voice_path: str,
    worker_id: int,
    device: str,
    batch_size: int,
    cleanup_config: Dict[str, Any],
) -> None:
    """Generate three candidate rounds in batches without loading ASR."""
    import gc
    import logging
    import os

    worker_logger = logging.getLogger("pocket_tts.audiobook.generator")
    tts_model = None
    try:
        from ..models.tts_model import TTSModel

        worker_logger.info("Staged TTS worker %s loading model", worker_id)
        tts_model = TTSModel.load_model(device=device)
        voice_state = tts_model.get_state_for_audio_prompt(voice_path, truncate=True)
        sample_rate = int(getattr(tts_model, "sample_rate", 24000))
        ready_queue.put({"status": "ready", "pid": os.getpid(), "worker_id": worker_id})

        while True:
            batch = task_queue.get()
            if batch is None:
                break
            for attempt in range(1, 4):
                batch_items = []
                for row in batch:
                    candidate = next(item for item in row["candidates"] if item["attempt"] == attempt)
                    params = row.get("tts_params", {}) or {}
                    chunk = SimpleNamespace(
                        text=row["text"],
                        tts_params={
                            "temperature": candidate["temp"],
                            "frames_after_eos": int(params.get("frames_after_eos", 2) or 2),
                            "eos_threshold": float(params.get("eos_threshold", -4.0) or -4.0),
                            "lsd_decode_steps": int(params.get("lsd_decode_steps", 1) or 1),
                        },
                        post_process=row.get("post_process", {}) or {},
                        pause_events=None,
                    )
                    batch_items.append((row["chunk_index"], chunk))
                generated = _generate_batch_chunk_audio(
                    tts_model,
                    voice_state,
                    batch_items,
                    batch_size,
                    False,
                    {},
                    worker_logger,
                    cleanup_config.get("enabled", True),
                    cleanup_config.get("use_silero_vad", True),
                    cleanup_config.get("speech_endpoint_threshold", 0.004),
                    cleanup_config.get("trimming_buffer_ms", 100),
                )
                for row in batch:
                    candidate = next(item for item in row["candidates"] if item["attempt"] == attempt)
                    audio = generated[row["chunk_index"]]
                    from pocket_tts.preprocessing.pause_injector import has_inline_pause_markers

                    # Marker plans were cleaned segment-by-segment before assembly.
                    if cleanup_config.get("enabled", True) and not has_inline_pause_markers(row["text"]):
                        from ..audio_processing.endpoint_cleanup import cleanup_audio_endpoint

                        audio = cleanup_audio_endpoint(
                            audio,
                            sample_rate,
                            threshold=cleanup_config.get("speech_endpoint_threshold", 0.004),
                            buffer_ms=cleanup_config.get("trimming_buffer_ms", 100),
                            use_silero=cleanup_config.get("use_silero_vad", True),
                        )
                    silence_duration = float(
                        (row.get("post_process", {}) or {}).get("silence_duration", 0.0) or 0.0
                    )
                    if silence_duration > 0:
                        silence = torch.zeros(int(silence_duration * sample_rate), device=audio.device)
                        if audio.ndim == 1:
                            audio = torch.cat([audio, silence], dim=0)
                        else:
                            audio = torch.cat(
                                [audio, silence.expand(audio.shape[0], -1)], dim=1
                            )
                    _write_worker_audio(audio, Path(candidate["audio_path"]), sample_rate)
                    result_queue.put({
                        "kind": "candidate_ready",
                        "chunk_index": row["chunk_index"],
                        "attempt": attempt,
                        "eos_detected": not getattr(tts_model, "last_generation_reached_limit", False),
                        "eos_step": getattr(tts_model, "last_generation_eos_step", None),
                        "max_generation_length": getattr(tts_model, "last_generation_max_len", None),
                    })
    except Exception as exc:
        worker_logger.exception("Staged TTS worker %s failed", worker_id)
        try:
            ready_queue.put({"status": "error", "error": str(exc), "worker_id": worker_id})
        except Exception:
            pass
        result_queue.put({"kind": "error", "error": str(exc), "worker_id": worker_id})
    finally:
        voice_state = None
        tts_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


# Dynamic queue-based worker for parallel processing
def _queue_worker(chunk_queue: Any, voice_path: str, output_dir: str, result_queue: Any, worker_id: int,
                  pause_injection_enabled: bool = False, pause_durations: Dict[str, float] = None,
                  device: str = "auto", cleanup_enabled: bool = True,
                  use_silero_vad: bool = True, endpoint_threshold: float = 0.004,
                  trimming_buffer_ms: int = 100, timing_dir: str | None = None,
                  batch_size: int = 1, ready_queue: Any = None,
                  cleanup_async: bool = True, max_pending: int = 12):
    """
    Worker process with model pooling - loads TTS model and voice once, processes multiple chunks.
    Eliminates model/voice reload overhead for subsequent chunks in the same worker.

    Args:
        chunk_queue: Queue containing phase-tagged generation tasks or shutdown signals
        voice_path: Path to voice file
        output_dir: Directory to save audio chunks
        result_queue: Queue to send results back to main process
        worker_id: ID of this worker for logging
        pause_injection_enabled: Whether to enable pause injection at punctuation
        pause_durations: Dict mapping punctuation to pause duration in seconds
        cleanup_enabled: Whether to trim generated audio tails.
        use_silero_vad: Whether to combine Silero VAD with RMS analysis.
        endpoint_threshold: RMS threshold used for endpoint detection.
        trimming_buffer_ms: Audio retained after detected speech endpoint.
        timing_dir: Directory for per-worker JSONL timing events.
        batch_size: Maximum number of compatible text units generated together.
        ready_queue: Queue used to signal completed worker initialization.
        cleanup_async: If True, hand cleanup+WAV to a CPU thread after each gen.
        max_pending: Backpressure limit for in-flight async postprocess jobs.
    """
    if pause_durations is None:
        pause_durations = {}
    import logging
    import os
    from pathlib import Path
    import tempfile

    from pocket_tts.preprocessing.pause_injector import (
        has_inline_pause_markers,
    )

    worker_logger = logging.getLogger("pocket_tts.audiobook.generator")
    ready_sent = False
    post_pool: _WorkerPostprocessPool | None = None

    try:
        from ..models.tts_model import TTSModel
        import torch

        _record_worker_event(timing_dir, worker_id, "worker_started")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _record_worker_event(
            timing_dir,
            worker_id,
            "cuda_memory",
            **_cuda_memory_snapshot("process_start"),
        )
        first_generation_recorded = False

        # LOAD MODEL ONCE AT WORKER STARTUP (major optimization)
        worker_logger.info(f"Worker {worker_id}: Loading TTS model (once per worker)")
        model_load_start = time.time()
        tts_model = TTSModel.load_model(device=device)
        # Inherit CUDA graph AR flag from env (parent sets POCKET_TTS_CUDA_GRAPHS from config).
        if os.environ.get("POCKET_TTS_CUDA_GRAPHS", "0").strip().lower() in (
            "1", "true", "yes", "on",
        ):
            tts_model.cuda_graphs_enabled = True
            tts_model.reset_cuda_graph_stats()
        model_load_time = time.time() - model_load_start
        worker_logger.info(f"Worker {worker_id}: Model loaded in {model_load_time:.3f}s on {tts_model.device}")
        graph_flag = bool(getattr(tts_model, "cuda_graphs_enabled", False))
        mimi_graph_env = os.environ.get("POCKET_TTS_MIMI_GRAPH", "").strip().lower()
        if mimi_graph_env in ("0", "false", "no", "off"):
            mimi_graph_flag = False
        elif mimi_graph_env in ("1", "true", "yes", "on"):
            mimi_graph_flag = True
        else:
            # Default: follow FlowLM cuda_graphs (same as TTSModel batch path)
            mimi_graph_flag = graph_flag
        worker_logger.info(
            "Worker %s: CUDA graphs enabled=%s mimi_graph=%s (env/config)",
            worker_id,
            graph_flag,
            mimi_graph_flag,
        )
        _record_worker_event(
            timing_dir,
            worker_id,
            "model_loaded",
            duration_ms=round(model_load_time * 1000, 3),
            device=str(tts_model.device),
            cuda_graphs_enabled=graph_flag,
            mimi_graph_enabled=mimi_graph_flag,
        )
        _record_worker_event(
            timing_dir,
            worker_id,
            "cuda_memory",
            **_cuda_memory_snapshot("model_loaded"),
        )

        # HANDLE VOICE CONVERSION IF NEEDED
        if os.path.isfile(voice_path):
            from ..data.voice_converter import VoicePromptConverter
            converter = VoicePromptConverter()
            temp_dir = Path(tempfile.gettempdir()) / f"pocket_tts_worker_{os.getpid()}"
            temp_dir.mkdir(exist_ok=True)
            converted_path = converter.convert(voice_path, temp_dir)
            worker_logger.info(f"Worker {worker_id}: Converted voice to {converted_path}")
            voice_path = str(converted_path)

        # LOAD VOICE STATE ONCE AT WORKER STARTUP (major optimization)
        worker_logger.info(f"Worker {worker_id}: Loading voice state (once per worker)")
        voice_load_start = time.time()
        voice_state = tts_model.get_state_for_audio_prompt(voice_path, truncate=True)
        voice_load_time = time.time() - voice_load_start
        voice_memory = _cuda_memory_snapshot("voice_state_loaded")
        _record_worker_event(
            timing_dir,
            worker_id,
            "cuda_memory",
            **voice_memory,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        settled_memory = _cuda_memory_snapshot("startup_settled")
        _record_worker_event(
            timing_dir,
            worker_id,
            "cuda_memory",
            **settled_memory,
        )
        worker_logger.info(f"Worker {worker_id}: Voice loaded in {voice_load_time:.3f}s")

        worker_logger.info(f"Worker {worker_id}: Ready to process chunks (no reload overhead)")
        _record_worker_event(
            timing_dir,
            worker_id,
            "worker_ready",
            voice_load_ms=round(voice_load_time * 1000, 3),
        )
        if ready_queue is not None:
            ready_queue.put({
                "status": "ready",
                "worker_id": worker_id,
                "pid": os.getpid(),
                "peak_device_free_mb": voice_memory.get("device_free_mb"),
                "settled_device_free_mb": settled_memory.get("device_free_mb"),
            })
            ready_sent = True

        sample_rate = int(getattr(tts_model, "sample_rate", 24000))
        if cleanup_async:
            post_pool = _WorkerPostprocessPool(
                worker_id=worker_id,
                result_queue=result_queue,
                timing_dir=timing_dir,
                output_dir=output_dir,
                sample_rate=sample_rate,
                cleanup_enabled=cleanup_enabled,
                use_silero_vad=use_silero_vad,
                endpoint_threshold=endpoint_threshold,
                trimming_buffer_ms=trimming_buffer_ms,
                max_pending=max_pending,
                worker_logger=worker_logger,
            )

        # TRACK PERFORMANCE AND ERRORS
        chunks_processed = 0
        total_generation_time = 0.0
        total_io_time = 0.0

        while True:
            try:
                chunk_data = chunk_queue.get()
            except Exception:
                worker_logger.warning(f"Worker {worker_id}: Queue read failed, exiting")
                break

            if chunk_data[0] == "shutdown":
                break

            if chunk_data[0] == "regeneration":
                regeneration_task = dict(chunk_data[1])
                regeneration_task["worker_id"] = worker_id
                regen_start = time.monotonic()
                regeneration_result = _regenerate_chunk_in_worker(
                    tts_model,
                    voice_state,
                    regeneration_task,
                    worker_logger,
                    attempt_callback=lambda event: result_queue.put({
                        "kind": "regeneration_attempt",
                        "event": event,
                    }),
                )
                result_queue.put({
                    "kind": "regeneration",
                    "result": regeneration_result,
                })
                _record_worker_event(
                    timing_dir,
                    worker_id,
                    "regeneration_completed",
                    chunk_index=regeneration_task.get("chunk_index"),
                    duration_ms=round((time.monotonic() - regen_start) * 1000, 3),
                )
                continue

            if chunk_data[0] == "original_batch":
                batch_items = list(chunk_data[1])
                batch_start = time.monotonic()
                for global_index, chunk in batch_items:
                    _record_worker_event(
                        timing_dir,
                        worker_id,
                        "chunk_started",
                        phase="original",
                        chunk_index=global_index,
                        batch_size=len(batch_items),
                    )
                try:
                    batch_audio = _generate_batch_chunk_audio(
                        tts_model,
                        voice_state,
                        batch_items,
                        batch_size,
                        pause_injection_enabled,
                        pause_durations,
                        worker_logger,
                        cleanup_enabled,
                        use_silero_vad,
                        endpoint_threshold,
                        trimming_buffer_ms,
                    )
                    if not first_generation_recorded:
                        _record_worker_event(
                            timing_dir,
                            worker_id,
                            "cuda_memory",
                            **_cuda_memory_snapshot("first_generation_complete"),
                        )
                        first_generation_recorded = True
                    shared_generation_time = time.monotonic() - batch_start
                    per_chunk_generation_time = shared_generation_time / len(batch_items)
                    graph_payload = _cuda_graph_stats_payload(tts_model)
                    for global_index, chunk in batch_items:
                        total_generation_time += per_chunk_generation_time
                        if post_pool is not None:
                            # Gen thread only hands off CPU audio; cleanup+WAV on post thread.
                            post_pool.submit(
                                batch_audio[global_index],
                                chunk,
                                global_index,
                                generation_ms=per_chunk_generation_time * 1000.0,
                                batch_size=len(batch_items),
                                graph_payload=graph_payload,
                            )
                            chunks_processed += 1
                        else:
                            chunk_start = time.monotonic()
                            chunk_path, cleanup_time, silence_time, io_time = _finalize_and_save_worker_audio(
                                batch_audio[global_index],
                                chunk,
                                global_index,
                                tts_model,
                                output_dir,
                                cleanup_enabled,
                                use_silero_vad,
                                endpoint_threshold,
                                trimming_buffer_ms,
                                sample_rate=sample_rate,
                            )
                            chunks_processed += 1
                            total_io_time += io_time
                            _record_worker_event(
                                timing_dir,
                                worker_id,
                                "chunk_completed",
                                phase="original",
                                chunk_index=global_index,
                                batch_size=len(batch_items),
                                generation_ms=round(per_chunk_generation_time * 1000, 3),
                                cleanup_ms=round(cleanup_time * 1000, 3),
                                silence_ms=round(silence_time * 1000, 3),
                                io_ms=round(io_time * 1000, 3),
                                async_postprocess=False,
                                total_ms=round((time.monotonic() - chunk_start) * 1000, 3),
                                **graph_payload,
                            )
                            result_queue.put({
                                "kind": "original",
                                "saved_path": chunk_path,
                                "chunk_idx": global_index,
                                "chunk_text": chunk.text[:50],
                            })
                    del batch_audio
                except Exception as e:
                    worker_logger.error(
                        "Worker %s: FAILED batch %s: %s",
                        worker_id,
                        [index for index, _ in batch_items],
                        e,
                    )
                    worker_logger.error(traceback.format_exc())
                    for global_index, chunk in batch_items:
                        result_queue.put({
                            "kind": "original",
                            "saved_path": None,
                            "chunk_idx": global_index,
                            "chunk_text": chunk.text[:50],
                            "error": str(e),
                        })
                continue

            _, global_index, chunk = chunk_data
            chunk_start = time.monotonic()
            _record_worker_event(
                timing_dir,
                worker_id,
                "chunk_started",
                phase="original",
                chunk_index=global_index,
            )

            try:
                # GENERATE AUDIO (NO MODEL/VOICE RELOAD OVERHEAD!)
                gen_start = time.time()

                tts_params = chunk.tts_params
                if hasattr(tts_params, 'temperature'):
                    temperature = tts_params.temperature
                    frames_after_eos = tts_params.frames_after_eos
                    eos_threshold = tts_params.eos_threshold
                    speed_factor = getattr(tts_params, 'speed_factor', 1.0)
                else:
                    temperature = tts_params.get('temperature', 0.7)
                    frames_after_eos = tts_params.get('frames_after_eos', 2)
                    eos_threshold = tts_params.get('eos_threshold', -4.0)
                    speed_factor = tts_params.get('speed_factor', 1.0)

                if hasattr(tts_model, 'temp'):
                    tts_model.temp = float(temperature)
                if hasattr(tts_model, 'eos_threshold'):
                    tts_model.eos_threshold = float(eos_threshold)

                # Shared plan normalization preserves manual marker boundaries.
                from pocket_tts.preprocessing.text_normalizer import (
                    flatten_newlines_for_json,
                    normalize_text_chunk_for_storage,
                )
                chunk_text = normalize_text_chunk_for_storage(chunk.text or "")
                chunk.text = chunk_text

                pause_plan_text = None
                if pause_injection_enabled:
                    from pocket_tts.preprocessing.pause_injector import (
                        inject_pauses_for_punctuation,
                    )
                    pause_plan_text = flatten_newlines_for_json(
                        inject_pauses_for_punctuation(chunk_text, pause_durations)
                    )
                elif has_inline_pause_markers(chunk_text):
                    pause_plan_text = chunk_text

                if pause_plan_text and has_inline_pause_markers(pause_plan_text):
                    from pocket_tts.preprocessing.pause_injector import (
                        generate_audio_with_pauses,
                        render_text_with_native_pauses,
                    )

                    def render_segment(text: str):
                        """Generate and postprocess one scalar worker pause segment."""
                        segment = tts_model.generate_audio(
                            voice_state, text, frames_after_eos=frames_after_eos
                        )
                        return _postprocess_inline_speech_segment(
                            segment,
                            {"speed_factor": speed_factor},
                            sample_rate,
                            cleanup_enabled,
                            use_silero_vad,
                            endpoint_threshold,
                            trimming_buffer_ms,
                        )

                    audio, pause_events = generate_audio_with_pauses(
                        tts_model, voice_state, pause_plan_text,
                        generate_segment=render_segment,
                    )
                    chunk.text = pause_plan_text
                    chunk._inline_pause_text = pause_plan_text
                    chunk._spoken_text = render_text_with_native_pauses(pause_plan_text)
                    chunk.pause_events = pause_events
                    chunk._pause_events = pause_events
                    chunk._inline_pause_preprocessed = True
                else:
                    chunk._inline_pause_text = None
                    chunk._spoken_text = chunk_text
                    chunk._inline_pause_preprocessed = False
                    audio = tts_model.generate_audio(
                        voice_state,
                        chunk_text,
                        frames_after_eos=frames_after_eos
                    )

                if not first_generation_recorded:
                    _record_worker_event(
                        timing_dir,
                        worker_id,
                        "cuda_memory",
                        **_cuda_memory_snapshot("first_generation_complete"),
                    )
                    first_generation_recorded = True

                gen_time = time.time() - gen_start
                total_generation_time += gen_time
                tts_timing = getattr(tts_model, "last_generation_timing", {})
                graph_payload = _cuda_graph_stats_payload(tts_model)
                # Fold AR timing into graph payload extras for post-thread events.
                timing_extras = {
                    "tokenization_ms": round(tts_timing.get("tokenization_ms", 0.0), 3),
                    "prompt_ms": round(tts_timing.get("prompt_ms", 0.0), 3),
                    "autoregressive_ms": round(tts_timing.get("autoregressive_ms", 0.0), 3),
                    "decoder_ms": round(tts_timing.get("decoder_ms", 0.0), 3),
                }
                graph_payload = {**graph_payload, **timing_extras}

                if post_pool is not None:
                    # Speed factor lives inside finalize; hand raw audio to post thread.
                    post_pool.submit(
                        audio,
                        chunk,
                        global_index,
                        generation_ms=gen_time * 1000.0,
                        batch_size=None,
                        graph_payload=graph_payload,
                    )
                    chunks_processed += 1
                    worker_logger.info(
                        "Worker %s: Chunk %s gen=%.3fs (async postprocess queued)",
                        worker_id,
                        global_index,
                        gen_time,
                    )
                    del audio
                else:
                    # Speed factor is applied inside finalize (single place for sync/async).
                    chunk_path, cleanup_time, silence_time, io_time = _finalize_and_save_worker_audio(
                        audio,
                        chunk,
                        global_index,
                        tts_model,
                        output_dir,
                        cleanup_enabled,
                        use_silero_vad,
                        endpoint_threshold,
                        trimming_buffer_ms,
                        sample_rate=sample_rate,
                    )
                    total_io_time += io_time
                    chunks_processed += 1
                    worker_logger.info(
                        f"Worker {worker_id}: Chunk {global_index} - "
                        f"gen={gen_time:.3f}s, io={io_time:.3f}s"
                    )
                    _record_worker_event(
                        timing_dir,
                        worker_id,
                        "chunk_completed",
                        phase="original",
                        chunk_index=global_index,
                        generation_ms=round(gen_time * 1000, 3),
                        cleanup_ms=round(cleanup_time * 1000, 3),
                        silence_ms=round(silence_time * 1000, 3),
                        io_ms=round(io_time * 1000, 3),
                        async_postprocess=False,
                        total_ms=round((time.monotonic() - chunk_start) * 1000, 3),
                        **graph_payload,
                    )
                    result_queue.put({
                        "kind": "original",
                        "saved_path": str(chunk_path),
                        "chunk_idx": global_index,
                        "chunk_text": chunk.text[:50],
                    })
                    del audio
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                # LOG ERROR BUT CONTINUE PROCESSING OTHER CHUNKS
                worker_logger.error(f"Worker {worker_id}: FAILED chunk {global_index}: {e}")
                worker_logger.error(traceback.format_exc())
                result_queue.put({
                    "kind": "original",
                    "saved_path": None,
                    "chunk_idx": global_index,
                    "chunk_text": chunk.text[:50] if chunk else "",
                    "error": str(e),
                })

        # Drain CPU postprocess so parent receives every chunk_completed / WAV.
        if post_pool is not None:
            worker_logger.info(
                "Worker %s: draining async postprocess queue...",
                worker_id,
            )
            post_pool.shutdown(wait=True)
            total_io_time += post_pool.total_io_time
            worker_logger.info(
                "Worker %s: postprocess done completed=%s failed=%s cleanup_s=%.2f io_s=%.2f",
                worker_id,
                post_pool.completed,
                post_pool.failed,
                post_pool.total_cleanup_time,
                post_pool.total_io_time,
            )

        # WORKER SUMMARY
        avg_gen = total_generation_time / chunks_processed if chunks_processed > 0 else 0
        avg_io = total_io_time / chunks_processed if chunks_processed > 0 else 0
        graph_payload = _cuda_graph_stats_payload(tts_model)
        worker_logger.info(
            "Worker %s: Exiting - processed %s chunks, avg_gen=%.3fs, avg_io=%.3fs, "
            "cuda_graphs enabled=%s captures_ok=%s fail=%s steps_graph=%s steps_eager=%s (%.1f%% graph), "
            "mimi_captures_ok=%s mimi_steps_graph=%s mimi_eager=%s (%.1f%% mimi_graph)",
            worker_id,
            chunks_processed,
            avg_gen,
            avg_io,
            graph_payload.get("cuda_graphs_enabled"),
            graph_payload.get("cuda_graph_captures_ok"),
            graph_payload.get("cuda_graph_captures_fail"),
            graph_payload.get("cuda_graph_steps_graph"),
            graph_payload.get("cuda_graph_steps_eager"),
            graph_payload.get("cuda_graph_steps_graph_pct") or 0.0,
            graph_payload.get("mimi_graph_captures_ok"),
            graph_payload.get("mimi_graph_steps_graph"),
            graph_payload.get("mimi_graph_steps_eager"),
            graph_payload.get("mimi_graph_steps_graph_pct") or 0.0,
        )
        _record_worker_event(
            timing_dir,
            worker_id,
            "worker_exiting",
            chunks_processed=chunks_processed,
            average_generation_ms=round(avg_gen * 1000, 3),
            average_io_ms=round(avg_io * 1000, 3),
            async_postprocess=bool(post_pool is not None),
            postprocess_completed=getattr(post_pool, "completed", chunks_processed),
            postprocess_failed=getattr(post_pool, "failed", 0),
            **graph_payload,
            **_cuda_memory_snapshot("worker_exit"),
        )

    except Exception as e:
        worker_logger.error(f"Worker {worker_id}: Fatal error: {e}")
        worker_logger.error(traceback.format_exc())
        if post_pool is not None:
            try:
                post_pool.shutdown(wait=True, timeout=120.0)
            except Exception:
                pass
        if ready_queue is not None and not ready_sent:
            ready_queue.put({"status": "error", "worker_id": worker_id, "error": str(e)})
        _record_worker_event(
            timing_dir,
            worker_id,
            "cuda_memory",
            **_cuda_memory_snapshot("fatal_error"),
        )
        result_queue.put({
            "kind": "original",
            "saved_path": None,
            "chunk_idx": -1,
            "chunk_text": "",
            "error": str(e),
        })
