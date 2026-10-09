#!/usr/bin/env python3
"""Run PocketGPU's legacy or opt-in isolated ASR verification pipeline.

``legacy`` preserves the existing in-process report-only behavior.  ``new``
runs Parakeet, Medium, and forced alignment in separate child processes.  Each
new-pipeline child records CUDA memory before loading, performs explicit CUDA
teardown, writes a process report, and exits before the next model can load.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PIPELINES = {
    "legacy": {
        "stage_one_engine": "faster_whisper",
        "stage_one_model": "base",
        "failure_filename": "asr_failures.json",
        "medium_filename": "asr_medium_validation_all.json",
        "alignment_filename": None,
    },
    "new": {
        "stage_one_engine": "parakeet",
        "stage_one_model": "nvidia/parakeet-tdt-0.6b-v3",
        "stage_two_model": "medium",
        "failure_filename": "asr_new_failures.json",
        "medium_filename": "asr_new_medium_verification.json",
        "alignment_filename": "asr_new_alignment_diagnostic.json",
    },
}

NEW_STAGES = ("stage_one", "medium", "alignment")
SECOND_STAGE_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v3",
)
STREAMING_AUDIT_NAME = "asr_during_tts_stage_one.json"


def _validate_second_stage_model(model_name: str) -> str:
    """Validate and return a supported faster-whisper Stage 2 model name."""
    normalized = str(model_name or "medium").strip()
    if normalized not in SECOND_STAGE_MODELS:
        allowed = ", ".join(SECOND_STAGE_MODELS)
        raise ValueError(f"Unsupported second-stage ASR model {normalized!r}; choose: {allowed}")
    return normalized


def _read_failure_rows(path: Path) -> list[dict[str, Any]]:
    """Read a Stage 1 failure list and reject malformed output clearly."""
    if not path.exists():
        raise RuntimeError(f"Stage 1 did not write its failure log: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Stage 1 wrote invalid JSON to {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise RuntimeError(f"Stage 1 failure log must be a JSON list: {path}")
    return [row for row in payload if isinstance(row, dict)]


def _read_streaming_stage_one_audit(tts_dir: Path) -> dict[str, Any]:
    """Read validated during-TTS Stage 1 evidence for Stage 2-only reuse.

    The Stage 4b path trusts only strict materialized evidence written after a
    complete drain. Any malformed or mismatched payload must fail closed before
    Stage 2 starts so the caller can fall back to the ordinary full pipeline.
    """
    audit_path = tts_dir / STREAMING_AUDIT_NAME
    if not audit_path.exists():
        raise RuntimeError(f"Streaming Stage 1 audit is missing: {audit_path}")
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Streaming Stage 1 audit is invalid JSON: {audit_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Streaming Stage 1 audit must be a JSON object: {audit_path}")
    if payload.get("stage_one_source") != "during_tts_streaming":
        raise RuntimeError("Streaming Stage 1 audit has unsupported source identifier")
    summary = payload.get("summary") or {}
    if not isinstance(summary, dict) or summary.get("complete") is not True:
        raise RuntimeError("Streaming Stage 1 audit must come from a complete drained summary")
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Streaming Stage 1 audit is missing terminal records")
    return payload


def _build_streaming_stage_one_report(
    audit_payload: dict[str, Any],
    failure_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Build synthetic Stage 1 report fields from precompleted streaming evidence."""
    summary = dict(audit_payload.get("summary") or {})
    records = [
        record
        for record in (audit_payload.get("records") or [])
        if isinstance(record, dict)
    ]
    candidate_count = int(audit_payload.get("candidate_count") or len(json.loads(failure_path.read_text(encoding="utf-8"))))
    backend_models = sorted({
        (str(record.get("backend") or ""), str(record.get("model") or ""))
        for record in records
    })
    backend = backend_models[0][0] if len(backend_models) == 1 else "mixed"
    model = backend_models[0][1] if len(backend_models) == 1 else "mixed"
    stage_one = {
        "entrypoint": "ASR.streaming_stage_one.materialize_stage_one_summary",
        "engine": backend,
        "model": model,
        "failure_log": str(failure_path),
        "summary": {
            "submitted": summary.get("submitted", len(records)),
            "completed": summary.get("completed", len(records)),
            "failed": summary.get("failed", candidate_count),
            "complete": True,
            "source": "during_tts_streaming",
        },
        "wall_s": summary.get("wall_s"),
        "source": "during_tts_streaming",
    }
    stage_one_process = {
        "stage": "stage_one",
        "fresh_process": False,
        "result": stage_one,
        "error": None,
        "returncode": 0,
        "process_exited_before_next_stage": True,
        "status_path": None,
        "log_path": str(audit_payload.get("summary", {}).get("event_log_path") or ""),
        "source": "during_tts_streaming",
        "wall_s": summary.get("wall_s"),
        "note": "Stage 1 completed before this runner started; this report reuses materialized evidence only.",
    }
    return stage_one, stage_one_process, candidate_count


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one deterministic, readable pipeline artifact under the TTS folder."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _cuda_memory_snapshot() -> dict[str, Any]:
    """Capture this process's CUDA allocator and driver memory state safely."""
    snapshot: dict[str, Any] = {
        "pid": os.getpid(),
        "cuda_available": False,
        "allocated_bytes": None,
        "reserved_bytes": None,
        "free_bytes": None,
        "total_bytes": None,
        "error": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return snapshot
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        snapshot.update({
            "cuda_available": True,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        })
    except Exception as exc:
        snapshot["error"] = str(exc)
    return snapshot


def _hard_cuda_teardown() -> dict[str, Any]:
    """Collect Python and CUDA state before this isolated child process exits."""
    details: dict[str, Any] = {
        "before": _cuda_memory_snapshot(),
        "gc_collected": 0,
        "actions": [],
        "errors": [],
        "after": None,
    }
    try:
        details["gc_collected"] = gc.collect()
        details["actions"].append("gc.collect")
    except Exception as exc:
        details["errors"].append(f"gc.collect: {exc}")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            details["actions"].append("torch.cuda.synchronize")
            torch.cuda.empty_cache()
            details["actions"].append("torch.cuda.empty_cache")
            torch.cuda.ipc_collect()
            details["actions"].append("torch.cuda.ipc_collect")
            torch.cuda.synchronize()
            details["actions"].append("torch.cuda.synchronize_after_cleanup")
    except Exception as exc:
        details["errors"].append(f"cuda_teardown: {exc}")
    details["after"] = _cuda_memory_snapshot()
    return details


def run_stage_one(
    tts_dir: Path,
    settings: dict[str, Any],
    threshold: float,
    language: str,
) -> dict[str, Any]:
    """Call production Stage 1 validation with this pipeline's explicit backend."""
    from ASR.asr_validator import run_batch_folder_validation

    failure_path = tts_dir / str(settings["failure_filename"])
    started = time.monotonic()
    summary = run_batch_folder_validation(
        tts_dir,
        failure_path,
        threshold,
        model_name=str(settings["stage_one_model"]),
        language=language,
        force_device="cuda",
        engine=str(settings["stage_one_engine"]),
        cpu_workers=4,
        pack_size=8,
        pack_silence_s=0.75,
    )
    return {
        "entrypoint": "ASR.asr_validator.run_batch_folder_validation",
        "engine": settings["stage_one_engine"],
        "model": settings["stage_one_model"],
        "failure_log": str(failure_path),
        "summary": summary,
        "wall_s": round(time.monotonic() - started, 3),
    }


def run_medium_verifier(
    failures: list[dict[str, Any]],
    tts_dir: Path,
    settings: dict[str, Any],
    threshold: float,
    language: str,
    model_name: str = "medium",
    require_gpu: bool = False,
) -> dict[str, Any]:
    """Call generator's packed selected-model verifier without regeneration work."""
    from pocket_tts.audiobook.generator import AudiobookGenerator

    model_name = _validate_second_stage_model(model_name)
    output_path = tts_dir / str(settings["medium_filename"])
    started = time.monotonic()
    generator = AudiobookGenerator.__new__(AudiobookGenerator)
    if not failures:
        empty_summary = {
            "attempted": 0,
            "verification_skipped": True,
            "verification_error": "no_stage_one_candidates",
        }
        generator._write_medium_verification_log(
            tts_dir,
            threshold,
            language,
            empty_summary,
            [],
            report_filename=str(settings["medium_filename"]),
        )
        return {
            "entrypoint": "pocket_tts.audiobook.generator.AudiobookGenerator._verify_failed_chunks_with_medium_asr",
            "model": model_name,
            "output": str(output_path),
            "attempted": 0,
            "active": False,
            "gpu_required": require_gpu,
            "reason": "no_stage_one_candidates",
            "wall_s": round(time.monotonic() - started, 3),
        }

    confirmed, summary = generator._verify_failed_chunks_with_medium_asr(
        failures,
        tts_dir,
        threshold,
        language,
        model_name=model_name,
        report_filename=str(settings["medium_filename"]),
        require_gpu=require_gpu,
    )
    return {
        "entrypoint": "pocket_tts.audiobook.generator.AudiobookGenerator._verify_failed_chunks_with_medium_asr",
        "model": model_name,
        "output": str(output_path),
        "attempted": len(failures),
        "active": not bool(summary.get("verification_skipped")),
        "gpu_required": require_gpu,
        "gpu_load_failed": summary.get("verification_error")
        == "gpu_second_stage_model_load_failed_no_cpu_fallback",
        "summary": summary,
        "confirmed_failures": len(confirmed),
        "wall_s": round(time.monotonic() - started, 3),
    }


def run_alignment_diagnostic(
    failures: list[dict[str, Any]],
    tts_dir: Path,
    settings: dict[str, Any],
    threshold: float,
    language: str,
) -> dict[str, Any]:
    """Run NeMo alignment as evidence-only output that cannot alter decisions."""
    from ASR.verification_backends import NemoForcedAligner

    output_path = tts_dir / str(settings["alignment_filename"])
    started = time.monotonic()
    jobs: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    for failure in failures:
        index = failure.get("chunk_index")
        if index is None:
            continue
        chunk_id = f"chunk_{int(index):05d}"
        audio_path = tts_dir / "audio_chunks" / f"{chunk_id}.wav"
        jobs.append({"audio_path": str(audio_path), "text": str(failure.get("original_text") or "")})
        rows.append({
            "chunk_index": index,
            "chunk_id": chunk_id,
            "original_text": failure.get("original_text", ""),
            "stage_one": {
                "transcribed_text": failure.get("transcribed_text", ""),
                "score": failure.get("score"),
                "explanation": failure.get("explanation", ""),
            },
        })

    aligner = NemoForcedAligner()
    try:
        evidence = aligner.align_batch(jobs, tts_dir / ".asr_new_alignment")
        evidence_by_audio = {item.audio_path: item.to_dict() for item in evidence}
        error = None
    except Exception as exc:
        evidence_by_audio = {}
        error = str(exc)

    available = 0
    conclusive = 0
    records = []
    for row, job in zip(rows, jobs):
        alignment = evidence_by_audio.get(job["audio_path"], {
            "available": False,
            "conclusive": False,
            "error": error or "No alignment result returned",
        })
        available += int(bool(alignment.get("available")))
        conclusive += int(bool(alignment.get("conclusive")))
        records.append({
            **row,
            "alignment": alignment,
            "decision": "diagnostic_only",
            "explanation": "Alignment evidence is recorded only; it cannot accept, fail, or regenerate this chunk.",
        })

    _write_json(output_path, {
        "report_type": "forced_alignment_diagnostic",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "threshold": threshold,
        "language": language,
        "decision_policy": "diagnostic_only_never_used_for_regeneration",
        "summary": {
            "attempted": len(rows),
            "available": available,
            "conclusive": conclusive,
            "backend_error": error,
            "model": aligner.model_name,
        },
        "records": records,
    })
    return {
        "entrypoint": "ASR.verification_backends.NemoForcedAligner.align_batch",
        "model": aligner.model_name,
        "output": str(output_path),
        "attempted": len(rows),
        "active": bool(jobs),
        "available": available,
        "conclusive": conclusive,
        "decision_policy": "diagnostic_only_never_used_for_regeneration",
        "wall_s": round(time.monotonic() - started, 3),
    }


def _run_internal_stage(
    stage: str,
    tts_dir: Path,
    threshold: float,
    language: str,
    second_stage_model: str,
    status_path: Path,
) -> int:
    """Execute exactly one new-pipeline model stage in its own process."""
    if stage not in NEW_STAGES:
        raise ValueError(f"Unknown isolated stage: {stage}")
    settings = dict(PIPELINES["new"])
    settings["stage_two_model"] = _validate_second_stage_model(second_stage_model)
    started = time.monotonic()
    status: dict[str, Any] = {
        "report_type": "isolated_asr_stage",
        "stage": stage,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "fresh_process": True,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cuda_before_stage": _cuda_memory_snapshot(),
        "result": None,
        "error": None,
    }
    exit_code = 0
    try:
        if stage == "stage_one":
            status["result"] = run_stage_one(tts_dir, settings, threshold, language)
        else:
            failures = _read_failure_rows(tts_dir / str(settings["failure_filename"]))
            if stage == "medium":
                status["result"] = run_medium_verifier(
                    failures,
                    tts_dir,
                    settings,
                    threshold,
                    language,
                    model_name=str(settings["stage_two_model"]),
                    require_gpu=True,
                )
            else:
                status["result"] = run_alignment_diagnostic(
                    failures, tts_dir, settings, threshold, language
                )
    except Exception as exc:
        exit_code = 2
        status["error"] = str(exc)
    finally:
        status["cuda_teardown"] = _hard_cuda_teardown()
        status["wall_s"] = round(time.monotonic() - started, 3)
        status["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json(status_path, status)
    return exit_code


def _run_isolated_stage(
    tts_dir: Path,
    stage: str,
    threshold: float,
    language: str,
    second_stage_model: str,
) -> dict[str, Any]:
    """Launch one fresh Python process and load its recorded stage boundary."""
    status_path = tts_dir / f"asr_new_{stage}_process.json"
    log_path = tts_dir / f"asr_new_{stage}.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(tts_dir),
        "--internal-stage",
        stage,
        "--status-file",
        str(status_path),
        "--threshold",
        str(threshold),
        "--language",
        language,
        "--second-stage-model",
        second_stage_model,
    ]
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        status = {
            "stage": stage,
            "fresh_process": True,
            "result": None,
            "error": "child_exited_without_status_file",
        }
    status.update({
        "command": command,
        "log_path": str(log_path),
        "status_path": str(status_path),
        "returncode": completed.returncode,
        "parent_wait_s": round(time.monotonic() - started, 3),
        "process_exited_before_next_stage": True,
    })
    # The child writes its own status before exiting. Persist parent-only facts
    # too, because the pipeline checks this file after the process boundary.
    _write_json(status_path, status)
    return status


def _run_new_isolated_pipeline(
    tts_dir: Path,
    threshold: float,
    language: str,
    run_alignment: bool = True,
    second_stage_model: str = "medium",
    use_existing_stage_one: bool = False,
) -> dict[str, Any]:
    """Run new mode with optional diagnostic-only forced alignment.

    Args:
        tts_dir: Book-local TTS directory containing chunk WAV and text files.
        threshold: Shared spoken-content comparison threshold.
        language: ASR language code.
        run_alignment: Whether to run the diagnostic-only alignment stage.
        second_stage_model: Faster-Whisper model used only for Stage 2.
        use_existing_stage_one: Reuse strict complete during-TTS Stage 1 evidence
            instead of launching the isolated Stage 1 child again.
    """
    second_stage_model = _validate_second_stage_model(second_stage_model)
    started = time.monotonic()
    failure_path = tts_dir / str(PIPELINES["new"]["failure_filename"])
    if use_existing_stage_one:
        audit_payload = _read_streaming_stage_one_audit(tts_dir)
        failures = _read_failure_rows(failure_path)
        stage_one, stage_one_process, candidate_count = _build_streaming_stage_one_report(
            audit_payload,
            failure_path,
        )
    else:
        stage_one_process = _run_isolated_stage(
            tts_dir, "stage_one", threshold, language, second_stage_model
        )
        if stage_one_process.get("returncode") != 0:
            raise RuntimeError(f"Parakeet child failed: {stage_one_process.get('error')}")
        failures = _read_failure_rows(failure_path)
        stage_one = stage_one_process.get("result") or {}
        candidate_count = len(failures)

    medium_process = _run_isolated_stage(
        tts_dir, "medium", threshold, language, second_stage_model
    )
    if medium_process.get("returncode") != 0:
        raise RuntimeError(f"Medium child failed: {medium_process.get('error')}")
    if run_alignment:
        alignment_process = _run_isolated_stage(
            tts_dir, "alignment", threshold, language, second_stage_model
        )
        if alignment_process.get("returncode") != 0:
            raise RuntimeError(f"Alignment child failed: {alignment_process.get('error')}")
        alignment = alignment_process.get("result") or {}
    else:
        alignment_process = None
        alignment = {
            "active": False,
            "decision_policy": "diagnostic_only_never_used_for_regeneration",
            "reason": "disabled_by_request",
        }

    medium = medium_process.get("result") or {}
    report_path = tts_dir / "asr_new_pipeline_report.json"
    outputs = [
        str(failure_path),
        str(tts_dir / str(PIPELINES["new"]["medium_filename"])),
        str(tts_dir / "asr_new_medium_failures.json"),
        str(tts_dir / "asr_new_medium_process.json"),
        str(report_path),
    ]
    if use_existing_stage_one:
        outputs.append(str(tts_dir / STREAMING_AUDIT_NAME))
    else:
        outputs.append(str(tts_dir / "asr_new_stage_one_process.json"))
    if run_alignment:
        outputs.extend([
            str(tts_dir / str(PIPELINES["new"]["alignment_filename"])),
            str(tts_dir / "asr_new_alignment_process.json"),
        ])
    report = {
        "report_type": "live_asr_pipeline_run",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pipeline": "new",
        "stage_one_source": "during_tts_streaming" if use_existing_stage_one else "postgen_isolated_child",
        "report_only": True,
        "regeneration_performed": False,
        "threshold": threshold,
        "language": language,
        "second_stage_model": second_stage_model,
        "alignment_enabled": run_alignment,
        "isolation": {
            "strategy": "one_fresh_subprocess_per_model_stage",
            "guarantee": "Each child performs explicit teardown and exits before the next child starts.",
        },
        "stage_one": stage_one,
        "stage_one_process": stage_one_process,
        "stage_one_audit_path": str(tts_dir / STREAMING_AUDIT_NAME) if use_existing_stage_one else None,
        "stage_one_failure_path": str(failure_path),
        "stage_one_candidate_count": candidate_count,
        "medium": medium,
        "medium_process": medium_process,
        "forced_alignment": alignment,
        "forced_alignment_process": alignment_process,
        "outputs": outputs,
        "wall_s": round(time.monotonic() - started, 3),
    }
    _write_json(report_path, report)
    return report


def _run_legacy_pipeline(
    tts_dir: Path,
    threshold: float,
    language: str,
) -> dict[str, Any]:
    """Preserve the pre-existing direct legacy report-only runner behavior."""
    settings = PIPELINES["legacy"]
    started = time.monotonic()
    stage_one = run_stage_one(tts_dir, settings, threshold, language)
    failures = _read_failure_rows(tts_dir / str(settings["failure_filename"]))
    medium = run_medium_verifier(failures, tts_dir, settings, threshold, language)
    report_path = tts_dir / "asr_legacy_pipeline_report.json"
    report = {
        "report_type": "live_asr_pipeline_run",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pipeline": "legacy",
        "report_only": True,
        "regeneration_performed": False,
        "threshold": threshold,
        "language": language,
        "stage_one": stage_one,
        "stage_one_candidate_count": len(failures),
        "medium": medium,
        "forced_alignment": None,
        "outputs": [
            str(tts_dir / str(settings["failure_filename"])),
            str(tts_dir / str(settings["medium_filename"])),
            str(report_path),
        ],
        "wall_s": round(time.monotonic() - started, 3),
    }
    _write_json(report_path, report)
    return report


def run_pipeline(
    tts_dir: Path,
    pipeline: str,
    threshold: float,
    language: str,
    run_alignment: bool = True,
    second_stage_model: str = "medium",
    use_existing_stage_one: bool = False,
) -> dict[str, Any]:
    """Run one report-only pipeline with optional New-model selection and diagnostics."""
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline}")
    if use_existing_stage_one and pipeline != "new":
        raise ValueError("use_existing_stage_one is supported only with pipeline='new'")
    tts_dir = tts_dir.resolve()
    if not (tts_dir / "audio_chunks").is_dir() or not (tts_dir / "text_chunks").is_dir():
        raise RuntimeError("TTS folder must contain audio_chunks/ and text_chunks/")
    if pipeline == "new":
        return _run_new_isolated_pipeline(
            tts_dir,
            threshold,
            language,
            run_alignment,
            second_stage_model,
            use_existing_stage_one,
        )
    return _run_legacy_pipeline(tts_dir, threshold, language)


def main() -> int:
    """Parse CLI arguments and run either the parent pipeline or one child stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tts_dir", type=Path, help="Folder containing audio_chunks/ and text_chunks/")
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="legacy")
    parser.add_argument("--threshold", type=float, default=0.68, help="Spoken-content pass threshold")
    parser.add_argument("--language", default="en", help="ASR language code")
    parser.add_argument(
        "--second-stage-model",
        choices=SECOND_STAGE_MODELS,
        default="medium",
        help="New pipeline only: Faster-Whisper model for Stage 2 (default: medium).",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="New pipeline only: skip diagnostic-only forced alignment.",
    )
    parser.add_argument(
        "--use-existing-stage-one",
        action="store_true",
        help="New pipeline only: reuse strict complete during-TTS Stage 1 evidence instead of rerunning Stage 1.",
    )
    parser.add_argument("--internal-stage", choices=NEW_STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--status-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.use_existing_stage_one and args.pipeline != "new":
        parser.error("--use-existing-stage-one requires --pipeline new")
    if args.internal_stage:
        if args.status_file is None:
            parser.error("--status-file is required with --internal-stage")
        exit_code = _run_internal_stage(
            args.internal_stage,
            args.tts_dir.resolve(),
            args.threshold,
            args.language,
            args.second_stage_model,
            args.status_file.resolve(),
        )
        # NeMo can leave CUDA worker threads alive after normal interpreter
        # shutdown. The status file is already complete, so force this child
        # to exit and release its CUDA context before the next model starts.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    try:
        report = run_pipeline(
            args.tts_dir,
            args.pipeline,
            args.threshold,
            args.language,
            run_alignment=not args.skip_alignment,
            second_stage_model=args.second_stage_model,
            use_existing_stage_one=args.use_existing_stage_one,
        )
    except Exception as exc:
        print(f"ASR pipeline failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
