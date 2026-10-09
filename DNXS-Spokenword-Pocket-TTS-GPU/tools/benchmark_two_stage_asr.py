#!/usr/bin/env python3
"""Offline benchmark runner for PocketGPU two-stage TTS speech verification.

This runner records environment readiness and evaluates existing labeled Stage 1
candidate rows.  With ``--run`` it invokes each installed Stage 1 backend on a
copy of the same TTS folder; it never regenerates audio.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ASR"))

from verification_backends import ParakeetTDTBackend  # noqa: E402


def _load_chunk_indexes(path: Path) -> set[int]:
    """Load labeled chunk indexes from a user-maintained JSON acceptance file."""
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    return {int(row["chunk_index"]) for row in rows if "chunk_index" in row}


def inspect_acceptance_data(tts_dir: Path) -> Dict[str, Any]:
    """Report whether pass/fail labels and their original WAVs are available."""
    pass_ids = _load_chunk_indexes(tts_dir / "pass_check.json")
    fail_ids = _load_chunk_indexes(tts_dir / "fail_check.json")
    audio_dir = tts_dir / "audio_chunks"

    def available(indexes: Iterable[int]) -> list[int]:
        """Return indexes whose current original chunk WAV exists on disk."""
        return [index for index in sorted(indexes) if (audio_dir / f"chunk_{index:05d}.wav").exists()]

    return {
        "pass_check_expected_accept": sorted(pass_ids),
        "fail_check_expected_confirm": sorted(fail_ids),
        "pass_check_wavs_available": available(pass_ids),
        "fail_check_wavs_available": available(fail_ids),
    }


def collect_environment_readiness() -> Dict[str, Any]:
    """Record optional model availability without installing or downloading anything."""
    return {
        "nemo_installed": importlib.util.find_spec("nemo") is not None,
        "parakeet_available": ParakeetTDTBackend.is_available(),
        "medium_verifier_available": importlib.util.find_spec("faster_whisper") is not None,
        "python": sys.version.split()[0],
    }


def run_stage_one(validator: Path, tts_dir: Path, engine: str, model: str, output: Path) -> Dict[str, Any]:
    """Run one Stage 1 backend against existing WAVs without entering regeneration."""
    command = [
        sys.executable, str(validator), "--batch-tts-dir", str(tts_dir),
        "--log-file", str(output), "--threshold", "0.7", "--model", model,
        "--engine", engine, "--device", "cuda", "--json",
    ]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    return {
        "engine": engine,
        "model": model,
        "log_file": str(output),
        "wall_s": time.monotonic() - started,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "candidate_count": len(json.loads(output.read_text(encoding="utf-8"))) if output.exists() else None,
    }


def run_stage_two_medium_consensus(tts_dir: Path, failure_log: Path) -> Dict[str, Any]:
    """Run the exact GUI Stage 2 Medium-consensus logic over a failure log."""
    started = time.monotonic()
    if not failure_log.exists():
        return {
            "available": False,
            "source_log": str(failure_log),
            "output": str(tts_dir / "asr_medium_verification.json"),
            "error": "source log missing",
            "wall_s": 0.0,
        }
    rows = json.loads(failure_log.read_text(encoding="utf-8"))
    from pocket_tts.audiobook.generator import AudiobookGenerator

    generator = AudiobookGenerator.__new__(AudiobookGenerator)
    confirmed, summary = generator._verify_failed_chunks_with_medium_asr(
        rows,
        tts_dir,
        threshold=0.7,
        language="en",
    )
    return {
        "available": not summary.get("verification_skipped", False),
        "source_log": str(failure_log),
        "output": str(tts_dir / "asr_medium_verification.json"),
        "candidate_count": len(rows),
        "accepted_by_medium": summary.get("verified_pass", 0),
        "accepted_not_proven_failure": summary.get("not_proven", 0),
        "confirmed_by_two_asr_models": len(confirmed),
        "verification_error": summary.get("verification_error"),
        "wall_s": time.monotonic() - started,
    }


def main() -> int:
    """Parse arguments, inspect labels, and optionally run benchmarkable Stage 1 backends."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tts_dir", type=Path, help="TTS folder containing audio_chunks and text_chunks")
    parser.add_argument("--run", action="store_true", help="Run installed Stage 1 backends; default is inspection only")
    parser.add_argument(
        "--stage-two-only",
        action="store_true",
        help="Skip Stage 1 and run independent Medium verification against an existing failure log",
    )
    parser.add_argument(
        "--source-log",
        type=Path,
        default=None,
        help="Failure log to feed into Stage 2 when --stage-two-only is set",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write benchmark JSON here")
    args = parser.parse_args()
    tts_dir = args.tts_dir.resolve()
    report: Dict[str, Any] = {
        "report_type": "two_stage_asr_benchmark",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tts_dir": str(tts_dir),
        "environment": collect_environment_readiness(),
        "acceptance_data": inspect_acceptance_data(tts_dir),
        "runs": [],
        "notes": [
            "No regeneration is performed by this runner.",
            "Independent Medium verification runs only after a Stage 1 candidate list exists.",
            "Do not treat an unavailable Medium verifier as evidence of a failed WAV.",
        ],
    }
    if args.stage_two_only:
        source_log = args.source_log or (tts_dir / "asr_failures.json")
        report["stage_two"] = run_stage_two_medium_consensus(
            tts_dir,
            source_log.resolve(),
        )
    elif args.run:
        validator = ROOT / "ASR" / "asr_validator.py"
        work_dir = tts_dir / ".asr_benchmarks"
        work_dir.mkdir(exist_ok=True)
        stage_one_runs = []
        for engine, model in (("whisper_cpp", "medium"), ("faster_whisper", "base"), ("parakeet", "base")):
            stage_one_runs.append(run_stage_one(
                validator, tts_dir, engine, model, work_dir / f"{engine}_failures.json",
            ))
        report["runs"] = stage_one_runs
        preferred_source = None
        for preferred in ("parakeet", "faster_whisper", "whisper_cpp"):
            for run in stage_one_runs:
                # The validator intentionally exits 1 when it found candidates.
                # A readable JSON list is a completed Stage 1 result, regardless
                # of whether every WAV passed.
                if run["engine"] == preferred and run["candidate_count"] is not None:
                    preferred_source = Path(run["log_file"])
                    break
            if preferred_source is not None:
                break
        if preferred_source is not None:
            report["stage_two"] = run_stage_two_medium_consensus(
                tts_dir,
                preferred_source,
            )
        else:
            report["stage_two"] = {
                "available": False,
                "source_log": str(preferred_source) if preferred_source is not None else "",
                "output": str(tts_dir / "asr_medium_verification.json"),
                "error": "No usable Stage 1 failure log",
            }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
