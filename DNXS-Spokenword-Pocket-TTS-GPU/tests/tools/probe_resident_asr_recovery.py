#!/usr/bin/env python3
"""Benchmark staged PocketTTS recovery with one resident higher-quality ASR model.

The probe first validates every row from a book-local ``asr_failures.json``
using the current production WAVs and one resident faster-whisper model.  It
then regenerates only the rows that still fail, validates every candidate with
that same model, and stops a row at its first passing candidate.  All generated
candidate WAVs and JSON evidence remain under ``tests/results``; book audio is
never changed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ASR_DIR = ROOT / "ASR"
if str(ASR_DIR) not in sys.path:
    sys.path.insert(0, str(ASR_DIR))


@dataclass
class CandidateResult:
    """Capture one candidate's generation and resident-ASR outcome."""

    attempt: int
    temperature: float | None
    score: float
    passed: bool
    audio_path: str
    transcript: str
    explanation: str


def parse_args() -> argparse.Namespace:
    """Parse one TTS folder plus bounded recovery benchmark settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-dir", type=Path, required=True)
    parser.add_argument("--model", default="medium", help="Resident ASR model.")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "tests/results/resident_asr_recovery",
        help="Parent directory for non-destructive candidate WAVs and evidence.",
    )
    return parser.parse_args()


def load_failures(tts_dir: Path) -> list[dict[str, Any]]:
    """Load canonical failure rows and reject incomplete book folders early."""
    failure_path = tts_dir / "asr_failures.json"
    if not failure_path.is_file():
        raise FileNotFoundError(f"Failure report not found: {failure_path}")
    rows = json.loads(failure_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Failure report must contain at least one JSON row: {failure_path}")
    required = (tts_dir / "audio_chunks", tts_dir / "text_chunks")
    for path in required:
        if not path.is_dir():
            raise FileNotFoundError(f"Required TTS directory not found: {path}")
    return rows


def load_voice_path(tts_dir: Path) -> Path:
    """Resolve this book's saved voice-conditioning WAV from chunk metadata."""
    metadata_path = tts_dir / "text_chunks/audiobook.chunks.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    voice_used = metadata.get("_metadata", {}).get("voice_used")
    if not voice_used:
        raise ValueError(f"voice_used is missing from {metadata_path}")
    voice_path = Path(str(voice_used))
    if not voice_path.is_absolute():
        voice_path = (ROOT / voice_path).resolve()
    if not voice_path.is_file():
        raise FileNotFoundError(f"Saved voice file not found: {voice_path}")
    return voice_path


def candidate_result(attempt: int, temperature: float | None, audio_path: Path, result: dict[str, Any]) -> CandidateResult:
    """Normalize validator output into JSON-safe evidence for one candidate."""
    return CandidateResult(
        attempt=attempt,
        temperature=temperature,
        score=float(result.get("score", 0.0)),
        passed=bool(result.get("passed", False)),
        audio_path=str(audio_path),
        transcript=str(result.get("transcribed_text", result.get("hyp_text_raw", ""))),
        explanation=str(result.get("explanation", result.get("error", ""))),
    )


def save_audio(audio: Any, audio_path: Path, sample_rate: int) -> None:
    """Write one generated tensor as the PCM WAV consumed by the ASR validator."""
    import numpy as np
    from scipy.io import wavfile

    samples = audio.detach().to("cpu").numpy().reshape(-1).clip(-1.0, 1.0)
    wavfile.write(audio_path, sample_rate, (samples * 32767.0).astype(np.int16))


def regenerate_candidate(
    model: Any,
    voice_state: Any,
    failure: dict[str, Any],
    attempt: int,
    output_path: Path,
) -> float:
    """Generate one normal-temperature-decrement candidate without touching book WAVs."""
    from pocket_tts.audio_processing.endpoint_cleanup import cleanup_audio_endpoint

    tts_params = failure.get("tts_params", {}) or {}
    initial_temp = float(tts_params.get("temperature", failure.get("original_temp", 0.7)))
    temperature = max(0.1, initial_temp - (attempt - 1) * 0.1)
    frames_after_eos = int(tts_params.get("frames_after_eos", 2))
    text = str(failure.get("text") or failure.get("original_text") or "")
    if not text:
        raise ValueError(f"Failure row {failure.get('chunk_index')} has no source text")
    if hasattr(model, "temp"):
        model.temp = temperature
    audio = model.generate_audio(voice_state, text, frames_after_eos=frames_after_eos)
    audio = cleanup_audio_endpoint(
        audio,
        int(getattr(model, "sample_rate", 24000)),
        threshold=0.004,
        buffer_ms=100,
        use_silero=True,
    )
    save_audio(audio, output_path, int(getattr(model, "sample_rate", 24000)))
    return temperature


def main() -> int:
    """Execute non-destructive resident-ASR recovery phases and save a summary."""
    args = parse_args()
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1")
    tts_dir = args.tts_dir.expanduser().resolve()
    failures = load_failures(tts_dir)
    voice_path = load_voice_path(tts_dir)
    run_dir = args.results_dir.expanduser().resolve() / time.strftime("%Y%m%d-%H%M%S")
    candidate_dir = run_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=False)

    from ASR.asr_validator import cleanup_asr_model, load_asr_model_adaptive, validate_single_chunk
    from pocket_tts.audiobook.generator import AudiobookGenerator
    from pocket_tts.config import ConfigManager

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    config = ConfigManager.load_config(str(ROOT / "pocket_tts/config/default_config.yaml"))
    generator = AudiobookGenerator(config=config)
    print("Loading TTS model and saved voice...", flush=True)
    generator._init_tts_model()
    voice_state = generator._load_voice(str(voice_path))
    print(f"Loading resident {args.model} ASR model...", flush=True)
    asr_model, device = load_asr_model_adaptive(args.model, force_device="cuda")
    if asr_model is None:
        raise RuntimeError(f"Could not load resident {args.model} ASR model")

    original_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    still_failed: list[tuple[dict[str, Any], CandidateResult]] = []
    try:
        for failure in failures:
            chunk_index = int(failure["chunk_index"])
            chunk_id = f"chunk_{chunk_index:05d}"
            original_path = tts_dir / "audio_chunks" / f"{chunk_id}.wav"
            result = validate_single_chunk(
                chunk_id, tts_dir, args.threshold, asr_model,
                audio_path_override=original_path, language=args.language,
            )
            original = candidate_result(0, None, original_path, result)
            row = {"chunk_index": chunk_index, "original": asdict(original), "attempts": []}
            rows.append(row)
            if original.passed:
                continue
            still_failed.append((failure, original))
        original_elapsed = time.perf_counter() - original_started

        regeneration_started = time.perf_counter()
        for failure, _original in still_failed:
            chunk_index = int(failure["chunk_index"])
            row = next(item for item in rows if item["chunk_index"] == chunk_index)
            for attempt in range(1, args.max_retries + 1):
                audio_path = candidate_dir / f"chunk_{chunk_index:05d}_attempt_{attempt}.wav"
                temperature = regenerate_candidate(
                    generator.tts_model, voice_state, failure, attempt, audio_path
                )
                result = validate_single_chunk(
                    f"chunk_{chunk_index:05d}", tts_dir, args.threshold, asr_model,
                    audio_path_override=audio_path, language=args.language,
                )
                candidate = candidate_result(attempt, temperature, audio_path, result)
                row["attempts"].append(asdict(candidate))
                if candidate.passed:
                    break
        regeneration_elapsed = time.perf_counter() - regeneration_started
    finally:
        cleanup_asr_model(asr_model)

    accepted_original = sum(bool(row["original"]["passed"]) for row in rows)
    accepted_regeneration = sum(
        any(attempt["passed"] for attempt in row["attempts"]) for row in rows
    )
    unresolved = len(rows) - accepted_original - accepted_regeneration
    summary = {
        "tts_dir": str(tts_dir),
        "model": args.model,
        "asr_device": device,
        "threshold": args.threshold,
        "input_failures": len(failures),
        "accepted_original": accepted_original,
        "regenerated_rows": len(still_failed),
        "accepted_regeneration": accepted_regeneration,
        "unresolved": unresolved,
        "attempts_generated": sum(len(row["attempts"]) for row in rows),
        "original_recheck_s": round(original_elapsed, 3),
        "regeneration_s": round(regeneration_elapsed, 3),
        "total_s": round(time.perf_counter() - started, 3),
        "rows": rows,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Evidence: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
