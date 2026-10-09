#!/usr/bin/env python3
"""Regenerate every row in a PocketTTS ``asr_failures.json`` report.

This is a headless entry point for PocketTTS's existing failed-chunk recovery
method.  It loads the book's saved voice, then calls
``AudiobookGenerator._reprocess_failed_chunks`` unchanged.  Therefore each
row gets the production sequence: original candidate plus every configured
regeneration attempt, endpoint cleanup, single-chunk ASR, and best-score
selection.

Example
-------
``venv/bin/python tests/tools/regenerate_asr_failures.py \
  --tts-dir 'Output/Doctor Who_ The Edge of Destruction/TTS' --max-retries 3``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse required book folder and production regeneration overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tts-dir",
        type=Path,
        required=True,
        help="Book-local TTS directory containing asr_failures.json.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Regeneration attempts per failed chunk (default: 3).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="ASR score threshold; defaults to configured production value.",
    )
    parser.add_argument(
        "--temp-decrement",
        type=float,
        default=None,
        help="Temperature reduction per retry; defaults to production value.",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help="Override configured faster-whisper model for attempt checks.",
    )
    return parser.parse_args()


def load_voice_path(tts_dir: Path) -> Path:
    """Read exact saved ``voice_used`` path from this book's chunk metadata."""
    metadata_path = tts_dir / "text_chunks" / "audiobook.chunks.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Chunk metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    voice_used = metadata.get("_metadata", {}).get("voice_used")
    if not voice_used:
        raise ValueError(f"voice_used is missing from {metadata_path}")
    voice_path = Path(str(voice_used)).expanduser()
    if not voice_path.is_absolute():
        voice_path = (ROOT / voice_path).resolve()
    if not voice_path.is_file():
        raise FileNotFoundError(f"Saved voice file not found: {voice_path}")
    return voice_path


def validate_inputs(tts_dir: Path) -> tuple[Path, int]:
    """Confirm this folder has complete saved data for in-place regeneration."""
    failure_path = tts_dir / "asr_failures.json"
    audio_dir = tts_dir / "audio_chunks"
    text_dir = tts_dir / "text_chunks"
    for path in (failure_path, audio_dir, text_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required TTS path not found: {path}")
    failures = json.loads(failure_path.read_text(encoding="utf-8"))
    if not isinstance(failures, list):
        raise ValueError(f"Failure report must be a JSON list: {failure_path}")
    if not failures:
        raise ValueError(f"No failed chunks in {failure_path}")
    return failure_path, len(failures)


def build_asr_config(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    """Apply explicit CLI overrides while retaining PocketTTS ASR defaults."""
    asr_config = dict(getattr(config, "asr_quality_control", {}) or {})
    asr_config["max_retries"] = args.max_retries
    asr_config["regen_asr_model"] = args.asr_model or asr_config.get("regen_asr_model") or "medium"
    if args.threshold is not None:
        asr_config["threshold"] = args.threshold
    if args.temp_decrement is not None:
        asr_config["temp_decrement"] = args.temp_decrement
    if args.asr_model:
        asr_config["model"] = args.asr_model
    return asr_config


def main() -> int:
    """Run production failed-chunk recovery and print an end-to-end duration."""
    args = parse_args()
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1 for regeneration")

    tts_dir = args.tts_dir.expanduser().resolve()
    failure_path, failure_count = validate_inputs(tts_dir)
    voice_path = load_voice_path(tts_dir)

    from pocket_tts.audiobook.generator import AudiobookGenerator
    from pocket_tts.config import ConfigManager

    config = ConfigManager.load_config(str(ROOT / "pocket_tts/config/default_config.yaml"))
    asr_config = build_asr_config(args, config)
    generator = AudiobookGenerator(config=config)
    dataset_paths = {
        "tts_dir": tts_dir,
        "audio_chunks_dir": tts_dir / "audio_chunks",
        "text_chunks_dir": tts_dir / "text_chunks",
    }

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print(f"Failures: {failure_count}")
    print(f"Voice: {voice_path}")
    print(f"Attempts per failure: {asr_config['max_retries']}")
    print(f"ASR model: {asr_config.get('model', 'base')}")
    print(f"Regen ASR model: {asr_config.get('regen_asr_model', 'medium')}")
    print("Loading production TTS model...", flush=True)
    started = time.perf_counter()
    generator._init_tts_model()
    voice_state = generator._load_voice(str(voice_path))
    investigation = generator._reprocess_failed_chunks(
        str(failure_path), voice_state, dataset_paths, asr_config
    )
    elapsed = time.perf_counter() - started
    print(
        f"Completed {failure_count} failures x {asr_config['max_retries']} attempts "
        f"in {elapsed:.2f}s; unresolved: {len(investigation)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
