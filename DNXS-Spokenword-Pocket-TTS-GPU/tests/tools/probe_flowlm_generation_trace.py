#!/usr/bin/env python3
"""Trace repeated Pocket TTS FlowLM generation and correlate it with ASR.

This probe follows the successful text-token boundary test.  It performs three
fresh model passes over one book input and writes WAVs, human-readable trace
logs, structured data, and a cross-run report.  For each generated chunk it
records a SHA-256 digest of the autoregressive FlowLM audio latents, EOS
position, maximum permitted generation length, a WAV digest, and the batch-ASR
result.  It does *not* claim that audio latents are decodable text: ASR is the
speech-level evidence.

CUDA graphs are disabled only for this diagnostic process.  That leaves the
normal eager FlowLM path intact while allowing the hook to observe every AR
step.  The production generator and its settings are never modified.

Example
-------
``venv/bin/python tests/tools/probe_flowlm_generation_trace.py \\
  --input 'input/His (A Dark Erotic Romance Novel) - Dark, Aubrey_output_part25.txt'``
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RESULTS_ROOT = ROOT / "tests" / "results" / "flowlm_generation_trace"
DEFAULT_INPUT = ROOT / "input" / "His (A Dark Erotic Romance Novel) - Dark, Aubrey_output_part25.txt"


@dataclass
class ChunkTrace:
    """One generated chunk's FlowLM, WAV, and ASR diagnostic evidence."""

    chunk_number: int
    original_text: str
    text_token_count: int
    text_token_sha256: str
    latent_step_count: int
    latent_sha256: str
    eos_step: int | None
    max_generation_length: int | None
    reached_limit: bool
    audio_seconds: float
    wav_sha256: str
    asr_status: str = "NOT_RUN"
    asr_similarity: float | None = None
    asr_transcript: str = ""
    asr_explanation: str = ""


class FlowLMLatentCapture:
    """Temporarily hash every eager autoregressive FlowLM latent for one chunk."""

    def __init__(self, model: Any) -> None:
        """Store model method and initialise one independent latent digest."""
        self.model = model
        self._original_method = model._run_flow_lm_and_increment_step
        self._digest = hashlib.sha256()
        self.step_count = 0
        self.eos_step: int | None = None

    def __enter__(self) -> "FlowLMLatentCapture":
        """Replace the eager FlowLM method with a bound trace wrapper."""
        def traced(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            """Call FlowLM then add this autoregressive latent to the digest."""
            latent, is_eos = self._original_method(*args, **kwargs)
            backbone = kwargs.get("backbone_input_latents")
            if backbone is not None and int(backbone.shape[1]) > 0:
                latent_bytes = (
                    latent.detach().to(dtype=torch.float32, device="cpu")
                    .contiguous().numpy().tobytes()
                )
                self._digest.update(latent_bytes)
                if bool(is_eos.item()) and self.eos_step is None:
                    self.eos_step = self.step_count
                self.step_count += 1
            return latent, is_eos

        self.model._run_flow_lm_and_increment_step = traced
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Restore original FlowLM method regardless of generation outcome."""
        self.model._run_flow_lm_and_increment_step = self._original_method

    @property
    def sha256(self) -> str:
        """Return the sequential SHA-256 digest of generated AR latent frames."""
        return self._digest.hexdigest()


def seed_run(seed: int) -> None:
    """Seed supported random generators without changing global deterministic settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_chunks(input_path: Path, min_words: int) -> list[tuple[int, str]]:
    """Use Pocket TTS sentence chunking so the probe matches normal input shape."""
    from pocket_tts.preprocessing.chunker import SmartChunker
    from pocket_tts.preprocessing.structure_detector import StructureDetector

    structure = StructureDetector().analyze(input_path.read_text(encoding="utf-8"))
    return [
        (chunk_number, chunk.text)
        for chunk_number, chunk in enumerate(
            SmartChunker(mode="sentence", min_words=min_words).chunk(structure),
            start=1,
        )
    ]


def make_failed_chunk_set(tts_dir: Path, failure_path: Path) -> tuple[list[tuple[int, str]], str]:
    """Load exact original text and voice from an existing GUI ASR failure set."""
    failures = json.loads(failure_path.read_text(encoding="utf-8"))
    chunk_numbers = sorted({int(row["chunk_index"]) for row in failures})
    chunks: list[tuple[int, str]] = []
    for chunk_number in chunk_numbers:
        text_path = tts_dir / "text_chunks" / f"chunk_{chunk_number:05d}.txt"
        if not text_path.is_file():
            raise FileNotFoundError(f"Failed chunk text missing: {text_path}")
        chunks.append((chunk_number, text_path.read_text(encoding="utf-8").strip()))
    metadata_path = tts_dir / "text_chunks" / "audiobook.chunks.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    voice = str(metadata.get("_metadata", {}).get("voice_used", ""))
    if not voice:
        raise ValueError(f"voice_used missing from {metadata_path}")
    return chunks, voice


def sha256_bytes(data: bytes) -> str:
    """Return a stable SHA-256 digest for small binary diagnostic artifacts."""
    return hashlib.sha256(data).hexdigest()


def save_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> str:
    """Write one mono PCM WAV and return its exact file digest."""
    from scipy.io import wavfile

    samples = audio.detach().to(dtype=torch.float32, device="cpu").numpy()
    if samples.ndim > 1:
        samples = samples.reshape(-1)
    pcm = np.clip(samples, -1.0, 1.0)
    wavfile.write(path, sample_rate, (pcm * 32767.0).astype(np.int16))
    return sha256_bytes(path.read_bytes())


def token_evidence(model: Any, text: str) -> tuple[int, str]:
    """Return text-token count and digest for correlation with prior boundary test."""
    ids = model.flow_lm.conditioner.prepare(text).tokens.detach().to("cpu").numpy()
    return int(ids.size), sha256_bytes(ids.tobytes())


def generate_run(
    run_number: int,
    chunks: Sequence[tuple[int, str]],
    output_dir: Path,
    voice: str,
    seed: int,
) -> list[ChunkTrace]:
    """Load a fresh GPU model, generate every chunk, and persist FlowLM traces."""
    from pocket_tts.models.tts_model import TTSModel

    seed_run(seed)
    run_root = output_dir / f"run_{run_number}"
    tts_dir = run_root / "TTS"
    audio_dir = tts_dir / "audio_chunks"
    text_dir = tts_dir / "text_chunks"
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    model = TTSModel.load_model(device="cuda")
    # The trace wrapper sees only eager calls; graph replay would hide AR frames.
    model.cuda_graphs_enabled = False
    os.environ["POCKET_TTS_CUDA_GRAPHS"] = "0"
    state = model.get_state_for_audio_prompt(voice, truncate=True)
    sample_rate = int(model.config.mimi.sample_rate)
    rows: list[ChunkTrace] = []
    try:
        for sequence_number, (chunk_number, text) in enumerate(chunks, start=1):
            wav_path = audio_dir / f"chunk_{chunk_number:05d}.wav"
            text_path = text_dir / f"chunk_{chunk_number:05d}.txt"
            text_path.write_text(text + "\n", encoding="utf-8")
            token_count, token_hash = token_evidence(model, text)
            with FlowLMLatentCapture(model) as capture:
                audio = model.generate_audio(state, text, copy_state=True)
            wav_hash = save_wav(wav_path, audio, sample_rate)
            rows.append(
                ChunkTrace(
                    chunk_number=chunk_number,
                    original_text=text,
                    text_token_count=token_count,
                    text_token_sha256=token_hash,
                    latent_step_count=capture.step_count,
                    latent_sha256=capture.sha256,
                    eos_step=getattr(model, "last_generation_eos_step", capture.eos_step),
                    max_generation_length=getattr(model, "last_generation_max_len", None),
                    reached_limit=bool(getattr(model, "last_generation_reached_limit", False)),
                    audio_seconds=round(float(audio.numel()) / sample_rate, 4),
                    wav_sha256=wav_hash,
                )
            )
            # Direct sequential tracing has no worker teardown boundary; release the
            # decoded tensor each chunk so a long probe cannot grow the CUDA cache.
            del audio
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Run {run_number}: generated {sequence_number}/{len(chunks)} (chunk {chunk_number:05d})", flush=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def run_asr(tts_dir: Path, failure_path: Path) -> dict[int, dict[str, Any]]:
    """Run existing batch ASR once and map its failures by one-based chunk number."""
    command = [
        str(ROOT / "venv" / "bin" / "python"), str(ROOT / "ASR" / "asr_validator.py"),
        "--batch-tts-dir", str(tts_dir), "--log-file", str(failure_path),
        "--threshold", "0.8", "--model", "base", "--language", "en",
        "--device", "cuda", "--cpu-workers", "1", "--engine", "faster_whisper",
        "--pack-size", "8",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (failure_path.parent / "asr_console.log").write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8"
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"ASR failed with exit code {completed.returncode}; see {failure_path.parent / 'asr_console.log'}")
    if not failure_path.exists():
        return {}
    failures = json.loads(failure_path.read_text(encoding="utf-8"))
    mapped: dict[int, dict[str, Any]] = {}
    for failure in failures:
        chunk_index = failure.get("chunk_index")
        if chunk_index is not None:
            mapped[int(chunk_index)] = failure
    return mapped


def attach_asr(rows: Iterable[ChunkTrace], failures: dict[int, dict[str, Any]]) -> None:
    """Mark each trace PASS or FAIL and carry ASR evidence into its row."""
    for row in rows:
        failure = failures.get(row.chunk_number)
        if failure is None:
            row.asr_status = "PASS"
            continue
        row.asr_status = "FAIL"
        row.asr_similarity = failure.get("score")
        row.asr_transcript = str(failure.get("transcribed_text", ""))
        row.asr_explanation = str(failure.get("explanation", ""))


def write_run_log(path: Path, run_number: int, rows: Iterable[ChunkTrace]) -> None:
    """Write readable per-chunk FlowLM and ASR evidence for one generated run."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Pocket TTS FlowLM Generation Trace\n")
        handle.write(f"Run: {run_number}\n\n")
        for row in rows:
            handle.write(f"Chunk: {row.chunk_number:05d}\n")
            handle.write(f"Original: {row.original_text}\n")
            handle.write(f"Text tokens: {row.text_token_count} | sha256={row.text_token_sha256}\n")
            handle.write(f"FlowLM latents: steps={row.latent_step_count} | sha256={row.latent_sha256}\n")
            handle.write(f"EOS: step={row.eos_step} | max={row.max_generation_length} | reached_limit={row.reached_limit}\n")
            handle.write(f"WAV: seconds={row.audio_seconds:.4f} | sha256={row.wav_sha256}\n")
            handle.write(f"ASR: {row.asr_status} | similarity={row.asr_similarity}\n")
            if row.asr_status == "FAIL":
                handle.write(f"ASR transcript: {row.asr_transcript}\n")
                handle.write(f"ASR explanation: {row.asr_explanation}\n")
            handle.write("\n")


def classify_chunk(rows: Sequence[ChunkTrace]) -> str:
    """Classify repeated evidence to isolate FlowLM, decoder, EOS, or ASR signals."""
    if any(row.reached_limit or row.eos_step is None for row in rows):
        return "EOS_OR_MAX_LIMIT_ANOMALY"
    if len({row.latent_sha256 for row in rows}) > 1:
        return "FLOWLM_LATENT_DIVERGENCE"
    if len({row.wav_sha256 for row in rows}) > 1:
        return "DECODER_OR_WAV_DIVERGENCE"
    if any(row.asr_status == "FAIL" for row in rows):
        return "STABLE_AUDIO_ASR_FAILURE"
    return "STABLE_PASS"


def write_report(path: Path, input_path: Path, run_rows: Sequence[Sequence[ChunkTrace]], seed: int) -> None:
    """Write cross-run stability findings and enumerate any suspect chunks."""
    groups = list(zip(*run_rows))
    classifications: dict[str, list[int]] = {}
    for group in groups:
        classifications.setdefault(classify_chunk(group), []).append(group[0].chunk_number)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Pocket TTS FlowLM Generation Trace Report\n")
        handle.write("=" * 44 + "\n")
        handle.write(f"Input: {input_path}\nRuns: {len(run_rows)}\nSeed per run: {seed}\n")
        handle.write("Trace mode: CUDA graphs disabled only to expose eager FlowLM AR frames.\n\n")
        for run_number, rows in enumerate(run_rows, start=1):
            handle.write(
                f"Run {run_number}: chunks={len(rows)}, ASR pass={sum(row.asr_status == 'PASS' for row in rows)}, "
                f"ASR fail={sum(row.asr_status == 'FAIL' for row in rows)}, "
                f"EOS/max anomalies={sum(row.reached_limit or row.eos_step is None for row in rows)}\n"
            )
        handle.write("\nCross-run classifications:\n")
        for label, chunk_numbers in classifications.items():
            handle.write(f"{label}: {len(chunk_numbers)}\n")
            if label != "STABLE_PASS":
                handle.write("  chunks: " + ", ".join(f"{number:05d}" for number in chunk_numbers) + "\n")
        verdict = "PASS: all chunks were stable across repeated FlowLM/WAV traces and passed ASR."
        if set(classifications) != {"STABLE_PASS"}:
            verdict = "INVESTIGATE: see non-STABLE_PASS classifications above."
        handle.write(f"\nVerdict: {verdict}\n")
        handle.write(
            "Interpretation: latent divergence points upstream at FlowLM sampling/state; identical latents with WAV divergence points at decoding; identical audio with ASR failure points at speech output rather than text tokenization.\n"
        )


def main() -> int:
    """Execute three fresh FlowLM/WAV/ASR runs and write diagnostic artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Mid-size book text input")
    parser.add_argument("--voice", default="alba", help="Built-in Pocket TTS voice or WAV path")
    parser.add_argument(
        "--tts-dir", type=Path, default=None,
        help="Existing GUI TTS folder; selects exact chunks listed in --failure-file",
    )
    parser.add_argument(
        "--failure-file", type=Path, default=None,
        help="Existing asr_failures.json; defaults to --tts-dir/asr_failures.json",
    )
    parser.add_argument("--min-words", type=int, default=5, help="Pocket TTS chunk merge floor")
    parser.add_argument("--runs", type=int, default=3, help="Fresh model passes to generate")
    parser.add_argument("--seed", type=int, default=20260729, help="Seed reused for every run")
    parser.add_argument(
        "--max-chunks", type=int, default=64,
        help="Mid-size diagnostic chunk cap; use 0 only for an entire input",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Artifact destination")
    args = parser.parse_args()
    if args.tts_dir is None and not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.runs != 3:
        raise ValueError("This diagnostic requires exactly three runs.")
    source_label = args.input.resolve()
    voice = args.voice
    if args.tts_dir is not None:
        failure_path = args.failure_file or args.tts_dir / "asr_failures.json"
        if not failure_path.is_file():
            raise FileNotFoundError(failure_path)
        chunks, source_voice = make_failed_chunk_set(args.tts_dir, failure_path)
        source_label = failure_path.resolve()
        if args.voice == "alba":
            voice = source_voice
    else:
        chunks = make_chunks(args.input, args.min_words)
    if args.max_chunks > 0:
        chunks = chunks[:args.max_chunks]
    if not chunks:
        raise ValueError("Input did not produce any chunks.")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or DEFAULT_RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[list[ChunkTrace]] = []
    for run_number in range(1, args.runs + 1):
        rows = generate_run(run_number, chunks, output_dir, voice, args.seed)
        failures = run_asr(output_dir / f"run_{run_number}" / "TTS", output_dir / f"run_{run_number}" / "asr_failures.json")
        attach_asr(rows, failures)
        write_run_log(output_dir / f"flowlm_trace_run_{run_number}.log", run_number, rows)
        (output_dir / f"run_{run_number}" / "trace.json").write_text(
            json.dumps([asdict(row) for row in rows], indent=2) + "\n", encoding="utf-8"
        )
        all_rows.append(rows)
    write_report(output_dir / "flowlm_trace_report.txt", source_label, all_rows, args.seed)
    (output_dir / "flowlm_trace_summary.json").write_text(
        json.dumps([[asdict(row) for row in rows] for rows in all_rows], indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote FlowLM traces, WAVs, ASR evidence, and report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
