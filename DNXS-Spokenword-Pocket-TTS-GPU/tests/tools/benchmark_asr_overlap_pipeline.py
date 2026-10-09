#!/usr/bin/env python3
"""Benchmark post-gen, CPU-overlap, and GPU-overlap ASR against real TTS work.

The harness reads an existing book fragment from ``input/`` and generates a
bounded number of its real chunks with the normal audiobook engine.  Every
scenario validates the same sample of already-generated book chunks, so the
ASR workload is identical while only its scheduling changes.

Scenarios
---------
``postgen_gpu``
    Full-speed TTS, then packed CUDA faster-whisper. Current production plan.
``cpu_overlap``
    Start CPU faster-whisper after the first generated WAV.  Pipeline wall time
    is the later of TTS completion and ASR completion.
``gpu_overlap``
    Same overlap but with CUDA ASR. This is experimental and exists to measure
    TTS slowdown or VRAM failure before considering it for production.

Example
-------
``venv/bin/python tests/tools/benchmark_asr_overlap_pipeline.py \\
  --input 'input/Thirteen Doctors 13 Stories - Naomi Alderm - Doctor Who output_part1.txt' \\
  --asr-source 'Output/Doctor Who_ The Edge of Destruction/TTS'``
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pocket_tts.audiobook.generator import AudiobookGenerator
from pocket_tts.config import ConfigManager
from pocket_tts.preprocessing.chunker import SmartChunker
from pocket_tts.preprocessing.structure_detector import StructureDetector


RESULTS_ROOT = ROOT / "tests" / "results" / "asr_overlap_pipeline"
ASR_SCRIPT = ROOT / "ASR" / "asr_validator.py"


def discover_chunk_stems(tts_dir: Path) -> List[str]:
    """Return chunk stems that have both audio and matching text sidecars."""
    audio_dir = tts_dir / "audio_chunks"
    text_dir = tts_dir / "text_chunks"
    return [
        wav.stem
        for wav in sorted(audio_dir.glob("chunk_*.wav"))
        if (text_dir / f"{wav.stem}.txt").is_file()
    ]


def build_asr_sample(source_tts: Path, dest: Path, chunk_count: int) -> Path:
    """Create a symlinked, fixed ASR corpus without copying source WAV files."""
    stems = discover_chunk_stems(source_tts)
    if not stems:
        raise FileNotFoundError(f"No audio/text chunk pairs under {source_tts}")
    selected = stems[: min(chunk_count, len(stems))]
    if dest.exists():
        shutil.rmtree(dest)
    audio_dest = dest / "audio_chunks"
    text_dest = dest / "text_chunks"
    audio_dest.mkdir(parents=True)
    text_dest.mkdir(parents=True)
    for stem in selected:
        (audio_dest / f"{stem}.wav").symlink_to(
            (source_tts / "audio_chunks" / f"{stem}.wav").resolve()
        )
        (text_dest / f"{stem}.txt").symlink_to(
            (source_tts / "text_chunks" / f"{stem}.txt").resolve()
        )
    return dest


def make_real_book_chunks(input_path: Path, chunk_limit: int) -> list[Any]:
    """Chunk real input-book prose and supply normal default TTS parameters."""
    text = input_path.read_text(encoding="utf-8")
    structure = StructureDetector().analyze(text)
    chunks = SmartChunker(mode="sentence", min_words=5).chunk(structure)
    chunks = chunks[: min(chunk_limit, len(chunks))]
    if not chunks:
        raise ValueError(f"No TTS chunks derived from {input_path}")
    for chunk in chunks:
        chunk.tts_params = {
            "temperature": 0.7,
            "frames_after_eos": 2,
            "eos_threshold": -4.0,
            "lsd_decode_steps": 4,
            "speed_factor": 1.0,
        }
        chunk.post_process = {"silence_duration": 0.0}
    return chunks


def build_generator(worker_count: int) -> AudiobookGenerator:
    """Build a production-like TTS generator with post-gen ASR disabled."""
    config = ConfigManager.load_config("pocket_tts/config/default_config.yaml")
    config.parallel["enabled"] = worker_count > 1
    config.parallel["max_workers"] = worker_count
    config.asr_quality_control["enabled"] = False
    config.m4b["enabled"] = False
    return AudiobookGenerator(config=config)


def asr_command(
    python_exe: Path,
    sample_tts: Path,
    scenario_dir: Path,
    device: str,
    model: str,
    workers: int,
) -> list[str]:
    """Build a full-folder ASR command for a fixed comparison corpus."""
    return [
        str(python_exe),
        str(ASR_SCRIPT),
        "--batch-tts-dir",
        str(sample_tts),
        "--log-file",
        str(scenario_dir / "asr_failures.json"),
        "--threshold",
        "0.8",
        "--model",
        model,
        "--language",
        "en",
        "--device",
        device,
        "--cpu-workers",
        str(workers),
        "--engine",
        "faster_whisper",
        "--pack-size",
        "8",
    ]


def run_asr_job(
    command: list[str], scenario_dir: Path, label: str
) -> Dict[str, Any]:
    """Run ASR synchronously and preserve stdout for later evidence review."""
    log_path = scenario_dir / f"{label}_asr.log"
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
    return {
        "wall_s": time.monotonic() - started,
        "returncode": completed.returncode,
        "log": str(log_path),
    }


def start_asr_job(
    command: list[str], scenario_dir: Path, label: str
) -> tuple[subprocess.Popen[str], Path, float]:
    """Start an ASR process concurrently and return its log path and start time."""
    log_path = scenario_dir / f"{label}_asr.log"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Keep handle alive on process so it closes after wait below.
    process._pocketgpu_log_file = log_file  # type: ignore[attr-defined]
    return process, log_path, time.monotonic()


def wait_for_first_wav(audio_dir: Path, generator_thread: threading.Thread) -> bool:
    """Wait for first real WAV publication while generation remains active."""
    deadline = time.monotonic() + 600.0
    while time.monotonic() < deadline:
        if any(audio_dir.glob("chunk_*.wav")):
            return True
        if not generator_thread.is_alive():
            return False
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for first WAV under {audio_dir}")


def run_scenario(
    *,
    name: str,
    schedule: str,
    chunks: list[Any],
    sample_tts: Path,
    python_exe: Path,
    worker_count: int,
    asr_device: str,
    asr_model: str,
    asr_workers: int,
) -> Dict[str, Any]:
    """Run one scheduling plan and return TTS, ASR, and pipeline wall times."""
    scenario_dir = RESULTS_ROOT / name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    source_name = f"asr_overlap_{name}.txt"
    output_paths = AudiobookGenerator.generate_output_paths(source_name, "alba")
    generator = build_generator(worker_count)
    tts_result: Dict[str, Any] = {}
    tts_error: List[str] = []

    def generate() -> None:
        """Run bounded real-book TTS and capture its result for the parent thread."""
        try:
            result = generator.generate_audiobook(
                chunks=chunks,
                voice_path="alba",
                output_path=str(output_paths["final_audio_path"]),
                source_file=source_name,
                save_dataset_chunks=True,
                total_start_time=time.time(),
            )
            tts_result.update(result)
        except Exception as exc:
            tts_error.append(f"{type(exc).__name__}: {exc}")

    command = asr_command(
        python_exe,
        sample_tts,
        scenario_dir,
        asr_device,
        asr_model,
        asr_workers,
    )
    pipeline_started = time.monotonic()
    tts_started = time.monotonic()
    tts_thread = threading.Thread(target=generate, name=f"tts-{name}")
    tts_thread.start()
    asr: Optional[Dict[str, Any]] = None

    if schedule == "postgen_gpu":
        tts_thread.join()
        tts_wall = time.monotonic() - tts_started
        asr = run_asr_job(command, scenario_dir, name)
    else:
        audio_dir = Path(output_paths["audio_chunks_dir"])
        first_wav_seen = wait_for_first_wav(audio_dir, tts_thread)
        process, log_path, asr_started = start_asr_job(command, scenario_dir, name)
        tts_thread.join()
        tts_wall = time.monotonic() - tts_started
        returncode = process.wait(timeout=900)
        log_file = process._pocketgpu_log_file  # type: ignore[attr-defined]
        log_file.close()
        asr = {
            "wall_s": time.monotonic() - asr_started,
            "returncode": returncode,
            "log": str(log_path),
            "first_wav_seen": first_wav_seen,
        }

    pipeline_wall = time.monotonic() - pipeline_started
    return {
        "scenario": name,
        "asr_device": asr_device,
        "asr_model": asr_model,
        "asr_workers": asr_workers,
        "tts_workers": worker_count,
        "tts_batch_size": 4,
        "asr_batch_size": 8,
        "tts_wall_s": tts_wall,
        "tts_success": bool(tts_result.get("success")) and not tts_error,
        "tts_result": tts_result,
        "tts_error": tts_error,
        "asr": asr,
        "pipeline_wall_s": pipeline_wall,
        "output_tts_dir": str(output_paths["tts_dir"]),
    }


def run_model_probe(
    *,
    sample_tts: Path,
    python_exe: Path,
    model: str,
    workers: int,
) -> Dict[str, Any]:
    """Measure post-gen CUDA ASR speed and failures for one model choice."""
    probe_dir = RESULTS_ROOT / "model_probe" / model
    probe_dir.mkdir(parents=True, exist_ok=True)
    result = run_asr_job(
        asr_command(python_exe, sample_tts, probe_dir, "cuda", model, workers),
        probe_dir,
        f"gpu_{model}",
    )
    failure_path = probe_dir / "asr_failures.json"
    failures = []
    if failure_path.exists():
        failures = json.loads(failure_path.read_text(encoding="utf-8"))
    result.update({"model": model, "failures": len(failures)})
    return result


def parse_models(spec: str) -> Iterable[str]:
    """Yield normalized, nonempty ASR model names from a comma list."""
    for model in spec.split(","):
        normalized = model.strip()
        if normalized:
            yield normalized


def main() -> int:
    """Execute scheduling and model probes, then write a single evidence JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Real input-book text file")
    parser.add_argument("--asr-source", type=Path, required=True, help="Existing TTS folder for fixed ASR workload")
    parser.add_argument("--tts-chunks", type=int, default=96, help="Real input chunks per TTS scenario")
    parser.add_argument("--asr-chunks", type=int, default=128, help="Fixed existing chunks per ASR job")
    parser.add_argument("--tts-workers", type=int, default=3, help="Parallel TTS workers")
    parser.add_argument("--asr-workers", type=int, default=4, help="ASR pipeline CPU worker budget")
    parser.add_argument("--asr-model", default="base", help="ASR model for scheduling scenarios")
    parser.add_argument("--model-probe", default="base,small", help="CUDA post-gen models to compare")
    parser.add_argument("--asr-python", type=Path, default=Path(sys.executable), help="Python with faster-whisper")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.asr_python.is_file():
        raise FileNotFoundError(args.asr_python)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    sample_tts = build_asr_sample(
        args.asr_source.resolve(), RESULTS_ROOT / run_id / "asr_sample", args.asr_chunks
    )
    chunks = make_real_book_chunks(args.input.resolve(), args.tts_chunks)

    scenarios = [
        ("postgen_gpu", "cuda"),
        ("cpu_overlap", "cpu"),
        ("gpu_overlap", "cuda"),
    ]
    rows = []
    for name, device in scenarios:
        print(f"\n=== {name} ===", flush=True)
        rows.append(
            run_scenario(
                name=f"{run_id}_{name}",
                schedule=name,
                chunks=chunks,
                sample_tts=sample_tts,
                python_exe=args.asr_python,
                worker_count=args.tts_workers,
                asr_device=device,
                asr_model=args.asr_model,
                asr_workers=args.asr_workers,
            )
        )

    probes = []
    for model in parse_models(args.model_probe):
        print(f"\n=== model probe: {model} ===", flush=True)
        probes.append(
            run_model_probe(
                sample_tts=sample_tts,
                python_exe=args.asr_python,
                model=model,
                workers=args.asr_workers,
            )
        )

    payload = {
        "run_id": run_id,
        "input": str(args.input.resolve()),
        "asr_source": str(args.asr_source.resolve()),
        "tts_chunks": len(chunks),
        "asr_chunks": len(discover_chunk_stems(sample_tts)),
        "scenarios": rows,
        "model_probes": probes,
    }
    result_path = RESULTS_ROOT / run_id / "summary.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
