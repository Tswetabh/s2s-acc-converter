"""Benchmark Pocket TTS throughput and resource usage with fixed workloads."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pocket_tts.audiobook.generator import AudiobookGenerator
from pocket_tts.config import ConfigManager
from pocket_tts.preprocessing.schema import (
    BoundaryType,
    ChunkMetadata,
    EmotionType,
)


BENCHMARK_TEXT = (
    "The measured benchmark sentence contains enough words to exercise text "
    "preparation, autoregressive latent generation, audio decoding, endpoint "
    "cleanup, and file output without relying on external input files."
)


class ResourceSampler:
    """Sample process CPU and first-GPU utilization during a benchmark run."""

    def __init__(self, interval_seconds: float = 0.25):
        """Initialize sampler with a fixed polling interval."""
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()

    def start(self) -> None:
        """Start collecting resource samples in a background thread."""
        self._process.cpu_percent(None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for the sampler thread to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        """Collect one resource sample per polling interval."""
        while not self._stop.is_set():
            sample = {
                "cpu_percent": self._process.cpu_percent(None),
                "ram_percent": psutil.virtual_memory().percent,
            }
            if torch.cuda.is_available():
                gpu_metrics = _query_gpu_metrics()
                sample.update(gpu_metrics)
            self.samples.append(sample)
            self._stop.wait(self.interval_seconds)


def _query_gpu_metrics() -> dict[str, float]:
    """Read aggregate GPU utilization and memory from nvidia-smi."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        utilization, used, total = result.stdout.strip().splitlines()[0].split(", ")
        return {
            "gpu_utilization_percent": float(utilization),
            "gpu_memory_used_mb": float(used),
            "gpu_memory_total_mb": float(total),
        }
    except Exception:
        return {}


def _make_chunks(count: int) -> list[ChunkMetadata]:
    """Create deterministic benchmark chunks with identical TTS settings."""
    chunks = []
    params = {
        "temperature": 0.7,
        "frames_after_eos": 2,
        "eos_threshold": -4.0,
        "lsd_decode_steps": 4,
        "speed_factor": 1.0,
    }
    for index in range(count):
        chunks.append(
            ChunkMetadata(
                index=index,
                text=f"{BENCHMARK_TEXT} Benchmark chunk {index + 1}.",
                word_count=len(BENCHMARK_TEXT.split()) + 3,
                character_count=len(BENCHMARK_TEXT) + 20,
                boundary_type=BoundaryType.SENTENCE_END,
                punctuation=".",
                start_position=index * 100,
                end_position=(index + 1) * 100,
                emotion=EmotionType.NEUTRAL,
                emotion_scores={"neutral": 1.0},
                emotion_confidence=1.0,
                tts_params=params,
                post_process={},
            )
        )
    return chunks


def _summarize_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    """Return peak and average resource values from sampler output."""
    if not samples:
        return {}
    keys = samples[0].keys()
    summary = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        summary[f"{key}_avg"] = sum(values) / len(values)
        summary[f"{key}_peak"] = max(values)
    return summary


def run_benchmark(worker_count: int, chunk_count: int) -> dict[str, Any]:
    """Run one warm benchmark using fixed chunks and requested worker cap."""
    config = ConfigManager.load_config("pocket_tts/config/default_config.yaml")
    config.parallel["enabled"] = worker_count > 1
    config.parallel["max_workers"] = worker_count
    config.asr_quality_control["enabled"] = False
    config.m4b["enabled"] = False
    config.audio_cleanup["enabled"] = False

    source_file = f"/tmp/pocketgpu_benchmark_{worker_count}.txt"
    output_path = f"/tmp/pocketgpu_benchmark_{worker_count}.wav"
    generator = AudiobookGenerator(config=config)
    chunks = _make_chunks(chunk_count)
    sampler = ResourceSampler()
    started = time.monotonic()
    total_start_time = time.time()
    sampler.start()
    try:
        result = generator.generate_audiobook(
            chunks=chunks,
            voice_path="alba",
            output_path=output_path,
            source_file=source_file,
            save_dataset_chunks=True,
            total_start_time=total_start_time,
        )
    finally:
        sampler.stop()

    elapsed = time.monotonic() - started
    result.update(
        {
            "benchmark_worker_count": worker_count,
            "benchmark_chunk_count": chunk_count,
            "benchmark_wall_time": elapsed,
            "benchmark_audio_seconds": result.get("audio_duration", 0.0),
            "benchmark_chars_per_second": sum(len(chunk.text) for chunk in chunks) / elapsed,
            "benchmark_words_per_second": sum(chunk.word_count for chunk in chunks) / elapsed,
            "resources": _summarize_samples(sampler.samples),
        }
    )
    return result


def main() -> None:
    """Run requested worker counts and print machine-readable benchmark JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--chunks", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = []
    for worker_count in args.workers:
        results.append(run_benchmark(worker_count, args.chunks))
    output = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
