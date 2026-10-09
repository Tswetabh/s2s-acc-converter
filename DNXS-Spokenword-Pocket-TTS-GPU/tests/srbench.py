"""Benchmark single- and multi-worker Pocket TTS GPU throughput.

Uses hardcoded short sentences and the built-in alba voice (no file input).

Usage (from repo root, main venv):
  PYTHONPATH=. venv/bin/python tests/srbench.py
  PYTHONPATH=. venv/bin/python tests/srbench.py --workers 3
  PYTHONPATH=. venv/bin/python tests/srbench.py --single-only
  PYTHONPATH=. venv/bin/python tests/srbench.py --multi-only --workers 2
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

torch.set_float32_matmul_precision("high")

BENCHMARK_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "In the distance, the mountains rose like ancient sentinels against the darkening sky.",
    "She opened the old leather journal and began to write about everything she had witnessed that day.",
    "The laboratory equipment hummed quietly as the scientist prepared the next phase of the experiment.",
    "Wind swept through the valley carrying the scent of pine and distant rain across the open meadow.",
    "Scientists discovered a new species of deep-sea fish living near hydrothermal vents at unprecedented depths.",
    "The old library smelled of aged paper and leather bindings, a scent that transported visitors to another era.",
    "Rain drummed steadily against the windowpane while she read aloud from the worn copy of her favorite novel.",
]

DEFAULT_WORKERS = 3
CHUNKS_PER_WORKER = 8


def _worker_fn(worker_id: int, num_chunks: int, result_dict) -> None:
    """Load one TTS model in a child process and generate num_chunks of audio."""
    # Spawn children do not inherit parent sys.path edits reliably; re-add root.
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ.setdefault("PYTHONPATH", root)

    import torch as _torch

    _torch.set_float32_matmul_precision("high")
    from pocket_tts.models.tts_model import TTSModel

    model = TTSModel.load_model(device="auto")
    voice = model.get_state_for_audio_prompt("alba", truncate=True)

    texts = [
        BENCHMARK_TEXTS[(worker_id * num_chunks + i) % len(BENCHMARK_TEXTS)]
        for i in range(num_chunks)
    ]

    durs = []
    for t in texts:
        cs = time.time()
        audio = model.generate_audio(voice, t, frames_after_eos=2)
        ce = time.time()
        dur = audio.shape[-1] / model.sample_rate
        durs.append((ce - cs, dur))
    result_dict[worker_id] = durs


def benchmark_single_worker() -> float:
    """Run single-process TTS on BENCHMARK_TEXTS; return realtime factor."""
    from pocket_tts.models.tts_model import TTSModel

    print("=" * 60)
    print("SINGLE WORKER BENCHMARK")
    print("=" * 60)
    print(f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    model = TTSModel.load_model(device="auto")
    print(f"Model loaded on {model.device}")

    voice = model.get_state_for_audio_prompt("alba", truncate=True)
    print("Voice loaded (built-in alba)\n")

    # Discard first gen so compile/cache cost stays out of timed loop.
    model.generate_audio(voice, "Warmup.", frames_after_eos=2)

    durs = []
    start = time.time()
    for i, t in enumerate(BENCHMARK_TEXTS):
        cs = time.time()
        audio = model.generate_audio(voice, t, frames_after_eos=2)
        ce = time.time()
        dur = audio.shape[-1] / model.sample_rate
        durs.append(dur)
        print(f"  Chunk {i + 1}: gen={ce - cs:.2f}s, audio={dur:.1f}s")

    elapsed = time.time() - start
    total_audio = sum(durs)
    rtf = total_audio / elapsed if elapsed > 0 else 0.0
    print(f"\nResult: {elapsed:.2f}s for {total_audio:.1f}s audio = {rtf:.1f}x realtime")
    return rtf


def benchmark_multi_worker(num_workers: int = DEFAULT_WORKERS) -> float:
    """Spawn num_workers TTS processes and return combined realtime factor."""
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    print("\n" + "=" * 60)
    print(f"MULTI-WORKER BENCHMARK ({num_workers} workers)")
    print("=" * 60)

    manager = mp.Manager()
    result_dict = manager.dict()

    start = time.time()
    procs = []
    for i in range(num_workers):
        p = mp.Process(
            target=_worker_fn,
            args=(i, CHUNKS_PER_WORKER, result_dict),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode not in (0, None):
            print(f"  WARNING: worker pid={p.pid} exitcode={p.exitcode}")

    elapsed = time.time() - start

    if not result_dict:
        print("\nResult: no worker results (all workers failed)")
        return 0.0

    total_audio = 0.0
    for wid in sorted(result_dict.keys()):
        for gen_time, audio_dur in result_dict[wid]:
            print(f"  Worker {wid}: gen={gen_time:.2f}s, audio={audio_dur:.1f}s")
            total_audio += audio_dur

    rtf = total_audio / elapsed if elapsed > 0 else 0.0
    print(f"\nResult: {elapsed:.2f}s for {total_audio:.1f}s audio = {rtf:.1f}x realtime")
    return rtf


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for worker count and which phases to run."""
    p = argparse.ArgumentParser(
        description="Pocket TTS GPU single/multi-worker throughput benchmark",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Multi-worker process count (default {DEFAULT_WORKERS}; 4060 Ti 8GB safe)",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--single-only",
        action="store_true",
        help="Run only the single-worker benchmark",
    )
    mode.add_argument(
        "--multi-only",
        action="store_true",
        help="Run only the multi-worker benchmark",
    )
    return p.parse_args(argv)


def _print_env() -> None:
    """Print torch/CUDA environment details for the summary block."""
    print("Environment info:")
    print(f"  OS: {sys.platform}")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  cuDNN: {torch.backends.cudnn.version()}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  float32_matmul_precision: {torch.get_float32_matmul_precision()}")


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse args, run selected benchmarks, print summary."""
    # Spawn avoids CUDA-fork corruption when child processes load GPU models.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = _parse_args(argv)
    run_single = not args.multi_only
    run_multi = not args.single_only

    single = None
    multi = None
    if run_single:
        single = benchmark_single_worker()
    if run_multi:
        multi = benchmark_multi_worker(num_workers=args.workers)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if single is not None:
        print(f"  Single worker:  {single:.1f}x realtime")
    if multi is not None:
        print(f"  Multi worker ({args.workers}): {multi:.1f}x realtime")
    if single is not None and multi is not None and single > 0:
        print(f"  Parallel speedup: {multi / single:.1f}x")
    print()
    _print_env()


if __name__ == "__main__":
    main()
