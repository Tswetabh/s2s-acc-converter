#!/usr/bin/env python3
"""Paired A/B speed tests using the real multi-worker audiobook pipeline.

Runs the same path as the GUI: AudiobookGenerator + spawn workers + queue
dispatch + per-worker model/voice pool + WAV chunk write. ASR and M4B are
forced off so wall time is generation, not quality control.

Modes:
  item2  — POCKET_TTS_AR_TIMER=context (old per-step timer) vs monotonic (Item #2 cut)
  batch  — batch_generation.enabled false vs true
  graphs — cuda_graphs.enabled false vs true (T3 multi-worker; keeps batch on)

Default book: Doctor Who Edge of Destruction GUI output (text chunks + voice).
Default workers: 4

Examples:
  venv/bin/python tools/ab_generation_speed.py item2 --workers 4 --limit 80
  venv/bin/python tools/ab_generation_speed.py batch --workers 4 --limit 80 --batch-size 4
  venv/bin/python tools/ab_generation_speed.py graphs --workers 4 --limit 80 --batch-size 4
  venv/bin/python tools/ab_generation_speed.py item2 --workers 5 --limit 0   # all chunks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

logger = logging.getLogger("ab_generation_speed")

DEFAULT_BOOK = PROJECT_ROOT / "Output" / "Doctor Who_ The Edge of Destruction"


def _discover_paths(book_dir: Path) -> tuple[Path, Path]:
    """Locate text_chunks and voice WAV under a GUI book output tree."""
    text_dir = book_dir / "TTS" / "text_chunks"
    if not text_dir.is_dir():
        raise FileNotFoundError(f"Missing text_chunks directory: {text_dir}")
    voice_candidates = sorted((book_dir / "TTS").glob("*_converted.wav"))
    if not voice_candidates:
        voice_candidates = sorted((book_dir / "TTS").glob("*.wav"))
    if not voice_candidates:
        raise FileNotFoundError(f"No voice WAV under {book_dir / 'TTS'}")
    return text_dir, voice_candidates[0]


def _load_chunk_texts(text_dir: Path, limit: int | None) -> list[tuple[str, str]]:
    """Load ordered (chunk_id, text) pairs from chunk_*.txt files."""
    paths = sorted(text_dir.glob("chunk_*.txt"))
    if not paths:
        raise FileNotFoundError(f"No chunk_*.txt files in {text_dir}")
    items = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            items.append((path.stem, text))
    if limit is not None:
        items = items[: max(0, limit)]
    if not items:
        raise RuntimeError("No non-empty chunk texts after filtering")
    return items


def _make_chunk_metadata(
    texts: list[tuple[str, str]],
    temperature: float,
    frames_after_eos: int,
    eos_threshold: float,
    lsd_decode_steps: int,
) -> list[ChunkMetadata]:
    """Build ChunkMetadata list matching worker/generator expectations."""
    chunks: list[ChunkMetadata] = []
    params = {
        "temperature": temperature,
        "frames_after_eos": frames_after_eos,
        "eos_threshold": eos_threshold,
        "lsd_decode_steps": lsd_decode_steps,
        "speed_factor": 1.0,
    }
    for index, (_chunk_id, text) in enumerate(texts):
        chunks.append(
            ChunkMetadata(
                index=index,
                text=text,
                word_count=len(text.split()),
                character_count=len(text),
                boundary_type=BoundaryType.SENTENCE_END,
                punctuation=".",
                start_position=index * 100,
                end_position=(index + 1) * 100,
                emotion=EmotionType.NEUTRAL,
                emotion_scores={"neutral": 1.0},
                emotion_confidence=1.0,
                tts_params=dict(params),
                post_process={"silence_duration": 0.0},
            )
        )
    return chunks


def _build_config(
    workers: int,
    batch_enabled: bool,
    batch_size: int,
    lsd_steps: int,
    cuda_graphs: bool = False,
) -> Any:
    """Load default config and force multi-worker gen settings (ASR/M4B off)."""
    config = ConfigManager.load_config("pocket_tts/config/default_config.yaml")
    config.parallel["enabled"] = workers > 1
    config.parallel["max_workers"] = max(1, workers)
    config.parallel["min_workers"] = 1
    config.asr_quality_control["enabled"] = False
    config.m4b["enabled"] = False
    # Keep cleanup on to match production path cost
    if not hasattr(config, "batch_generation") or config.batch_generation is None:
        config.batch_generation = {}
    config.batch_generation["enabled"] = bool(batch_enabled)
    config.batch_generation["batch_size"] = max(2, int(batch_size)) if batch_enabled else max(2, int(batch_size))
    # When batch disabled, still keep batch_size for config shape; enabled=false
    config.batch_generation["batch_regeneration"] = False
    if not hasattr(config, "quality") or config.quality is None:
        config.quality = {}
    config.quality["lsd_steps"] = int(lsd_steps)
    if not hasattr(config, "device") or config.device is None:
        config.device = {}
    config.device["preferred"] = "cuda"
    if not hasattr(config, "cuda_graphs") or config.cuda_graphs is None:
        config.cuda_graphs = {}
    config.cuda_graphs["enabled"] = bool(cuda_graphs)
    config.cuda_graphs["fallback_eager"] = True
    return config


def _parse_worker_timing(timing_dir: Path) -> dict[str, Any]:
    """Aggregate worker_*.jsonl generation times from one run."""
    gen_ms: list[float] = []
    workers: dict[str, int] = {}
    if not timing_dir.is_dir():
        return {"generation_ms_sum": 0.0, "generation_ms_count": 0, "workers_seen": 0}
    for path in sorted(timing_dir.glob("worker_*.jsonl")):
        workers[path.stem] = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "chunk_completed" and event.get("phase") == "original":
                workers[path.stem] += 1
                if "generation_ms" in event:
                    gen_ms.append(float(event["generation_ms"]))
    return {
        "generation_ms_sum": sum(gen_ms),
        "generation_ms_mean": (sum(gen_ms) / len(gen_ms)) if gen_ms else 0.0,
        "generation_ms_count": len(gen_ms),
        "workers_seen": len(workers),
        "chunks_per_worker": workers,
    }


def _run_one(
    label: str,
    chunks: list[ChunkMetadata],
    voice_path: Path,
    workers: int,
    batch_enabled: bool,
    batch_size: int,
    lsd_steps: int,
    ar_timer: str,
    run_root: Path,
    cuda_graphs: bool = False,
) -> dict[str, Any]:
    """Execute one full multi-worker generation and collect wall/gen metrics."""
    run_dir = run_root / label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Workers re-import modules under spawn; env must be set before Process.start.
    os.environ["POCKET_TTS_AR_TIMER"] = ar_timer
    if cuda_graphs:
        os.environ["POCKET_TTS_CUDA_GRAPHS"] = "1"
    else:
        os.environ.pop("POCKET_TTS_CUDA_GRAPHS", None)

    config = _build_config(workers, batch_enabled, batch_size, lsd_steps, cuda_graphs=cuda_graphs)
    generator = AudiobookGenerator(config=config)

    # Unique source stem so generate_output_paths does not collide between A/B runs.
    source_file = str(run_dir / f"ab_{label}.txt")
    Path(source_file).write_text(
        "\n".join(chunk.text for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    # Output path is rewritten into Output/<stem>/ when save_dataset_chunks=True.
    output_path = str(run_dir / f"{label}_final.wav")

    print(
        f"\n=== RUN {label} ===\n"
        f"  workers={workers} batch_enabled={batch_enabled} "
        f"batch_size={batch_size} ar_timer={ar_timer} cuda_graphs={cuda_graphs}\n"
        f"  chunks={len(chunks)} voice={voice_path.name}\n"
        f"  out={run_dir}"
    )

    wall_start = time.perf_counter()
    total_start = time.time()
    result = generator.generate_audiobook(
        chunks=chunks,
        voice_path=str(voice_path),
        output_path=output_path,
        source_file=source_file,
        save_dataset_chunks=True,
        total_start_time=total_start,
    )
    wall_s = time.perf_counter() - wall_start

    # Dataset paths: generator.generate_output_paths places under Output/<stem>/...
    # Recover worker_timing from result paths or by scanning run-related Output.
    timing_stats = {"generation_ms_sum": 0.0, "generation_ms_count": 0, "workers_seen": 0}
    dataset_dir = None
    if result.get("success"):
        # Prefer paths returned by generator
        for key in ("output_path", "final_audio_path", "audio_path"):
            if result.get(key):
                candidate = Path(result[key]).resolve()
                # .../Book/TTS/final.wav or similar
                tts_dir = candidate.parent if candidate.parent.name == "TTS" else candidate.parent
                if (tts_dir / "worker_timing").is_dir():
                    dataset_dir = tts_dir
                    break
                if (tts_dir.parent / "TTS" / "worker_timing").is_dir():
                    dataset_dir = tts_dir.parent / "TTS"
                    break
        if dataset_dir is None:
            # Search recent Output trees named after source stem
            stem = Path(source_file).stem
            matches = sorted(
                PROJECT_ROOT.glob(f"Output/**/{stem}*/TTS/worker_timing"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not matches:
                matches = sorted(
                    PROJECT_ROOT.glob("Output/**/TTS/worker_timing"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            if matches:
                dataset_dir = matches[0].parent
        if dataset_dir is not None:
            timing_stats = _parse_worker_timing(dataset_dir / "worker_timing")
            # Copy timing into run_dir for archival
            dest = run_dir / "worker_timing"
            if dest.exists():
                shutil.rmtree(dest)
            if (dataset_dir / "worker_timing").is_dir():
                shutil.copytree(dataset_dir / "worker_timing", dest)

    record = {
        "label": label,
        "success": bool(result.get("success")),
        "wall_s": wall_s,
        "chunk_processing_time": result.get("chunk_processing_time"),
        "realtime_factor": result.get("realtime_factor"),
        "total_realtime_factor": result.get("total_realtime_factor"),
        "audio_duration": result.get("audio_duration"),
        "chunks_completed": result.get("chunks_completed") or result.get("chunks_processed"),
        "workers_requested": workers,
        "batch_enabled": batch_enabled,
        "cuda_graphs": cuda_graphs,
        "ar_timer": ar_timer,
        "worker_timing": timing_stats,
        "generator_result_keys": sorted(result.keys()),
        "reason": result.get("reason"),
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
    }
    print(
        f"  done success={record['success']} wall_s={wall_s:.2f} "
        f"chunk_process={record['chunk_processing_time']} "
        f"rtf={record['realtime_factor']} "
        f"worker_gen_sum_ms={timing_stats.get('generation_ms_sum', 0):.0f} "
        f"workers_seen={timing_stats.get('workers_seen')}"
    )
    return record


def _print_item2(a: dict[str, Any], b: dict[str, Any]) -> None:
    """Print baseline vs Item #2 cut comparison."""
    print("\n=== ITEM2 COMPARISON (baseline context timer vs Item #2 cut) ===")
    for key in ("wall_s", "chunk_processing_time", "realtime_factor"):
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            print(f"{key}: baseline={av} cut={bv}")
            continue
        avf = float(av)
        bvf = float(bv)
        if key == "realtime_factor":
            # higher is better
            pct = ((bvf - avf) / avf * 100.0) if avf else 0.0
            print(f"{key}: baseline={avf:.3f} cut={bvf:.3f} delta%={pct:+.2f} (higher better)")
        else:
            pct = ((bvf - avf) / avf * 100.0) if avf else 0.0
            print(f"{key}: baseline={avf:.3f} cut={bvf:.3f} delta%={pct:+.2f} (lower better for time)")
    ag = float(a.get("worker_timing", {}).get("generation_ms_sum") or 0.0)
    bg = float(b.get("worker_timing", {}).get("generation_ms_sum") or 0.0)
    if ag:
        print(
            f"worker generation_ms sum: baseline={ag:.0f} cut={bg:.0f} "
            f"delta%={((bg - ag) / ag * 100.0):+.2f}"
        )
    print(
        f"workers_seen: baseline={a.get('worker_timing', {}).get('workers_seen')} "
        f"cut={b.get('worker_timing', {}).get('workers_seen')}"
    )


def _print_batch(a: dict[str, Any], b: dict[str, Any]) -> None:
    """Print scalar vs batch comparison."""
    print("\n=== BATCH COMPARISON (scalar vs batch_generation.enabled) ===")
    for key in ("wall_s", "chunk_processing_time", "realtime_factor"):
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            print(f"{key}: scalar={av} batch={bv}")
            continue
        avf = float(av)
        bvf = float(bv)
        if key == "realtime_factor":
            pct = ((bvf - avf) / avf * 100.0) if avf else 0.0
            print(f"{key}: scalar={avf:.3f} batch={bvf:.3f} delta%={pct:+.2f} (higher better)")
        else:
            pct = ((bvf - avf) / avf * 100.0) if avf else 0.0
            print(f"{key}: scalar={avf:.3f} batch={bvf:.3f} delta%={pct:+.2f} (lower better for time)")



def _print_graphs(a: dict[str, Any], b: dict[str, Any]) -> None:
    """Print CUDA graphs off vs on comparison."""
    print("\n=== CUDA GRAPHS COMPARISON (off vs on) ===")
    for key in ("wall_s", "chunk_processing_time", "realtime_factor"):
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            print(f"{key}: off={av} on={bv}")
            continue
        avf = float(av)
        bvf = float(bv)
        if key == "realtime_factor":
            pct = ((bvf - avf) / avf * 100.0) if avf else 0.0
            print(f"{key}: off={avf:.3f} on={bvf:.3f} delta%={pct:+.2f} (higher better)")
        else:
            pct = ((bvf - avf) / avf * 100.0) if avf else 0.0
            print(f"{key}: off={avf:.3f} on={bvf:.3f} delta%={pct:+.2f} (lower better for time)")
    ag = float(a.get("worker_timing", {}).get("generation_ms_sum") or 0.0)
    bg = float(b.get("worker_timing", {}).get("generation_ms_sum") or 0.0)
    if ag:
        print(
            f"worker generation_ms sum: off={ag:.0f} on={bg:.0f} "
            f"delta%={((bg - ag) / ag * 100.0):+.2f}"
        )
    print(
        f"workers_seen: off={a.get('worker_timing', {}).get('workers_seen')} "
        f"on={b.get('worker_timing', {}).get('workers_seen')}"
    )

def main() -> None:
    """CLI: multi-worker Item #2 or batch A/B on a real book directory."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["item2", "batch", "graphs"])
    parser.add_argument(
        "--book-dir",
        type=Path,
        default=DEFAULT_BOOK,
        help="GUI book output dir with TTS/text_chunks and voice",
    )
    parser.add_argument("--voice", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4, help="Worker count (default 4)")
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Max chunks (0 = all). Default 80 for a practical multi-worker A/B.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lsd-steps", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--frames-after-eos", type=int, default=2)
    parser.add_argument("--eos-threshold", type=float, default=-3.0)
    parser.add_argument(
        "--order",
        choices=["baseline-first", "cut-first", "scalar-first", "batch-first"],
        default=None,
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    book_dir = args.book_dir.resolve()
    text_dir, default_voice = _discover_paths(book_dir)
    voice_path = (args.voice or default_voice).resolve()
    limit = None if args.limit == 0 else args.limit
    text_items = _load_chunk_texts(text_dir, limit)
    chunks = _make_chunk_metadata(
        text_items,
        args.temperature,
        args.frames_after_eos,
        args.eos_threshold,
        args.lsd_steps,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_root = PROJECT_ROOT / "tests" / "results" / f"ab_mw_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Book:     {book_dir}")
    print(f"Voice:    {voice_path}")
    print(f"Chunks:   {len(chunks)} (limit={args.limit})")
    print(f"Workers:  {args.workers}")
    print(f"Mode:     {args.mode}")
    print(f"Run root: {run_root}")

    if args.mode == "item2":
        order = args.order or "baseline-first"
        sequence = [
            ("baseline_context_timer", "context", False),
            ("item2_cut_monotonic", "monotonic", False),
        ]
        if order == "cut-first":
            sequence = list(reversed(sequence))
        runs = []
        for label, ar_timer, _batch in sequence:
            runs.append(
                _run_one(
                    label=label,
                    chunks=chunks,
                    voice_path=voice_path,
                    workers=args.workers,
                    batch_enabled=False,
                    batch_size=args.batch_size,
                    lsd_steps=args.lsd_steps,
                    ar_timer=ar_timer,
                    run_root=run_root,
                    cuda_graphs=False,
                )
            )
        by_label = {row["label"]: row for row in runs}
        baseline = by_label["baseline_context_timer"]
        cut = by_label["item2_cut_monotonic"]
        _print_item2(baseline, cut)
        comparison = {"baseline": baseline, "item2_cut": cut}
    elif args.mode == "batch":
        order = args.order or "scalar-first"
        sequence = [
            ("scalar_no_batch", "monotonic", False),
            ("batch_enabled", "monotonic", True),
        ]
        if order == "batch-first":
            sequence = list(reversed(sequence))
        runs = []
        for label, ar_timer, batch_enabled in sequence:
            runs.append(
                _run_one(
                    label=label,
                    chunks=chunks,
                    voice_path=voice_path,
                    workers=args.workers,
                    batch_enabled=batch_enabled,
                    batch_size=args.batch_size,
                    lsd_steps=args.lsd_steps,
                    ar_timer=ar_timer,
                    run_root=run_root,
                    cuda_graphs=False,
                )
            )
        by_label = {row["label"]: row for row in runs}
        scalar = by_label["scalar_no_batch"]
        batch = by_label["batch_enabled"]
        _print_batch(scalar, batch)
        comparison = {"scalar": scalar, "batch": batch}
    else:
        # graphs: batch ON (production), graphs off vs on
        order = args.order or "baseline-first"
        sequence = [
            ("graphs_off", False),
            ("graphs_on", True),
        ]
        if order == "cut-first":
            sequence = list(reversed(sequence))
        runs = []
        for label, graphs_on in sequence:
            runs.append(
                _run_one(
                    label=label,
                    chunks=chunks,
                    voice_path=voice_path,
                    workers=args.workers,
                    batch_enabled=True,
                    batch_size=args.batch_size,
                    lsd_steps=args.lsd_steps,
                    ar_timer="monotonic",
                    run_root=run_root,
                    cuda_graphs=graphs_on,
                )
            )
        by_label = {row["label"]: row for row in runs}
        off = by_label["graphs_off"]
        on = by_label["graphs_on"]
        _print_graphs(off, on)
        comparison = {"graphs_off": off, "graphs_on": on}

    # Restore production default env
    os.environ["POCKET_TTS_AR_TIMER"] = "monotonic"
    os.environ.pop("POCKET_TTS_CUDA_GRAPHS", None)

    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "book_dir": str(book_dir),
            "voice": str(voice_path),
            "chunk_count": len(chunks),
            "limit": args.limit,
            "workers_requested": args.workers,
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "run_root": str(run_root),
        },
        "runs": runs,
        "comparison": comparison,
    }
    out = args.output or (run_root / f"ab_{args.mode}_workers{args.workers}.json")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
