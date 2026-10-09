#!/usr/bin/env python3
"""Calibrate optimal GPU ASR worker count for this machine.

Benches faster-whisper on a sample of existing audiobook chunks at several
worker counts, picks the knee of the throughput curve (diminishing returns),
and optionally writes ``gpu_workers_after_tts`` into the project config.

Why this exists
---------------
Multi-worker GPU ASR is SM-bound, not only VRAM-bound. A 4060 Ti peaks near
4 workers; a 5090 may still gain at 6–8. Shipping one static default is wrong;
this probe sets the right number per GPU.

Example
-------
::

    python tests/tools/calibrate_asr_workers.py \\
      --tts-dir "Output/Doctor Who_ The Edge of Destruction/TTS" \\
      --model base --workers 1,2,3,4,6 --sample-chunks 64 \\
      --write-config pocket_tts/config/default_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "tests" / "results" / "asr_worker_calibrate"
DEFAULT_ASR_PYTHON = Path(sys.executable)
DEFAULT_ASR_SCRIPT = ROOT / "ASR" / "asr_validator.py"
DEFAULT_CONFIG = ROOT / "pocket_tts" / "config" / "default_config.yaml"


def _nvidia_smi() -> Dict[str, int]:
    """Return GPU memory and util from nvidia-smi (MiB / percent)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        used, free, total, util = [x.strip() for x in out.split(",")]
        return {
            "used_mib": int(float(used)),
            "free_mib": int(float(free)),
            "total_mib": int(float(total)),
            "util_pct": int(float(util)),
        }
    except Exception:
        return {"used_mib": 0, "free_mib": 0, "total_mib": 0, "util_pct": 0}


def _gpu_name() -> str:
    """Return first GPU product name, or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        return out.splitlines()[0].strip() if out else "unknown"
    except Exception:
        return "unknown"


def discover_chunk_stems(tts_dir: Path) -> List[str]:
    """Return sorted chunk stems that have both wav and text sidecars.

    Args:
        tts_dir: Parent TTS directory with audio_chunks and text_chunks.

    Returns:
        Sorted list of stems like ``chunk_00001``.
    """
    audio = tts_dir / "audio_chunks"
    text = tts_dir / "text_chunks"
    if not audio.is_dir() or not text.is_dir():
        raise FileNotFoundError(f"Need audio_chunks and text_chunks under {tts_dir}")
    stems = []
    for wav in sorted(audio.glob("chunk_*.wav")):
        stem = wav.stem
        if (text / f"{stem}.txt").is_file():
            stems.append(stem)
    return stems


def build_sample_tree(
    tts_dir: Path,
    stems: Sequence[str],
    dest: Path,
) -> Path:
    """Build a temp TTS tree with symlinks to a subset of chunks.

    Args:
        tts_dir: Source TTS directory.
        stems: Chunk stems to include.
        dest: Destination root (created).

    Returns:
        Path to the sample TTS directory (contains audio_chunks / text_chunks).
    """
    sample = dest / "TTS"
    audio_d = sample / "audio_chunks"
    text_d = sample / "text_chunks"
    audio_d.mkdir(parents=True, exist_ok=True)
    text_d.mkdir(parents=True, exist_ok=True)
    src_audio = tts_dir / "audio_chunks"
    src_text = tts_dir / "text_chunks"
    for stem in stems:
        wav_src = (src_audio / f"{stem}.wav").resolve()
        txt_src = (src_text / f"{stem}.txt").resolve()
        wav_dst = audio_d / f"{stem}.wav"
        txt_dst = text_d / f"{stem}.txt"
        if wav_dst.exists() or wav_dst.is_symlink():
            wav_dst.unlink()
        if txt_dst.exists() or txt_dst.is_symlink():
            txt_dst.unlink()
        wav_dst.symlink_to(wav_src)
        txt_dst.symlink_to(txt_src)
    return sample


def parse_worker_list(spec: str) -> List[int]:
    """Parse ``1,2,3,4,6`` or ``1-6`` into a sorted unique worker list.

    Args:
        spec: Comma-separated ints and/or inclusive ranges.

    Returns:
        Sorted unique worker counts >= 1.
    """
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    uniq = sorted({max(1, w) for w in out})
    if not uniq:
        raise ValueError("worker list empty")
    return uniq


def pick_recommended(
    rows: Sequence[Dict[str, Any]],
    min_gain_pct: float = 8.0,
    within_peak_pct: float = 5.0,
) -> Dict[str, Any]:
    """Choose worker count from throughput rows.

    Policy:
      1. Drop failed rows.
      2. Find peak chunks/s.
      3. Prefer the **smallest** worker count within ``within_peak_pct`` of peak
         (saves VRAM when extra workers buy almost nothing).
      4. Also record the knee: last count whose step-up gain was >= min_gain_pct.

    Args:
        rows: Per-worker result dicts with workers, chunks_per_s, ok.
        min_gain_pct: Minimum % gain vs previous count to count as useful.
        within_peak_pct: Prefer fewer workers if within this % of peak thruput.

    Returns:
        Dict with recommended, peak, knee, and rationale.
    """
    ok = [r for r in rows if r.get("ok") and r.get("chunks_per_s", 0) > 0]
    if not ok:
        return {
            "recommended": 1,
            "peak_workers": 1,
            "knee_workers": 1,
            "rationale": "no successful runs; default 1",
        }

    peak = max(ok, key=lambda r: r["chunks_per_s"])
    peak_cps = peak["chunks_per_s"]
    floor = peak_cps * (1.0 - within_peak_pct / 100.0)

    # Smallest W within X% of peak
    near_peak = [r for r in ok if r["chunks_per_s"] >= floor]
    recommended = min(near_peak, key=lambda r: r["workers"])

    # Knee: walk ascending; last step with gain >= min_gain_pct
    ordered = sorted(ok, key=lambda r: r["workers"])
    knee = ordered[0]
    for prev, cur in zip(ordered, ordered[1:]):
        prev_cps = prev["chunks_per_s"] or 1e-9
        gain = (cur["chunks_per_s"] - prev_cps) / prev_cps * 100.0
        cur = dict(cur)
        cur["gain_pct_vs_prev"] = round(gain, 2)
        if gain >= min_gain_pct:
            knee = cur
        else:
            break

    rationale = (
        f"peak {peak['workers']}w @ {peak_cps:.2f} ch/s; "
        f"recommend {recommended['workers']}w "
        f"({recommended['chunks_per_s']:.2f} ch/s, "
        f"within {within_peak_pct:g}% of peak); "
        f"knee(gain>={min_gain_pct:g}%)={knee['workers']}w"
    )
    return {
        "recommended": int(recommended["workers"]),
        "peak_workers": int(peak["workers"]),
        "knee_workers": int(knee["workers"]),
        "peak_chunks_per_s": peak_cps,
        "recommended_chunks_per_s": recommended["chunks_per_s"],
        "rationale": rationale,
    }


def run_one_bench(
    *,
    asr_python: Path,
    asr_script: Path,
    sample_tts: Path,
    workers: int,
    model: str,
    device: str,
    engine: str,
    expected: int,
    run_dir: Path,
    timeout_s: float,
) -> Dict[str, Any]:
    """Run one offline multi-worker ASR pass and return timing stats.

    Args:
        asr_python: Path to ASR venv python.
        asr_script: Path to asr_validator.py.
        sample_tts: Sample TTS dir with audio/text chunks.
        workers: Concurrent ASR workers (each loads own model).
        model: faster-whisper model name.
        device: cuda or cpu.
        engine: ASR engine name.
        expected: Expected chunk count (stop condition).
        run_dir: Directory for this run's log + failures.
        timeout_s: Hard timeout for the subprocess.

    Returns:
        Result dict with ok, wall_s, chunks_per_s, etc.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    fail_path = run_dir / "failures.json"
    if fail_path.exists():
        fail_path.unlink()

    audio = sample_tts / "audio_chunks"
    cmd = [
        str(asr_python),
        str(asr_script),
        "--monitor-folder",
        str(audio),
        "--tts-dir",
        str(sample_tts),
        "--expected-count",
        str(expected),
        "--model",
        model,
        "--device",
        device,
        "--cpu-workers",
        str(workers),
        "--gpu-workers-after-tts",
        "0",
        "--threshold",
        "0.8",
        "--language",
        "en",
        "--engine",
        engine,
        "--log-file",
        str(fail_path),
    ]

    smi_before = _nvidia_smi()
    t0 = time.time()
    peak_used = smi_before.get("used_mib", 0)
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        text=True,
    )
    # Poll for completion + peak VRAM
    deadline = t0 + timeout_s
    completed_wall: Optional[float] = None
    while time.time() < deadline:
        smi = _nvidia_smi()
        peak_used = max(peak_used, smi.get("used_mib", 0))
        if proc.poll() is not None:
            break
        # Early exit if log has completion line (process may linger on join)
        try:
            text = log_path.read_text(errors="ignore")
        except Exception:
            text = ""
        m = re.search(r"processed (\d+) chunks in ([\d.]+)s", text)
        if m and int(m.group(1)) >= expected:
            completed_wall = float(m.group(2))
            # Give process a moment to exit cleanly
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        time.sleep(0.5)
    else:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:
            pass

    wall_outer = time.time() - t0
    log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    m = re.search(r"processed (\d+) chunks in ([\d.]+)s", log_text)
    loaded = len(re.findall(r"Successfully loaded", log_text))
    fatal = "Failed to load" in log_text or "out of memory" in log_text.lower()
    worker_fatal = log_text.count("ASR worker fatal")

    if m:
        n_done = int(m.group(1))
        wall = float(m.group(2))
    else:
        n_done = len(re.findall(r": PASSED \(score|: FAILED \(score", log_text))
        wall = wall_outer
        if completed_wall is not None:
            wall = completed_wall

    ok = (
        m is not None
        and n_done >= expected
        and not fatal
        and worker_fatal == 0
        and wall > 0
    )
    cps = (n_done / wall) if wall > 0 else 0.0
    return {
        "workers": workers,
        "ok": ok,
        "chunks_done": n_done,
        "chunks_expected": expected,
        "wall_s": round(wall, 2),
        "chunks_per_s": round(cps, 3),
        "models_loaded_log": loaded,
        "peak_vram_used_mib": peak_used,
        "smi_before": smi_before,
        "smi_after": _nvidia_smi(),
        "fatal": fatal or worker_fatal > 0,
        "log": str(log_path),
        "returncode": proc.returncode,
    }


def write_config_gpu_workers(config_path: Path, gpu_workers: int) -> None:
    """Set ``gpu_workers_after_tts`` and mark ``worker_calibration.done: true``.

    Args:
        config_path: Path to default_config.yaml (or user config).
        gpu_workers: Recommended post-gen GPU ASR worker count.
    """
    # Prefer shared first-run helper so flag + value stay in sync.
    sys.path.insert(0, str(ROOT))
    try:
        from pocket_tts.asr_worker_calibration import mark_calibration_done

        mark_calibration_done(config_path, int(gpu_workers))
        return
    except Exception:
        pass

    import yaml

    text = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    asr = data.setdefault("asr_quality_control", {})
    if not isinstance(asr, dict):
        asr = {}
        data["asr_quality_control"] = asr
    parallel = asr.setdefault("parallel", {})
    if not isinstance(parallel, dict):
        parallel = {}
        asr["parallel"] = parallel
    parallel["gpu_workers_after_tts"] = int(gpu_workers)
    cal = asr.setdefault("worker_calibration", {})
    if not isinstance(cal, dict):
        cal = {}
        asr["worker_calibration"] = cal
    cal["done"] = True
    cal["recommended"] = int(gpu_workers)
    config_path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry: bench ASR worker counts and optionally write config."""
    parser = argparse.ArgumentParser(
        description="Calibrate GPU ASR worker count from real chunk throughput",
    )
    parser.add_argument(
        "--tts-dir",
        type=Path,
        required=True,
        help="TTS dir with audio_chunks + text_chunks (e.g. Edge book TTS)",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="faster-whisper model (default: base)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for calibration (default: cuda)",
    )
    parser.add_argument(
        "--engine",
        default="faster_whisper",
        help="ASR engine (default: faster_whisper)",
    )
    parser.add_argument(
        "--workers",
        default="1,2,3,4,6",
        help="Worker counts to test (default: 1,2,3,4,6)",
    )
    parser.add_argument(
        "--sample-chunks",
        type=int,
        default=64,
        help="Number of chunks to sample for each trial (default: 64)",
    )
    parser.add_argument(
        "--min-gain-pct",
        type=float,
        default=8.0,
        help="Knee: min %% thruput gain vs prior worker count (default: 8)",
    )
    parser.add_argument(
        "--within-peak-pct",
        type=float,
        default=5.0,
        help="Recommend fewest workers within this %% of peak thruput (default: 5)",
    )
    parser.add_argument(
        "--timeout-per-trial",
        type=float,
        default=600.0,
        help="Timeout seconds per worker trial (default: 600)",
    )
    parser.add_argument(
        "--asr-python",
        type=Path,
        default=Path(sys.executable),
        help="Python for ASR (default: current interpreter / main venv)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Where to write calibration JSON/logs",
    )
    parser.add_argument(
        "--write-config",
        type=Path,
        default=None,
        help="If set, write gpu_workers_after_tts into this YAML config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only; do not run benches",
    )
    args = parser.parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    tts_dir = args.tts_dir.resolve()
    if not tts_dir.is_dir():
        print(f"ERROR: tts-dir not found: {tts_dir}", file=sys.stderr)
        return 1
    if not args.asr_python.is_file():
        print(f"ERROR: ASR python not found: {args.asr_python}", file=sys.stderr)
        return 1
    if not DEFAULT_ASR_SCRIPT.is_file():
        print(f"ERROR: asr_validator not found: {DEFAULT_ASR_SCRIPT}", file=sys.stderr)
        return 1

    stems_all = discover_chunk_stems(tts_dir)
    if not stems_all:
        print("ERROR: no matching audio/text chunk pairs", file=sys.stderr)
        return 1
    n_sample = max(1, min(int(args.sample_chunks), len(stems_all)))
    # Evenly stride sample across the book so short/long mix is representative
    if n_sample >= len(stems_all):
        stems = list(stems_all)
    else:
        step = len(stems_all) / n_sample
        stems = [stems_all[int(i * step)] for i in range(n_sample)]

    worker_list = parse_worker_list(args.workers)
    gpu = _gpu_name()
    smi0 = _nvidia_smi()

    print("=== ASR worker calibration ===", flush=True)
    print(f"GPU:     {gpu}", flush=True)
    print(f"VRAM:    {smi0['used_mib']} used / {smi0['free_mib']} free / {smi0['total_mib']} total MiB", flush=True)
    print(f"TTS dir: {tts_dir}", flush=True)
    print(f"Model:   {args.model}  device={args.device}  engine={args.engine}", flush=True)
    print(f"Sample:  {len(stems)} / {len(stems_all)} chunks", flush=True)
    print(f"Workers: {worker_list}", flush=True)
    print(
        f"Policy:  min_gain={args.min_gain_pct:g}%  within_peak={args.within_peak_pct:g}%",
        flush=True,
    )

    if args.dry_run:
        print("dry-run: exit", flush=True)
        return 0

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = args.results_dir / stamp
    work.mkdir(parents=True, exist_ok=True)
    sample_tts = build_sample_tree(tts_dir, stems, work / "sample")

    rows: List[Dict[str, Any]] = []
    for w in worker_list:
        print(f"\n--- trial workers={w} ---", flush=True)
        row = run_one_bench(
            asr_python=args.asr_python,
            asr_script=DEFAULT_ASR_SCRIPT,
            sample_tts=sample_tts,
            workers=w,
            model=args.model,
            device=args.device,
            engine=args.engine,
            expected=len(stems),
            run_dir=work / f"w{w}",
            timeout_s=float(args.timeout_per_trial),
        )
        rows.append(row)
        status = "OK" if row["ok"] else "FAIL"
        print(
            f"  [{status}] wall={row['wall_s']}s  "
            f"{row['chunks_per_s']} ch/s  "
            f"done={row['chunks_done']}/{row['chunks_expected']}  "
            f"peak_vram≈{row['peak_vram_used_mib']} MiB",
            flush=True,
        )
        # Brief cooldown so VRAM releases between trials
        time.sleep(1.5)

    pick = pick_recommended(
        rows,
        min_gain_pct=float(args.min_gain_pct),
        within_peak_pct=float(args.within_peak_pct),
    )
    # Annotate step gains for report
    ordered = sorted([r for r in rows if r.get("ok")], key=lambda r: r["workers"])
    for prev, cur in zip(ordered, ordered[1:]):
        prev_cps = prev["chunks_per_s"] or 1e-9
        cur["gain_pct_vs_prev"] = round(
            (cur["chunks_per_s"] - prev_cps) / prev_cps * 100.0, 2
        )

    report = {
        "test": "calibrate_asr_workers",
        "timestamp": stamp,
        "gpu_name": gpu,
        "smi_baseline": smi0,
        "tts_dir": str(tts_dir),
        "model": args.model,
        "device": args.device,
        "engine": args.engine,
        "sample_chunks": len(stems),
        "total_chunks_available": len(stems_all),
        "worker_list": worker_list,
        "min_gain_pct": args.min_gain_pct,
        "within_peak_pct": args.within_peak_pct,
        "trials": rows,
        "pick": pick,
    }

    out_json = work / "calibration.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Also write latest pointer
    latest = args.results_dir / "latest.json"
    latest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== RESULTS ===", flush=True)
    print(f"{'w':>4} {'wall':>8} {'ch/s':>8} {'gain':>8}  status", flush=True)
    for r in sorted(rows, key=lambda x: x["workers"]):
        gain = r.get("gain_pct_vs_prev")
        gain_s = f"{gain:+.1f}%" if isinstance(gain, (int, float)) else "  —"
        print(
            f"{r['workers']:>4} {r['wall_s']:>7.1f}s {r['chunks_per_s']:>7.2f} "
            f"{gain_s:>8}  {'OK' if r['ok'] else 'FAIL'}",
            flush=True,
        )
    print(f"\n{pick['rationale']}", flush=True)
    print(f"RECOMMENDED gpu_workers_after_tts = {pick['recommended']}", flush=True)
    print(f"Wrote {out_json}", flush=True)

    if args.write_config:
        cfg = args.write_config.resolve()
        if not cfg.is_file():
            print(f"ERROR: config not found: {cfg}", file=sys.stderr)
            return 1
        write_config_gpu_workers(cfg, int(pick["recommended"]))
        print(
            f"Updated {cfg} → parallel.gpu_workers_after_tts={pick['recommended']}",
            flush=True,
        )

    return 0 if any(r.get("ok") for r in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
