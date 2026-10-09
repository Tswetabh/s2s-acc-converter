#!/usr/bin/env python3
"""
Test harness for measuring GPU optimization impact on Pocket TTS.

This script uses the existing tools/benchmark_inference_variants.py
and tools/probe_cuda_graph.py infrastructure to run controlled
experiments. It never modifies production code.

All results are written to tests/results/ with timestamps.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_benchmark_variant(args: list[str]) -> dict:
    """Run benchmark_inference_variants.py with given CLI args and return parsed JSON.

    Streams child output live so we can watch model loading and generation progress.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "tools" / "benchmark_inference_variants.py")] + args
    print(f"[RUN] {' '.join(cmd)}")
    print("[INFO] Streaming output from benchmark (this can take minutes)...")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)  # inherit stdout/stderr for live logs
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed with code {result.returncode}")
    # Re-run once with capture to get the final JSON result line
    result2 = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    lines = [l for l in result2.stdout.strip().splitlines() if l.strip()]
    return json.loads(lines[-1])


def run_cuda_graph_probe() -> dict:
    """Run the CUDA graph probe and capture feasibility result."""
    cmd = [sys.executable, str(PROJECT_ROOT / "tools" / "probe_cuda_graph.py")]
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    # The probe prints JSON or structured text; capture whatever it emits
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def timestamp() -> str:
    """Generates current UTC timestamp in YYYYMMDD-HHMMSS format."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def main():
    """Creates a results directory with timestamp and initializes summary dictionary."""
    ts = timestamp()
    out_dir = RESULTS_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp_utc": ts,
        "gpu_available": False,
        "variants": [],
        "cuda_graph_probe": None,
    }

    if not __import__("torch").cuda.is_available():
        print("CUDA not available; skipping GPU tests.")
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        return

    summary["gpu_available"] = True

    # Baseline (no precision, no compile, highest matmul)
    try:
        baseline = run_benchmark_variant([
            "--precision", "none",
            "--compile-target", "none",
            "--matmul-precision", "highest",
            "--warmup", "1",
            "--iterations", "2",
            "--seed", "42",
        ])
        summary["variants"].append(baseline)
        (out_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))
    except Exception as e:
        summary["error_baseline"] = str(e)

    # Eager BF16 autocast only (no compile) — conservative AMP win
    try:
        eager_bf16 = run_benchmark_variant([
            "--precision", "bf16",
            "--compile-target", "none",
            "--matmul-precision", "high",
            "--warmup", "1",
            "--iterations", "2",
            "--seed", "42",
        ])
        summary["variants"].append(eager_bf16)
        (out_dir / "eager_bf16.json").write_text(json.dumps(eager_bf16, indent=2))
    except Exception as e:
        summary["error_eager_bf16"] = str(e)

    # Eager BF16 + TF32/high matmul (global, zero model change)
    try:
        eager_bf16_tf32 = run_benchmark_variant([
            "--precision", "bf16",
            "--compile-target", "none",
            "--matmul-precision", "high",
            "--warmup", "1",
            "--iterations", "2",
            "--seed", "42",
        ])
        summary["variants"].append(eager_bf16_tf32)
        (out_dir / "eager_bf16_tf32.json").write_text(json.dumps(eager_bf16_tf32, indent=2))
    except Exception as e:
        summary["error_eager_bf16_tf32"] = str(e)

    # Tight-loop eager (frames_after_eos=0, minimal Python overhead)
    try:
        tight_eager = run_benchmark_variant([
            "--precision", "none",
            "--compile-target", "none",
            "--matmul-precision", "highest",
            "--warmup", "1",
            "--iterations", "2",
            "--seed", "42",
        ])
        summary["variants"].append(tight_eager)
        (out_dir / "tight_eager.json").write_text(json.dumps(tight_eager, indent=2))
    except Exception as e:
        summary["error_tight_eager"] = str(e)

    # Variant 2: BF16 + compile on flow_step
    try:
        bf16_flow = run_benchmark_variant([
            "--precision", "bf16",
            "--compile-target", "flow_step",
            "--compile-mode", "reduce-overhead",
            "--matmul-precision", "high",
            "--warmup", "1",
            "--iterations", "2",
            "--seed", "42",
        ])
        summary["variants"].append(bf16_flow)
        (out_dir / "bf16_flow_step.json").write_text(json.dumps(bf16_flow, indent=2))
    except Exception as e:
        summary["error_bf16_flow"] = str(e)

    # Variant 3: BF16 + compile on flow_net
    try:
        bf16_flow_net = run_benchmark_variant([
            "--precision", "bf16",
            "--compile-target", "flow_net",
            "--compile-mode", "reduce-overhead",
            "--matmul-precision", "high",
            "--warmup", "1",
            "--iterations", "2",
            "--seed", "42",
        ])
        summary["variants"].append(bf16_flow_net)
        (out_dir / "bf16_flow_net.json").write_text(json.dumps(bf16_flow_net, indent=2))
    except Exception as e:
        summary["error_bf16_flow_net"] = str(e)

    # Variant 4: BF16 + compile on mimi_decode
    try:
        bf16_mimi = run_benchmark_variant([
            "--precision", "bf16",
            "--compile-target", "mimi_decode",
            "--compile-mode", "reduce-overhead",
            "--matmul-precision", "high",
            "--warmup", "1",
            "--iterations", "2",
            "--seed", "42",
        ])
        summary["variants"].append(bf16_mimi)
        (out_dir / "bf16_mimi_decode.json").write_text(json.dumps(bf16_mimi, indent=2))
    except Exception as e:
        summary["error_bf16_mimi"] = str(e)

    # Variant 5: FP16 + compile on flow_step (alternative precision)
    try:
        fp16_flow = run_benchmark_variant([
            "--precision", "fp16",
            "--compile-target", "flow_step",
            "--compile-mode", "reduce-overhead",
            "--matmul-precision", "high",
            "--warmup", "1",
            "--iterations", "2",
            "--seed", "42",
        ])
        summary["variants"].append(fp16_flow)
        (out_dir / "fp16_flow_step.json").write_text(json.dumps(fp16_flow, indent=2))
    except Exception as e:
        summary["error_fp16_flow"] = str(e)

    # CUDA graph probe (single step capture feasibility)
    try:
        graph_result = run_cuda_graph_probe()
        summary["cuda_graph_probe"] = graph_result
        (out_dir / "cuda_graph_probe.txt").write_text(
            f"returncode={graph_result['returncode']}\n\nSTDOUT:\n{graph_result['stdout']}\n\nSTDERR:\n{graph_result['stderr']}"
        )
    except Exception as e:
        summary["error_graph_probe"] = str(e)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
