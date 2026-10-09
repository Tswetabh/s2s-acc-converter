#!/usr/bin/env python3
"""
Minimal single-purpose test for Item #2 only.

Measures the effect of removing per-step display_execution_time
from _autoregressive_generation.

Runs the benchmark tool twice with identical eager settings and
compares autoregressive_ms.

This script replaces the abandoned multi-variant orchestrator.
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH = PROJECT_ROOT / "tools" / "benchmark_inference_variants.py"

def run_once(label: str) -> dict:
    """Runs a specified command with given parameters and captures its output."""
    cmd = [
        sys.executable, str(BENCH),
        "--precision", "none",
        "--compile-target", "none",
        "--matmul-precision", "highest",
        "--warmup", "1",
        "--iterations", "2",
        "--seed", "42",
    ]
    print(f"[{label}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"[{label}] FAILED (code {result.returncode})")
        print("STDERR (last 500 chars):", result.stderr[-500:])
        raise RuntimeError(f"Benchmark failed: {result.returncode}")

    # The benchmark tool prints a single pretty-printed JSON object.
    # We capture the entire stdout and find the JSON block.
    output = result.stdout
    # Find the first '{' and the last '}' to extract the JSON object
    start = output.find("{")
    end = output.rfind("}") + 1
    if start != -1 and end > start:
        json_str = output[start:end]
        try:
            data = json.loads(json_str)
            if "model_timing" in data:
                return data
        except json.JSONDecodeError as e:
            print(f"[{label}] JSON parse error: {e}")
            print(f"[{label}] Extracted string (first 500): {json_str[:500]}")

    print(f"[{label}] Full stdout (last 2000 chars):\n{result.stdout[-2000:]}")
    print(f"[{label}] Full stderr (last 500 chars):\n{result.stderr[-500:]}")
    raise RuntimeError(f"[{label}] No valid JSON with model_timing found")

def main():
    """Executes a benchmark test comparing baseline and modified models."""
    print("=== Item #2 Overhead Reduction Test (Eager only) ===")
    baseline = run_once("BASELINE")
    modified = run_once("MODIFIED")

    b_ms = baseline["model_timing"]["autoregressive_ms"]
    m_ms = modified["model_timing"]["autoregressive_ms"]

    print("\n=== RESULTS ===")
    print(f"Baseline autoregressive_ms: {b_ms:.1f}")
    print(f"Modified autoregressive_ms: {m_ms:.1f}")
    delta = m_ms - b_ms
    pct = (delta / b_ms) * 100 if b_ms > 0 else 0
    print(f"Difference: {delta:+.1f} ms ({pct:+.1f}%)")

    if delta < 0:
        print("→ Overhead reduction observed.")
    else:
        print("→ No reduction or regression.")

if __name__ == "__main__":
    main()
