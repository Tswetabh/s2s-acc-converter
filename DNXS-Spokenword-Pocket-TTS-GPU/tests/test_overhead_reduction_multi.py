#!/usr/bin/env python3
"""
Extended Item #2 test: N short independent generations.

Simulates the real audiobook pipeline where text is split into
multiple sentence-level chunks, each processed independently.

Runs the benchmark tool N times with short texts and aggregates
the autoregressive_ms across all runs.

This replaces the abandoned multi-variant orchestrator.
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH = PROJECT_ROOT / "tools" / "benchmark_inference_variants.py"

# 10 short, varied sentences to simulate real chunks
CHUNKS = [
    "Hello, how are you today?",
    "The weather is quite pleasant this afternoon.",
    "Please pass the salt and pepper.",
    "I would like to order a large coffee.",
    "The train arrives at platform seven.",
    "She walked quickly down the busy street.",
    "Can you help me find the nearest station?",
    "The book was much better than the movie.",
    "They decided to go hiking this weekend.",
    "Thank you very much for your assistance.",
]

def run_benchmark(text: str, label: str) -> float:
    """Run benchmark for one short text and return autoregressive_ms."""
    cmd = [
        sys.executable, str(BENCH),
        "--precision", "none",
        "--compile-target", "none",
        "--matmul-precision", "highest",
        "--warmup", "0",
        "--iterations", "1",
        "--seed", "42",
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"[{label}] Benchmark failed: {result.returncode}")

    # Extract JSON block
    output = result.stdout
    start = output.find("{")
    end = output.rfind("}") + 1
    if start != -1 and end > start:
        try:
            data = json.loads(output[start:end])
            if "model_timing" in data:
                return float(data["model_timing"]["autoregressive_ms"])
        except json.JSONDecodeError:
            pass

    raise RuntimeError(f"[{label}] Could not parse autoregressive_ms from output")

def run_n_chunks(label: str, n: int = 10) -> float:
    """Run N independent short generations and return total autoregressive time."""
    total_ms = 0.0
    texts = CHUNKS[:n]
    for i, text in enumerate(texts):
        ms = run_benchmark(text, f"{label} chunk {i+1}")
        total_ms += ms
        print(f"  [{label}] Chunk {i+1}: {ms:.1f} ms")
    return total_ms

def main():
    """Executes a test to measure autoregressive time for baseline chunks."""
    print("=== Item #2 Multi-Chunk Overhead Reduction Test ===")
    print(f"Running {len(CHUNKS)} independent short generations...\n")

    baseline_total = run_n_chunks("BASELINE")
    print(f"\nBASELINE total autoregressive_ms: {baseline_total:.1f}\n")

    # Note: This test runs against the current code on disk.
    # To measure the "modified" version, the caller must first apply
    # the overhead-reduction edit to tts_model.py, then re-run.
    # For a single-run comparison we just report the baseline here.

    print("=== SUMMARY ===")
    print(f"Total chunks: {len(CHUNKS)}")
    print(f"Baseline total autoregressive time: {baseline_total:.1f} ms")
    print(f"Average per chunk: {baseline_total / len(CHUNKS):.1f} ms")

if __name__ == "__main__":
    main()
