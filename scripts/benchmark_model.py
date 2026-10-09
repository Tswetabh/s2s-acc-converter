"""CLI script to benchmark a single TTS model across standardized test cases."""

import sys
import os
import subprocess

# Auto-redirect to project virtual environment on Drive D if executed with global Python
venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts", "python.exe"))
if os.path.exists(venv_python):
    curr_exe = os.path.normcase(os.path.abspath(sys.executable))
    target_exe = os.path.normcase(os.path.abspath(venv_python))
    if curr_exe != target_exe:
        result = subprocess.run([venv_python] + sys.argv, check=False)
        sys.exit(result.returncode)

import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.registry import get_model, list_available_models
from app.core.benchmark import BenchmarkEngine
from app.core.inference import InferenceEngine

def main():
    parser = argparse.ArgumentParser(description="Unified TTS Single Model Benchmark CLI")
    parser.add_argument("--model", type=str, required=True, help="Model ID (e.g. kokoro, piper, qwen3tts)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--test-cases", nargs="+", default=None, help="Specific test case IDs to run (default: all)")

    args = parser.parse_args()

    model_adapter = get_model(args.model)
    if not model_adapter:
        print(f"[ERROR] Unknown model '{args.model}'. Available: {list_available_models()}")
        sys.exit(1)

    print(f"\n[BENCHMARK] Starting standardized benchmark for: {model_adapter.name}")
    print(f"[INFO] Hardware profile: {model_adapter.hardware_classification}")

    health = model_adapter.health_check()
    print(f"[HEALTH] {health['status']} - {health['reason']}")

    results = BenchmarkEngine.benchmark_model(
        model_adapter=model_adapter,
        test_case_ids=args.test_cases,
        device=args.device,
        skip_cloud_recommended=False,
    )

    print("\n" + "=" * 65)
    print(f"BENCHMARK SUMMARY FOR {model_adapter.name.upper()}")
    print("-" * 65)
    print(f"{'Test Case':<18} {'Status':<14} {'RTF':<8} {'Gen (s)':<10} {'Peak VRAM (GB)':<14}")
    print("-" * 65)

    for r in results:
        t_id = r.get("test_case", "")
        status = r.get("status", "")
        rtf = str(r.get("rtf", 0.0))
        gen_s = str(r.get("generation_time_sec", 0.0))
        vram = str(r.get("vram_peak_gb", 0.0))
        print(f"{t_id:<18} {status:<14} {rtf:<8} {gen_s:<10} {vram:<14}")

    print("=" * 65)
    print("[INFO] Results appended to outputs/benchmarks/results.csv\n")

if __name__ == "__main__":
    main()
