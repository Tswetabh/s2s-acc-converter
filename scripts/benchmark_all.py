"""CLI script to benchmark all registered TTS models with error isolation."""

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

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.registry import get_all_models
from app.core.benchmark import BenchmarkEngine
from app.config import HW_CLOUD_RECOMMENDED

def main():
    parser = argparse.ArgumentParser(description="Unified TTS Batch Benchmark CLI")
    parser.add_argument("--local-only", action="store_true", help="Skip models classified as CLOUD_RECOMMENDED")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--quick", action="store_true", help="Run only the core 'normal_01' test case per model")

    args = parser.parse_args()

    models = get_all_models()
    test_cases = ["normal_01"] if args.quick else None

    print("\n" + "=" * 65)
    print("UNIFIED TTS EVALUATION PLATFORM -- BATCH BENCHMARK")
    print(f"Total Registered Models: {len(models)}")
    print(f"Local-Only Filter:       {args.local_only}")
    print(f"Test Suite:              {'Quick (normal_01)' if args.quick else 'Full (12 categories)'}")
    print("=" * 65 + "\n")

    summary_rows = []

    for i, model_adapter in enumerate(models, 1):
        meta = model_adapter.get_metadata()
        m_name = meta["name"]
        m_class = meta["hardware_classification"]

        if args.local_only and m_class == HW_CLOUD_RECOMMENDED:
            print(f"[{i:02d}/{len(models):02d}] {m_name:<28} [CLOUD] (Skipped via --local-only)")
            summary_rows.append((m_name, "SKIPPED_CLOUD", "-", "-", "-"))
            continue

        print(f"[{i:02d}/{len(models):02d}] Testing {m_name}...")
        health = model_adapter.health_check()

        if health["status"] not in ["READY", "MEMORY_LIMITED", "CUDA_UNAVAILABLE"]:
            print(f"         Status: [{health['status']}] ({health['reason']})")
            summary_rows.append((m_name, health["status"], "-", "-", "-"))
            continue

        try:
            results = BenchmarkEngine.benchmark_model(
                model_adapter=model_adapter,
                test_case_ids=test_cases,
                device=args.device,
                skip_cloud_recommended=args.local_only,
            )
            success_runs = [r for r in results if r["status"] == "SUCCESS"]
            if success_runs:
                avg_rtf = round(sum(r["rtf"] for r in success_runs) / len(success_runs), 3)
                peak_vram = max(r["vram_peak_gb"] for r in success_runs)
                print(f"         [SUCCESS] (Avg RTF: {avg_rtf}, Peak VRAM: {peak_vram} GB)")
                summary_rows.append((m_name, "SUCCESS", str(avg_rtf), str(peak_vram), f"{len(success_runs)} tests"))
            else:
                last_err = results[0]["error"] if results else "Unknown error"
                print(f"         [INCOMPLETE]: {last_err[:50]}")
                summary_rows.append((m_name, "INCOMPLETE", "-", "-", last_err[:25]))
        except Exception as e:
            print(f"         [FAILED]: {str(e)}")
            summary_rows.append((m_name, "FAILED", "-", "-", str(e)[:25]))

    print("\n" + "=" * 70)
    print("BATCH BENCHMARK COMPLETION SUMMARY")
    print("-" * 70)
    print(f"{'Model Name':<28} {'Status':<14} {'Avg RTF':<10} {'Peak VRAM':<12} {'Notes'}")
    print("-" * 70)
    for row in summary_rows:
        print(f"{row[0]:<28} {row[1]:<14} {row[2]:<10} {row[3]:<12} {row[4]}")
    print("=" * 70)
    print("[INFO] Full telemetry persisted in outputs/benchmarks/results.csv\n")

if __name__ == "__main__":
    main()
