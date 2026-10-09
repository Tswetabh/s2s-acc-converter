#!/usr/bin/env python3
"""VRAM stress: 1 TTS worker + progressive ASR (base) workers 1..N.

Loads one TTS model on CUDA (main project env), then in a separate ASR/venv
process loads faster-whisper base models one at a time up to --max-asr
(default 4), reporting free/used VRAM after each step.

Mirrors production: TTS in main process, ASR in ASR/venv (separate process,
shared GPU memory).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "tests" / "results" / "vram_stress_1tts_asr"


def _nvidia_smi() -> dict:
    """Return GPU memory stats in MiB via nvidia-smi."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used, free, total, util = [x.strip() for x in out.split(",")]
    return {
        "used_mib": int(float(used)),
        "free_mib": int(float(free)),
        "total_mib": int(float(total)),
        "util_pct": int(float(util)),
    }


def _tts_holder_script() -> str:
    """Python source run in main venv: load 1 TTS model and wait on stdin."""
    return r"""
import sys, time, os
sys.path.insert(0, os.environ.get("POCKETGPU_ROOT", "."))
import torch
from pocket_tts.models.tts_model import TTSModel

def vram():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated(0) / (1024**2)
        reserved = torch.cuda.memory_reserved(0) / (1024**2)
        return alloc, reserved
    return 0.0, 0.0

print("TTS_HOLDER: loading TTSModel on cuda...", flush=True)
t0 = time.time()
model = TTSModel.load_model(device="cuda")
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.time() - t0
alloc, reserved = vram()
print(f"TTS_HOLDER: READY load_s={elapsed:.2f} torch_alloc_mib={alloc:.0f} torch_reserved_mib={reserved:.0f}", flush=True)
print("TTS_HOLDER: waiting (send 'quit' on stdin to exit)", flush=True)
# Keep model alive; block on stdin
for line in sys.stdin:
    if line.strip().lower() in ("quit", "exit", "q"):
        break
print("TTS_HOLDER: exiting", flush=True)
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""


def _asr_loader_script(max_asr: int, model_name: str) -> str:
    """Python source run in ASR/venv: load N Whisper models, report after each."""
    # model_name and max_asr injected as literals
    return f"""
import json, sys, time, gc
import torch
from faster_whisper import WhisperModel

MAX_ASR = {int(max_asr)}
MODEL = {model_name!r}
models = []

def smi():
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    used, free, total, util = [x.strip() for x in out.split(",")]
    return {{
        "used_mib": int(float(used)),
        "free_mib": int(float(free)),
        "total_mib": int(float(total)),
        "util_pct": int(float(util)),
    }}

def torch_vram():
    if not torch.cuda.is_available():
        return {{"alloc_mib": 0, "reserved_mib": 0}}
    torch.cuda.synchronize()
    return {{
        "alloc_mib": round(torch.cuda.memory_allocated(0) / (1024**2)),
        "reserved_mib": round(torch.cuda.memory_reserved(0) / (1024**2)),
    }}

print("ASR_LOADER: start", flush=True)
results = []
baseline = smi()
print("ASR_STEP " + json.dumps({{"n_asr": 0, "status": "baseline_in_asr_proc", "smi": baseline, "torch": torch_vram()}}), flush=True)

for i in range(1, MAX_ASR + 1):
    step = {{"n_asr": i, "model": MODEL}}
    t0 = time.time()
    try:
        m = WhisperModel(MODEL, device="cuda", compute_type="float16")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        models.append(m)
        step["status"] = "ok"
        step["load_s"] = round(time.time() - t0, 2)
    except Exception as e:
        step["status"] = "fail"
        step["error"] = f"{{type(e).__name__}}: {{e}}"
        step["load_s"] = round(time.time() - t0, 2)
        step["smi"] = smi()
        step["torch"] = torch_vram()
        print("ASR_STEP " + json.dumps(step), flush=True)
        results.append(step)
        break
    step["smi"] = smi()
    step["torch"] = torch_vram()
    # delta vs previous ok step
    print("ASR_STEP " + json.dumps(step), flush=True)
    results.append(step)

print("ASR_DONE " + json.dumps({{"loaded": len(models), "steps": results}}), flush=True)
# hold models until parent closes stdin
for line in sys.stdin:
    if line.strip().lower() in ("quit", "exit", "q"):
        break
print("ASR_LOADER: exiting", flush=True)
models.clear()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""


def main() -> int:
    """Run 1-TTS + progressive ASR VRAM stress and write JSON results."""
    parser = argparse.ArgumentParser(description="VRAM stress: 1 TTS + N ASR base workers")
    parser.add_argument("--max-asr", type=int, default=4, help="Max ASR workers to load (default 4)")
    parser.add_argument("--model", default="base", help="faster-whisper model name (default base)")
    parser.add_argument(
        "--tts-python",
        default=str(ROOT / "venv" / "bin" / "python"),
        help="Python for TTS holder (main project venv)",
    )
    parser.add_argument(
        "--asr-python",
        default=str(ROOT / "ASR" / "venv" / "bin" / "python"),
        help="Python for ASR loader (ASR/venv)",
    )
    parser.add_argument(
        "--out",
        default=str(RESULTS_DIR / "vram_stress.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timeline = []

    def snap(label: str, **extra) -> dict:
        """Record one VRAM snapshot into timeline and print it."""
        row = {"label": label, "smi": _nvidia_smi(), "t": time.time(), **extra}
        timeline.append(row)
        s = row["smi"]
        print(
            f"[{label}] used={s['used_mib']} MiB  free={s['free_mib']} MiB  "
            f"total={s['total_mib']} MiB  util={s['util_pct']}%",
            flush=True,
        )
        return row

    print("=== VRAM stress: 1 TTS + ASR base × 1..%d ===" % args.max_asr, flush=True)
    snap("0_baseline")

    # --- TTS holder ---
    env = os.environ.copy()
    env["POCKETGPU_ROOT"] = str(ROOT)
    # Avoid spawning many CPU threads during load
    env.setdefault("OMP_NUM_THREADS", "4")
    tts_proc = subprocess.Popen(
        [args.tts_python, "-u", "-c", _tts_holder_script()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
        env=env,
        bufsize=1,
    )
    print(f"TTS holder PID={tts_proc.pid}", flush=True)

    tts_ready = False
    tts_meta = {}
    deadline = time.time() + 300
    while time.time() < deadline:
        line = tts_proc.stdout.readline()
        if not line and tts_proc.poll() is not None:
            break
        if line:
            print(line.rstrip(), flush=True)
            if "TTS_HOLDER: READY" in line:
                tts_ready = True
                # parse load_s=.. torch_alloc_mib=..
                for part in line.split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        tts_meta[k] = v
                break
    if not tts_ready:
        print("ERROR: TTS holder failed to become READY", flush=True)
        try:
            tts_proc.kill()
        except Exception:
            pass
        return 1

    after_tts = snap("1_after_1_tts", tts_meta=tts_meta)
    free_after_tts = after_tts["smi"]["free_mib"]
    used_after_tts = after_tts["smi"]["used_mib"]
    tts_delta = used_after_tts - timeline[0]["smi"]["used_mib"]
    print(f"  → TTS footprint ≈ {tts_delta} MiB  free after TTS={free_after_tts} MiB", flush=True)

    # --- ASR progressive loader ---
    asr_proc = subprocess.Popen(
        [args.asr_python, "-u", "-c", _asr_loader_script(args.max_asr, args.model)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
        env=env,
        bufsize=1,
    )
    print(f"ASR loader PID={asr_proc.pid}", flush=True)

    asr_steps = []
    asr_done = None
    deadline = time.time() + 600
    while time.time() < deadline:
        line = asr_proc.stdout.readline()
        if not line and asr_proc.poll() is not None:
            break
        if not line:
            time.sleep(0.05)
            continue
        print(line.rstrip(), flush=True)
        if line.startswith("ASR_STEP "):
            step = json.loads(line[len("ASR_STEP ") :])
            asr_steps.append(step)
            n = step.get("n_asr", 0)
            status = step.get("status")
            smi = step.get("smi", {})
            snap(
                f"2_asr_{n}_{status}",
                n_asr=n,
                asr_step=step,
            )
            if status == "fail":
                print(
                    f"  → OOM/FAIL at ASR worker #{n}: {step.get('error')}",
                    flush=True,
                )
                break
            if n >= 1:
                delta = smi.get("used_mib", 0) - used_after_tts
                print(
                    f"  → cumulative ASR VRAM ≈ {delta} MiB  "
                    f"free={smi.get('free_mib')} MiB  "
                    f"per-worker~{delta // max(1, n)} MiB",
                    flush=True,
                )
        elif line.startswith("ASR_DONE "):
            asr_done = json.loads(line[len("ASR_DONE ") :])
            break

    # Final snap while both still alive
    final = snap("3_final_both_loaded", n_asr_loaded=len([s for s in asr_steps if s.get("status") == "ok"]))

    # Teardown
    for proc, name in ((asr_proc, "ASR"), (tts_proc, "TTS")):
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
                proc.wait(timeout=30)
        except Exception as e:
            print(f"  {name} clean exit failed ({e}); killing", flush=True)
            try:
                proc.kill()
            except Exception:
                pass

    time.sleep(1.0)
    snap("4_after_teardown")

    ok_asr = [s for s in asr_steps if s.get("status") == "ok" and s.get("n_asr", 0) > 0]
    fail_asr = [s for s in asr_steps if s.get("status") == "fail"]

    summary = {
        "test": "vram_stress_1tts_asr",
        "tts_workers": 1,
        "asr_model": args.model,
        "asr_max_requested": args.max_asr,
        "asr_loaded_ok": len(ok_asr),
        "asr_failed_at": fail_asr[0]["n_asr"] if fail_asr else None,
        "tts_delta_mib": tts_delta,
        "free_after_tts_mib": free_after_tts,
        "final_used_mib": final["smi"]["used_mib"],
        "final_free_mib": final["smi"]["free_mib"],
        "asr_steps": asr_steps,
        "timeline": [
            {
                "label": r["label"],
                "used_mib": r["smi"]["used_mib"],
                "free_mib": r["smi"]["free_mib"],
                "n_asr": r.get("n_asr"),
            }
            for r in timeline
        ],
        "verdict": (
            f"1 TTS + {len(ok_asr)} ASR ({args.model}) fit"
            + (f"; failed at #{fail_asr[0]['n_asr']}" if fail_asr else f" (all {args.max_asr} OK)")
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== VERDICT ===", flush=True)
    print(summary["verdict"], flush=True)
    print(f"Wrote {out_path}", flush=True)
    for r in summary["timeline"]:
        print(
            f"  {r['label']:28s} used={r['used_mib']:5d}  free={r['free_mib']:5d}",
            flush=True,
        )
    return 0 if not fail_asr else 2


if __name__ == "__main__":
    sys.exit(main())
