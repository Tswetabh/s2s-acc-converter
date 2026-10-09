"""CLI script to run inference on a single TTS model."""

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
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.registry import get_model, list_available_models
from app.core.inference import InferenceEngine
from app.config import AUDIO_OUTPUT_DIR, DEFAULT_REFERENCE_AUDIO

def main():
    parser = argparse.ArgumentParser(description="Unified TTS Single Model Inference CLI")
    parser.add_argument("--model", type=str, required=True, help="Model ID (e.g., kokoro, piper, qwen3tts, etc.)")
    parser.add_argument("--text", type=str, default="Artificial intelligence is changing the way people learn and work.", help="Text to synthesize")
    parser.add_argument("--output", type=str, default=None, help="Output WAV path (default: auto in outputs/audio/<model>/)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--speaker", type=str, default=None, help="Speaker identifier / voice preset")
    parser.add_argument("--language", type=str, default="en", help="Language code (e.g. en, hi, zh)")
    parser.add_argument("--reference-audio", type=str, default=None, help="Path to reference audio for voice cloning")

    args = parser.parse_args()

    model_adapter = get_model(args.model)
    if not model_adapter:
        print(f"[ERROR] Unknown model '{args.model}'.")
        print("Available models:", ", ".join(list_available_models()))
        sys.exit(1)

    print(f"\n[INFO] Running inference for: {model_adapter.name}")
    print(f"[INFO] Hardware Profile: {model_adapter.hardware_classification}")

    health = model_adapter.health_check()
    print(f"[STATUS] Health check: {health['status']} ({health['reason']})")

    out_path = args.output
    if not out_path:
        out_dir = AUDIO_OUTPUT_DIR / args.model
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(out_dir / f"{args.model}_{timestamp}.wav")

    ref_audio = args.reference_audio
    if not ref_audio and model_adapter.supports_voice_cloning and DEFAULT_REFERENCE_AUDIO.exists():
        ref_audio = str(DEFAULT_REFERENCE_AUDIO)

    try:
        metrics = InferenceEngine.run_generation(
            model_adapter=model_adapter,
            text=args.text,
            output_path=out_path,
            device=args.device,
            language=args.language,
            speaker=args.speaker,
            reference_audio=ref_audio,
        )

        print("\n" + "=" * 45)
        print(f"GENERATION SUCCESSFUL: {model_adapter.name}")
        print("-" * 45)
        print(f"{'Output Path:':<22} {metrics['output_path']}")
        print(f"{'Load Time:':<22} {metrics['load_time_sec']} s")
        print(f"{'Generation Time:':<22} {metrics['generation_time_sec']} s")
        print(f"{'Audio Duration:':<22} {metrics['audio_duration_sec']} s")
        print(f"{'Real-Time Factor:':<22} {metrics['rtf']} (RTF)")
        print(f"{'TTFA:':<22} {metrics['ttfa_sec']} s")
        print(f"{'Peak VRAM:':<22} {metrics['vram_peak_gb']} GB")
        print(f"{'Peak RAM:':<22} {metrics['ram_peak_gb']} GB")
        print(f"{'Sample Rate:':<22} {metrics['sample_rate']} Hz")
        print("=" * 45 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Generation failed: {str(e)}")
        sys.exit(1)
    finally:
        InferenceEngine.unload_active_model()

if __name__ == "__main__":
    main()
