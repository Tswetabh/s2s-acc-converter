"""Zero-Shot Voice Cloning CLI & Interactive Script.

Clones any voice from a short reference audio clip (3-10s) using state-of-the-art
neural TTS architectures (F5-TTS, CosyVoice, XTTS-v2).
"""

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
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Point HuggingFace cache to secondary D drive
os.environ["HF_HOME"] = "D:/tts-poc-cache/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
scripts_dir = str(Path(sys.executable).parent)
if scripts_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")

from app.core.audio_utils import prepare_reference_audio, get_audio_metadata
from app.core.gpu_monitor import GPUMonitor
from app.config import AUDIO_OUTPUT_DIR, REFERENCE_VOICES_DIR

PRESET_VOICES = {
    "hindi": {
        "path": "data/reference_voices/hindi_f_tyagi.wav",
        "description": "🇮🇳 Hindi Female Voice (Tyagi, 5.2s)",
        "default_ref_text": "क्रिप्या बिना टिकेट यात्रा ना करें यहां दंडनिय अपराध है"
    },
    "tyagi": {
        "path": "data/reference_voices/hindi_f_tyagi.wav",
        "description": "🇮🇳 Hindi Female Voice (Tyagi, 5.2s)",
        "default_ref_text": "क्रिप्या बिना टिकेट यात्रा ना करें यहां दंडनिय अपराध है"
    },
    "female": {
        "path": "data/reference_voices/reference_01.wav",
        "description": "👩 English Female Voice (Studio, 8.5s)",
        "default_ref_text": "Hello, welcome to the speech evaluation benchmark. This reference voice is used to evaluate zero-shot cloning fidelity."
    },
    "male": {
        "path": "data/reference_voices/reference_02.wav",
        "description": "👨 English Male Voice (Studio, 8.3s)",
        "default_ref_text": "Deep learning and transformer architectures enable computers to synthesize natural human speech with expressive intonation."
    }
}

def resolve_reference_audio(ref_input: str) -> tuple[str, str]:
    """Resolve preset shortcut or file path, returning (audio_path, known_ref_text)."""
    key = ref_input.strip().lower()
    if key in PRESET_VOICES:
        preset = PRESET_VOICES[key]
        return str(Path(preset["path"]).resolve()), preset["default_ref_text"]
    
    clean_path = ref_input.strip().strip('"').strip("'")
    p = Path(clean_path)
    if not p.exists():
        raise FileNotFoundError(f"Reference audio file not found: {clean_path}")
    
    fname = p.name.lower()
    if "tyagi" in fname or "hindi" in fname:
        return str(p.resolve()), PRESET_VOICES["hindi"]["default_ref_text"]
    elif "reference_01" in fname:
        return str(p.resolve()), PRESET_VOICES["female"]["default_ref_text"]
    elif "reference_02" in fname:
        return str(p.resolve()), PRESET_VOICES["male"]["default_ref_text"]
    
    return str(p.resolve()), ""

def ensure_reference_text(audio_path: str, user_ref_text: str = "") -> str:
    """Ensure reference audio has text. If empty, transcribe on GPU and immediately evict Whisper."""
    if user_ref_text and user_ref_text.strip():
        return user_ref_text.strip()
    
    p = Path(audio_path)
    fname = p.name.lower()
    if "tyagi" in fname or "hindi" in fname:
        return PRESET_VOICES["hindi"]["default_ref_text"]
    elif "reference_01" in fname:
        return PRESET_VOICES["female"]["default_ref_text"]
    elif "reference_02" in fname:
        return PRESET_VOICES["male"]["default_ref_text"]

    print("[INFO] Transcribing reference audio with Whisper (auto-evicting from VRAM after)...")
    import soundfile as sf
    from transformers import pipeline
    import torch, gc

    data, sr = sf.read(audio_path)
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    res = asr({"raw": data, "sampling_rate": sr})
    transcribed_text = res["text"].strip()
    print(f"[INFO] Transcribed reference speech: \"{transcribed_text}\"")

    # CRITICAL: Evict Whisper completely before F5-TTS starts!
    del asr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    return transcribed_text

def run_f5tts_cloning(
    ref_audio_path: str,
    target_text: str,
    output_path: str,
    ref_text: str = "",
    device: str = "cuda",
    nfe_step: int = 32,
    speed: float = 1.0
) -> dict:
    """Run real F5-TTS Flow Matching zero-shot cloning."""
    print(f"[INFO] Initializing F5-TTS model on {device.upper()}...")
    t_load_start = time.perf_counter()
    from f5_tts.api import F5TTS
    f5 = F5TTS(device=device, hf_cache_dir="D:/tts-poc-cache/huggingface")
    load_time = round(time.perf_counter() - t_load_start, 2)
    print(f"[INFO] F5-TTS loaded in {load_time}s.")

    print(f"[INFO] Running Flow Matching synthesis (NFE steps: {nfe_step})...")
    t_gen_start = time.perf_counter()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        with torch.inference_mode():
            wav, sr, _ = f5.infer(
                ref_file=ref_audio_path,
                ref_text=ref_text,
                gen_text=target_text,
                file_wave=output_path,
                nfe_step=nfe_step,
                speed=speed,
            )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print("\n[WARN] GPU VRAM pressure detected. Offloading Vocos vocoder to CPU and finalizing waveform...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            f5.vocoder = f5.vocoder.to("cpu")
            with torch.inference_mode():
                wav, sr, _ = f5.infer(
                    ref_file=ref_audio_path,
                    ref_text=ref_text,
                    gen_text=target_text,
                    file_wave=output_path,
                    nfe_step=nfe_step,
                    speed=speed,
                )
        else:
            raise

    gen_time = round(time.perf_counter() - t_gen_start, 2)

    meta = get_audio_metadata(output_path)
    audio_dur = meta.get("duration_sec", 0.0)
    rtf = round(gen_time / max(0.001, audio_dur), 3)

    return {
        "load_time_sec": load_time,
        "gen_time_sec": gen_time,
        "audio_duration_sec": audio_dur,
        "rtf": rtf,
        "sample_rate": sr,
    }

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Voice Cloning CLI")
    parser.add_argument(
        "--ref-audio", "-r", type=str, default=None,
        help="Path to reference audio file (.mp3, .wav, .flac) or preset ('tyagi', 'female', 'male')"
    )
    parser.add_argument(
        "--text", "-t", type=str, default=None,
        help="Text for the cloned voice to synthesize"
    )
    parser.add_argument(
        "--ref-text", type=str, default="",
        help="Optional transcript of what is spoken in the reference audio clip"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="f5tts", choices=["f5tts", "cosyvoice2", "xtts"],
        help="Zero-shot cloning model to use (default: f5tts)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output WAV destination path"
    )
    parser.add_argument(
        "--steps", type=int, default=16,
        help="Diffusion / Flow Matching denoising steps (default: 16 for fast laptop inference, or 32 for max fidelity)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speech speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda", choices=["cuda", "cpu", "auto"],
        help="Device to run on (default: cuda)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("      🎙️  ZERO-SHOT VOICE CLONING STUDIO  🎙️")
    print("=" * 60)

    # Interactive prompt if parameters missing
    ref_audio_input = args.ref_audio
    if not ref_audio_input:
        print("\nPreset voices available:")
        for k, v in PRESET_VOICES.items():
            print(f"  • '{k}': {v['description']}")
        default_ref = r"C:\Users\tripa\Downloads\Hindi_F_Tyagi.mp3"
        prompt_str = f"\nEnter reference audio path or preset [default: '{default_ref}']: "
        user_choice = input(prompt_str).strip()
        ref_audio_input = user_choice if user_choice else default_ref

    target_text = args.text
    if not target_text:
        default_text = "नमस्ते, यह मेरा नया क्लोन किया हुआ आवाज़ है। यह मॉडल मेरे अंदाज़ में बोल रहा है।"
        user_text = input(f"\nEnter text to synthesize [default: '{default_text}']: ").strip()
        target_text = user_text if user_text else default_text

    # Resolve audio path
    try:
        raw_ref_path, preset_ref_text = resolve_reference_audio(ref_audio_input)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    ref_text = ensure_reference_text(raw_ref_path, user_ref_text=args.ref_text or preset_ref_text)

    # Sanitize and prepare reference audio (trim > 10s to protect RTX 3050 VRAM)
    cloned_dir = AUDIO_OUTPUT_DIR / "cloned"
    cloned_dir.mkdir(parents=True, exist_ok=True)
    temp_prep_path = cloned_dir / f"ref_prepared_{int(time.time())}.wav"

    print(f"\n[STEP 1/3] Preprocessing Reference Voice: {Path(raw_ref_path).name}")
    prep = prepare_reference_audio(
        input_source=raw_ref_path,
        output_path=temp_prep_path,
        max_duration_sec=10.0,
        target_sample_rate=24000
    )
    if prep["was_trimmed"]:
        print(f"  ✂️  Original audio was {prep['orig_duration_sec']}s. Safely trimmed to {prep['final_duration_sec']}s for RTX 3050 VRAM safety.")
    else:
        print(f"  ✅ Reference audio validated ({prep['final_duration_sec']}s, 24kHz mono).")

    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = args.output or str(cloned_dir / f"cloned_{args.model}_{timestamp}.wav")

    # GPU Monitor Baseline
    GPUMonitor.cleanup()
    GPUMonitor.reset_peak_stats()
    base_state = GPUMonitor.capture_state()
    base_vram = base_state["vram_used_gb"]

    print(f"\n[STEP 2/3] Synthesizing Voice with {args.model.upper()}...")
    print(f"  Target Text: \"{target_text}\"")

    try:
        if args.model == "f5tts":
            metrics = run_f5tts_cloning(
                ref_audio_path=prep["output_path"],
                target_text=target_text,
                output_path=out_file,
                ref_text=ref_text,
                device=args.device if args.device != "auto" else "cuda",
                nfe_step=args.steps,
                speed=args.speed
            )
        else:
            from app.registry import get_model
            from app.core.inference import InferenceEngine
            adapter = get_model(args.model)
            if not adapter:
                raise ValueError(f"Model '{args.model}' not found in registry.")
            metrics = InferenceEngine.run_generation(
                model_adapter=adapter,
                text=target_text,
                output_path=out_file,
                device=args.device,
                reference_audio=prep["output_path"]
            )
    except Exception as e:
        print(f"\n[ERROR] Voice cloning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        peak_vram = GPUMonitor.get_peak_vram_gb(base_vram)
        # Step 3: Enforce GPU Memory cleanup
        GPUMonitor.cleanup()

    print("\n" + "=" * 60)
    print("      🎉  VOICE CLONING COMPLETED SUCCESSFULLY  🎉")
    print("=" * 60)
    print(f"  {'Output Audio:':<22} {out_file}")
    print(f"  {'Speech Duration:':<22} {metrics['audio_duration_sec']} seconds")
    print(f"  {'Generation Time:':<22} {metrics['gen_time_sec']} seconds")
    print(f"  {'Real-Time Factor:':<22} {metrics['rtf']} RTF")
    print(f"  {'Peak GPU VRAM:':<22} {peak_vram} GB")
    print(f"  {'Sample Rate:':<22} {metrics['sample_rate']} Hz")
    print("=" * 60)
    print(f"\n[TIP] You can play the audio directly: {out_file}\n")

if __name__ == "__main__":
    main()
