#!/usr/bin/env python3
"""Unified Voice Pipeline: M4A/Audio -> STT (Faster-Whisper) -> TTS (Pocket TTS).

Workflow:
  1. Input Audio Selection:
     - Accepts explicit --audio path (supports .m4a, .wav, .mp3, etc.)
     - OR automatically scans the recordings directory:
       'C:\\Users\\tripa\\Documents\\Sound Recordings'
       (or custom path via --recordings-dir / SOUND_RECORDINGS_DIR)
     - Interactive selection via --select, or picks the newest recording by default.
     - Fallback default: 'D:\\tts-poc\\Testinput\\Anoop voice line.wav'.
  2. Audio Conversion:
     - Automatically converts .m4a or any compressed audio to 16kHz mono WAV.
  3. Speech-to-Text (STT):
     - Transcribes using Faster-Whisper ('tiny' for instant speed or 'turbo' for precision).
  4. Text-to-Speech (TTS):
     - Voice-clones using reference voice (default: 'Anoop voice line.wav' or user choice).
     - Synthesizes transcribed text and writes final output WAV.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Optional, Tuple

# Auto-redirect to dedicated virtual environment if executed with global/other Python
ROOT = Path(__file__).resolve().parents[1]
TARGET_PYTHON = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU" / ".venv" / "Scripts" / "python.exe"
if TARGET_PYTHON.exists():
    curr_exe = os.path.normcase(os.path.abspath(sys.executable))
    target_exe = os.path.normcase(str(TARGET_PYTHON.resolve()))
    if curr_exe != target_exe:
        result = subprocess.run([str(TARGET_PYTHON)] + sys.argv, check=False)
        sys.exit(result.returncode)

# Reconfigure stdout/stderr for clean UTF-8 terminal printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Setup project paths
ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.audio_convert import convert_to_wav
from models.stt.faster_whisper.adapter import FasterWhisperAdapter
import soundfile as sf
import torch
from pocket_tts.models.tts_model import TTSModel

# Default paths
DEFAULT_RECORDINGS_DIR = Path(
    os.getenv("SOUND_RECORDINGS_DIR", r"C:\Users\tripa\Documents\Sound Recordings")
)
DEFAULT_FALLBACK_AUDIO = ROOT / "Testinput" / "Anoop voice line.wav"
DEFAULT_REFERENCE_VOICE = ROOT / "Testinput" / "american accent.wav"
DEFAULT_OUTPUT_DIR = ROOT / "Output"


def find_recordings(directory: Path) -> List[Path]:
    """Find all supported audio files in the given directory, sorted newest first."""
    if not directory.exists() or not directory.is_dir():
        return []
    valid_exts = {".m4a", ".wav", ".mp3", ".ogg", ".flac", ".aac", ".wma"}
    files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    # Sort by modification time (most recent first)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files


def choose_recording_interactively(recordings: List[Path]) -> Optional[Path]:
    """Display interactive menu to let the user select a recording."""
    print("\n" + "=" * 60)
    print("Found Recordings:")
    for idx, f in enumerate(recordings, start=1):
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_kb = f.stat().st_size / 1024
        print(f"  [{idx}] {f.name} ({size_kb:.1f} KB, modified: {mtime})")
    print("=" * 60)

    while True:
        try:
            choice = input(f"Select recording number (1-{len(recordings)}) or 'q' to cancel: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice.lower() in ("q", "quit", "exit"):
            return None
        if choice.isdigit():
            num = int(choice)
            if 1 <= num <= len(recordings):
                return recordings[num - 1]
        print(f"Invalid selection. Please enter a number between 1 and {len(recordings)}.")


def resolve_input_audio(
    audio_arg: Optional[str],
    recordings_dir: Path,
    interactive: bool = False,
    fallback_path: Path = DEFAULT_FALLBACK_AUDIO,
) -> Tuple[Path, str]:
    """Resolve input audio path and return (resolved_path, source_description)."""
    # 1. Explicit audio file provided
    if audio_arg:
        p = Path(audio_arg).resolve()
        if p.is_file():
            return p, f"Explicit input: {p.name}"
        raise FileNotFoundError(f"Specified audio file not found: {p}")

    # 2. Check recordings directory
    recordings = find_recordings(recordings_dir)
    if recordings:
        if interactive:
            selected = choose_recording_interactively(recordings)
            if selected:
                return selected, f"User-selected from {recordings_dir}: {selected.name}"
            print("[INFO] Selection cancelled, using default fallback.")
        else:
            newest = recordings[0]
            return newest, f"Latest recording from {recordings_dir}: {newest.name}"

    # 3. Fallback default
    if fallback_path.exists():
        return fallback_path, f"Default fallback: {fallback_path.name}"

    raise FileNotFoundError(
        f"No audio file found. Neither in recordings dir ({recordings_dir}) "
        f"nor fallback path ({fallback_path})."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified S2S Pipeline: Laptop M4A/Audio -> Faster-Whisper STT -> Pocket TTS"
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        default=None,
        help="Path to input audio file (.m4a, .wav, etc.). If omitted, looks in recordings folder.",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=DEFAULT_RECORDINGS_DIR,
        help=f"Directory where new recordings are saved (default: {DEFAULT_RECORDINGS_DIR})",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Interactively select which recording to use from the recordings directory.",
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        default=None,
        help="Reference audio for TTS voice cloning (default: 'Testinput/american accent.wav').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path for generated output WAV file.",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default="tiny",
        choices=["tiny", "turbo", "large-v3-turbo"],
        help="Faster-Whisper model to use: 'tiny' (~75MB, fastest) or 'turbo' (high accuracy). Default: tiny.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for STT & TTS inference (default: auto).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    total_pipeline_start = time.perf_counter()

    print("=" * 65)
    print("   UNIFIED SPEECH-TO-SPEECH WORKFLOW PIPELINE")
    print("   Input Recording -> Conversion -> STT -> TTS Clone")
    print("=" * 65)

    # -------------------------------------------------------------
    # STAGE 1: INGESTION & AUDIO RESOLUTION
    # -------------------------------------------------------------
    print("\n[STAGE 1/4] Resolving Input Audio...")
    t_resolve_start = time.perf_counter()
    input_file, source_desc = resolve_input_audio(
        audio_arg=args.audio,
        recordings_dir=args.recordings_dir,
        interactive=args.select,
        fallback_path=DEFAULT_FALLBACK_AUDIO,
    )
    t_resolve = time.perf_counter() - t_resolve_start
    print(f"  Source : {source_desc}")
    print(f"  File   : {input_file}")
    print(f"  Latency: {t_resolve * 1000:.1f} ms")

    # -------------------------------------------------------------
    # STAGE 2: AUDIO CONVERSION (.m4a -> .wav)
    # -------------------------------------------------------------
    print("\n[STAGE 2/4] Validating & Converting Audio Format...")
    t_convert_start = time.perf_counter()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converted = False
    if input_file.suffix.lower() != ".wav":
        converted_wav_path = DEFAULT_OUTPUT_DIR / f"converted_{input_file.stem}.wav"
        print(f"  Converting {input_file.suffix} -> WAV (16kHz mono)...")
        wav_audio_path = convert_to_wav(input_file, converted_wav_path, target_sr=16000)
        converted = True
        print(f"  Converted WAV ready: {wav_audio_path}")
    else:
        wav_audio_path = input_file
        print(f"  Input is already a WAV file: {wav_audio_path}")
    t_convert = time.perf_counter() - t_convert_start
    print(f"  Conversion Latency: {t_convert * 1000:.1f} ms" + (" (skipped, native WAV)" if not converted else ""))

    # -------------------------------------------------------------
    # STAGE 3: SPEECH-TO-TEXT (STT)
    # -------------------------------------------------------------
    print(f"\n[STAGE 3/4] Transcribing with Faster-Whisper ('{args.stt_model}')...")
    stt_stage_start = time.perf_counter()

    # Determine STT device and compute type
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        stt_device = "cuda"
        stt_compute = "float16"
    else:
        stt_device = "cpu"
        stt_compute = "int8"

    stt_load_start = time.perf_counter()
    stt_adapter = FasterWhisperAdapter(
        model_path=args.stt_model,
        device=stt_device,
        compute_type=stt_compute,
    )
    stt_adapter.load()
    t_stt_load = time.perf_counter() - stt_load_start

    stt_infer_start = time.perf_counter()
    stt_result = stt_adapter.transcribe(str(wav_audio_path))
    t_stt_infer = time.perf_counter() - stt_infer_start
    t_stt_total = time.perf_counter() - stt_stage_start

    transcribed_text = stt_result.get("text", "").strip()
    detected_lang = stt_result.get("language", "unknown")
    audio_len = stt_result.get("duration", 0.0)
    stt_rtf = t_stt_infer / max(audio_len, 0.001)

    print("\n" + "-" * 55)
    print(f"Detected Language  : {detected_lang}")
    print(f"Input Audio Length : {audio_len:.2f}s")
    print(f"STT Model Load     : {t_stt_load * 1000:.1f} ms ({t_stt_load:.2f}s)")
    print(f"STT Inference Time : {t_stt_infer * 1000:.1f} ms ({t_stt_infer:.2f}s) | RTF: {stt_rtf:.2f}x")
    print(f"Total STT Latency  : {t_stt_total * 1000:.1f} ms ({t_stt_total:.2f}s)")
    print(f"Transcribed Text   :\n\"{transcribed_text}\"")
    print("-" * 55)

    if not transcribed_text:
        print("[WARNING] No speech detected in input audio. Exiting.")
        return 0

    # Evict STT from VRAM before loading TTS to preserve GPU memory
    stt_adapter.unload()

    # -------------------------------------------------------------
    # STAGE 4: TEXT-TO-SPEECH (TTS) VOICE CLONING
    # -------------------------------------------------------------
    print("\n[STAGE 4/4] Synthesizing Speech with Pocket TTS...")
    tts_stage_start = time.perf_counter()

    # Resolve reference voice
    if args.reference:
        ref_path = Path(args.reference).resolve()
    else:
        ref_path = DEFAULT_REFERENCE_VOICE

    if not ref_path.exists():
        print(f"  [WARN] Reference voice not found at {ref_path}. Using fallback Alba voice.")
        ref_path = None
    else:
        # If reference audio is M4A, convert it as well
        if ref_path.suffix.lower() != ".wav":
            ref_path = convert_to_wav(ref_path)
        print(f"  Reference Voice : {ref_path.name}")

    # Determine TTS device
    tts_device = "cpu"  # Pocket TTS runs comfortably on CPU or CUDA
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        tts_device = "cuda"

    tts_load_start = time.perf_counter()
    tts_model = TTSModel.load_model(device=tts_device)
    t_tts_load = time.perf_counter() - tts_load_start
    print(f"  TTS Model loaded in {t_tts_load:.2f}s ({t_tts_load * 1000:.1f} ms)")

    # Voice conditioning
    t_cond_start = time.perf_counter()
    if ref_path and tts_model.has_voice_cloning:
        voice_state = tts_model.get_state_for_audio_prompt(str(ref_path))
        print("  Voice conditioning created from reference voice.")
    else:
        fallback_voice = "alba"
        voice_state = tts_model.get_state_for_audio_prompt(fallback_voice)
        print(f"  Using built-in voice: {fallback_voice}")
    t_tts_cond = time.perf_counter() - t_cond_start

    # Output file path
    if args.output:
        output_wav = Path(args.output).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_wav = DEFAULT_OUTPUT_DIR / f"pipeline_output_{ts}.wav"

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # Generation / Synthesis
    gen_start = time.perf_counter()
    audio = tts_model.generate_audio(voice_state, text_to_generate=transcribed_text)
    t_tts_gen = time.perf_counter() - gen_start
    t_tts_total = time.perf_counter() - tts_stage_start

    audio_cpu = audio.cpu()
    sf.write(str(output_wav), audio_cpu.squeeze().numpy(), tts_model.sample_rate)
    out_duration = audio_cpu.shape[-1] / tts_model.sample_rate
    tts_rtf = t_tts_gen / max(out_duration, 0.001)

    total_pipeline_time = time.perf_counter() - total_pipeline_start
    overall_rtf = total_pipeline_time / max(out_duration, 0.001)

    print("\n" + "=" * 65)
    print("                 PIPELINE LATENCY & BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"  {'STAGE':<30} {'TIME (s)':>10} {'LATENCY (ms)':>15} {'METRIC':>12}")
    print("  " + "-" * 61)
    print(f"  {'1. Audio Ingestion/Resolve':<30} {t_resolve:>9.3f}s {t_resolve * 1000:>13.1f} ms {'-':>12}")
    print(f"  {'2. Audio Conversion (M4A->WAV)':<30} {t_convert:>9.3f}s {t_convert * 1000:>13.1f} ms {f'In: {audio_len:.1f}s':>12}")
    print(f"  {'3. STT Total (Faster-Whisper)':<30} {t_stt_total:>9.3f}s {t_stt_total * 1000:>13.1f} ms {f'RTF: {stt_rtf:.2f}x':>12}")
    print(f"     ├─ Model Load             {t_stt_load:>9.3f}s {t_stt_load * 1000:>13.1f} ms")
    print(f"     └─ Audio Transcription    {t_stt_infer:>9.3f}s {t_stt_infer * 1000:>13.1f} ms")
    print(f"  {'4. TTS Total (Pocket TTS)':<30} {t_tts_total:>9.3f}s {t_tts_total * 1000:>13.1f} ms {f'RTF: {tts_rtf:.2f}x':>12}")
    print(f"     ├─ Model Load             {t_tts_load:>9.3f}s {t_tts_load * 1000:>13.1f} ms")
    print(f"     ├─ Voice Conditioning     {t_tts_cond:>9.3f}s {t_tts_cond * 1000:>13.1f} ms")
    print(f"     └─ Speech Synthesis       {t_tts_gen:>9.3f}s {t_tts_gen * 1000:>13.1f} ms {f'Out: {out_duration:.1f}s':>12}")
    print("  " + "-" * 61)
    print(f"  {'TOTAL END-TO-END PIPELINE':<30} {total_pipeline_time:>9.3f}s {total_pipeline_time * 1000:>13.1f} ms {f'RTF: {overall_rtf:.2f}x':>12}")
    print("=" * 65)
    print(f"  Output File : {output_wav}")
    print(f"  Synthesized : {out_duration:.2f}s of audio")
    print("=" * 65 + "\n")

    return 0



if __name__ == "__main__":
    sys.exit(main())
