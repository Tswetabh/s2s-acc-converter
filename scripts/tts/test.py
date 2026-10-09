#!/usr/bin/env python3
"""Pocket TTS - Voice Cloning & Generation Entry Point.

Loads Pocket TTS and generates speech using either custom voice reference
or built-in voices. Uses repo-relative paths.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add repo root and engine dir to sys.path
ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import soundfile as sf
import torch

from pocket_tts.models.tts_model import TTSModel

# Default repo-relative paths
DEFAULT_REFERENCE = ROOT / "Testinput" / "Swetabh_Input.wav"
DEFAULT_OUTPUT = ROOT / "Output" / "cloned_output.wav"


def parse_args():
    parser = argparse.ArgumentParser(description="Pocket TTS Voice Cloning & Inference")
    parser.add_argument(
        "--reference",
        "-r",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Path to reference audio file (.wav)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for output audio (.wav)",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Single text string to generate (non-interactive mode)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Pocket TTS - Voice Cloning")
    print("=" * 60)

    # 1. Device detection
    print("\n[1/3] Checking GPU...")
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            device = "cuda"
            print("CUDA available: YES")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            device = "cpu"
            print("CUDA available: NO (requested cuda but not available; using CPU)")
    else:
        device = "cpu"
        print("Using CPU")

    # 2. Check reference voice
    print("\n[2/3] Checking reference voice...")
    ref_audio = args.reference.resolve()
    if not ref_audio.is_file():
        # Fallback check inside Testinput
        alt_wavs = list((ROOT / "Testinput").glob("*.wav"))
        if alt_wavs:
            ref_audio = alt_wavs[0]
            print(f"Notice: Default not found, using: {ref_audio}")
        else:
            print(f"Warning: Reference audio not found at: {ref_audio}")
            ref_audio = None
    else:
        print(f"Reference: {ref_audio}")

    # 3. Load model
    print("\n[3/3] Loading Pocket TTS model...")
    start = time.time()
    model = TTSModel.load_model(device=device)
    print(f"Model loaded in {time.time() - start:.2f}s")

    # 4. Voice conditioning
    print("\nProcessing voice conditioning...")
    start = time.time()

    if model.has_voice_cloning and ref_audio is not None:
        voice_state = model.get_state_for_audio_prompt(str(ref_audio))
        print(f"Reference voice processed in {time.time() - start:.2f}s")
        print("\n" + "=" * 60)
        print("VOICE CLONING READY (Custom reference voice)")
        print("=" * 60)
    else:
        fallback_voice = "alba"
        if not model.has_voice_cloning:
            print("\n[NOTICE] Gated voice-cloning weights were not loaded.")
            print(f"         Using built-in voice '{fallback_voice}' as fallback.")
            print("         To enable custom voice cloning:")
            print("           1. Accept terms at: https://huggingface.co/kyutai/pocket-tts")
            print("           2. Log in with: python -m huggingface_hub.cli.hf login (or set HF_TOKEN)")
        else:
            print(f"No reference audio provided. Using built-in voice '{fallback_voice}'.")
        voice_state = model.get_state_for_audio_prompt(fallback_voice)
        print(f"Built-in voice '{fallback_voice}' loaded in {time.time() - start:.2f}s")
        print("\n" + "=" * 60)
        print(f"TTS READY (Built-in voice: {fallback_voice})")
        print("=" * 60)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate_single_text(text_content: str):
        print(f"\nGenerating: {text_content[:60]}...")
        gen_start = time.time()
        audio = model.generate_audio(
            voice_state,
            text_to_generate=text_content,
        )
        gen_time = time.time() - gen_start
        audio = audio.cpu()

        sf.write(
            str(output_path),
            audio.squeeze().numpy(),
            model.sample_rate,
        )
        duration = audio.shape[-1] / model.sample_rate
        print("\n[SUCCESS] Generated successfully")
        print(f"Output: {output_path}")
        print(f"Audio duration: {duration:.2f}s")
        print(f"Generation time: {gen_time:.2f}s")
        print(f"RTF: {gen_time / max(duration, 0.001):.2f}x")

    # If single text provided via CLI
    if args.text:
        generate_single_text(args.text)
        return 0

    # Interactive generation loop
    while True:
        try:
            text = input("\nEnter text (or 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if text.lower() in ["exit", "quit", "q"]:
            break
        if not text:
            continue

        generate_single_text(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
