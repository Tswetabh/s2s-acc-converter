#!/usr/bin/env python3
"""Faster-Whisper STT Transcription Entry Point.

Loads the local CT2 model from weights/stt/faster-whisper-large-v3-turbo-ct2
and transcribes the given audio file using portable repository-relative paths.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Union

from faster_whisper import WhisperModel

# Repository root (two levels up from scripts/stt/)
ROOT = Path(__file__).resolve().parents[2]

LOCAL_MODELS = {
    "turbo": ROOT / "weights" / "stt" / "faster-whisper-large-v3-turbo-ct2",
    "large-v3-turbo": ROOT / "weights" / "stt" / "faster-whisper-large-v3-turbo-ct2",
    "tiny": ROOT / "weights" / "stt" / "faster-whisper-tiny",
}
MODEL_PATH = LOCAL_MODELS["turbo"]

DEFAULT_AUDIO = ROOT / "Testinput" / "Swetabh_Input.wav"


def resolve_model_path(model_identifier: Union[str, Path, None] = None) -> Union[str, Path]:
    """Resolve model key ('tiny', 'turbo') or custom path/name."""
    if model_identifier is None:
        return MODEL_PATH
    key = str(model_identifier).lower()
    if key in LOCAL_MODELS and LOCAL_MODELS[key].exists():
        return LOCAL_MODELS[key]
    if isinstance(model_identifier, Path) and model_identifier.exists():
        return model_identifier
    if isinstance(model_identifier, str) and Path(model_identifier).exists():
        return Path(model_identifier)
    return model_identifier


def load_model(
    model_path: Optional[Union[str, Path]] = None,
    device: str = "cuda",
    compute_type: str = "float16",
) -> WhisperModel:
    """Load Faster-Whisper model from local weights directory or identifier."""
    target_path = resolve_model_path(model_path)
    if isinstance(target_path, Path) and not target_path.exists():
        raise FileNotFoundError(
            f"Faster-Whisper weights directory not found at: {target_path}\n"
            f"Available local models: {list(LOCAL_MODELS.keys())}"
        )

    try:
        model = WhisperModel(
            str(target_path),
            device=device,
            compute_type=compute_type,
        )
        return model
    except Exception as exc:
        if device == "cuda":
            print(
                f"[WARN] Failed to initialize WhisperModel on CUDA ({exc}). "
                f"Falling back to CPU int8..."
            )
            return WhisperModel(
                str(target_path),
                device="cpu",
                compute_type="int8",
            )
        raise


def load_audio_array(audio_path: Path, target_sr: int = 16000):
    """Load audio file to 16kHz mono float32 numpy array."""
    import numpy as np
    try:
        import soundfile as sf
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = np.mean(data, axis=1)
        data = np.asarray(data, dtype=np.float32)
        if sr != target_sr:
            try:
                import librosa
                data = librosa.resample(data, orig_sr=int(sr), target_sr=target_sr).astype(np.float32)
            except ImportError:
                import scipy.signal
                num_samples = int(len(data) * target_sr / sr)
                data = scipy.signal.resample(data, num_samples).astype(np.float32)
        return data
    except Exception:
        # Fall back to passing path directly to model.transcribe
        return str(audio_path)


def transcribe_audio(
    audio_path: Path,
    model: Optional[WhisperModel] = None,
    language: Optional[str] = None,
    beam_size: int = 5,
    device: str = "cuda",
    compute_type: str = "float16",
) -> dict:
    """Transcribe an audio file and return text and segment metadata."""
    audio_path = Path(audio_path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model is None:
        print(f"Loading model from: {MODEL_PATH}")
        start_load = time.time()
        model = load_model(MODEL_PATH, device=device, compute_type=compute_type)
        print(f"Model loaded in {time.time() - start_load:.2f}s")

    print(f"Transcribing: {audio_path}")
    start_transcribe = time.time()

    audio_input = load_audio_array(audio_path)

    segments, info = model.transcribe(
        audio_input,
        beam_size=beam_size,
        language=language,
    )

    results = []
    full_text_parts = []
    for segment in segments:
        seg_dict = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        results.append(seg_dict)
        full_text_parts.append(segment.text.strip())

    elapsed = time.time() - start_transcribe
    full_text = " ".join(full_text_parts)

    return {
        "text": full_text,
        "segments": results,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "transcription_time_sec": elapsed,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio using local Faster-Whisper model."
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=Path,
        default=None,
        help="Path to input audio file (defaults to sample in Testinput/)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="turbo",
        help="Model choice ('tiny', 'turbo') or path to local CT2 weights (default: turbo)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Legacy alias for --model",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Inference device (default: cuda)",
    )
    parser.add_argument(
        "--compute-type",
        "-c",
        default="float16",
        help="CTranslate2 compute type (e.g. float16, int8, float32)",
    )
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="Language code (e.g. 'en', 'hi'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to write transcribed text output",
    )

    args = parser.parse_args(argv)

    # Determine input audio
    audio_path = args.audio
    if audio_path is None:
        if DEFAULT_AUDIO.exists():
            audio_path = DEFAULT_AUDIO
        else:
            # Check for any .wav in Testinput/
            test_wavs = list((ROOT / "Testinput").glob("*.wav"))
            if test_wavs:
                audio_path = test_wavs[0]
            else:
                print("Error: No audio file specified and no default found in Testinput/")
                return 1

    model_choice = args.model_dir or args.model
    resolved_model = resolve_model_path(model_choice)

    print("=" * 60)
    print("Faster-Whisper STT")
    print(f"Model: {model_choice} -> {resolved_model}")
    print(f"Audio file: {audio_path}")
    print(f"Device: {args.device} | Compute type: {args.compute_type}")
    print("=" * 60)

    try:
        model = load_model(
            model_path=model_choice,
            device=args.device,
            compute_type=args.compute_type,
        )
        res = transcribe_audio(
            audio_path=audio_path,
            model=model,
            language=args.language,
        )

        print("\n" + "-" * 60)
        print(f"Detected Language: {res['language']} (p={res['language_probability']:.2f})")
        print(f"Audio Duration: {res['duration']:.2f}s")
        print(f"Transcription Time: {res['transcription_time_sec']:.2f}s")
        print("-" * 60)
        print("Transcription:")
        print(res["text"])
        print("-" * 60)

        if args.output:
            out_path = args.output.resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(res["text"], encoding="utf-8")
            print(f"Saved transcription to: {out_path}")

        return 0
    except Exception as exc:
        print(f"\n[ERROR] Transcription failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
