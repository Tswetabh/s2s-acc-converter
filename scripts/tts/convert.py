#!/usr/bin/env python3
"""Audio Converter Utility for Pocket TTS.

Converts arbitrary input audio (MP3, M4A, WAV, FLAC, etc.) to 24 kHz mono WAV,
which is the native format for Pocket TTS voice reference conditioning.
Uses portable repo-relative paths and CLI arguments.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def convert_audio(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 24000,
    channels: int = 1,
) -> Path:
    """Convert audio file to mono WAV at target sample rate."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try pydub if installed
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(input_path))
        audio = audio.set_channels(channels)
        audio = audio.set_frame_rate(sample_rate)
        audio.export(str(output_path), format="wav")
        return output_path
    except Exception:
        pass

    # 2. Try ffmpeg (system or bundled in repo)
    ffmpeg_candidates = [
        os.environ.get("POCKET_TTS_FFMPEG_PATH"),
        str(ROOT / "ffmpeg" / "ffmpeg.exe"),
        "ffmpeg",
    ]
    for ff in ffmpeg_candidates:
        if ff and (Path(ff).is_file() or subprocess.run(["where", ff] if sys.platform == "win32" else ["which", ff], capture_output=True).returncode == 0):
            cmd = [
                ff, "-y", "-i", str(input_path),
                "-ac", str(channels),
                "-ar", str(sample_rate),
                str(output_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and output_path.exists():
                return output_path

    # 3. Try PyAV (av)
    try:
        import av
        import numpy as np
        import soundfile as sf
        import scipy.signal

        container = av.open(str(input_path))
        stream = container.streams.audio[0]
        in_sr = stream.codec_context.sample_rate

        frames = []
        for frame in container.decode(audio=0):
            frames.append(frame.to_ndarray())
        container.close()

        if frames:
            data = np.concatenate(frames, axis=1)  # (channels, samples)
            if data.dtype != np.float32:
                # Normalize integer formats
                if np.issubdtype(data.dtype, np.integer):
                    max_val = float(np.iinfo(data.dtype).max)
                    data = data.astype(np.float32) / max_val
                else:
                    data = data.astype(np.float32)
            if data.shape[0] > 1:
                data = data.mean(axis=0)
            else:
                data = data.squeeze(0)

            if in_sr != sample_rate:
                num_samples = int(len(data) * sample_rate / in_sr)
                data = scipy.signal.resample(data, num_samples).astype(np.float32)

            sf.write(str(output_path), data, sample_rate)
            return output_path
    except Exception:
        pass

    # 4. Try soundfile + scipy
    try:
        import numpy as np
        import soundfile as sf
        import scipy.signal

        data, sr = sf.read(str(input_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != sample_rate:
            num_samples = int(len(data) * sample_rate / sr)
            data = scipy.signal.resample(data, num_samples)
        data = np.asarray(data, dtype=np.float32)
        sf.write(str(output_path), data, sample_rate)
        return output_path
    except Exception as exc:
        raise RuntimeError(
            f"Failed to convert {input_path} using available backends: {exc}"
        )


def main():
    parser = argparse.ArgumentParser(description="Convert audio to mono 24 kHz WAV for Pocket TTS")
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Path to input audio file",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="Path to output WAV file",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=24000,
        help="Sample rate (default: 24000)",
    )

    args = parser.parse_args()

    # Determine default paths
    input_file = args.input_file
    if not input_file:
        # Check Testinput/ for an m4a or wav
        m4a_candidates = list((ROOT / "Testinput").glob("*.m4a"))
        if m4a_candidates:
            input_file = m4a_candidates[0]
        else:
            print("Usage: python convert.py <input_audio> [output_audio.wav]")
            return 1

    input_path = Path(input_file).resolve()
    if args.output_file:
        output_path = Path(args.output_file).resolve()
    else:
        output_path = input_path.with_suffix(".wav")

    print(f"Converting: {input_path}")
    print(f"Target: {output_path} ({args.sr} Hz, mono)")

    try:
        out = convert_audio(input_path, output_path, sample_rate=args.sr)
        print("Converted successfully!")
        print(f"Output: {out}")
        return 0
    except Exception as exc:
        print(f"Conversion error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
