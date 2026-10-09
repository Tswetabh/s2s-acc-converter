"""Audio utilities for processing, verification, and measurement."""

import os
import wave
import struct
from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np

def get_audio_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect WAV audio file duration, sample rate, channels, and file size."""
    path = Path(file_path)
    if not path.exists():
        return {
            "exists": False,
            "duration_sec": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "file_size_mb": 0.0,
        }

    file_size_mb = round(os.path.getsize(path) / (1024 * 1024), 4)

    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            duration = round(frames / float(rate), 3) if rate > 0 else 0.0
            return {
                "exists": True,
                "duration_sec": duration,
                "sample_rate": rate,
                "channels": channels,
                "file_size_mb": file_size_mb,
            }
    except Exception:
        # Fallback to soundfile if wave fails
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return {
                "exists": True,
                "duration_sec": round(info.duration, 3),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "file_size_mb": file_size_mb,
            }
        except Exception:
            return {
                "exists": True,
                "duration_sec": 0.0,
                "sample_rate": 0,
                "channels": 1,
                "file_size_mb": file_size_mb,
            }

def save_numpy_to_wav(
    audio_data: Union[np.ndarray, Any],
    output_path: Union[str, Path],
    sample_rate: int = 24000
) -> str:
    """Save raw float/int audio array to PCM 16-bit WAV cleanly."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert torch tensor if needed
    if hasattr(audio_data, "detach"):
        audio_data = audio_data.detach().cpu().numpy()

    audio_data = np.asarray(audio_data)

    # Flatten if 2D single channel
    if audio_data.ndim > 1:
        if audio_data.shape[0] == 1:
            audio_data = audio_data.squeeze(0)
        elif audio_data.shape[-1] == 1:
            audio_data = audio_data.squeeze(-1)

    # Sanitize float audio (-1.0 to 1.0)
    if np.issubdtype(audio_data.dtype, np.floating):
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
        audio_data = np.clip(audio_data, -1.0, 1.0)
        int16_data = (audio_data * 32767.0).astype(np.int16)
    else:
        int16_data = audio_data.astype(np.int16)

    # Try soundfile
    try:
        import soundfile as sf
        sf.write(str(path), int16_data, sample_rate, subtype="PCM_16")
        return str(path)
    except Exception:
        pass

    # Standard library wave fallback
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())

    return str(path)


def prepare_reference_audio(
    input_source: Union[str, Path, bytes, Any],
    output_path: Union[str, Path],
    max_duration_sec: float = 10.0,
    target_sample_rate: int = 24000,
) -> Dict[str, Any]:
    """
    Sanitizes, downmixes to mono, trims to safe length, and normalizes reference audio.
    Protects 6GB VRAM from OOM crashes caused by uploading full podcast/audio files.
    """
    import io
    import soundfile as sf
    from scipy import signal

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Load audio data from file or buffer
    if isinstance(input_source, (bytes, bytearray)):
        data, orig_sr = sf.read(io.BytesIO(input_source))
    elif hasattr(input_source, "getbuffer"):
        data, orig_sr = sf.read(io.BytesIO(input_source.getbuffer()))
    elif hasattr(input_source, "read"):
        data, orig_sr = sf.read(input_source)
    else:
        data, orig_sr = sf.read(str(input_source))

    # Convert to mono if multichannel
    if data.ndim > 1:
        data = data.mean(axis=1)

    orig_duration = round(len(data) / float(orig_sr), 2)
    was_trimmed = False

    # Trim to max duration (e.g., 10 seconds)
    max_samples = int(max_duration_sec * orig_sr)
    if len(data) > max_samples:
        data = data[:max_samples]
        was_trimmed = True

    # Resample if needed
    if orig_sr != target_sample_rate:
        num_target_samples = int(len(data) * target_sample_rate / orig_sr)
        data = signal.resample(data, num_target_samples)
        curr_sr = target_sample_rate
    else:
        curr_sr = orig_sr

    # Peak normalize
    max_val = np.max(np.abs(data)) if len(data) > 0 else 0
    if max_val > 0.01:
        data = data / max_val * 0.95

    # Convert to 16-bit PCM and write
    int16_data = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(str(out_p), int16_data, curr_sr, subtype="PCM_16")

    final_duration = round(len(int16_data) / float(curr_sr), 2)
    file_size_mb = round(os.path.getsize(out_p) / (1024 * 1024), 2)

    return {
        "output_path": str(out_p),
        "orig_duration_sec": orig_duration,
        "final_duration_sec": final_duration,
        "sample_rate": curr_sr,
        "was_trimmed": was_trimmed,
        "file_size_mb": file_size_mb,
    }
