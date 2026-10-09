"""
Voice Prompt Converter
Converts any audio format to the TTS model's required format: 16-bit PCM, 24000 Hz, Mono WAV.
"""

import subprocess
import logging
import os
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# Suppresses the console window Windows would otherwise flash for every
# ffmpeg call, since ffmpeg is a console-subsystem program launched from a
# windowless (pythonw) parent. No-op (0) on non-Windows.
_NOCONSOLE = getattr(subprocess, "CREATE_NO_WINDOW", 0)

class VoicePromptConverter:
    """Converts audio files to the required TTS voice prompt format."""

    def __init__(self):
        """Initialize the converter with FFmpeg path."""
        # POCKET_TTS_FFMPEG_PATH is set by the Windows launcher when ffmpeg
        # isn't on PATH and a private copy was downloaded instead.
        self.ffmpeg_path = os.environ.get("POCKET_TTS_FFMPEG_PATH", "ffmpeg")

    def convert(self, input_path: Union[str, Path], output_dir: Union[str, Path]) -> Path:
        """
        Convert an audio file to 16-bit PCM, 24kHz, Mono WAV format.

        Args:
            input_path: Path to the input audio file (any format)
            output_dir: Directory to save the converted WAV file

        Returns:
            Path to the converted WAV file

        Raises:
            RuntimeError: If FFmpeg conversion fails
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename based on input
        output_filename = f"{input_path.stem}_converted.wav"
        output_path = output_dir / output_filename

        # Check if already exists and skip if so
        if output_path.exists():
            logger.info(f"Converted file already exists: {output_path}")
            return output_path

        logger.info(f"Converting {input_path} to TTS format...")

        # FFmpeg command for conversion
        cmd = [
            self.ffmpeg_path,
            "-y",  # Overwrite output files
            "-i", str(input_path),  # Input file
            "-ac", "1",  # Mono
            "-ar", "24000",  # 24kHz sample rate
            "-acodec", "pcm_s16le",  # 16-bit signed little-endian PCM
            str(output_path)  # Output file
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                creationflags=_NOCONSOLE
            )

            if result.returncode != 0:
                error_msg = f"FFmpeg conversion failed: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info(f"Conversion successful: {output_path}")
            return output_path

        except subprocess.TimeoutExpired:
            error_msg = "FFmpeg conversion timed out"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during conversion: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def validate_conversion(self, output_path: Path) -> bool:
        """
        Validate that the converted file meets TTS requirements.

        Args:
            output_path: Path to the converted WAV file

        Returns:
            True if valid, False otherwise
        """
        try:
            import wave
            with wave.open(str(output_path), 'rb') as wav:
                # Check format requirements
                if wav.getnchannels() != 1:
                    logger.error(f"Converted file has {wav.getnchannels()} channels, expected 1")
                    return False
                if wav.getframerate() != 24000:
                    logger.error(f"Converted file has {wav.getframerate()} Hz, expected 24000")
                    return False
                if wav.getsampwidth() != 2:  # 16-bit = 2 bytes
                    logger.error(f"Converted file has {wav.getsampwidth()*8}-bit depth, expected 16-bit")
                    return False
                if wav.getcomptype() != 'NONE':
                    logger.error("Converted file is compressed, expected uncompressed PCM")
                    return False

            logger.info(f"Converted file validation passed: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error validating converted file: {e}")
            return False