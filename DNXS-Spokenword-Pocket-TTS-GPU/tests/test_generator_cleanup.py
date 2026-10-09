"""Regression coverage for audiobook run-start cleanup behavior."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pocket_tts.audiobook.generator import AudiobookGenerator


class AudiobookCleanupTests(unittest.TestCase):
    """Ensure new runs start from empty chunk dirs and clean top-level artifacts."""

    def test_cleanup_wipes_chunk_dirs_and_preserves_run_logs(self) -> None:
        """Remove stale chunk artifacts while keeping historical run logs."""
        with TemporaryDirectory() as temp_dir:
            tts_dir = Path(temp_dir)
            audio_dir = tts_dir / "audio_chunks"
            text_dir = tts_dir / "text_chunks"
            audio_dir.mkdir()
            text_dir.mkdir()

            (audio_dir / "chunk_00001.wav").write_bytes(b"old-audio")
            (audio_dir / "Failed").mkdir()
            (audio_dir / "Failed" / "stale.wav").write_bytes(b"old-failed")
            (audio_dir / "nested").mkdir()
            (audio_dir / "nested" / "leftover.wav").write_bytes(b"nested")
            (text_dir / "chunk_00001.txt").write_text("old text", encoding="utf-8")
            (text_dir / "audiobook.chunks.json").write_text("{}", encoding="utf-8")
            (text_dir / "nested").mkdir()
            (text_dir / "nested" / "leftover.txt").write_text("nested", encoding="utf-8")

            (tts_dir / "asr_new_failures.json").write_text("[]", encoding="utf-8")
            (tts_dir / "asr_new_pipeline.log").write_text("old pipeline", encoding="utf-8")
            (tts_dir / "asr_stage2_investigation.log").write_text("old review", encoding="utf-8")
            (tts_dir / "run_20260815_090613.log").write_text("keep me", encoding="utf-8")
            (tts_dir / "run_20260815_091000.log").write_text("keep me too", encoding="utf-8")
            (tts_dir / "notes.txt").write_text("leave alone", encoding="utf-8")

            generator = AudiobookGenerator.__new__(AudiobookGenerator)
            generator._cleanup_existing_chunks(audio_dir, text_dir)

            self.assertTrue(audio_dir.is_dir())
            self.assertTrue(text_dir.is_dir())
            self.assertTrue((audio_dir / "Failed").is_dir())
            self.assertEqual(list(audio_dir.iterdir()), [audio_dir / "Failed"])
            self.assertEqual(list(text_dir.iterdir()), [])
            self.assertFalse((tts_dir / "asr_new_failures.json").exists())
            self.assertFalse((tts_dir / "asr_new_pipeline.log").exists())
            self.assertFalse((tts_dir / "asr_stage2_investigation.log").exists())
            self.assertTrue((tts_dir / "run_20260815_090613.log").exists())
            self.assertTrue((tts_dir / "run_20260815_091000.log").exists())
            self.assertTrue((tts_dir / "notes.txt").exists())


if __name__ == "__main__":
    unittest.main()
