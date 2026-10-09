"""Regression coverage for whisper.cpp build prompting and fallback behavior."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pocket_tts.gui.main_window import AudiobookGenerator


class _ComboStub:
    """Minimal combo-box stub for whisper.cpp prompt tests."""

    def __init__(self, *, data=None, text=None, options=None):
        """Store combo-box state and the available text choices."""
        self._data = data
        self._text = text if text is not None else data
        self._options = list(options or [])
        self.current_index_calls = []
        self.current_text_calls = []

    def currentData(self):
        """Return configured item data."""
        return self._data

    def currentText(self):
        """Return configured display text."""
        return self._text

    def findText(self, text):
        """Return the index for a visible text value when present."""
        try:
            return self._options.index(text)
        except ValueError:
            return -1

    def setCurrentIndex(self, index):
        """Record index changes and mirror the corresponding visible text."""
        self.current_index_calls.append(index)
        if 0 <= index < len(self._options):
            self._text = self._options[index]

    def setCurrentText(self, text):
        """Record text changes and update current text."""
        self.current_text_calls.append(text)
        self._text = text


class _CheckStub:
    """Minimal checkbox stub for whisper.cpp prompt tests."""

    def __init__(self, checked):
        """Store a fixed checked state."""
        self._checked = checked

    def isChecked(self):
        """Return fixed checked state."""
        return self._checked


class WhisperCppPromptTests(unittest.TestCase):
    """Verify legacy whisper.cpp prompt only appears when ASR QC is active."""

    def test_generate_gate_skips_prompt_when_asr_is_disabled(self) -> None:
        """ASR-off Generate path must not open whisper.cpp build prompt."""
        dummy = AudiobookGenerator.__new__(AudiobookGenerator)
        dummy.asr_enabled_check = _CheckStub(False)
        dummy.asr_pipeline_combo = _ComboStub(data="postgen_faster_whisper")

        self.assertFalse(AudiobookGenerator._should_prompt_cpp_gpu_build(dummy))

    def test_no_declines_build_and_falls_back_to_faster_whisper(self) -> None:
        """No on whisper.cpp prompt should switch live engine to Faster-Whisper."""
        dummy = AudiobookGenerator.__new__(AudiobookGenerator)
        dummy.results_text = []
        dummy.asr_engine_combo = _ComboStub(
            text="whisper_cpp",
            options=["whisper_cpp", "faster_whisper", "parakeet"],
        )

        with patch("pocket_tts.gui.main_window._cpp_cuda_build_available", return_value=False), patch(
            "pocket_tts.gui.main_window.QMessageBox.warning",
            return_value=0,
        ):
            result = AudiobookGenerator._confirm_cpp_gpu_if_needed(dummy)

        self.assertTrue(result)
        self.assertEqual(dummy.asr_engine_combo.currentText(), "faster_whisper")
        self.assertEqual(dummy.asr_engine_combo.current_index_calls, [1])
        self.assertIn("faster_whisper for this run", dummy.results_text[0])


if __name__ == "__main__":
    unittest.main()
