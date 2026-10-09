"""Regression tests for device-aware CPU ASR GUI settings."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pocket_tts.gui.main_window import AudiobookGenerator


class _Combo:
    """Provide the small subset of a combo-box used by these contract tests."""

    def __init__(self, *, data=None, text=None):
        """Store independent item data and display text values."""
        self._data = data
        self._text = text if text is not None else data

    def currentData(self):
        """Return the configured current item data."""
        return self._data

    def currentText(self):
        """Return the configured current display text."""
        return self._text


class _Spin:
    """Provide a fixed numeric GUI value for a headless contract test."""

    def __init__(self, value):
        """Store the configured numeric control value."""
        self._value = value

    def value(self):
        """Return the fixed numeric value."""
        return self._value


class _Check:
    """Provide a fixed checked state for a headless contract test."""

    def __init__(self, checked):
        """Store whether the simulated checkbox is checked."""
        self._checked = checked

    def isChecked(self):
        """Return the fixed checkbox state."""
        return self._checked


class CpuAsrGuiTests(unittest.TestCase):
    """Verify CPU TTS selection cannot retain GPU-only ASR configuration."""

    def test_apply_forces_cpu_safe_legacy_faster_whisper(self):
        """Persist CPU ASR as one base Faster-Whisper post-generation worker."""
        dummy = AudiobookGenerator.__new__(AudiobookGenerator)
        dummy.config = SimpleNamespace(asr_quality_control={})
        dummy.device_combo = _Combo(text="cpu")
        dummy.asr_enabled_check = _Check(True)
        dummy.asr_pipeline_combo = _Combo(data="postgen_parakeet")
        dummy.asr_second_stage_model_combo = _Combo(text="large-v3")
        dummy.asr_alignment_diagnostic_check = _Check(True)
        dummy.asr_engine_combo = _Combo(text="parakeet")
        dummy.asr_model_combo = _Combo(text="large-v3")
        dummy.asr_language_combo = _Combo(data="en")
        dummy.asr_threshold_spin = _Spin(0.77)
        dummy.asr_max_retries_spin = _Spin(5)
        dummy.asr_temp_decrement_spin = _Spin(0.2)
        dummy.asr_gpu_workers_spin = _Spin(8)

        AudiobookGenerator._apply_asr_gui_to_config(dummy)

        asr = dummy.config.asr_quality_control
        self.assertEqual(asr["device"], "cpu")
        self.assertEqual(asr["pipeline"], "legacy")
        self.assertEqual(asr["stage_one"], {"schedule": "postgen", "engine": "faster_whisper"})
        self.assertEqual(asr["engine"], "faster_whisper")
        self.assertEqual(asr["model"], "base")
        self.assertEqual(asr["parallel"]["cpu_workers"], 1)
