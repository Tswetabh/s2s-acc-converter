"""Regression coverage for safe comma-pause GUI defaults and warnings."""

from types import SimpleNamespace
from unittest.mock import patch

from pocket_tts.config import ConfigManager
from pocket_tts.gui.main_window import AudiobookGenerator


class _SpinStub:
    """Provide the QDoubleSpinBox value API needed by pause settings tests."""

    def __init__(self, value: float) -> None:
        """Store one deterministic spinner value."""
        self._value = float(value)

    def value(self) -> float:
        """Return the spinner's current numeric value."""
        return self._value


def _window_with_pause_spinners(values: dict[str, float]) -> AudiobookGenerator:
    """Build an uninitialized GUI instance for pure pause-setting tests."""
    window = AudiobookGenerator.__new__(AudiobookGenerator)
    window._pause_spinners = {
        punct: _SpinStub(value) for punct, value in values.items()
    }
    window._comma_pause_warning_shown = False
    return window


def test_config_defaults_comma_pause_to_zero() -> None:
    """Default and headless-free configuration must disable automatic commas."""
    config = ConfigManager.load_config(None)

    assert config.pause_injection["punctuation_durations"][","] == 0.0


def test_save_pause_durations_force_comma_to_zero() -> None:
    """Saving configuration cannot persist a live non-zero comma override."""
    window = _window_with_pause_spinners({",": 0.25, ".": 0.5})

    durations = window._pause_durations_for_config(force_comma_zero=True)

    assert durations == {",": 0.0, ".": 0.5}


def test_comma_warning_is_shown_once_until_reset() -> None:
    """Warn on the first non-zero attempt and warn again after returning to zero."""
    window = SimpleNamespace(_comma_pause_warning_shown=False)

    with patch("pocket_tts.gui.main_window.QMessageBox.warning") as warning:
        AudiobookGenerator._on_comma_pause_changed(window, 0.15)
        AudiobookGenerator._on_comma_pause_changed(window, 0.20)
        assert warning.call_count == 1

        AudiobookGenerator._on_comma_pause_changed(window, 0.0)
        AudiobookGenerator._on_comma_pause_changed(window, 0.15)
        assert warning.call_count == 2
