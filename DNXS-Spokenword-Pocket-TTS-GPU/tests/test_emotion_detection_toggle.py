"""Regression coverage for disabling emotion detection from the Main GUI."""

from types import SimpleNamespace
import wave

import torch

from pocket_tts.audiobook.generator import AudiobookGenerator
from pocket_tts.gui.main_window import AudiobookGenerator as MainWindow
from pocket_tts.preprocessing.emotion_analyzer import EmotionAnalyzer
from pocket_tts.preprocessing.parameter_mapper import ParameterMapper
from pocket_tts.preprocessing.schema import BoundaryType, EmotionType


class _ValueControl:
    """Provide Qt-style value and checked accessors for GUI snapshot tests."""

    def __init__(self, value):
        """Store a fixed test control value."""
        self._value = value

    def value(self):
        """Return the configured numeric control value."""
        return self._value

    def isChecked(self):
        """Return the configured checkbox state."""
        return self._value


def _mapper_config():
    """Build minimal config needed to map neutral TTS parameters."""
    return SimpleNamespace(
        speed_variation={
            "enabled": True,
            "emotion_speed_modifiers": {"sadness": -0.15, "neutral": 0.0},
        },
        mitigation={"short_sentence_threshold": 0, "temperature_short_sentence": 0.5},
    )


def test_disabled_detection_uses_neutral_metadata_and_parameters():
    """Avoid emotion-derived temperature and speed changes without model analysis."""
    emotion = EmotionAnalyzer.neutral_result()
    mapper = ParameterMapper(config=_mapper_config())

    params = mapper.calculate_params(
        emotion=emotion["emotion"],
        punctuation=".",
        boundary_type=BoundaryType.SENTENCE_END,
        word_count=12,
        emotion_scores=emotion["scores"],
    )

    assert emotion["emotion"] is EmotionType.NEUTRAL
    assert emotion["scores"] == {
        "joy": 0.0, "surprise": 0.0, "anger": 0.0, "neutral": 1.0,
        "sadness": 0.0, "fear": 0.0, "disgust": 0.0,
    }
    assert params.temperature == 0.7
    assert params.speed_factor == 1.0


def test_main_snapshot_carries_emotion_detection_switch():
    """Capture the live checkbox state for Main and immutable Batch jobs."""
    window = SimpleNamespace(
        chunking_mode_combo=SimpleNamespace(currentText=lambda: "sentence"),
        min_words_spin=_ValueControl(4),
        target_words_spin=_ValueControl(50),
        config=SimpleNamespace(chunking={"respect_boundaries": True}),
        temperature_spin=_ValueControl(0.7),
        eos_threshold_spin=_ValueControl(-3.0),
        frames_after_eos_spin=_ValueControl(2),
        lsd_steps_spin=_ValueControl(5),
        speed_variation_check=_ValueControl(True),
        emotion_detection_check=_ValueControl(False),
        pause_injection_check=_ValueControl(True),
        _pause_spinners={".": _ValueControl(0.5)},
        sentence_pause_spin=_ValueControl(400),
        paragraph_pause_spin=_ValueControl(700),
        chapter_pause_spin=_ValueControl(1000),
    )

    snapshot = MainWindow._build_runtime_settings_snapshot(window)

    assert snapshot["speed_variation"] is True
    assert snapshot["emotion_detection_enabled"] is False
    assert snapshot["pause_injection_enabled"] is True
    assert snapshot["pause_durations"] == {".": 0.5}


def test_disabled_detection_still_creates_valid_pcm_wav(tmp_path):
    """Write neutral-generation audio through production save path and verify WAV metadata."""
    output_path = tmp_path / "emotion_detection_disabled.wav"
    generator = SimpleNamespace(tts_model=SimpleNamespace(sample_rate=24000))
    audio = torch.zeros(2400, dtype=torch.float32)
    AudiobookGenerator._save_audio(generator, audio, str(output_path))

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 2400
