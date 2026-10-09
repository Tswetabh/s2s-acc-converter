"""Regression coverage for GUI ASR summary output."""

import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pocket_tts.gui.main_window import (
    AudiobookGenerator,
    _format_completion_timing_summary,
    _normalize_stage_one_pipeline_selection,
    _stage_one_pipeline_parts,
)
from pocket_tts.config import ConfigManager


class _ComboStub:
    """Minimal combo-box stub for headless GUI contract tests."""

    def __init__(self, *, data=None, text=None):
        """Initializes a spin-box stub that tracks enabled state without value mutation."""
        self._data = data
        self._text = text if text is not None else data

    def currentData(self):
        """Return the configured item data."""
        return self._data

    def currentText(self):
        """Return the configured display text."""
        return self._text


class _SpinStub:
    """Minimal spin-box stub that tracks enabled state without value mutation."""

    def __init__(self, value):
        """Initializes a combo stub with data and text, tracking enabled and value states."""
        self._value = value
        self.enabled_calls = []
        self.set_value_calls = []

    def value(self):
        """Return the stored numeric value."""
        return self._value

    def setEnabled(self, enabled):
        """Record enable/disable transitions."""
        self.enabled_calls.append(enabled)

    def setValue(self, value):
        """Record explicit value writes for mutation assertions."""
        self.set_value_calls.append(value)
        self._value = value


class _CheckStub:
    """Minimal checkbox stub for headless GUI tests."""

    def __init__(self, checked):
        """Initializes a label stub that records enabled state changes."""
        self._checked = checked

    def isChecked(self):
        """Return the fixed checked state."""
        return self._checked


class _LabelStub:
    """Minimal label stub that records enabled-state changes."""

    def __init__(self):
        """Initializes an AudiobookGenerator dummy with stub widgets for Stage 1 tests."""
        self.enabled_calls = []

    def setEnabled(self, enabled):
        """Record enable/disable transitions."""
        self.enabled_calls.append(enabled)


def _build_stage_one_gating_dummy(selection: str):
    """Create a headless GUI stub with all gating widgets for Stage 1 tests."""
    dummy = AudiobookGenerator.__new__(AudiobookGenerator)
    dummy.asr_pipeline_combo = _ComboStub(data=selection)
    dummy.asr_engine_combo = _LabelStub()
    dummy.asr_model_combo = _LabelStub()
    dummy.asr_max_retries_spin = _LabelStub()
    dummy.asr_temp_decrement_spin = _LabelStub()
    dummy.asr_alignment_diagnostic_check = _LabelStub()
    dummy.asr_second_stage_model_combo = _LabelStub()
    dummy.asr_gpu_workers_label = _LabelStub()
    dummy.asr_gpu_workers_spin = _SpinStub(7)
    dummy.results_text = []
    return dummy


class GuiAsrSummaryTests(unittest.TestCase):
    """Ensure GUI report surfaces timing, model, and regeneration counts."""

    def test_append_asr_summary_lines_includes_models_and_counts(self) -> None:
        """Render ASR summary lines from finished generation metadata."""
        dummy = AudiobookGenerator.__new__(AudiobookGenerator)
        dummy.results_text = []

        AudiobookGenerator._append_asr_summary_lines(
            dummy,
            {
                "asr_stage_one_time": 12.4,
                "asr_stage_two_time": 8.6,
                "asr_stage_one_model": "parakeet",
                "asr_stage_two_model": "medium",
                "asr_regeneration_model": "medium",
                "asr_stage_one_fails": 17,
                "asr_stage_two_fails": 6,
                "asr_remaining_fails": 2,
                "regeneration_file_count": 6,
                "regeneration_tts_workers": 2,
                "regeneration_asr_workers": 1,
                "regeneration_batch_size": 4,
                "regeneration_batch_count": 2,
                "medium_verification_summary": {
                    "attempted": 17,
                    "verified_pass": 11,
                    "verified_fail": 6,
                },
            },
        )

        self.assertEqual(
            dummy.results_text,
            [
                "ASR stage times:",
                "  Stage 1: 00:00:12",
                "  Stage 2: 00:00:09",
                "ASR models:",
                "  Stage 1: parakeet",
                "  Stage 2: medium",
                "  Regeneration: medium",
                "ASR fails:",
                "  Stage 1: 17",
                "  Stage 2: 6",
                "  Remaining after regen: 2",
                "Regeneration workload:",
                "  Files regenerated: 6",
                "  Workers: TTS 2, ASR 1",
                "  Batches: 2 (batch size 4)",
            ],
        )

    def test_completion_timing_summary_uses_requested_compact_order(self) -> None:
        """Show Stage 1, Stage 2, chunk, and end-to-end timing in one row."""
        self.assertEqual(
            _format_completion_timing_summary(
                {
                    "asr_stage_one_time": 12.4,
                    "asr_stage_two_time": 8.6,
                    "realtime_factor": 4.25,
                    "chunk_processing_time": 3661.2,
                    "total_realtime_factor": 3.5,
                    "processing_time": 3672.7,
                }
            ),
            "1-00:00:12 | 2-00:00:09 | 4.25x 01:01:01 | 3.50x 01:01:13",
        )

    def test_finished_run_retains_compact_timing_in_progress_row(self) -> None:
        """Keep completed timing visible instead of resetting the progress row."""
        dummy = SimpleNamespace(
            start_btn=Mock(),
            stop_btn=Mock(),
            play_btn=Mock(),
            generation_progress_bar=Mock(),
            chunk_progress_label=Mock(),
            time_elapsed_label=Mock(),
            eta_label=Mock(),
            completion_timing_label=Mock(),
            results_text=[],
            _append_asr_summary_lines=Mock(),
            _maybe_start_asr_worker_calibration=Mock(),
            setWindowTitle=Mock(),
        )
        result = {
            "success": True,
            "output_path": "/tmp/book.wav",
            "chunks_processed": 7,
            "total_chunks": 7,
            "asr_stage_one_time": 12,
            "asr_stage_two_time": 9,
            "realtime_factor": 4.25,
            "chunk_processing_time": 61,
            "total_realtime_factor": 3.5,
            "processing_time": 73,
        }

        with patch("pocket_tts.gui.main_window.QTimer.singleShot"):
            AudiobookGenerator.on_generation_finished(dummy, result)

        dummy.generation_progress_bar.setValue.assert_called_with(100)
        dummy.chunk_progress_label.setText.assert_called_with("Chunks: 7/7")
        dummy.time_elapsed_label.setText.assert_called_with("Elapsed: 00:01:13")
        dummy.eta_label.setText.assert_called_with("ETA: complete")
        dummy.completion_timing_label.setText.assert_called_with(
            "1-00:00:12 | 2-00:00:09 | 4.25x 00:01:01 | 3.50x 00:01:13"
        )

    def test_completion_timing_summary_treats_missing_stage_timings_as_zero(self) -> None:
        """Optional Stage 1 or Stage 2 timings should render as zero instead of crashing."""
        self.assertEqual(
            _format_completion_timing_summary(
                {
                    "asr_stage_one_time": None,
                    "asr_stage_two_time": None,
                    "realtime_factor": 4.25,
                    "chunk_processing_time": 61,
                    "total_realtime_factor": 3.5,
                    "processing_time": 73,
                }
            ),
            "1-00:00:00 | 2-00:00:00 | 4.25x 00:01:01 | 3.50x 00:01:13",
        )

    def test_stage_one_pipeline_restore_maps_legacy_and_new_fallbacks(self) -> None:
        """Restore Stage 1 selector from new nested config or old pipeline-only saves."""
        self.assertEqual(
            _normalize_stage_one_pipeline_selection(
                {"stage_one": {"schedule": "postgen", "engine": "faster_whisper"}}
            ),
            "postgen_faster_whisper",
        )
        self.assertEqual(
            _normalize_stage_one_pipeline_selection(
                {"stage_one": {"schedule": "postgen", "engine": "parakeet"}}
            ),
            "postgen_parakeet",
        )
        self.assertEqual(
            _normalize_stage_one_pipeline_selection(
                {"stage_one": {"schedule": "during_tts", "engine": "faster_whisper"}}
            ),
            "during_tts_faster_whisper",
        )
        self.assertEqual(
            _normalize_stage_one_pipeline_selection(
                {"stage_one": {"schedule": "during_tts", "engine": "parakeet"}}
            ),
            "during_tts_parakeet",
        )
        # Compatibility fallback for older saved configs with no stage_one dict.
        self.assertEqual(
            _normalize_stage_one_pipeline_selection({"pipeline": "legacy"}),
            "postgen_faster_whisper",
        )
        self.assertEqual(
            _normalize_stage_one_pipeline_selection({"pipeline": "new"}),
            "postgen_parakeet",
        )

    def test_apply_asr_gui_to_config_writes_stage_one_and_compat_pipeline(self) -> None:
        """Persist all four Stage 1 GUI choices with inert nested schedule metadata."""
        expected = {
            "postgen_faster_whisper": ("postgen", "faster_whisper", "legacy"),
            "postgen_parakeet": ("postgen", "parakeet", "new"),
            "during_tts_faster_whisper": ("during_tts", "faster_whisper", "legacy"),
            "during_tts_parakeet": ("during_tts", "parakeet", "new"),
        }
        for selection, (schedule, engine, compat_pipeline) in expected.items():
            with self.subTest(selection=selection):
                dummy = AudiobookGenerator.__new__(AudiobookGenerator)
                dummy.config = SimpleNamespace(asr_quality_control={})
                dummy.asr_enabled_check = _CheckStub(True)
                dummy.asr_pipeline_combo = _ComboStub(data=selection)
                dummy.asr_second_stage_model_combo = _ComboStub(text="small")
                dummy.asr_alignment_diagnostic_check = _CheckStub(True)
                dummy.asr_engine_combo = _ComboStub(text="whisper_cpp")
                dummy.asr_model_combo = _ComboStub(text="base")
                dummy.asr_language_combo = _ComboStub(data="en")
                dummy.asr_threshold_spin = _SpinStub(0.77)
                dummy.asr_max_retries_spin = _SpinStub(5)
                dummy.asr_temp_decrement_spin = _SpinStub(0.2)
                dummy.asr_gpu_workers_spin = _SpinStub(6)

                AudiobookGenerator._apply_asr_gui_to_config(dummy)

                asr = dummy.config.asr_quality_control
                self.assertEqual(asr["pipeline"], compat_pipeline)
                self.assertEqual(
                    asr["stage_one"],
                    {"schedule": schedule, "engine": engine},
                )
                self.assertEqual(asr["second_stage_model"], "small")
                self.assertEqual(asr["engine"], "whisper_cpp")
                self.assertEqual(asr["parallel"]["gpu_workers_after_tts"], 6)

    def test_stage_one_postgen_parakeet_gates_controls_like_new_pipeline(self) -> None:
        """Parakeet Stage 1 keeps verifier controls on and legacy controls off."""
        dummy = _build_stage_one_gating_dummy("postgen_parakeet")

        AudiobookGenerator._on_asr_pipeline_changed(dummy)

        self.assertEqual(dummy.asr_engine_combo.enabled_calls, [False])
        self.assertEqual(dummy.asr_model_combo.enabled_calls, [False])
        self.assertEqual(dummy.asr_alignment_diagnostic_check.enabled_calls, [True])
        self.assertEqual(dummy.asr_second_stage_model_combo.enabled_calls, [True])
        self.assertEqual(dummy.asr_max_retries_spin.enabled_calls, [True])
        self.assertEqual(dummy.asr_temp_decrement_spin.enabled_calls, [True])
        self.assertEqual(dummy.asr_gpu_workers_label.enabled_calls, [False])
        self.assertEqual(dummy.asr_gpu_workers_spin.enabled_calls, [False])
        self.assertEqual(dummy.asr_gpu_workers_spin.value(), 7)
        self.assertEqual(dummy.asr_gpu_workers_spin.set_value_calls, [])

    def test_stage_one_postgen_faster_whisper_gates_controls_like_legacy_pipeline(self) -> None:
        """Post-gen Faster-Whisper keeps legacy controls on and Parakeet controls off."""
        dummy = _build_stage_one_gating_dummy("postgen_faster_whisper")

        AudiobookGenerator._on_asr_pipeline_changed(dummy)

        self.assertEqual(dummy.asr_engine_combo.enabled_calls, [True])
        self.assertEqual(dummy.asr_model_combo.enabled_calls, [True])
        self.assertEqual(dummy.asr_alignment_diagnostic_check.enabled_calls, [False])
        self.assertEqual(dummy.asr_second_stage_model_combo.enabled_calls, [False])
        self.assertEqual(dummy.asr_max_retries_spin.enabled_calls, [True])
        self.assertEqual(dummy.asr_temp_decrement_spin.enabled_calls, [True])
        self.assertEqual(dummy.asr_gpu_workers_label.enabled_calls, [True])
        self.assertEqual(dummy.asr_gpu_workers_spin.enabled_calls, [True])
        self.assertEqual(dummy.asr_gpu_workers_spin.value(), 7)
        self.assertEqual(dummy.asr_gpu_workers_spin.set_value_calls, [])

    def test_stage_one_during_tts_disables_postgen_worker_controls_without_mutation(self) -> None:
        """Disable post-generation worker widgets for during-TTS selections only."""
        dummy = _build_stage_one_gating_dummy("during_tts_parakeet")

        AudiobookGenerator._on_asr_pipeline_changed(dummy)

        self.assertEqual(dummy.asr_engine_combo.enabled_calls, [False])
        self.assertEqual(dummy.asr_model_combo.enabled_calls, [False])
        self.assertEqual(dummy.asr_alignment_diagnostic_check.enabled_calls, [True])
        self.assertEqual(dummy.asr_second_stage_model_combo.enabled_calls, [True])
        self.assertEqual(dummy.asr_max_retries_spin.enabled_calls, [True])
        self.assertEqual(dummy.asr_temp_decrement_spin.enabled_calls, [True])
        self.assertEqual(dummy.asr_gpu_workers_label.enabled_calls, [False])
        self.assertEqual(dummy.asr_gpu_workers_spin.enabled_calls, [False])
        self.assertEqual(dummy.asr_gpu_workers_spin.value(), 7)
        self.assertEqual(dummy.asr_gpu_workers_spin.set_value_calls, [])
        self.assertIn("post-gen ASR still remains authoritative", dummy.results_text[-1])

        dummy.asr_pipeline_combo = _ComboStub(data="during_tts_faster_whisper")
        AudiobookGenerator._on_asr_pipeline_changed(dummy)
        self.assertEqual(dummy.asr_engine_combo.enabled_calls[-1], True)
        self.assertEqual(dummy.asr_model_combo.enabled_calls[-1], True)
        self.assertEqual(dummy.asr_alignment_diagnostic_check.enabled_calls[-1], False)
        self.assertEqual(dummy.asr_second_stage_model_combo.enabled_calls[-1], False)
        self.assertEqual(dummy.asr_gpu_workers_label.enabled_calls[-1], False)
        self.assertEqual(dummy.asr_gpu_workers_spin.enabled_calls[-1], False)
        self.assertEqual(dummy.asr_gpu_workers_spin.value(), 7)

    def test_stage_one_batch_snapshot_retains_new_nested_fields(self) -> None:
        """Carry Stage 1 contract fields into batch config snapshots."""
        dummy = AudiobookGenerator.__new__(AudiobookGenerator)
        dummy.config = ConfigManager.load_config(None)
        dummy.voice_combo = _ComboStub(text="alba (default)")
        dummy.results_text = []
        dummy.batch_tab = None
        dummy.asr_enabled_check = _CheckStub(True)
        dummy.asr_pipeline_combo = _ComboStub(data="during_tts_parakeet")
        dummy.asr_second_stage_model_combo = _ComboStub(text="medium")
        dummy.asr_alignment_diagnostic_check = _CheckStub(False)
        dummy.asr_engine_combo = _ComboStub(text="parakeet")
        dummy.asr_model_combo = _ComboStub(text="base")
        dummy.asr_language_combo = _ComboStub(data="en")
        dummy.asr_threshold_spin = _SpinStub(0.81)
        dummy.asr_max_retries_spin = _SpinStub(3)
        dummy.asr_temp_decrement_spin = _SpinStub(0.1)
        dummy.asr_gpu_workers_spin = _SpinStub(4)
        dummy.temperature_spin = _SpinStub(0.8)
        dummy.eos_threshold_spin = _SpinStub(-3.0)
        dummy.frames_after_eos_spin = _SpinStub(2)
        dummy.chunking_mode_combo = _ComboStub(text="paragraph")
        dummy.min_words_spin = _SpinStub(35)
        dummy.target_words_spin = _SpinStub(50)
        dummy.lsd_steps_spin = _SpinStub(2)
        dummy.sentence_pause_spin = _SpinStub(160)
        dummy.paragraph_pause_spin = _SpinStub(320)
        dummy.chapter_pause_spin = _SpinStub(800)
        dummy.pause_injection_check = _CheckStub(True)
        dummy._pause_spinners = {".": _SpinStub(0.5), ",": _SpinStub(0.18)}
        dummy.speed_variation_check = _CheckStub(True)
        dummy.emotion_detection_check = _CheckStub(True)
        dummy.device_combo = _ComboStub(text="cuda")
        dummy.max_workers_spin = _SpinStub(6)
        dummy.batch_size_spin = _SpinStub(2)
        dummy._m4b_gui_settings = lambda: {"write_wav": True, "enabled": True}
        dummy._build_runtime_settings_snapshot = lambda: {
            "chunking_mode": "paragraph",
            "min_words": 35,
            "target_words": 50,
            "respect_boundaries": True,
            "temperature": 0.8,
            "eos_threshold": -3.0,
            "frames_after_eos": 2,
            "lsd_steps": 2,
            "speed_variation": True,
            "emotion_detection_enabled": True,
            "pause_injection_enabled": True,
            "pause_durations": {".": 0.5, ",": 0.18},
            "sentence_pause_ms": 160,
            "paragraph_pause_ms": 320,
            "chapter_pause_ms": 800,
        }

        _, _, config_snapshot = AudiobookGenerator.build_batch_job_snapshot(dummy)

        self.assertEqual(
            config_snapshot.asr_quality_control["stage_one"],
            {"schedule": "during_tts", "engine": "parakeet"},
        )
        self.assertEqual(config_snapshot.asr_quality_control["pipeline"], "new")
        self.assertEqual(config_snapshot.batch_generation["batch_size"], 2)
        self.assertEqual(
            _stage_one_pipeline_parts("during_tts_parakeet"),
            {
                "selection": "during_tts_parakeet",
                "schedule": "during_tts",
                "engine": "parakeet",
                "compat_pipeline": "new",
            },
        )

    def test_live_batch_size_updates_generation_configuration(self) -> None:
        """Use the unsaved Performance spinner value for the next generation."""
        dummy = AudiobookGenerator.__new__(AudiobookGenerator)
        dummy.config = ConfigManager.load_config(None)
        dummy.batch_size_spin = _SpinStub(6)

        AudiobookGenerator._apply_batch_size_gui_to_config(dummy)

        self.assertEqual(dummy.config.batch_generation["batch_size"], 6)
