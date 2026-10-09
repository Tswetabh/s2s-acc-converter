"""
Audiobook Generator GUI Application
Main Qt-based interface for text-to-speech audiobook generation.
"""

import sys
import os
import json
import logging
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QComboBox,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QTextEdit,
    QCheckBox, QMessageBox, QSizePolicy, QTabWidget, QDialog, QLineEdit,
    QDialogButtonBox, QTextBrowser,
)
from qtpy.QtCore import Qt, QThread, Signal, QTimer
from qtpy.QtGui import QPalette, QColor, QPixmap

from pocket_tts.preprocessing.structure_detector import StructureDetector
from pocket_tts.preprocessing.chunker import SmartChunker
from pocket_tts.preprocessing.emotion_analyzer import EmotionAnalyzer
from pocket_tts.preprocessing.parameter_mapper import ParameterMapper
from pocket_tts.preprocessing.schema import BoundaryType
from pocket_tts.audiobook.generator import AudiobookGenerator as AudiobookGeneratorEngine
from pocket_tts.config import ConfigManager
from pocket_tts.asr_worker_calibration import (
    is_calibration_done,
    run_first_calibration,
)
from pocket_tts.gui.parameter_help import help_for
from pocket_tts.audio.metadata import COVER_ART_FILTER, DEFAULT_ARTIST
from pocket_tts.regeneration_scheduler import (
    STAGE_TWO_DISABLED,
    model_rank,
    recommended_stage_two_model,
)

# ASR backend / model choices (aligned with standalone ASR/asr_gui.py)
ASR_BACKENDS = ["faster_whisper", "whisper_cpp", "parakeet"]
CPU_ASR_MODEL = "base"
CPU_ASR_WORKERS = 1
ASR_MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v3",
]
STAGE_TWO_DISABLED_LABEL = "Disabled"
SECOND_STAGE_ASR_MODELS = (STAGE_TWO_DISABLED_LABEL,) + tuple(ASR_MODELS)

COMMA_PAUSE_DEFAULT_SECONDS = 0.0
COMMA_PAUSE_WARNING_TITLE = "Comma Pause Warning"
COMMA_PAUSE_WARNING_TEXT = (
    "Non-zero automatic comma pauses are known to cause TTS hallucinations. "
    "They may cause excessive regenerations when ASR is active and may require "
    "custom regeneration using the Regeneration tab. The non-zero value will "
    "apply only to the current run and will not be saved."
)


def _stage_two_config_value(label: str | None) -> str:
    """Map the Stage 2 combo label to the stored config token."""
    text = str(label or "").strip()
    if text.lower() in {"", "disabled", "none", "off"}:
        return STAGE_TWO_DISABLED
    return text


def _stage_two_combo_label(value: str | None) -> str:
    """Map a stored Stage 2 token to the combo display label."""
    if str(value or "").strip().lower() in {"", "disabled", "none", "off"}:
        return STAGE_TWO_DISABLED_LABEL
    return str(value or "medium").strip()


def _format_duration_hms(seconds: float | int | None) -> str:
    """Format optional integer or fractional seconds as fixed-width ``HH:MM:SS`` text.

    Completion reports can omit optional ASR stage timings, especially when a
    later stage reuses precomputed evidence. Missing or malformed optional
    values should render as zero here rather than crashing the GUI callback.
    """
    try:
        total_seconds = max(0, int(round(float(seconds or 0))))
    except (TypeError, ValueError):
        total_seconds = 0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def _format_completion_timing_summary(result: Dict[str, Any]) -> str:
    """Build the compact completed-run timing string shown at progress-row right."""
    stage_one = _format_duration_hms(result.get("asr_stage_one_time", 0))
    stage_two = _format_duration_hms(result.get("asr_stage_two_time", 0))
    chunk_time = _format_duration_hms(result.get("chunk_processing_time", 0))
    total_time = _format_duration_hms(result.get("processing_time", 0))
    chunk_speed = float(result.get("realtime_factor", 0) or 0)
    total_speed = float(result.get("total_realtime_factor", 0) or 0)
    return (
        f"1-{stage_one} | 2-{stage_two} | "
        f"{chunk_speed:.2f}x {chunk_time} | {total_speed:.2f}x {total_time}"
    )


def _cpp_cuda_build_available() -> bool:
    """True if pywhispercpp was built with libggml-cuda (GPU)."""
    try:
        asr_dir = Path(__file__).resolve().parents[2] / "ASR"
        if str(asr_dir) not in sys.path:
            sys.path.insert(0, str(asr_dir))
        from cpp_cuda_setup import cpp_cuda_build_available

        return bool(cpp_cuda_build_available(sys.executable))
    except Exception:
        return False


class AudiobookGenerator(QMainWindow):
    """Main GUI application for audiobook generation."""

    # Signals for thread communication
    generation_progress = Signal(dict)     # progress updates
    generation_finished = Signal(str)      # output file path
    asr_calibration_finished = Signal(dict)  # first-run worker calibration result

    # GUI settings file location
    SETTINGS_FILE = Path.home() / ".pocket_tts_gui_config.json"
    CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"

    def __init__(self):
        """Initializes the PocketTTS application. Sets up logging, configuration, and UI components. Loads GUI settings to determine last used directories for text and voice files. Initializes user interface and sets up signal connections. Returns None."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = ConfigManager.load_config('pocket_tts/config/default_config.yaml')
        self.current_structure = None
        self.current_chunks = None
        self.last_output_path = None
        self._asr_calibrating = False
        self._comma_pause_warning_shown = False

        # Load saved directory settings or use defaults
        settings = self._load_gui_settings()
        self.last_text_dir = settings.get('last_text_dir', str(Path.home()))
        self.last_voice_dir = settings.get('last_voice_dir', str(Path.home()))
        self.last_custom_voice = settings.get('last_custom_voice', None)

        self.init_ui()
        self.setup_connections()
        self.asr_calibration_finished.connect(self._on_asr_calibration_finished)
        # First-run GPU ASR worker calibration (once; gated by config flag)
        QTimer.singleShot(800, self._maybe_start_asr_worker_calibration)

    def _load_gui_settings(self) -> Dict[str, Any]:
        """Load GUI settings from persistent storage."""
        try:
            if self.SETTINGS_FILE.exists():
                with open(self.SETTINGS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load GUI settings: {e}")
        return {}

    def _save_gui_settings(self):
        """Save GUI settings to persistent storage."""
        try:
            settings = {
                'last_text_dir': self.last_text_dir,
                'last_voice_dir': self.last_voice_dir,
                'last_custom_voice': self.last_custom_voice,
            }
            with open(self.SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save GUI settings: {e}")

    def _save_current_config(self):
        """Save current parameter values from GUI to default_config.yaml."""
        try:
            pause_durations = self._pause_durations_for_config(force_comma_zero=True)
            self.config.tts_core.update({
                'temperature': self.temperature_spin.value(),
                'eos_threshold': self.eos_threshold_spin.value(),
                'frames_after_eos': self.frames_after_eos_spin.value(),
            })
            self.config.device['preferred'] = self.device_combo.currentText()
            self.config.chunking.update({
                'mode': self.chunking_mode_combo.currentText(),
                'min_words': self.min_words_spin.value(),
                'target_words': self.target_words_spin.value(),
            })
            self.config.quality['lsd_steps'] = self.lsd_steps_spin.value()
            self.config.pauses['base_durations'].update({
                'sentence_end': self.sentence_pause_spin.value(),
                'paragraph_break': self.paragraph_pause_spin.value(),
                'chapter_start': self.chapter_pause_spin.value(),
            })
            self.config.pause_injection.update({
                'enabled': self.pause_injection_check.isChecked(),
                'punctuation_durations': pause_durations,
            })
            self.config.speed_variation['enabled'] = self.speed_variation_check.isChecked()
            self.config.emotion['enabled'] = self.emotion_detection_check.isChecked()
            self.config.m4b.update(self._m4b_gui_settings())
            self.config.parallel['max_workers'] = self.max_workers_spin.value()
            self._apply_batch_size_gui_to_config()
            # Persist ASR + GUI worker count (user Save only — not auto-saved each run)
            self._apply_asr_gui_to_config(persist_workers=True)
            config_path = self.CONFIG_PATH
            self.config.save(str(config_path))
            self.results_text.append(f"✅ Configuration saved to {config_path.name}")
            comma_spin = self._pause_spinners[","]
            if comma_spin.value() > COMMA_PAUSE_DEFAULT_SECONDS:
                self.results_text.append(
                    "ℹ️ Comma pause remains 0.00 s in saved configuration; "
                    "the non-zero value is live-only."
                )
        except Exception as e:
            self.results_text.append(f"❌ Failed to save configuration: {e}")

    def _m4b_gui_settings(self) -> dict:
        """Return current audiobook export checkboxes/spinners for config."""
        write_m4b = bool(self.m4b_write_m4b_check.isChecked())
        write_mp3 = bool(self.m4b_write_mp3_check.isChecked())
        write_wav = bool(self.m4b_write_wav_check.isChecked())
        return {
            "write_m4b": write_m4b,
            "write_mp3": write_mp3,
            "write_wav": write_wav,
            # Legacy master flag: any export format selected.
            "enabled": write_m4b or write_mp3 or write_wav,
            "normalization_type": self.m4b_norm_combo.currentText(),
            "chapterize": self.m4b_chapterize_check.isChecked(),
            "chapter_mode": str(self.m4b_chapter_mode_combo.currentData()),
            "max_chapter_minutes": int(self.m4b_chapter_minutes_spin.value()),
            "metadata": self._metadata_dialog_values(),
        }

    def _update_chapter_export_controls(self, _value: Any = None) -> None:
        """Explain and enable chapter controls for the active export strategy.

        Qt sends either a bool, integer, or spin-box value to this shared slot;
        controls remain runtime truth, so the parameter is deliberately ignored.
        """
        chapterize = self.m4b_chapterize_check.isChecked()
        mode = str(self.m4b_chapter_mode_combo.currentData() or "headings_only")
        minutes = int(self.m4b_chapter_minutes_spin.value())
        self.m4b_chapter_mode_combo.setEnabled(chapterize)
        self.m4b_chapter_mode_note.setEnabled(chapterize)

        if not chapterize:
            self.m4b_chapter_minutes_label.setText("Minutes:")
            self.m4b_chapter_minutes_spin.setEnabled(False)
            self.m4b_chapter_minutes_spin.setSpecialValueText("Not used")
            self.m4b_chapter_mode_note.setText(
                "Chapter TOC is off. M4B and MP3 export as one complete book."
            )
        elif mode == "headings_or_minutes":
            self.m4b_chapter_minutes_label.setText("Fallback chapter (min):")
            self.m4b_chapter_minutes_spin.setEnabled(True)
            self.m4b_chapter_minutes_spin.setSpecialValueText("Set minutes")
            suffix = (
                " Set a minute value to enable fallback splitting."
                if minutes == 0
                else ""
            )
            self.m4b_chapter_mode_note.setText(
                "Uses detected Part/Chapter headings unchanged. If no heading is "
                "found, splits the book near paragraph or sentence ends using this "
                "minute length." + suffix
            )
        elif mode == "headings_with_max":
            self.m4b_chapter_minutes_label.setText("Maximum chapter (min):")
            self.m4b_chapter_minutes_spin.setEnabled(True)
            self.m4b_chapter_minutes_spin.setSpecialValueText("No maximum")
            self.m4b_chapter_mode_note.setText(
                "Uses detected headings, then splits any chapter longer than this "
                "minute length near paragraph or sentence ends."
            )
        else:
            self.m4b_chapter_minutes_label.setText("Minutes:")
            self.m4b_chapter_minutes_spin.setEnabled(False)
            self.m4b_chapter_minutes_spin.setSpecialValueText("Not used")
            self.m4b_chapter_mode_note.setText(
                "Uses only detected Part/Chapter headings. If none are found, the "
                "whole book remains one chapter."
            )

    def _metadata_dialog_values(self) -> dict:
        """Return configured Metadata-dialog fields with the required artist default."""
        raw = dict((getattr(self.config, "m4b", {}) or {}).get("metadata") or {})
        values = {
            key: str(raw.get(key, "") or "")
            for key in (
                "author",
                "series",
                "series_number",
                "title",
                "composer",
                "year",
                "genre",
            "artist",
            "description",
            "cover_path",
        )
        }
        values["artist"] = values["artist"] or DEFAULT_ARTIST
        return values

    def _metadata_cover_start_dir(self) -> str:
        """Return selected book-text folder as the cover picker starting directory."""
        selected_text = self.text_file_path.text()
        if selected_text and selected_text != "No file selected":
            path = Path(selected_text)
            if path.is_file():
                return str(path.parent)
        return str(getattr(self, "last_text_dir", Path.home()))

    @staticmethod
    def _update_metadata_cover_thumbnail(image_path: str, thumbnail: QLabel) -> None:
        """Display selected cover art as a bounded thumbnail or an empty placeholder."""
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            thumbnail.clear()
            thumbnail.setText("No cover selected")
            return
        thumbnail.setText("")
        thumbnail.setPixmap(
            pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def open_metadata_dialog(self, checked: bool = False) -> None:
        """Edit export metadata after accepting QPushButton's ignored checked signal."""
        _ = checked  # QPushButton.clicked always supplies a checked boolean.
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Meta Data")
        dialog.setMinimumWidth(620)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        values = self._metadata_dialog_values()
        widgets = {}
        fields = (
            ("author", "Author:"),
            ("series", "Series:"),
            ("series_number", "Series #:"),
            ("title", "Title:"),
            ("composer", "Composer:"),
            ("year", "Year:"),
            ("genre", "Genre:"),
            ("artist", "Artist:"),
        )
        for key, label in fields:
            field = QLineEdit(values[key])
            widgets[key] = field
            form.addRow(label, field)

        description = QTextEdit(values["description"])
        description.setFixedHeight(90)
        widgets["description"] = description
        form.addRow("Description:", description)

        cover_field = QLineEdit(values["cover_path"])
        cover_field.setReadOnly(True)
        cover_button = QPushButton("Browse Image...")
        cover_thumbnail = QLabel()
        cover_thumbnail.setFixedSize(150, 150)
        cover_thumbnail.setAlignment(Qt.AlignCenter)
        cover_thumbnail.setStyleSheet("border: 1px solid #777; color: #BBBBBB;")
        self._update_metadata_cover_thumbnail(values["cover_path"], cover_thumbnail)

        def choose_cover_image(checked: bool = False) -> None:
            """Pick cover art after accepting QPushButton's ignored checked signal."""
            _ = checked  # Browse Image uses QPushButton.clicked(bool).
            from .qt_file_dialogs import get_open_file_name

            image_path, _ = get_open_file_name(
                dialog,
                "Select Cover Image",
                self._metadata_cover_start_dir(),
                COVER_ART_FILTER,
            )
            if image_path:
                cover_field.setText(image_path)
                self._update_metadata_cover_thumbnail(image_path, cover_thumbnail)

        cover_button.clicked.connect(choose_cover_image)
        cover_layout = QVBoxLayout()
        cover_picker_layout = QHBoxLayout()
        cover_picker_layout.addWidget(cover_field)
        cover_picker_layout.addWidget(cover_button)
        cover_layout.addLayout(cover_picker_layout)
        cover_layout.addWidget(cover_thumbnail, alignment=Qt.AlignLeft)
        form.addRow("Cover image:", cover_layout)
        layout.addLayout(form)

        note = QLabel(
            "JPEG and PNG cover art embeds in M4B and MP3. WAV receives text "
            "metadata only; it cannot carry embedded cover art."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #BBBBBB;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return

        metadata = {
            key: (
                widget.toPlainText().strip()
                if isinstance(widget, QTextEdit)
                else widget.text().strip()
            )
            for key, widget in widgets.items()
        }
        metadata["cover_path"] = cover_field.text().strip()
        if not hasattr(self.config, "m4b") or self.config.m4b is None:
            self.config.m4b = {}
        self.config.m4b["metadata"] = metadata
        self.results_text.append(
            "✓ Export metadata updated for this run. Save Configuration to persist it."
        )

    def _apply_batch_size_gui_to_config(self) -> None:
        """Copy the live TTS batch-size spinner into the next-run configuration."""
        batch_generation = getattr(self.config, "batch_generation", None)
        if not isinstance(batch_generation, dict):
            batch_generation = {}
            self.config.batch_generation = batch_generation
        batch_generation["batch_size"] = int(self.batch_size_spin.value())

    def _apply_asr_gui_to_config(self, persist_workers: bool = False) -> None:
        """Copy current ASR GUI controls into self.config for a generation run.

        Args:
            persist_workers: If True, also store GPU worker spin into config
                (used when user clicks Save Configuration). Always applied for
                the in-memory config used by the current run.
        """
        if not hasattr(self.config, "asr_quality_control") or self.config.asr_quality_control is None:
            self.config.asr_quality_control = {}
        asr = self.config.asr_quality_control
        asr["enabled"] = self.asr_enabled_check.isChecked()
        asr_device = self._effective_asr_device()
        engine = "faster_whisper"
        if hasattr(self, "asr_engine_combo"):
            engine = (self.asr_engine_combo.currentText() or "faster_whisper").strip()
        if engine in ("whisper_cpp", "whisper-cpp", "cpp"):
            engine = "whisper_cpp"
        elif engine in ("parakeet", "parakeet_tdt", "nemo_parakeet"):
            engine = "parakeet"
        else:
            engine = "faster_whisper"
        if asr_device == "cpu":
            engine = "faster_whisper"
        # Single two-stage flow: always the legacy post-gen runner.
        asr["pipeline"] = "legacy"
        asr["stage_one"] = {
            "schedule": "postgen",
            "engine": engine,
        }
        second_stage_model = "medium"
        if hasattr(self, "asr_second_stage_model_combo"):
            second_stage_model = _stage_two_config_value(
                self.asr_second_stage_model_combo.currentText()
            )
        asr["second_stage_model"] = second_stage_model
        asr["alignment_diagnostic_enabled"] = False
        asr["device"] = asr_device
        asr["engine"] = engine
        asr["model"] = (
            CPU_ASR_MODEL
            if asr_device == "cpu"
            else self.asr_model_combo.currentText()
        )
        asr["language"] = self.asr_language_combo.currentData() or "en"
        asr["threshold"] = self.asr_threshold_spin.value()
        asr["max_retries"] = self.asr_max_retries_spin.value()
        asr["temp_decrement"] = self.asr_temp_decrement_spin.value()
        parallel = asr.get("parallel")
        if not isinstance(parallel, dict):
            parallel = {}
            asr["parallel"] = parallel
        # CPU mode uses one int8 model. More copies exhaust RAM and make a
        # no-GPU system slower rather than increasing decode throughput.
        worker_count = (
            CPU_ASR_WORKERS
            if asr_device == "cpu"
            else int(self.asr_gpu_workers_spin.value())
        )
        parallel["gpu_workers_after_tts"] = worker_count
        parallel["cpu_workers"] = worker_count
        _ = persist_workers  # flag reserved: run always uses GUI; save uses same path

    def _format_asr_start_status(self) -> str:
        """Return the Generate-start ASR status line, or empty when ASR is off."""
        asr = getattr(self.config, "asr_quality_control", {}) or {}
        if not asr.get("enabled"):
            return ""
        return (
            f"ASR: stage1={asr.get('engine')} model={asr.get('model')} "
            f"stage2={asr.get('second_stage_model')} "
            f"threshold={asr.get('threshold')} "
            f"gpu_workers_after_gen={((asr.get('parallel') or {}).get('gpu_workers_after_tts'))}"
        )

    def _should_prompt_cpp_gpu_build(self) -> bool:
        """Return True when Generate should gate whisper.cpp behind a build prompt."""
        if not hasattr(self, "asr_enabled_check") or not self.asr_enabled_check.isChecked():
            return False
        if not hasattr(self, "asr_engine_combo"):
            return False
        engine = (self.asr_engine_combo.currentText() or "").strip()
        return engine in ("whisper_cpp", "whisper-cpp", "cpp")

    def _effective_asr_device(self) -> str:
        """Return the safe ASR device implied by TTS selection and CUDA availability."""
        # A live window always has a device combo. CUDA preserves existing
        # behavior for minimal headless callers that do not create one.
        selected = "cuda"
        instance_state = vars(self)
        device_combo = instance_state.get("device_combo")
        if device_combo is not None:
            selected = str(device_combo.currentText() or "auto").strip().lower()
        else:
            config = instance_state.get("config")
            if isinstance(getattr(config, "device", None), dict):
                selected = str(config.device.get("preferred", "auto") or "auto").strip().lower()
        if selected == "cpu":
            return "cpu"
        if selected == "cuda":
            return "cuda"
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _on_tts_device_changed(self, _device: str = "") -> None:
        """Synchronize ASR choices whenever the user changes the TTS device."""
        self._sync_asr_controls_to_device()

    def _sync_asr_controls_to_device(self) -> None:
        """Lock ASR to the one supported no-GPU configuration when CPU is active."""
        if not hasattr(self, "asr_engine_combo"):
            return
        cpu_mode = self._effective_asr_device() == "cpu"
        if cpu_mode:
            engine_index = self.asr_engine_combo.findText("faster_whisper")
            if engine_index >= 0:
                self.asr_engine_combo.setCurrentIndex(engine_index)
            model_index = self.asr_model_combo.findText(CPU_ASR_MODEL)
            if model_index >= 0:
                self.asr_model_combo.setCurrentIndex(model_index)
            self.asr_gpu_workers_spin.setValue(CPU_ASR_WORKERS)
            self.asr_gpu_workers_label.setText("CPU ASR workers:")
            cpu_tip = (
                "CPU ASR uses one Faster-Whisper base model in int8 after TTS. "
                "This avoids duplicate model copies and works without CUDA."
            )
            self.asr_gpu_workers_label.setToolTip(cpu_tip)
            self.asr_gpu_workers_spin.setToolTip(cpu_tip)
            for widget in (
                self.asr_engine_combo,
                self.asr_model_combo,
                self.asr_gpu_workers_spin,
            ):
                widget.setEnabled(False)
            if hasattr(self, "asr_second_stage_model_combo"):
                self.asr_second_stage_model_combo.setEnabled(True)
            return

        self.asr_engine_combo.setEnabled(True)
        self.asr_gpu_workers_label.setText("GPU ASR workers (after gen):")
        gpu_tip = (
            "After TTS finishes (0=off). No ASR during gen. "
            "Faster-Whisper uses this as its GPU pipeline helper-worker budget."
        )
        self.asr_gpu_workers_label.setToolTip(gpu_tip)
        self.asr_gpu_workers_spin.setToolTip(gpu_tip)
        self._on_asr_engine_changed()

    def _confirm_cpp_gpu_if_needed(self) -> bool:
        """If whisper_cpp needs a CUDA build, ask to build it or fall back safely.

        Yes runs ``ASR/install_pywhispercpp_cuda.sh`` with progress in the terminal.
        No switches the live ASR engine to ``faster_whisper`` for this run only.

        Returns:
            True if generation should continue; False only when build fails.
        """
        if not hasattr(self, "asr_engine_combo"):
            return True
        engine = (self.asr_engine_combo.currentText() or "").strip()
        if engine not in ("whisper_cpp", "whisper-cpp", "cpp"):
            return True
        # Post-gen ASR runs on CUDA; need CUDA pywhispercpp
        if _cpp_cuda_build_available():
            return True

        reply = QMessageBox.warning(
            self,
            "Build whisper.cpp GPU?",
            (
                "whisper.cpp GPU support is not installed yet.\n\n"
                "YES = build it from source now (GGML_CUDA).\n"
                "  • Often 5–15+ minutes\n"
                "  • Needs CUDA toolkit (nvcc) and may fail\n"
                "  • Progress appears in this window's log\n\n"
                "NO = cancel. Use backend faster_whisper for GPU ASR instead "
                "(recommended; works without a source build).\n\n"
                "Build whisper.cpp GPU support now?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            self.results_text.append(
                "↩ whisper_cpp GPU declined; using faster_whisper for this run."
            )
            engine_index = self.asr_engine_combo.findText("faster_whisper")
            if engine_index >= 0:
                self.asr_engine_combo.setCurrentIndex(engine_index)
            else:
                self.asr_engine_combo.setCurrentText("faster_whisper")
            return True

        # Hide parameters so Results terminal has room for build log (user's manual habit)
        self._expand_results_terminal_for_progress()

        self.results_text.append(
            "🔧 Building CUDA pywhispercpp (source) — do not close the app…"
        )
        self.results_text.append("   This can take 5–15+ minutes. Progress lines follow.")
        self.setWindowTitle("Audiobook Generator - Building whisper.cpp CUDA…")
        self.generation_progress_bar.setRange(0, 0)
        QApplication.processEvents()

        try:
            asr_dir = Path(__file__).resolve().parents[2] / "ASR"
            if str(asr_dir) not in sys.path:
                sys.path.insert(0, str(asr_dir))
            from cpp_cuda_setup import build_cpp_cuda

            def _line(msg: str) -> None:
                """Append build log line and keep UI responsive."""
                self.results_text.append(f"   {msg}")
                QApplication.processEvents()

            ok, msg = build_cpp_cuda(
                progress_cb=_line,
                python_exe=sys.executable,
            )
            self.generation_progress_bar.setRange(0, 100)
            self.setWindowTitle("Audiobook Generator")
            if not ok:
                QMessageBox.critical(
                    self,
                    "whisper.cpp GPU build failed",
                    f"{msg}\n\nUse ASR Backend = faster_whisper for GPU ASR.",
                )
                self.results_text.append(f"❌ CUDA build failed: {msg}")
                return False
            self.results_text.append(f"✅ {msg}")
            return True
        except Exception as exc:
            self.generation_progress_bar.setRange(0, 100)
            self.setWindowTitle("Audiobook Generator")
            QMessageBox.critical(
                self,
                "whisper.cpp GPU build failed",
                f"{exc}\n\nUse ASR Backend = faster_whisper for GPU ASR.",
            )
            self.results_text.append(f"❌ CUDA build error: {exc}")
            return False

    def _set_stage_two_combo(self, model_name: str) -> None:
        """Select a Stage 2 combo item without re-entering the auto-set handler."""
        combo = getattr(self, "asr_second_stage_model_combo", None)
        if combo is None:
            return
        label = _stage_two_combo_label(model_name)
        index = combo.findText(label)
        blocker = getattr(combo, "blockSignals", None)
        if callable(blocker):
            blocker(True)
        try:
            if index >= 0:
                combo.setCurrentIndex(index)
            elif hasattr(combo, "setCurrentText"):
                combo.setCurrentText(label)
        finally:
            if callable(blocker):
                blocker(False)

    def _on_asr_engine_changed(self, _index: int = 0) -> None:
        """Enable Stage 1 model for Whisper engines and refresh Stage 2 default."""
        engine = self.asr_engine_combo.currentText() if hasattr(self, "asr_engine_combo") else ""
        uses_parakeet = engine in ("parakeet", "parakeet_tdt", "nemo_parakeet")
        model_combo = getattr(self, "asr_model_combo", None)
        if model_combo is not None and self._effective_asr_device() != "cpu":
            model_combo.setEnabled(not uses_parakeet)
        second_stage_combo = getattr(self, "asr_second_stage_model_combo", None)
        if second_stage_combo is not None:
            second_stage_combo.setEnabled(True)
        gpu_workers_spin = getattr(self, "asr_gpu_workers_spin", None)
        if gpu_workers_spin is not None and self._effective_asr_device() != "cpu":
            gpu_workers_spin.setEnabled(True)
        gpu_workers_label = getattr(self, "asr_gpu_workers_label", None)
        if gpu_workers_label is not None:
            gpu_workers_label.setEnabled(True)
        if uses_parakeet:
            self._set_stage_two_combo("medium")
            return
        if engine in ("whisper_cpp", "whisper-cpp", "cpp"):
            if not _cpp_cuda_build_available() and hasattr(self, "results_text"):
                self.results_text.append(
                    "ℹ whisper_cpp selected: CUDA build not installed yet. "
                    "Generate will ask to BUILD it now (Yes) or cancel (No)."
                )
            model = self.asr_model_combo.currentText() if model_combo is not None else ""
            if str(model).startswith("distil") and hasattr(self, "results_text"):
                self.results_text.append(
                    f"ℹ whisper_cpp has no true distil weights; '{model}' maps to nearest ggml size."
                )

    def _on_stage_one_model_changed(self, _index: int = 0) -> None:
        """Auto-set Stage 2 from the locked ladder, with popups when Stage 1 is medium+."""
        if not hasattr(self, "asr_model_combo") or not hasattr(self, "asr_second_stage_model_combo"):
            return
        engine = ""
        if hasattr(self, "asr_engine_combo"):
            engine = (self.asr_engine_combo.currentText() or "").strip()
        if engine in ("parakeet", "parakeet_tdt", "nemo_parakeet"):
            self._set_stage_two_combo("medium")
            return
        stage_one = (self.asr_model_combo.currentText() or "base").strip()
        recommended = recommended_stage_two_model(stage_one, engine=engine)
        if recommended == STAGE_TWO_DISABLED:
            QMessageBox.information(
                self,
                "Stage 2 model",
                "no larger model — Stage 2 Disabled",
            )
        elif model_rank(stage_one) >= model_rank("medium"):
            QMessageBox.information(
                self,
                "Stage 2 model",
                f"Stage 2 must be higher — setting to {recommended}",
            )
        self._set_stage_two_combo(recommended)

    def _maybe_start_asr_worker_calibration(self) -> None:
        """Kick off one-shot first-run ASR worker calibration if not done yet."""
        asr = getattr(self.config, "asr_quality_control", None) or {}
        if is_calibration_done(asr):
            return
        if self._asr_calibrating:
            return
        self._asr_calibrating = True
        self.results_text.append(
            "🔧 First-run ASR worker calibration starting "
            "(once only; sets GPU ASR workers default)..."
        )

        def _worker() -> None:
            """Background thread body for first-run ASR worker calibration."""
            model = "base"
            try:
                if hasattr(self, "asr_model_combo"):
                    model = self.asr_model_combo.currentText() or "base"
            except Exception:
                model = "base"
            result = run_first_calibration(
                config_path=self.CONFIG_PATH,
                model=model,
            )
            self.asr_calibration_finished.emit(result)

        import threading

        threading.Thread(target=_worker, name="asr-worker-cal", daemon=True).start()

    def _on_asr_calibration_finished(self, result: Dict[str, Any]) -> None:
        """Apply first-run calibration result to GUI and in-memory config.

        Args:
            result: Dict from run_first_calibration (ok / deferred / recommended).
        """
        self._asr_calibrating = False
        if result.get("deferred"):
            self.results_text.append(
                f"ℹ️ ASR calibration deferred: {result.get('message', 'no sample audio yet')}"
            )
            return
        if not result.get("ok"):
            self.results_text.append(
                f"⚠️ ASR calibration failed: {result.get('message', 'unknown error')}"
            )
            return
        rec = int(result.get("recommended") or 4)
        # Seed GUI default from calibration; user can still change for each run
        if hasattr(self, "asr_gpu_workers_spin"):
            self.asr_gpu_workers_spin.setValue(rec)
        asr = getattr(self.config, "asr_quality_control", None)
        if not isinstance(asr, dict):
            self.config.asr_quality_control = {}
            asr = self.config.asr_quality_control
        parallel = asr.get("parallel")
        if not isinstance(parallel, dict):
            parallel = {}
            asr["parallel"] = parallel
        parallel["gpu_workers_after_tts"] = rec
        asr["worker_calibration"] = {
            "done": True,
            "recommended": rec,
            "last_run": result.get("message"),
        }
        self.results_text.append(
            f"✅ ASR worker calibration done: GPU workers after gen = {rec} "
            f"(GUI can override per run without saving)"
        )

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Audiobook Generator")
        # Default size only if maximize fails; do not pin geometry (fights showMaximized)
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Create Generate tab (existing UI)
        self.create_generate_tab()

        # Create Regenerate tab (new UI)
        self.create_regenerate_tab()

        # Create Batch Processing tab
        self.create_batch_tab()

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_generate_tab(self):
        """Create the Generate Audiobook tab (existing UI)."""
        generate_widget = QWidget()
        self.tab_widget.addTab(generate_widget, "Generate Audiobook")

        # Main layout for generate tab
        layout = QVBoxLayout(generate_widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Create header with file selectors and image
        self.create_header_section(layout)

        # Create sections
        self.create_parameters_section(layout)
        self.create_progress_section(layout)
        self.create_results_section(layout)

    def create_regenerate_tab(self):
        """Create the Regenerate Chunks tab (new UI)."""
        from .regenerate_tab import RegenerateTab
        regenerate_tab = RegenerateTab(config=self.config)
        self.tab_widget.addTab(regenerate_tab, "Regenerate Chunks")

    def create_batch_tab(self):
        """Create the Batch Processing tab."""
        from .batch_tab import BatchTab
        self.batch_tab = BatchTab(config=self.config, main_window=self)
        self.tab_widget.addTab(self.batch_tab, "Batch Processing")

    def create_header_section(self, parent_layout):
        """Create header section with file selectors and image."""
        # Main horizontal layout for header
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        # Left side: File selection group plus inline controls row.
        left_column = QWidget()
        left_column_layout = QVBoxLayout(left_column)
        left_column_layout.setContentsMargins(0, 0, 0, 0)
        left_column_layout.setSpacing(4)

        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        file_layout.setContentsMargins(4, 4, 4, 4)
        file_layout.setSpacing(4)

        # Text file selection - label on left, selector and button on right
        text_file_layout = QHBoxLayout()
        text_file_layout.setContentsMargins(0, 0, 0, 0)
        text_file_layout.setSpacing(4)
        text_label = QLabel("Text File:")
        text_label.setFixedWidth(60)  # Fixed width for label alignment
        text_file_layout.addWidget(text_label)
        self.text_file_path = QLabel("No file selected")
        self.text_file_path.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
        text_file_layout.addWidget(self.text_file_path)
        self.browse_text_btn = QPushButton("Browse...")
        self.browse_text_btn.clicked.connect(self.browse_text_file)
        text_file_layout.addWidget(self.browse_text_btn)

        # Voice selection - label on left, selector and button on right
        voice_file_layout = QHBoxLayout()
        voice_file_layout.setContentsMargins(0, 0, 0, 0)
        voice_file_layout.setSpacing(4)
        voice_label = QLabel("Voice:")
        voice_label.setFixedWidth(60)  # Fixed width for label alignment
        voice_file_layout.addWidget(voice_label)
        self.voice_combo = QComboBox()
        self.voice_combo.addItem("alba (default)")
        self.voice_combo.addItem("marius")
        self.voice_combo.addItem("javert")
        self.voice_combo.addItem("jean")
        self.voice_combo.addItem("fantine")
        self.voice_combo.addItem("cosette")
        self.voice_combo.addItem("eponine")
        self.voice_combo.addItem("azelma")
        self.voice_combo.addItem("Custom WAV...")
        # Restore last custom voice if it still exists
        if self.last_custom_voice and os.path.exists(self.last_custom_voice):
            voice_name = f"Custom: {Path(self.last_custom_voice).name}"
            self.voice_combo.addItem(voice_name, self.last_custom_voice)
            self.voice_combo.setCurrentText(voice_name)
        voice_file_layout.addWidget(self.voice_combo)
        self.browse_voice_btn = QPushButton("Browse...")
        self.browse_voice_btn.clicked.connect(self.browse_voice_file)
        voice_file_layout.addWidget(self.browse_voice_btn)

        # Add both selectors to file group
        file_layout.addLayout(text_file_layout)
        file_layout.addLayout(voice_file_layout)

        self.params_checkbox = QCheckBox("Show Parameters")
        self.params_checkbox.setChecked(True)  # Show parameters by default
        self.save_config_button = QPushButton("Save Configuration")
        self.save_config_button.setToolTip("Save current parameter values to default_config.yaml")
        self.save_config_button.clicked.connect(self._save_current_config)
        self.metadata_button = QPushButton("Meta Data")
        self.metadata_button.setStyleSheet("color: #00FF00;")
        self.metadata_button.setToolTip(
            "Set M4B, MP3, and supported WAV metadata for the next export."
        )
        self.metadata_button.clicked.connect(self.open_metadata_dialog)

        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(8)
        controls_row.addWidget(self.params_checkbox)
        controls_row.addWidget(self.save_config_button)
        controls_row.addWidget(self.metadata_button)
        controls_row.addStretch(1)

        left_column_layout.addWidget(file_group)
        left_column_layout.addLayout(controls_row)
        left_column_layout.addStretch(1)

        # Right side: Image with left margin buffer
        image_label = QLabel()
        image_path = Path(__file__).parent.parent.parent / "docs" / "icon.png"

        if image_path.exists():
            pixmap = QPixmap(str(image_path))
            # Keep the wide artwork's natural shape so its container does not
            # create a tall blank header beside the file selectors.
            scaled_pixmap = pixmap.scaled(260, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setFixedSize(260, 130)
            image_label.setScaledContents(False)
            image_label.setAlignment(Qt.AlignCenter)
        else:
            # Fallback if image not found
            image_label.setText("Image\nNot Found")
            image_label.setFixedSize(260, 130)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet("border: 1px solid #ccc;")

        left_column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        file_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        # Add to header layout with spacing (1 inch = ~96 pixels at standard DPI)
        header_layout.addWidget(left_column, stretch=1, alignment=Qt.AlignTop)
        header_layout.addSpacing(20)
        header_layout.addWidget(image_label, alignment=Qt.AlignTop)

        parent_layout.addLayout(header_layout)

    def create_parameters_section(self, parent_layout):
        """Create collapsible parameters section."""
        # Container for checkbox and parameters
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Parameters group (not checkable)
        self.params_group = QGroupBox("Parameters")
        # Use horizontal layout for 7-column structure
        layout = QHBoxLayout(self.params_group)
        layout.setAlignment(Qt.AlignTop)

        # Column 0 (NEW): TTS Core Parameters
        tts_core_group, tts_core_layout = self.create_param_section(
            "TTS Core Parameters", "tts_core"
        )

        # Temperature spinner
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 1.5)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setDecimals(2)
        tts_core_value = self.config.tts_core.get('temperature', 0.7) if hasattr(self.config, 'tts_core') else 0.7
        self.temperature_spin.setValue(tts_core_value)
        tts_core_layout.addRow("Temperature:", self.temperature_spin)

        # EOS Threshold spinner
        self.eos_threshold_spin = QDoubleSpinBox()
        self.eos_threshold_spin.setRange(-10.0, 0.0)
        self.eos_threshold_spin.setSingleStep(0.5)
        self.eos_threshold_spin.setDecimals(1)
        eos_value = self.config.tts_core.get('eos_threshold', -4.0) if hasattr(self.config, 'tts_core') else -4.0
        self.eos_threshold_spin.setValue(eos_value)
        tts_core_layout.addRow("EOS Threshold:", self.eos_threshold_spin)

        # Frames After EOS spinner
        self.frames_after_eos_spin = QSpinBox()
        self.frames_after_eos_spin.setRange(0, 10)
        frames_value = self.config.tts_core.get('frames_after_eos', 2) if hasattr(self.config, 'tts_core') else 2
        self.frames_after_eos_spin.setValue(frames_value)
        tts_core_layout.addRow("Frames After EOS:", self.frames_after_eos_spin)

        # Column 1: Chunking parameters
        chunking_group, chunking_layout = self.create_param_section(
            "Chunking Settings", "chunking"
        )

        self.chunking_mode_combo = QComboBox()
        self.chunking_mode_combo.addItem("sentence")
        self.chunking_mode_combo.addItem("paragraph")
        self.chunking_mode_combo.addItem("word_target")
        self.chunking_mode_combo.setToolTip(self._chunking_mode_tooltip())
        chunking_layout.addRow("Mode:", self.chunking_mode_combo)

        self.min_words_spin = QSpinBox()
        self.min_words_spin.setRange(1, 100)
        self.min_words_spin.setToolTip(self._chunking_min_words_tooltip("sentence"))
        chunking_layout.addRow("Min Words:", self.min_words_spin)

        self.target_words_spin = QSpinBox()
        self.target_words_spin.setRange(1, 999)
        self.target_words_spin.setValue(self.config.chunking.get('target_words', 50))
        self.target_words_spin.setToolTip(self._chunking_target_words_tooltip())
        self.target_words_label = QLabel("Target Words:")
        chunking_layout.addRow(self.target_words_label, self.target_words_spin)
        self.chunking_mode_combo.currentTextChanged.connect(
            self._update_chunking_mode_controls
        )

        # Set default values from config
        self.chunking_mode_combo.setCurrentText(self.config.chunking['mode'])
        self.min_words_spin.setValue(self.config.chunking['min_words'])
        self._update_chunking_mode_controls(self.chunking_mode_combo.currentText())

        # Column 2: Post-Processing Pauses Group
        pause_group, pause_layout = self.create_param_section(
            "Pause Durations (ms)", "pauses"
        )

        self.sentence_pause_spin = QSpinBox()
        self.sentence_pause_spin.setRange(0, 2000)
        self.sentence_pause_spin.setSingleStep(50)
        self.sentence_pause_spin.setValue(self.config.pauses['base_durations']['sentence_end'])
        pause_layout.addRow("Sentence End:", self.sentence_pause_spin)

        self.paragraph_pause_spin = QSpinBox()
        self.paragraph_pause_spin.setRange(0, 5000)
        self.paragraph_pause_spin.setSingleStep(100)
        self.paragraph_pause_spin.setValue(self.config.pauses['base_durations']['paragraph_break'])
        pause_layout.addRow("Paragraph Break:", self.paragraph_pause_spin)

        self.chapter_pause_spin = QSpinBox()
        self.chapter_pause_spin.setRange(0, 10000)
        self.chapter_pause_spin.setSingleStep(500)
        self.chapter_pause_spin.setValue(self.config.pauses['base_durations']['chapter_start'])
        pause_layout.addRow("Chapter Start:", self.chapter_pause_spin)

        # Column 3: Quality parameters
        quality_group, quality_layout = self.create_param_section(
            "Quality Settings", "quality"
        )

        self.lsd_steps_spin = QSpinBox()
        self.lsd_steps_spin.setRange(1, 30)
        self.lsd_steps_spin.setValue(self.config.quality.get('lsd_steps', 2))
        quality_layout.addRow("LSD Steps:", self.lsd_steps_spin)
        lsd_note = QLabel("(5-10 optimal for quality vs speed)")
        lsd_note.setStyleSheet("font-size: 10px; color: gray;")
        quality_layout.addRow("", lsd_note)

        self.speed_variation_check = QCheckBox("Enable speed variation")
        speed_config = self.config.speed_variation if hasattr(self.config, 'speed_variation') else {}
        self.speed_variation_check.setChecked(speed_config.get('enabled', True))
        quality_layout.addRow("", self.speed_variation_check)

        self.emotion_detection_check = QCheckBox("Enable emotion detection")
        self.emotion_detection_check.setToolTip(
            "When off, Pocket TTS does not load or run the emotion model. All "
            "chunks use neutral TTS parameters."
        )
        emotion_config = self.config.emotion if hasattr(self.config, 'emotion') else {}
        self.emotion_detection_check.setChecked(emotion_config.get('enabled', True))
        quality_layout.addRow("", self.emotion_detection_check)

        # Column 3b: Pause Injection Settings
        pause_injection_group, pause_injection_layout = self.create_param_section(
            "Pause Injection (s)", "pause_injection"
        )

        # Enable/disable checkbox
        self.pause_injection_check = QCheckBox("Enable punctuation pauses")
        pi_config = self.config.pause_injection if hasattr(self.config, 'pause_injection') else {}
        self.pause_injection_check.setChecked(pi_config.get('enabled', True))
        pause_injection_layout.addRow(self.pause_injection_check)

        # Create spinners for each punctuation mark
        durations = pi_config.get('punctuation_durations', {})
        durations = dict(durations)
        durations[","] = COMMA_PAUSE_DEFAULT_SECONDS
        self.config.pause_injection["punctuation_durations"] = durations
        self._pause_spinners = {}

        PUNCT_LABELS = [
            ('.', 'Period (.)'),
            ('!', 'Exclamation (!)'),
            ('?', 'Question (?)'),
            (',', 'Comma (,)'),
            ('...', 'Ellipsis (...)'),
            ('--', 'Em Dash (--)'),
            (';', 'Semicolon (;)'),
            (':', 'Colon (:)'),
        ]

        for punct, label in PUNCT_LABELS:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 3.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setSuffix(" s")
            spin.setValue(
                COMMA_PAUSE_DEFAULT_SECONDS
                if punct == ","
                else durations.get(punct, 0.0)
            )
            spin.setEnabled(self.pause_injection_check.isChecked())
            self._pause_spinners[punct] = spin
            pause_injection_layout.addRow(label + ":", spin)
            if punct == ",":
                spin.valueChanged.connect(self._on_comma_pause_changed)

        # Connect checkbox to enable/disable spinners
        self.pause_injection_check.toggled.connect(self._on_pause_injection_toggled)

        # Column 4: Audiobook export (only checked formats are built)
        m4b_group, m4b_layout = self.create_param_section(
            "Audiobook Export", "m4b"
        )

        # Migrate older configs: enabled + output_mode → per-format flags.
        m4b_cfg = dict(getattr(self.config, "m4b", {}) or {})
        write_m4b = m4b_cfg.get("write_m4b")
        write_mp3 = m4b_cfg.get("write_mp3")
        write_wav = m4b_cfg.get("write_wav")
        if write_m4b is None and write_mp3 is None and write_wav is None:
            enabled = bool(m4b_cfg.get("enabled", False))
            mode = str(m4b_cfg.get("output_mode", "m4b") or "m4b").lower()
            write_wav = True  # historical default always wrote the full WAV
            write_m4b = enabled and mode in {"m4b", "both"}
            write_mp3 = enabled and mode in {"mp3", "both"}

        self.m4b_write_m4b_check = QCheckBox("M4B")
        self.m4b_write_m4b_check.setChecked(bool(write_m4b))
        self.m4b_write_m4b_check.setToolTip(
            "Build a single M4B from chunk WAVs (chapter TOC when Chapterize is on). "
            "Does not require writing a full WAV."
        )
        m4b_layout.addRow(self.m4b_write_m4b_check)

        self.m4b_write_mp3_check = QCheckBox("MP3")
        self.m4b_write_mp3_check.setChecked(bool(write_mp3))
        self.m4b_write_mp3_check.setToolTip(
            "With Chapterize: one MP3 per chapter under chapters/. "
            "Without: one full-book MP3 from chunk WAVs."
        )
        m4b_layout.addRow(self.m4b_write_mp3_check)

        self.m4b_write_wav_check = QCheckBox("WAV")
        self.m4b_write_wav_check.setChecked(bool(write_wav) if write_wav is not None else True)
        self.m4b_write_wav_check.setToolTip(
            "Write the full stitched WAV. Skip this for faster e2e when only M4B/MP3 are needed."
        )
        m4b_layout.addRow(self.m4b_write_wav_check)

        self.m4b_chapterize_check = QCheckBox("Chapterize")
        self.m4b_chapterize_check.setChecked(bool(m4b_cfg.get("chapterize", False)))
        self.m4b_chapterize_check.setToolTip(
            "Create an audiobook chapter TOC. Choose the strategy below to use "
            "detected headings, fallback minutes, or a maximum chapter length."
        )
        m4b_layout.addRow(self.m4b_chapterize_check)

        configured_chapter_mode = str(m4b_cfg.get("chapter_mode") or "").lower()
        if configured_chapter_mode not in {
            "headings_only", "headings_or_minutes", "headings_with_max",
        }:
            # Keep existing saved minute-cap behavior until user changes it explicitly.
            configured_chapter_mode = (
                "headings_with_max"
                if int(m4b_cfg.get("max_chapter_minutes", 0) or 0) > 0
                else "headings_only"
            )
        self.m4b_chapter_mode_combo = QComboBox()
        self.m4b_chapter_mode_combo.addItem("Headings only", "headings_only")
        self.m4b_chapter_mode_combo.addItem(
            "Headings, otherwise minutes", "headings_or_minutes"
        )
        self.m4b_chapter_mode_combo.addItem(
            "Headings + maximum duration", "headings_with_max"
        )
        self.m4b_chapter_mode_combo.setCurrentIndex(
            self.m4b_chapter_mode_combo.findData(configured_chapter_mode)
        )
        self.m4b_chapter_mode_combo.setToolTip(
            "Headings, otherwise minutes keeps heading chapters intact. It uses "
            "the minute setting only when no Part/Chapter heading is detected."
        )
        m4b_layout.addRow("Chapter strategy:", self.m4b_chapter_mode_combo)

        self.m4b_chapter_minutes_spin = QSpinBox()
        self.m4b_chapter_minutes_spin.setRange(0, 180)
        self.m4b_chapter_minutes_spin.setSingleStep(5)
        self.m4b_chapter_minutes_spin.setValue(
            int(m4b_cfg.get("max_chapter_minutes", 0) or 0)
        )
        self.m4b_chapter_minutes_label = QLabel()
        self.m4b_chapter_minutes_spin.setToolTip(
            "Minute-based chapters prefer paragraph ends, then sentence ends."
        )
        m4b_layout.addRow(self.m4b_chapter_minutes_label, self.m4b_chapter_minutes_spin)

        self.m4b_chapter_mode_note = QLabel()
        self.m4b_chapter_mode_note.setWordWrap(True)
        self.m4b_chapter_mode_note.setStyleSheet(
            "color: #c9d8e8; background: #263445; border: 1px solid #42617c; "
            "border-radius: 6px; padding: 7px;"
        )
        m4b_layout.addRow("", self.m4b_chapter_mode_note)
        self.m4b_chapterize_check.toggled.connect(self._update_chapter_export_controls)
        self.m4b_chapter_mode_combo.currentIndexChanged.connect(
            self._update_chapter_export_controls
        )
        self.m4b_chapter_minutes_spin.valueChanged.connect(
            self._update_chapter_export_controls
        )
        self._update_chapter_export_controls()

        self.m4b_norm_combo = QComboBox()
        self.m4b_norm_combo.addItems(["none", "peak", "loudness", "simple"])
        self.m4b_norm_combo.setCurrentText(m4b_cfg.get('normalization_type', 'peak'))
        m4b_layout.addRow("Normalization:", self.m4b_norm_combo)

        # Column 5: Performance Settings
        performance_group, performance_layout = self.create_param_section(
            "Performance Settings", "performance"
        )

        # Device selector
        import torch
        device_config = getattr(self.config, 'device', {})
        if isinstance(device_config, dict):
            current_device = device_config.get('preferred', 'auto')
        else:
            current_device = 'auto'

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.device_combo.setCurrentText(current_device)

        device_info = ""
        if torch.cuda.is_available():
            device_info = f" (GPU: {torch.cuda.get_device_name(0)})"
        self.device_combo.setToolTip(f"Select compute device.{device_info}")
        performance_layout.addRow("Device:", self.device_combo)

        # Calculate worker limit - VRAM-based on GPU, CPU-based on CPU
        import psutil
        physical_cores = psutil.cpu_count(logical=False)
        logical_cpus = psutil.cpu_count(logical=True)
        current_device = self.device_combo.currentText()

        if current_device in ('cuda', 'auto') and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_limit = max(1, int((vram_gb - 2.0) / 1.5))
            cpu_limit = max(1, (physical_cores or 4) - 1)
            cpu_based_limit = max(vram_limit, cpu_limit)  # Take the higher of VRAM or CPU
            tooltip_text = (f"GPU mode: {vram_gb:.1f} GB VRAM allows ~{vram_limit} workers. "
                           f"CPU has {physical_cores or '?'} cores. Using {cpu_based_limit}.")
        elif physical_cores:
            cpu_based_limit = max(1, physical_cores - 1)
            tooltip_text = (f"CPU mode. Physical cores: {physical_cores}, Logical CPUs: {logical_cpus}. "
                           f"Limited to {cpu_based_limit} workers.")
        else:
            cpu_count = os.cpu_count() or 4
            cpu_based_limit = max(1, cpu_count - 5)
            tooltip_text = (f"CPU mode. CPU count: {cpu_count}. "
                           f"Limited to {cpu_based_limit} workers.")

        print(f"DEBUG: Max workers spinner range: 1 to {cpu_based_limit} (physical_cores={physical_cores}, logical_cpus={logical_cpus})")

        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, cpu_based_limit)  # Limited by CPU formula
        self.max_workers_spin.setValue(self.config.parallel.get('max_workers', cpu_based_limit))
        self.max_workers_spin.setToolTip(tooltip_text)
        performance_layout.addRow("Max Workers:", self.max_workers_spin)

        batch_generation = getattr(self.config, "batch_generation", {}) or {}
        configured_batch_size = max(
            2, int(batch_generation.get("batch_size", 4) or 4)
        )
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(2, 16)
        self.batch_size_spin.setValue(configured_batch_size)
        self.batch_size_spin.setToolTip(
            "Maximum compatible chunks sent to each TTS worker together. "
            "This value applies to the next Generate or Add to Batch action, "
            "without saving the configuration. Larger batches use more VRAM; "
            "the final or incompatible group can be smaller."
        )
        performance_layout.addRow("TTS Batch Size:", self.batch_size_spin)

        # Store original config value for reset reference
        self.original_max_workers = self.config.parallel.get('max_workers', 4)

        # Initialize max workers override
        self.max_workers_override = None

        # Connect spinner to update override
        self.max_workers_spin.valueChanged.connect(self.on_max_workers_changed)

        # Column 6: ASR Quality Control
        asr_group, asr_layout = self.create_param_section(
            "ASR Quality Control", "asr"
        )

        self.asr_enabled_check = QCheckBox("Enable ASR Quality Control")
        asr_enabled = self.config.asr_quality_control.get('enabled', False) if hasattr(self.config, 'asr_quality_control') else False
        self.asr_enabled_check.setChecked(asr_enabled)
        asr_layout.addRow(self.asr_enabled_check)

        self.asr_engine_combo = QComboBox()
        self.asr_engine_combo.addItems(ASR_BACKENDS)
        saved_engine = "faster_whisper"
        if hasattr(self.config, "asr_quality_control"):
            saved_engine = str(
                self.config.asr_quality_control.get("engine", "faster_whisper") or "faster_whisper"
            ).strip()
        if saved_engine in ("whisper_cpp", "whisper-cpp", "cpp"):
            saved_engine = "whisper_cpp"
        elif saved_engine in ("parakeet", "parakeet_tdt", "nemo_parakeet"):
            saved_engine = "parakeet"
        else:
            saved_engine = "faster_whisper"
        eng_idx = self.asr_engine_combo.findText(saved_engine)
        self.asr_engine_combo.setCurrentIndex(eng_idx if eng_idx >= 0 else 0)
        self.asr_engine_combo.setToolTip(
            "Stage 1 engine.\n"
            "faster_whisper: CTranslate2 Whisper.\n"
            "whisper_cpp: ggml via pywhispercpp; GPU needs CUDA source build.\n"
            "parakeet: fixed 0.6b NeMo detector (no Stage 1 model knob)."
        )
        self.asr_engine_combo.currentIndexChanged.connect(self._on_asr_engine_changed)
        asr_layout.addRow("Stage 1:", self.asr_engine_combo)

        self.asr_model_combo = QComboBox()
        self.asr_model_combo.addItems(ASR_MODELS)
        saved_asr_model = self.config.asr_quality_control.get('model', 'base')
        idx = self.asr_model_combo.findText(saved_asr_model)
        if idx >= 0:
            self.asr_model_combo.setCurrentIndex(idx)
        else:
            self.asr_model_combo.setCurrentText("base")
        self.asr_model_combo.setToolTip(
            "Stage 1 model for Faster-Whisper and whisper.cpp. "
            "Parakeet ignores this and uses its fixed 0.6b checkpoint."
        )
        self.asr_model_combo.currentIndexChanged.connect(self._on_stage_one_model_changed)
        asr_layout.addRow("Stage 1 model:", self.asr_model_combo)

        self.asr_second_stage_model_combo = QComboBox()
        self.asr_second_stage_model_combo.addItems(SECOND_STAGE_ASR_MODELS)
        saved_second_stage_model = "medium"
        if hasattr(self.config, "asr_quality_control"):
            saved_second_stage_model = str(
                self.config.asr_quality_control.get("second_stage_model", "medium") or "medium"
            ).strip()
        second_stage_label = _stage_two_combo_label(saved_second_stage_model)
        second_stage_index = self.asr_second_stage_model_combo.findText(second_stage_label)
        self.asr_second_stage_model_combo.setCurrentIndex(
            second_stage_index if second_stage_index >= 0 else SECOND_STAGE_ASR_MODELS.index("medium")
        )
        self.asr_second_stage_model_combo.setToolTip(
            "Independent Faster-Whisper Stage 2 verifier. Default medium. "
            "Must be larger than Stage 1, or Disabled to skip verification."
        )
        asr_layout.addRow("Stage 2 model:", self.asr_second_stage_model_combo)

        # GPU ASR workers after TTS gen (calibrator seeds default; GUI wins per run)
        self.asr_gpu_workers_spin = QSpinBox()
        self.asr_gpu_workers_spin.setRange(0, 16)
        self.asr_gpu_workers_spin.setMinimumWidth(96)
        self.asr_gpu_workers_spin.setToolTip(
            "After TTS finishes (0=off). No ASR during gen.\n"
            "faster_whisper: CPU load/score budget for 1× GPU pipeline "
            "(pack=8 + solo-retry on fail). Try 8.\n"
            "whisper_cpp: number of concurrent CUDA model copies. "
            "Use 1–3 on 8GB for medium/large."
        )
        parallel_asr = (self.config.asr_quality_control or {}).get("parallel") or {}
        self.asr_gpu_workers_spin.setValue(
            int(parallel_asr.get("gpu_workers_after_tts", 4) or 0)
        )
        self.asr_gpu_workers_label = QLabel("GPU ASR workers (after gen):")
        self.asr_gpu_workers_label.setToolTip(self.asr_gpu_workers_spin.toolTip())
        asr_layout.addRow(self.asr_gpu_workers_label, self.asr_gpu_workers_spin)
        self.device_combo.currentTextChanged.connect(self._on_tts_device_changed)
        self._sync_asr_controls_to_device()

        # Language codes for faster-whisper (display label → ISO code)
        self._asr_language_options = [
            ("English", "en"),
            ("Spanish", "es"),
            ("French", "fr"),
            ("German", "de"),
            ("Italian", "it"),
            ("Portuguese", "pt"),
            ("Dutch", "nl"),
            ("Polish", "pl"),
            ("Russian", "ru"),
            ("Japanese", "ja"),
            ("Chinese", "zh"),
            ("Korean", "ko"),
            ("Arabic", "ar"),
            ("Hindi", "hi"),
            ("Turkish", "tr"),
            ("Swedish", "sv"),
            ("Danish", "da"),
            ("Norwegian", "no"),
            ("Finnish", "fi"),
            ("Czech", "cs"),
            ("Greek", "el"),
            ("Hebrew", "he"),
            ("Hungarian", "hu"),
            ("Romanian", "ro"),
            ("Ukrainian", "uk"),
            ("Vietnamese", "vi"),
            ("Thai", "th"),
            ("Indonesian", "id"),
        ]
        self.asr_language_combo = QComboBox()
        for label, code in self._asr_language_options:
            self.asr_language_combo.addItem(label, code)
        saved_lang = self.config.asr_quality_control.get('language', 'en') if hasattr(self.config, 'asr_quality_control') else 'en'
        lang_index = self.asr_language_combo.findData(saved_lang)
        self.asr_language_combo.setCurrentIndex(lang_index if lang_index >= 0 else 0)
        asr_layout.addRow("Language:", self.asr_language_combo)

        self.asr_threshold_spin = QDoubleSpinBox()
        self.asr_threshold_spin.setRange(0.0, 1.0)
        self.asr_threshold_spin.setSingleStep(0.05)
        self.asr_threshold_spin.setMinimumWidth(96)
        self.asr_threshold_spin.setValue(self.config.asr_quality_control.get('threshold', 0.85) if hasattr(self.config, 'asr_quality_control') else 0.85)
        asr_layout.addRow("Threshold:", self.asr_threshold_spin)

        self.asr_max_retries_spin = QSpinBox()
        self.asr_max_retries_spin.setRange(0, 10)
        self.asr_max_retries_spin.setMinimumWidth(96)
        self.asr_max_retries_spin.setToolTip(
            "0 runs ASR reporting only; values above 0 allow regeneration attempts."
        )
        self.asr_max_retries_spin.setValue(self.config.asr_quality_control.get('max_retries', 3) if hasattr(self.config, 'asr_quality_control') else 3)
        asr_layout.addRow("Max Retries:", self.asr_max_retries_spin)

        self.asr_temp_decrement_spin = QDoubleSpinBox()
        self.asr_temp_decrement_spin.setRange(0.01, 0.5)
        self.asr_temp_decrement_spin.setSingleStep(0.01)
        self.asr_temp_decrement_spin.setMinimumWidth(96)
        self.asr_temp_decrement_spin.setValue(self.config.asr_quality_control.get('temp_decrement', 0.1) if hasattr(self.config, 'asr_quality_control') else 0.1)
        asr_layout.addRow("Temp Decrement:", self.asr_temp_decrement_spin)

        self._on_asr_engine_changed()

        # Add subgroups to main parameters layout (8 columns, TTS core first)
        layout.addWidget(tts_core_group, alignment=Qt.AlignTop)
        layout.addWidget(chunking_group, alignment=Qt.AlignTop)
        layout.addWidget(pause_group, alignment=Qt.AlignTop)
        layout.addWidget(quality_group, alignment=Qt.AlignTop)
        layout.addWidget(pause_injection_group, alignment=Qt.AlignTop)
        layout.addWidget(m4b_group, alignment=Qt.AlignTop)
        layout.addWidget(performance_group, alignment=Qt.AlignTop)
        layout.addWidget(asr_group, alignment=Qt.AlignTop)

        # Add parameters group to container and set initially hidden
        container_layout.addWidget(self.params_group)
        self.params_group.setVisible(True)  # Show parameters by default

        parent_layout.addWidget(container)
        # Let Results claim any leftover vertical space.
        parent_layout.setStretch(0, 0)
        parent_layout.setStretch(1, 0)
        parent_layout.setStretch(2, 0)
        parent_layout.setStretch(3, 1)

    def create_collapsible_group(self, title: str) -> QGroupBox:
        """Create a plain group box (legacy helper; prefer create_param_section)."""
        group = QGroupBox(title)
        return group

    def create_param_section(
        self, title: str, help_key: str
    ) -> tuple:
        """Build a parameter column with title and a ? help button.

        Args:
            title: Section heading shown next to the help button.
            help_key: Key into ``parameter_help.PARAMETER_SECTION_HELP``.

        Returns:
            (group_widget, form_layout) — add rows to the form layout.
        """
        group = QGroupBox()
        # Keep short sections compact instead of stretching them to the
        # height of long sections such as Pause Injection and ASR.
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        outer = QVBoxLayout(group)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-weight: bold;")
        title_lbl.setWordWrap(False)
        title_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        title_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        header.addWidget(title_lbl, stretch=1)

        help_btn = QPushButton("?")
        help_btn.setFixedSize(22, 22)
        help_btn.setToolTip(f"Help: {title}")
        help_btn.setStyleSheet(
            "QPushButton { font-weight: bold; border-radius: 11px; "
            "background: #3a3a3a; color: #7ec8ff; border: 1px solid #555; }"
            "QPushButton:hover { background: #4a4a4a; color: #fff; }"
        )
        # Capture key/title for the slot (avoid late-binding loop bugs)
        help_btn.clicked.connect(
            lambda _checked=False, k=help_key, t=title: self._show_param_section_help(k, t)
        )
        header.addWidget(help_btn, stretch=0, alignment=Qt.AlignVCenter)
        outer.addLayout(header)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(form)
        return group, form

    def _show_param_section_help(self, help_key: str, title: str) -> None:
        """Open a scrollable dialog with section + setting explanations.

        Args:
            help_key: Lookup key for help text.
            title: Dialog window title.
        """
        body = help_for(help_key)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Help — {title}")
        dlg.setMinimumSize(480, 420)
        lay = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        # Plain text with simple structure; escape not needed (our content is static)
        html_lines = [f"<h2>{title}</h2>"]
        for line in body.strip().splitlines():
            if not line.strip():
                html_lines.append("<br>")
            elif line.startswith("• "):
                html_lines.append(f"<p style='margin-left:12px'>{line}</p>")
            elif line.startswith("  "):
                html_lines.append(
                    f"<p style='margin-left:28px; color:#ccc'>{line.strip()}</p>"
                )
            elif line in (
                "What this section does",
                "Settings",
                "Tips",
            ) or line.endswith("Settings") and len(line) < 40:
                html_lines.append(f"<h3>{line}</h3>")
            else:
                html_lines.append(f"<p>{line}</p>")
        browser.setHtml("\n".join(html_lines))
        lay.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        lay.addWidget(buttons)
        dlg.exec_()

    def create_progress_section(self, parent_layout):
        """Create chunk/ETA controls with completed-run metrics at far right."""
        group = QGroupBox("Generation Progress")
        layout = QVBoxLayout(group)

        # Progress info
        progress_layout = QHBoxLayout()

        self.chunk_progress_label = QLabel("Chunks: 0/0")
        self.chunk_progress_label.setStyleSheet("color: #00FF00;")
        progress_layout.addWidget(self.chunk_progress_label)

        self.time_elapsed_label = QLabel("Elapsed: 0:00")
        self.time_elapsed_label.setStyleSheet("color: #00FF00;")
        progress_layout.addWidget(self.time_elapsed_label)

        self.eta_label = QLabel("ETA: --:--")
        self.eta_label.setStyleSheet("color: #00FF00;")
        progress_layout.addWidget(self.eta_label)

        progress_layout.addStretch(1)
        self.completion_timing_label = QLabel("")
        self.completion_timing_label.setStyleSheet("color: #00FF00;")
        self.completion_timing_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.completion_timing_label.setToolTip(
            "Completed run: 1 = ASR Stage 1, 2 = ASR Stage 2, then "
            "chunk-generation realtime factor and time, followed by "
            "end-to-end realtime factor and time."
        )
        progress_layout.addWidget(self.completion_timing_label)

        layout.addLayout(progress_layout)

        # Progress bar
        self.generation_progress_bar = QProgressBar()
        self.generation_progress_bar.setRange(0, 100)
        layout.addWidget(self.generation_progress_bar)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Generate Audiobook")
        self.start_btn.clicked.connect(self.start_generation)
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)

        self.add_to_batch_btn = QPushButton("Add to Batch")
        self.add_to_batch_btn.clicked.connect(self.add_to_batch)
        self.add_to_batch_btn.setEnabled(False)
        button_layout.addWidget(self.add_to_batch_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setEnabled(False)
        # Avoid accidental Space/Enter activation when Start is disabled and
        # focus lands on Stop during generation (common false-cancel path).
        from qtpy.QtCore import Qt as _Qt
        self.stop_btn.setFocusPolicy(_Qt.NoFocus)
        button_layout.addWidget(self.stop_btn)

        self.play_btn = QPushButton("Play Last Audio")
        self.play_btn.clicked.connect(self.play_last_audio)
        self.play_btn.setEnabled(False)
        button_layout.addWidget(self.play_btn)

        layout.addLayout(button_layout)

        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        parent_layout.addWidget(group)

    def create_results_section(self, parent_layout):
        """Create results section."""
        group = QGroupBox("Results")
        layout = QVBoxLayout(group)
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        # Default state: Expanding (since parameters are hidden by default)
        self.results_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Set bright green text color
        self.results_text.setStyleSheet("QTextEdit { color: #00FF00; }")
        layout.addWidget(self.results_text)

        parent_layout.addWidget(group, stretch=1)

    def setup_connections(self):
        """Setup signal connections."""
        self.generation_progress.connect(self.on_generation_progress)
        self.generation_finished.connect(self.on_generation_finished)
        self.params_checkbox.stateChanged.connect(self.on_parameters_toggled)

    def on_parameters_toggled(self, state):
        """Show or hide the parameters panel; grow Results when hidden.

        When maximized/fullscreen, skip adjustSize() so the window does not
        shrink off the maximized state (that lag is what users see under load).
        """
        is_checked = state == Qt.Checked
        self.params_group.setVisible(is_checked)

        # Results remains the flexible terminal area in both states. The
        # Parameters panel should not force its output view to a short strip.
        self.results_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_text.setMaximumHeight(16777215)

        if self.isMaximized() or self.isFullScreen():
            # Keep full-screen/maximized; only reflow central layout
            if self.centralWidget() is not None:
                self.centralWidget().updateGeometry()
        else:
            # Non-maximized: resize window to fit content
            self.adjustSize()

    def _update_chunking_mode_controls(self, mode: str) -> None:
        """Show/lock chunking spinners to match mode semantics."""
        is_word_target = mode == "word_target"
        is_paragraph = mode == "paragraph"
        self.target_words_label.setVisible(is_word_target)
        self.target_words_spin.setVisible(is_word_target)
        # paragraph: min_words has no effect — gray it out.
        self.min_words_spin.setEnabled(not is_paragraph)
        self.min_words_spin.setToolTip(self._chunking_min_words_tooltip(mode))
        self.target_words_spin.setToolTip(self._chunking_target_words_tooltip())
        if hasattr(self, "chunking_mode_combo"):
            self.chunking_mode_combo.setToolTip(self._chunking_mode_tooltip())

    def _chunking_mode_tooltip(self) -> str:
        """Return help text for chunking mode semantics."""
        return (
            "sentence: min_words is a soft merge floor. Pack full sentences "
            "until the floor is met; a chunk may exceed it, never mid-sentence.\n"
            "paragraph: ignore min_words; one chunk per paragraph (\\n\\n breaks).\n"
            "word_target: min_words soft floor, target_words hard cap. Pack full "
            "sentences up to the cap without exceeding it unless one sentence alone "
            "is longer. Blank lines / paragraph breaks do NOT force a new chunk — "
            "blocks under the cap pack into one TTS chunk."
        )

    def _chunking_min_words_tooltip(self, mode: str) -> str:
        """Return min_words help for the active mode."""
        if mode == "paragraph":
            return "Not used in paragraph mode (split only on paragraph breaks)."
        if mode == "word_target":
            return (
                "Soft lower floor for word_target: prefer packing at least this "
                "many words when the next sentence still fits under target_words."
            )
        return (
            "Soft merge floor for sentence mode: keep adding full sentences until "
            "this many words, then flush. Last sentence may overshoot."
        )

    def _chunking_target_words_tooltip(self) -> str:
        """Return target_words help text."""
        return (
            "Hard upper bound for word_target only. Do not add another full "
            "sentence if it would exceed this count, unless the chunk is empty "
            "(one long sentence stays whole)."
        )

    def _build_runtime_settings_snapshot(self) -> dict:
        """Capture current GUI chunking/TTS settings for one generate or batch job.

        Config file is load/save only; this snapshot is the source of truth for
        the live run, including unsaved spinner values.
        """
        mode = self.chunking_mode_combo.currentText()
        return {
            "chunking_mode": mode,
            "min_words": self.min_words_spin.value(),
            "target_words": self.target_words_spin.value(),
            "respect_boundaries": bool(
                self.config.chunking.get("respect_boundaries", True)
            ),
            "temperature": self.temperature_spin.value(),
            "eos_threshold": self.eos_threshold_spin.value(),
            "frames_after_eos": self.frames_after_eos_spin.value(),
            "lsd_steps": self.lsd_steps_spin.value(),
            "speed_variation": self.speed_variation_check.isChecked(),
            "emotion_detection_enabled": self.emotion_detection_check.isChecked(),
            "pause_injection_enabled": self.pause_injection_check.isChecked(),
            "pause_durations": {
                punct: spin.value() for punct, spin in self._pause_spinners.items()
            },
            "sentence_pause_ms": self.sentence_pause_spin.value(),
            "paragraph_pause_ms": self.paragraph_pause_spin.value(),
            "chapter_pause_ms": self.chapter_pause_spin.value(),
        }

    def _expand_results_terminal_for_progress(self) -> None:
        """Uncheck Show Parameters and flush layout so the Results log is large.

        Used before long blocking work (e.g. CUDA pywhispercpp build) so progress
        lines are visible. Manual uncheck can lag a few seconds under load; we
        process events twice so the reflow paints before the build starts.
        """
        if hasattr(self, "params_checkbox") and self.params_checkbox.isChecked():
            self.params_checkbox.setChecked(False)
        elif hasattr(self, "params_group") and hasattr(self, "results_text"):
            # Already unchecked — still force Results to expand
            self.params_group.setVisible(False)
            self.results_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.results_text.setMaximumHeight(16777215)
            if self.centralWidget() is not None:
                self.centralWidget().updateGeometry()

        # Let Qt finish hide/reflow before a long blocking job steals the UI thread
        QApplication.processEvents()
        if self.centralWidget() is not None:
            self.centralWidget().update()
        QApplication.processEvents()

    def on_max_workers_changed(self, value):
        """Handle max workers spinner change."""
        # Store override value (None means use config default)
        if value != self.original_max_workers:
            self.max_workers_override = value
            print(f"DEBUG: GUI max_workers_override set to {value} (config default: {self.original_max_workers})")
        else:
            self.max_workers_override = None
            print(f"DEBUG: GUI max_workers_override reset to None (config default: {self.original_max_workers})")

    def _on_pause_injection_toggled(self, checked: bool):
        """Handle pause injection checkbox toggle."""
        for spin in self._pause_spinners.values():
            spin.setEnabled(checked)

    def _on_comma_pause_changed(self, value: float) -> None:
        """Warn once when a live comma pause is changed above the safe default."""
        if float(value) <= COMMA_PAUSE_DEFAULT_SECONDS:
            self._comma_pause_warning_shown = False
            return
        if self._comma_pause_warning_shown:
            return
        self._comma_pause_warning_shown = True
        QMessageBox.warning(
            self,
            COMMA_PAUSE_WARNING_TITLE,
            COMMA_PAUSE_WARNING_TEXT,
        )

    def _pause_durations_for_config(self, *, force_comma_zero: bool = False) -> dict:
        """Collect GUI punctuation pauses, optionally forcing comma to zero for saves."""
        durations = {
            punct: float(spin.value())
            for punct, spin in self._pause_spinners.items()
        }
        if force_comma_zero:
            durations[","] = COMMA_PAUSE_DEFAULT_SECONDS
        return durations

    def browse_text_file(self):
        """Browse for text file (hidden files/folders hidden by default)."""
        from .qt_file_dialogs import get_open_file_name

        file_path, _ = get_open_file_name(
            self, "Select Text File", self.last_text_dir, "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.last_text_dir = str(Path(file_path).parent)  # Remember directory
            self._save_gui_settings()  # Save settings immediately
            self.text_file_path.setText(file_path)
            self.results_text.append(f"📄 Selected: {os.path.basename(file_path)}")
            self.results_text.append("Ready to generate audiobook")
            self.start_btn.setEnabled(True)
            self.add_to_batch_btn.setEnabled(True)

    def browse_voice_file(self):
        """Browse for voice file (hidden files/folders hidden by default)."""
        from .qt_file_dialogs import get_open_file_name

        file_path, _ = get_open_file_name(
            self,
            "Select Voice File",
            self.last_voice_dir,
            "Audio Files (*.wav *.mp3 *.m4a *.flac);;WAV Files (*.wav);;All Files (*)",
        )
        if file_path:
            self.last_voice_dir = str(Path(file_path).parent)  # Remember directory
            self._save_gui_settings()  # Save settings immediately
            # Add to combo box if not already there
            voice_name = f"Custom: {Path(file_path).name}"
            if self.voice_combo.findText(voice_name) == -1:
                # Store display name and full path as item data
                self.voice_combo.addItem(voice_name, file_path)
            else:
                # Update the data for existing item
                index = self.voice_combo.findText(voice_name)
                self.voice_combo.setItemData(index, file_path)
            self.voice_combo.setCurrentText(voice_name)
            self.last_custom_voice = file_path
            self._save_gui_settings()

    def start_generation(self):
        """Start one generation run and clear metrics retained from prior success."""
        # Prevent multiple concurrent generations
        if (hasattr(self, 'generation_thread') and
            self.generation_thread is not None and
            isinstance(self.generation_thread, GenerationThread) and
            hasattr(self.generation_thread, 'isRunning') and
            self.generation_thread.isRunning()):
            self.results_text.append("⚠️ Generation already in progress")
            return

        total_start_time = time.time()

        # Get text file
        text_file = self.text_file_path.text()
        if text_file == "No file selected" or not os.path.exists(text_file):
            self.results_text.append("❌ Error: Please select a valid text file")
            return

        # Live GUI snapshot is source of truth (unsaved spinners included).
        runtime = self._build_runtime_settings_snapshot()
        pi_enabled = runtime["pause_injection_enabled"]

        # When pause injection is on, zero out boundary pauses so silence_duration=0 in chunk metadata
        if pi_enabled:
            sentence_pause_ms = 0
            paragraph_pause_ms = 0
            chapter_pause_ms = 0
        else:
            sentence_pause_ms = runtime["sentence_pause_ms"]
            paragraph_pause_ms = runtime["paragraph_pause_ms"]
            chapter_pause_ms = runtime["chapter_pause_ms"]

        # Sync live controls to config before preprocessing builds TTS parameters.
        if not hasattr(self.config, 'speed_variation') or not isinstance(self.config.speed_variation, dict):
            self.config.speed_variation = {}
        self.config.speed_variation['enabled'] = runtime["speed_variation"]
        self.config.emotion['enabled'] = runtime["emotion_detection_enabled"]

        # Automatically preprocess text
        self.results_text.append("🔍 Analyzing text...")
        success = self._preprocess_text(
            text_file,
            sentence_pause_ms,
            paragraph_pause_ms,
            chapter_pause_ms,
            runtime_settings=runtime,
        )
        if not success:
            return  # Error already displayed in _preprocess_text

        # Get voice selection
        voice_selection = self.voice_combo.currentText()
        if voice_selection.startswith("Custom:"):
            # Retrieve full path from combo box item data
            current_index = self.voice_combo.currentIndex()
            voice_path = self.voice_combo.itemData(current_index)
            if voice_path is None:
                # Fallback for legacy items without data (shouldn't happen)
                voice_path = voice_selection.replace("Custom: ", "")
            if not os.path.exists(voice_path):
                self.results_text.append(f"❌ Error: Voice file not found: {voice_path}")
                return
        else:
            voice_path = voice_selection.split(" ")[0]  # Get first word (voice name)

        # Generate automatic output path based on input filename and voice
        from pocket_tts.audiobook.generator import AudiobookGenerator
        dataset_paths = AudiobookGenerator.generate_output_paths(text_file, voice_path)
        output_path = str(dataset_paths['final_audio_path'])

        # Display output directory and filename to user
        self.results_text.append(f"📁 Output directory: {dataset_paths['output_dir']}")
        self.results_text.append(f"🎵 Final audio: {Path(output_path).name}")

        # Helper function to convert milliseconds to frames for TTS engine
        ms_to_frames = lambda ms: int((ms / 1000) * 24000)

        # Collect parameters from GUI snapshot (convert ms to frames for generation)
        params = {
            'voice_path': voice_path,
            'output_path': output_path,
            'source_file': text_file,  # Add source file path for JSON metadata
            'total_start_time': total_start_time,
            'chunking_mode': runtime['chunking_mode'],
            'min_words': runtime['min_words'],
            'target_words': runtime['target_words'],
            'respect_boundaries': runtime['respect_boundaries'],
            'lsd_steps': runtime['lsd_steps'],
            'speed_variation': runtime['speed_variation'],
            'emotion_detection_enabled': runtime['emotion_detection_enabled'],
            'pause_injection_enabled': pi_enabled,
            'pause_durations': dict(runtime['pause_durations']),
        }

        # When pause injection is enabled, zero out boundary silence
        if pi_enabled:
            params['sentence_pause'] = ms_to_frames(0)
            params['paragraph_pause'] = ms_to_frames(0)
            params['chapter_pause'] = ms_to_frames(0)
        else:
            params['sentence_pause'] = ms_to_frames(self.sentence_pause_spin.value())
            params['paragraph_pause'] = ms_to_frames(self.paragraph_pause_spin.value())
            params['chapter_pause'] = ms_to_frames(self.chapter_pause_spin.value())

        # Update config with audiobook export settings
        if not hasattr(self.config, 'm4b'):
            self.config.m4b = {}
        self.config.m4b.update(self._m4b_gui_settings())
        # Ensure other defaults are present
        if 'speed' not in self.config.m4b: self.config.m4b['speed'] = 1.0
        if 'sample_rate' not in self.config.m4b: self.config.m4b['sample_rate'] = 24000
        if 'target_db' not in self.config.m4b: self.config.m4b['target_db'] = -1.5

        # ASR settings from GUI for this run only (not written to yaml unless Save).
        # Stage 1 compatibility still decides whether legacy whisper.cpp checks apply.
        if self._should_prompt_cpp_gpu_build() and not self._confirm_cpp_gpu_if_needed():
            return
        self._apply_asr_gui_to_config(persist_workers=False)
        self._apply_batch_size_gui_to_config()

        # Update config with device settings
        if not hasattr(self.config, 'device') or not isinstance(self.config.device, dict):
            self.config.device = {}
        self.config.device['preferred'] = self.device_combo.currentText()

        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.generation_progress_bar.setValue(0)
        self.chunk_progress_label.setText("Chunks: 0/0")
        self.time_elapsed_label.setText("Elapsed: 0:00")
        self.eta_label.setText("ETA: --:--")
        self.completion_timing_label.setText("")
        self.results_text.append("\n▶️ Starting generation...")
        self.results_text.append(f"Output: {output_path}")
        asr_status = self._format_asr_start_status()
        if asr_status:
            self.results_text.append(asr_status)
        batch_generation = getattr(self.config, "batch_generation", {}) or {}
        if batch_generation.get("enabled", False):
            self.results_text.append(
                f"Performance: TTS batch size={batch_generation['batch_size']}"
            )

        # Disable start button during generation
        self.start_btn.setEnabled(False)

        # Start generation thread
        self.generation_thread = GenerationThread(self.current_chunks, params, self.config, self.max_workers_override)
        self.generation_thread.progress.connect(self.on_generation_progress)
        self.generation_thread.finished.connect(self.on_generation_finished)
        self.generation_thread.start()

    def build_batch_job_snapshot(self):
        """Build voice path, run params, and config snapshot from current Main GUI.

        Used by Add to Batch and Batch-tab Add Folder so every queued .txt gets
        the same Main-tab settings and per-file output folder naming.

        Returns:
            ``(voice_path, params_template, config_snapshot)`` or
            ``(None, None, None)`` if voice/settings cannot be resolved.
            ``params_template`` has no fixed source_file/output_path; callers set
            those per text file via ``add_job``.
        """
        voice_selection = self.voice_combo.currentText()
        if voice_selection.startswith("Custom:"):
            voice_path = self.voice_combo.itemData(self.voice_combo.currentIndex())
            if not voice_path:
                voice_path = voice_selection.replace("Custom: ", "")
            if not os.path.isfile(str(voice_path)):
                msg = f"❌ Error: Voice file not found: {voice_path}"
                self.results_text.append(msg)
                if getattr(self, "batch_tab", None) is not None:
                    self.batch_tab.log_message(msg)
                return None, None, None
        else:
            voice_path = voice_selection.split(" ")[0]

        self._apply_asr_gui_to_config(persist_workers=False)
        self._apply_batch_size_gui_to_config()
        config_snapshot = deepcopy(self.config)
        config_snapshot.tts_core.update({
            "temperature": self.temperature_spin.value(),
            "eos_threshold": self.eos_threshold_spin.value(),
            "frames_after_eos": self.frames_after_eos_spin.value(),
        })
        config_snapshot.chunking.update({
            "mode": self.chunking_mode_combo.currentText(),
            "min_words": self.min_words_spin.value(),
            "target_words": self.target_words_spin.value(),
        })
        config_snapshot.quality["lsd_steps"] = self.lsd_steps_spin.value()
        config_snapshot.pauses["base_durations"].update({
            "sentence_end": self.sentence_pause_spin.value(),
            "paragraph_break": self.paragraph_pause_spin.value(),
            "chapter_start": self.chapter_pause_spin.value(),
        })
        config_snapshot.pause_injection.update({
            "enabled": self.pause_injection_check.isChecked(),
            "punctuation_durations": {
                punct: spin.value() for punct, spin in self._pause_spinners.items()
            },
        })
        config_snapshot.speed_variation["enabled"] = self.speed_variation_check.isChecked()
        config_snapshot.emotion["enabled"] = self.emotion_detection_check.isChecked()
        config_snapshot.m4b.update(self._m4b_gui_settings())
        config_snapshot.device["preferred"] = self.device_combo.currentText()
        config_snapshot.parallel["max_workers"] = self.max_workers_spin.value()

        runtime = self._build_runtime_settings_snapshot()
        pi_enabled = runtime["pause_injection_enabled"]
        params = {
            "voice_path": voice_path,
            "chunking_mode": runtime["chunking_mode"],
            "min_words": runtime["min_words"],
            "target_words": runtime["target_words"],
            "respect_boundaries": runtime["respect_boundaries"],
            "temperature": runtime["temperature"],
            "eos_threshold": runtime["eos_threshold"],
            "frames_after_eos": runtime["frames_after_eos"],
            "lsd_steps": runtime["lsd_steps"],
            "speed_variation": runtime["speed_variation"],
            "emotion_detection_enabled": runtime["emotion_detection_enabled"],
            "pause_injection_enabled": pi_enabled,
            "pause_durations": dict(runtime["pause_durations"]),
            "sentence_pause_ms": 0 if pi_enabled else runtime["sentence_pause_ms"],
            "paragraph_pause_ms": 0 if pi_enabled else runtime["paragraph_pause_ms"],
            "chapter_pause_ms": 0 if pi_enabled else runtime["chapter_pause_ms"],
        }
        return voice_path, params, config_snapshot

    def add_to_batch(self):
        """Queue the current Main-tab input and settings as an immutable batch job."""
        text_file = self.text_file_path.text()
        if text_file == "No file selected" or not os.path.isfile(text_file):
            self.results_text.append("❌ Error: Please select a valid text file")
            return

        voice_path, params, config_snapshot = self.build_batch_job_snapshot()
        if not voice_path:
            return

        params = dict(params)
        params["source_file"] = text_file
        job = self.batch_tab.add_job(text_file, voice_path, params, config_snapshot)
        self.results_text.append(
            f"Queued batch job: {Path(job.output_path).parent.name}"
        )

    def _preprocess_text(
        self,
        text_file: str,
        sentence_pause_ms: int,
        paragraph_pause_ms: int,
        chapter_pause_ms: int,
        runtime_settings: dict | None = None,
    ) -> bool:
        """Preprocess text file using a live runtime settings snapshot.

        Args:
            text_file: Path to the book text.
            sentence_pause_ms: Boundary silence for sentence ends (may be 0).
            paragraph_pause_ms: Boundary silence for paragraphs (may be 0).
            chapter_pause_ms: Boundary silence for chapters (may be 0).
            runtime_settings: GUI snapshot; if None, built from current controls.
                Config file is not used for chunking on the live path.
        """
        try:
            runtime = runtime_settings or self._build_runtime_settings_snapshot()
            # Read text file
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # Periods on bare line ends BEFORE structure/chunk merge, so a short
            # title line still has a TTS pause cue after it is joined to body text.
            # Newlines are preserved here for paragraph detection; flatten is later.
            from pocket_tts.preprocessing.text_normalizer import (
                ensure_terminal_punctuation_on_lines,
            )
            text = ensure_terminal_punctuation_on_lines(text)

            # Initialize components from snapshot (unsaved spinners apply now).
            detector = StructureDetector()
            chunker = SmartChunker(
                mode=runtime["chunking_mode"],
                min_words=int(runtime["min_words"]),
                respect_boundaries=bool(runtime.get("respect_boundaries", True)),
                target_words=int(runtime["target_words"]),
            )
            emotion_detection_enabled = bool(runtime["emotion_detection_enabled"])
            analyzer = EmotionAnalyzer() if emotion_detection_enabled else None

            # Use the passed-in boundary pause values (may be zeroed if pause injection is on)
            boundary_pauses = {
                BoundaryType.SENTENCE_END: sentence_pause_ms,
                BoundaryType.PARAGRAPH_BREAK: paragraph_pause_ms,
                BoundaryType.CHAPTER_START: chapter_pause_ms
            }

            # Initialize mapper with custom boundary values and base TTS parameters
            mapper = ParameterMapper(
                config=self.config,
                boundary_pauses=boundary_pauses,
                base_temperature=float(runtime["temperature"]),
                base_eos_threshold=float(runtime["eos_threshold"]),
                base_frames_after_eos=int(runtime["frames_after_eos"]),
            )

            # Process text
            self.results_text.append("📖 Detecting structure...")
            structure = detector.analyze(text)

            self.results_text.append("✂️ Creating chunks...")
            chunks = chunker.chunk(structure)

            # Canonical [Xs] spacing early so emotion,
            # chapter stamping, JSON, and TTS all see the same text the generator
            # will speak. Glued Chapter[4s]One becomes Chapter [4s] One.
            from pocket_tts.preprocessing.text_normalizer import flatten_newlines_for_json

            for chunk in chunks:
                chunk.text = flatten_newlines_for_json(chunk.text or "")

            # Skip model loading and analysis when the Main-tab checkbox is off.
            if chunks:
                if emotion_detection_enabled:
                    self.results_text.append("🧠 Analyzing emotions...")
                    texts_to_analyze = [chunk.text for chunk in chunks]
                    emotion_results = analyzer.analyze_batch(texts_to_analyze)
                else:
                    self.results_text.append(
                        "⏭️ Emotion detection disabled; using neutral TTS parameters..."
                    )
                    emotion_results = [EmotionAnalyzer.neutral_result() for _ in chunks]

                # Map emotions to parameters
                self.results_text.append("⚙️ Mapping parameters...")
                for chunk, emotion in zip(chunks, emotion_results):
                    params = mapper.calculate_params(
                        emotion=emotion['emotion'],
                        punctuation=chunk.punctuation,
                        boundary_type=chunk.boundary_type,
                        word_count=chunk.word_count,
                        emotion_scores=emotion['scores']
                    )

                    # Get silence duration (ms) -> convert to seconds for storage
                    silence_duration_ms = mapper.calculate_silence_duration_ms(chunk.boundary_type)
                    silence_duration_sec = silence_duration_ms / 1000.0

                    # Convert TTSParams object to dictionary
                    chunk.tts_params = {
                        'temperature': params.temperature,
                        'frames_after_eos': params.frames_after_eos,
                        'eos_threshold': params.eos_threshold,
                        'lsd_decode_steps': params.lsd_decode_steps,
                        'speed_factor': params.speed_factor
                    }
                    chunk.emotion = emotion['emotion']
                    chunk.emotion_scores = emotion['scores']
                    chunk.emotion_confidence = emotion['confidence']

                    # Store post-processing parameters
                    chunk.post_process = {
                        'silence_duration': silence_duration_sec
                    }

            # Store results
            self.current_structure = structure
            self.current_chunks = chunks

            chunk_count = len(chunks)
            self.results_text.append(f"✅ Analysis complete: {chunk_count} chunks created")

            return True

        except Exception as e:
            self.results_text.append(f"❌ Preprocessing failed: {str(e)}")
            return False

    def stop_generation(self):
        """Stop audiobook generation (explicit Stop button only)."""
        if (hasattr(self, 'generation_thread') and
            self.generation_thread is not None and
            isinstance(self.generation_thread, GenerationThread) and
            hasattr(self.generation_thread, 'isRunning') and
            self.generation_thread.isRunning()):
            import logging
            logging.getLogger(__name__).info(
                "Stop button pressed — requesting generation cancel"
            )
            self.generation_thread.stop()
            self.stop_btn.setEnabled(False)
            self.start_btn.setEnabled(True)

    def on_generation_progress(self, progress_data):
        """Update generation progress and phase terminal lines."""
        phase = (progress_data.get("phase") or "").strip().lower()
        phase_icons = {
            "starting": "▶",
            "tts": "🎙",
            "asr": "🔍",
            "regen": "♻",
            "done": "✓",
        }

        # Explicit pipeline phases (starting / tts / asr / regen / done)
        if phase in phase_icons:
            msg = progress_data.get("message") or phase
            icon = phase_icons[phase]
            key = f"{phase}:{msg}"
            if getattr(self, "_last_phase_msg", None) != key:
                self._last_phase_msg = key
                self.results_text.append(f"{icon} [{phase.upper()}] {msg}")
            if phase == "asr":
                self.chunk_progress_label.setText("ASR check…")
                self.eta_label.setText("ASR: not stuck")
                self.generation_progress_bar.setRange(0, 0)
                self.setWindowTitle("Audiobook Generator - ASR check…")
            elif phase == "regen":
                self.chunk_progress_label.setText("Regeneration…")
                self.setWindowTitle("Audiobook Generator - Regeneration…")
            elif phase == "tts":
                self.setWindowTitle("Audiobook Generator - TTS generation…")
            elif phase == "starting":
                self.setWindowTitle("Audiobook Generator - Starting…")
            elif phase == "done":
                self.generation_progress_bar.setRange(0, 100)
                self.setWindowTitle("Audiobook Generator")
            # Continue to update chunk counters if present
            if "current_chunk" not in progress_data and not progress_data.get("asr_status"):
                return

        if 'regeneration' in progress_data:
            event = progress_data['regeneration']
            self.results_text.append(
                f"♻ [REGEN] {event['chunk_num']} attempt "
                f"{event['attempt']}/{event['max_retries']}: "
                f"score={event['score']:.3f}, passed={event['passed']}"
            )
            self.chunk_progress_label.setText("Regeneration…")
            self.setWindowTitle("Audiobook Generator - Regeneration…")
            return

        # Post-gen ASR status (download / load / validate) — keep UI alive
        if progress_data.get("asr_status"):
            msg = progress_data.get("message") or "ASR running…"
            self.chunk_progress_label.setText("ASR check…")
            self.eta_label.setText("ASR: not stuck")
            self.generation_progress_bar.setRange(0, 0)  # indeterminate pulse
            self.setWindowTitle("Audiobook Generator - ASR check…")
            # Throttle log spam: only append when message changes
            if getattr(self, "_last_asr_status_msg", None) != msg:
                self._last_asr_status_msg = msg
                self.results_text.append(f"🔍 [ASR] {msg}")
            return

        print(f"DEBUG: Progress callback received - current={progress_data.get('current_chunk', 0)}, total={progress_data.get('total_chunks', 1)}")
        current = progress_data.get('current_chunk', 0)
        total = progress_data.get('total_chunks', 1)
        elapsed = progress_data.get('elapsed_seconds', 0)

        # Update progress bar
        if total > 0:
            self.generation_progress_bar.setRange(0, 100)
            percentage = int((current / total) * 100)
            self.generation_progress_bar.setValue(percentage)
        else:
            self.generation_progress_bar.setRange(0, 0)  # Indeterminate

        # Update labels
        self.chunk_progress_label.setText(f"Chunks: {current}/{total}")

        elapsed_str = f"{elapsed // 60}:{elapsed % 60:02d}"
        self.time_elapsed_label.setText(f"Elapsed: {elapsed_str}")

        if 'eta_seconds' in progress_data:
            eta = progress_data['eta_seconds']
            eta_str = f"{eta // 60}:{eta % 60:02d}"
            self.eta_label.setText(f"ETA: {eta_str}")
        else:
            self.eta_label.setText("ETA: --:--")

        # Update window title with progress
        if total > 0:
            self.setWindowTitle(f"Audiobook Generator - TTS {percentage}%")
        else:
            self.setWindowTitle("Audiobook Generator - Processing...")

        # Do NOT call QApplication.processEvents() here. Progress already arrives
        # on the GUI thread via Qt queued signals. processEvents() re-enters the
        # event loop and can activate Stop (focus/Space) or close handlers mid-run,
        # logging a false "cancelled by user" while parallel gen keeps running.

    def on_generation_finished(self, result):
        """Present a finished run while retaining its timing summary in progress UI."""
        # Clean up thread reference safely
        if hasattr(self, 'generation_thread') and self.generation_thread is not None:
            thread = self.generation_thread
            self.generation_thread = None
            thread.wait()
            thread.deleteLater()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if result.get('success', False):
            # Success
            self.results_text.append("\n✓ Audiobook generated successfully!")
            self.results_text.append(f"Output: {result['output_path']}")
            self.last_output_path = result['output_path']
            self.play_btn.setEnabled(True)

            if 'audio_duration' in result:
                duration_seconds = result['audio_duration']
                duration_str = f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}"
                self.results_text.append(f"Duration: {duration_str}")

            def format_clock_time(seconds):
                """Format a wall duration as m:ss or h:mm:ss for GUI lines."""
                total_seconds = max(0, int(round(float(seconds or 0))))
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds_part = divmod(remainder, 60)
                if hours:
                    return f"{hours}:{minutes:02d}:{seconds_part:02d}"
                return f"{minutes}:{seconds_part:02d}"

            if 'realtime_factor' in result:
                speed = result['realtime_factor']
                chunk_time = result.get('chunk_processing_time')
                if chunk_time is not None:
                    self.results_text.append(
                        f"Chunk generation speed: {speed:.1f}x realtime  "
                        f"{format_clock_time(chunk_time)}"
                    )
                else:
                    self.results_text.append(f"Chunk generation speed: {speed:.1f}x realtime")

            if 'total_realtime_factor' in result:
                total_speed = result['total_realtime_factor']
                self.results_text.append(f"End-to-end speed: {total_speed:.1f}x realtime")

            def format_phase_time(seconds):
                """Format a processing-phase duration for the GUI terminal."""
                total_seconds = max(0, int(round(seconds)))
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds_part = divmod(remainder, 60)
                return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"

            self._append_asr_summary_lines(result)

            plan_lines = result.get("regeneration_plan_summary") or []
            if plan_lines:
                self.results_text.append("Regeneration plan:")
                for line in plan_lines:
                    self.results_text.append(f"  {line}")

            if 'chunks_processed' in result and 'total_chunks' in result:
                self.results_text.append(f"Chunks: {result['chunks_processed']}/{result['total_chunks']}")

            if 'processing_time' in result:
                proc_time = result['processing_time']
                time_str = f"{int(proc_time // 60)}:{int(proc_time % 60):02d}"
                self.results_text.append(f"End-to-end processing time: {time_str}")

            completed = int(result.get('chunks_processed', 0))
            total = int(result.get('total_chunks', completed))
            self.generation_progress_bar.setValue(100)
            self.chunk_progress_label.setText(f"Chunks: {completed}/{total}")
            self.time_elapsed_label.setText(
                f"Elapsed: {_format_duration_hms(result.get('processing_time', 0))}"
            )
            self.eta_label.setText("ETA: complete")
            self.completion_timing_label.setText(
                _format_completion_timing_summary(result)
            )

            # If first-run ASR cal was deferred (no sample book), try again now
            QTimer.singleShot(1500, self._maybe_start_asr_worker_calibration)

        else:
            # Error or cancellation
            reason = result.get('reason', 'unknown_error')
            if reason == 'cancelled':
                self.results_text.append("\n⚠️ Generation cancelled by user")
            else:
                self.results_text.append(f"\n❌ Generation failed: {reason}")

            if 'chunks_completed' in result:
                self.results_text.append(f"Chunks completed: {result['chunks_completed']}")

        if not result.get('success', False):
            # A failed run has no complete timing record to retain in the row.
            self.generation_progress_bar.setValue(0)
            self.chunk_progress_label.setText("Chunks: 0/0")
            self.time_elapsed_label.setText("Elapsed: 0:00")
            self.eta_label.setText("ETA: --:--")
            self.completion_timing_label.setText("")
        self.setWindowTitle("Audiobook Generator")

        if result.get('success', False) and result.get('asr_investigation_required'):
            investigation_log = result['asr_investigation_required']
            if investigation_log:
                self._show_asr_investigation_popup(
                    investigation_log,
                    result.get('output_path', ''),
                    result=result,
                )

    def _append_asr_summary_lines(self, result):
        """Append ASR stage timing, models, fail counts, and regeneration counts.

        Fail counts are listed in audit order: Stage 1 → Stage 2 → remaining
        after regeneration (successful regens are not counted as remaining).
        """
        def format_phase_time(seconds):
            """Format a processing-phase duration for the GUI terminal."""
            total_seconds = max(0, int(round(seconds)))
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds_part = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"

        stage_one_time = result.get("asr_stage_one_time")
        stage_two_time = result.get("asr_stage_two_time")
        stage_one_model = result.get("asr_stage_one_model")
        stage_two_model = result.get("asr_stage_two_model")
        regen_model = result.get("asr_regeneration_model")
        regen_files = result.get("regeneration_file_count")
        regen_tts_workers = result.get("regeneration_tts_workers")
        regen_asr_workers = result.get("regeneration_asr_workers")
        regen_batch_size = result.get("regeneration_batch_size")
        regen_batch_count = result.get("regeneration_batch_count")
        medium_summary = result.get("medium_verification_summary") or {}
        stage_one_fails = result.get("asr_stage_one_fails")
        stage_two_fails = result.get("asr_stage_two_fails")
        remaining_fails = result.get("asr_remaining_fails")
        if stage_one_fails is None:
            stage_one_fails = medium_summary.get("attempted")
            if stage_one_fails is None:
                stage_one_fails = medium_summary.get("stage_one_candidates")
        if stage_two_fails is None:
            stage_two_fails = medium_summary.get("verified_fail")
            if stage_two_fails is None:
                stage_two_fails = medium_summary.get("medium_confirmed_failures")
        if remaining_fails is None:
            investigation = result.get("asr_investigation_required") or []
            if isinstance(investigation, list):
                remaining_fails = len(investigation)

        if stage_one_time is not None or stage_two_time is not None:
            self.results_text.append("ASR stage times:")
            if stage_one_time is not None:
                self.results_text.append(
                    f"  Stage 1: {format_phase_time(stage_one_time)}"
                )
            if stage_two_time is not None:
                self.results_text.append(
                    f"  Stage 2: {format_phase_time(stage_two_time)}"
                )
        if stage_one_model or stage_two_model or regen_model:
            self.results_text.append("ASR models:")
            if stage_one_model:
                self.results_text.append(f"  Stage 1: {stage_one_model}")
            if stage_two_model:
                self.results_text.append(f"  Stage 2: {stage_two_model}")
            if regen_model:
                self.results_text.append(f"  Regeneration: {regen_model}")

        if (
            stage_one_fails is not None
            or stage_two_fails is not None
            or remaining_fails is not None
        ):
            self.results_text.append("ASR fails:")
            if stage_one_fails is not None:
                self.results_text.append(f"  Stage 1: {int(stage_one_fails)}")
            if stage_two_fails is not None:
                self.results_text.append(f"  Stage 2: {int(stage_two_fails)}")
            if remaining_fails is not None:
                self.results_text.append(
                    f"  Remaining after regen: {int(remaining_fails)}"
                )

        if regen_files is not None or regen_tts_workers is not None or regen_asr_workers is not None:
            self.results_text.append("Regeneration workload:")
            if regen_files is not None:
                self.results_text.append(f"  Files regenerated: {regen_files}")
            if regen_tts_workers is not None or regen_asr_workers is not None:
                self.results_text.append(
                    "  Workers: "
                    f"TTS {regen_tts_workers if regen_tts_workers is not None else 0}, "
                    f"ASR {regen_asr_workers if regen_asr_workers is not None else 0}"
                )
            if regen_batch_count is not None:
                batch_text = f"  Batches: {regen_batch_count}"
                if regen_batch_size:
                    batch_text += f" (batch size {regen_batch_size})"
                self.results_text.append(batch_text)

    def _show_asr_investigation_popup(self, investigation_log: list, output_path: str, result=None):
        """Show popup when final regenerated chunks still failed ASR threshold."""
        from pathlib import Path

        result = result or {}
        chunk_count = len(investigation_log) if investigation_log is not None else 0
        remaining = result.get("asr_remaining_fails")
        if remaining is None:
            remaining = chunk_count
        stage_one_fails = result.get("asr_stage_one_fails")
        stage_two_fails = result.get("asr_stage_two_fails")
        medium_summary = result.get("medium_verification_summary") or {}
        if stage_one_fails is None:
            stage_one_fails = medium_summary.get("attempted") or medium_summary.get(
                "stage_one_candidates"
            )
        if stage_two_fails is None:
            stage_two_fails = medium_summary.get("verified_fail") or medium_summary.get(
                "medium_confirmed_failures"
            )

        tts_dir = Path(output_path).parent / "TTS" if output_path else Path(".")
        log_path = tts_dir / "asr_investigation.log"

        lines = [
            f"Stage 1 fails: {stage_one_fails if stage_one_fails is not None else '?'}",
            f"Stage 2 fails: {stage_two_fails if stage_two_fails is not None else '?'}",
            f"Remaining after regen: {int(remaining)}",
            "",
            "Remaining count excludes chunks fixed by regeneration.",
            "",
            f"See investigation log at:\n{log_path}",
        ]

        msg = QMessageBox(self)
        msg.setWindowTitle("ASR Quality Investigation Required")
        msg.setIcon(QMessageBox.Warning)
        msg.setText(
            f"{int(remaining)} final failed chunk(s) remained below ASR quality threshold."
        )
        msg.setInformativeText("\n".join(lines))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def play_last_audio(self):
        """Play the last generated audiobook file."""
        if not self.last_output_path:
            return

        if not Path(self.last_output_path).exists():
            QMessageBox.warning(self, "File Not Found",
                f"Audio file not found: {Path(self.last_output_path).name}")
            return

        try:
            system = platform.system()
            if system == "Linux":
                subprocess.Popen(["xdg-open", self.last_output_path])
            elif system == "Darwin":
                subprocess.Popen(["open", self.last_output_path])
            elif system == "Windows":
                os.startfile(self.last_output_path)

        except Exception as e:
            QMessageBox.warning(self, "Playback Error",
                f"Could not play audio: {str(e)}")

    def closeEvent(self, event):
        """Clean up generation thread on window close."""
        if hasattr(self, 'generation_thread') and self.generation_thread is not None and self.generation_thread.isRunning():
            import logging
            logging.getLogger(__name__).info(
                "Main window closeEvent — stopping generation thread"
            )
            self.generation_thread.stop()
            self.generation_thread.wait()
        event.accept()


class GenerationThread(QThread):
    """Background thread for audiobook generation."""

    progress = Signal(dict)
    finished = Signal(dict)  # Changed to emit result dict

    def __init__(self, chunks, params, config=None, max_workers_override=None):
        """Initializes an audio book generation thread.
        Args:
        chunks (list): List of audio chunks.
        params (dict): Parameters for the audiobook generation.
        config (Config, optional): Configuration object containing settings for parallel processing.
        max_workers_override (int, optional): Override value for maximum workers.
        Returns: None
        """
        super().__init__()
        self.chunks = chunks
        self.params = params
        self.config = config
        self.max_workers_override = max_workers_override
        self.generator = None

        # Debug logging for override pipeline
        print(f"DEBUG: GenerationThread.__init__ - max_workers_override={max_workers_override}")
        if config and hasattr(config, 'parallel'):
            print(f"DEBUG: GenerationThread.__init__ - config.parallel={config.parallel}")

    def run(self):
        """Run audiobook generation."""
        try:
            print(f"DEBUG: GenerationThread.run() - max_workers_override={self.max_workers_override}")
            if self.config:
                print(f"DEBUG: GenerationThread.run() - config.parallel={self.config.parallel}")

            # Apply max_workers override to config if set
            if self.max_workers_override is not None:
                print(f"DEBUG: Applying override to config: setting max_workers to {self.max_workers_override}")
                import dataclasses

                # Convert config to dict using dataclasses (gets all fields automatically)
                config_dict = dataclasses.asdict(self.config)

                # Remove private attributes that shouldn't be passed to __init__
                config_dict.pop('_config_path', None)

                # Apply the max_workers override
                config_dict['parallel']['max_workers'] = self.max_workers_override

                from pocket_tts.preprocessing.schema import Config
                self.config = Config(**config_dict)
                print(f"DEBUG: Modified config parallel.max_workers={self.config.parallel.get('max_workers')}")

            # Use the modified config
            print(f"DEBUG: Creating AudiobookGenerator with config parallel.max_workers={self.config.parallel.get('max_workers', 'not found')}")
            self.generator = AudiobookGeneratorEngine(config=self.config)

            # Set pause injection parameters on the generator
            self.generator._pause_injection_enabled = self.params.get('pause_injection_enabled', False)
            self.generator._pause_durations = self.params.get('pause_durations', {})

            # Progress callback
            def progress_callback(progress_data):
                """Emits progress data for a long-running audiobook generation task.
                Args:
                progress_data (dict): Data containing progress information.
                Returns:
                None
                """
                self.progress.emit(progress_data)

            # Generate audiobook
            result = self.generator.generate_audiobook(
                chunks=self.chunks,
                voice_path=self.params['voice_path'],
                output_path=self.params['output_path'],
                progress_callback=progress_callback,
                source_file=self.params.get('source_file', 'unknown'),
                save_dataset_chunks=True,
                total_start_time=self.params.get('total_start_time'),
                run_settings=self.params,
            )

            self.finished.emit(result)

        except Exception as e:
            error_result = {
                'success': False,
                'reason': str(e)
            }
            self.finished.emit(error_result)

    def stop(self):
        """Stop generation.
        Args:
        self (object): The instance of the class.
        Returns: None
        """
        """Stop generation."""
        if self.generator:
            self.generator.cancel_generation()


def _setup_app_log():
    """Attach app log at GUI launch (fresh file). Each generation resets again."""
    from pocket_tts.utils.utils import reset_pocket_tts_app_log

    reset_pocket_tts_app_log()


def main():
    """Main entry point for GUI application."""
    _setup_app_log()
    app = QApplication(sys.argv)
    app.setApplicationName("Audiobook Generator")
    app.setApplicationVersion("1.0.0")

    # Set up styling
    app.setStyle("Fusion")

    # Dark theme palette (optional)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    # Tooltips use palette role colors; keep them readable on dark theme.
    palette.setColor(QPalette.ToolTipBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = AudiobookGenerator()
    # Maximize on open. Some WMs ignore the first maximize until the window is
    # mapped, so re-assert after the event loop starts.
    window.showMaximized()
    window.setWindowState(window.windowState() | Qt.WindowMaximized)

    def _ensure_maximized() -> None:
        """Force maximized state after the window manager maps the frame."""
        if not window.isMaximized():
            window.showMaximized()
        window.setWindowState(window.windowState() | Qt.WindowMaximized)

    QTimer.singleShot(0, _ensure_maximized)
    QTimer.singleShot(100, _ensure_maximized)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
