#!/usr/bin/env python3
"""Standalone ASR batch GUI for PocketGPU.

Mimics the Turbo/standalone ASR validator layout:
  - Browse a book or TTS folder
  - Optionally Process ASR Fails from any JSON report
    (``asr_new_failures.json``, ``asr_new_medium_failures.json``,
    legacy ``asr_failures.json``, or another selected ``*.json`` / ``*.*`` file)
  - Dropdowns: backend, model, device (CPU/GPU/Auto), language
  - Workers spin
  - Run Validation
  - All logs written into the selected book/folder

Launch::

    ASR/venv/bin/python ASR/asr_gui.py
    # or: ./ASR/run.sh
"""

from __future__ import annotations

import json
import logging
import queue
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Local imports (run from ASR/ or project root)
_ASR_DIR = Path(__file__).resolve().parent
if str(_ASR_DIR) not in sys.path:
    sys.path.insert(0, str(_ASR_DIR))

from asr_validator import (  # noqa: E402
    cleanup_asr_model,
    discover_chunks,
    format_time,
    load_asr_model_adaptive,
    run_pipeline_batch_validation,
    validate_single_chunk,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKENDS = [
    "faster_whisper",  # CTranslate2; supports distil-* HuggingFace models
    "whisper_cpp",     # pywhispercpp / ggml (maps distil-* → nearest .en size)
]


def cpp_cuda_build_available() -> bool:
    """Return True if installed pywhispercpp includes a CUDA ggml backend."""
    try:
        from cpp_cuda_setup import cpp_cuda_build_available as _probe

        return bool(_probe(sys.executable))
    except Exception:
        return False


def ask_and_build_cpp_cuda(parent, log_fn=None, status_fn=None) -> bool:
    """Ask user to build CUDA pywhispercpp; on Yes run the build with progress.

    Args:
        parent: Tk parent for dialogs.
        log_fn: Optional callable(str) for log lines.
        status_fn: Optional callable(str) for status bar.

    Returns:
        True if CUDA build is present after this call; False if No or build failed.
    """
    if cpp_cuda_build_available():
        return True

    msg = (
        "whisper.cpp GPU support is not installed yet.\n\n"
        "YES = build it from source now (GGML_CUDA).\n"
        "  • Often 5–15+ minutes\n"
        "  • Needs CUDA toolkit (nvcc) and may fail\n"
        "  • Progress appears in the log\n\n"
        "NO = cancel. Use backend faster_whisper + device cuda for GPU ASR "
        "(recommended; no source build).\n\n"
        "Build whisper.cpp GPU support now?"
    )
    if not messagebox.askyesno(
        "Build whisper.cpp GPU?",
        msg,
        parent=parent,
        icon="warning",
    ):
        if log_fn:
            log_fn("↩ Build cancelled. Use faster_whisper + cuda for GPU ASR.")
        return False

    if log_fn:
        log_fn("🔧 Building CUDA pywhispercpp — do not close this window…")
    if status_fn:
        status_fn("Building whisper.cpp CUDA (5–15+ min)…")

    try:
        from cpp_cuda_setup import build_cpp_cuda

        def _line(line: str) -> None:
            if log_fn:
                log_fn(f"   {line}")
            if parent is not None:
                try:
                    parent.update_idletasks()
                except Exception:
                    pass

        ok, result_msg = build_cpp_cuda(
            progress_cb=_line,
            python_exe=sys.executable,
        )
        if not ok:
            messagebox.showerror(
                "Build failed",
                f"{result_msg}\n\nUse faster_whisper + cuda for GPU ASR.",
                parent=parent,
            )
            if log_fn:
                log_fn(f"❌ CUDA build failed: {result_msg}")
            return False
        if log_fn:
            log_fn(f"✅ {result_msg}")
        if status_fn:
            status_fn("CUDA build complete")
        return True
    except Exception as exc:
        messagebox.showerror(
            "Build failed",
            f"{exc}\n\nUse faster_whisper + cuda for GPU ASR.",
            parent=parent,
        )
        if log_fn:
            log_fn(f"❌ CUDA build error: {exc}")
        return False

# faster-whisper names. Distil requires download on first use (~HF hub).
MODELS = [
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

# whisper.cpp has no distil ggml in pywhispercpp map — GUI still lists same
# models; backend maps distil → small.en / medium.en / large-v3.

DEVICES = ["cpu", "cuda", "auto"]

LANGUAGES = [
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Japanese", "ja"),
    ("Chinese", "zh"),
    ("Korean", "ko"),
    ("Auto-detect", None),
]

CONFIG_FILE = _ASR_DIR / "asr_gui_config.json"

# Known reports searched when Process ASR Fails is on and no file was picked.
# Newer New-pipeline files are preferred over the legacy Stage 1 list.
KNOWN_FAILURE_REPORTS = (
    "asr_new_medium_failures.json",
    "asr_new_failures.json",
    "asr_failures.json",
)

# File-dialog filters: show every JSON first; user can switch to *.*
FAILURE_FILE_TYPES = (
    ("JSON files", "*.json"),
    ("All files", "*.*"),
)


def resolve_paths(selected: Path) -> Tuple[Optional[Path], Optional[Path], str]:
    """Resolve TTS dir (audio/text) and log dir from user selection.

    Args:
        selected: Folder the user browsed (book root, TTS, or audio_chunks).

    Returns:
        (tts_dir, log_dir, error_message). log_dir is where logs are written
        (the folder the user selected, or its parent book root when possible).
    """
    p = selected.resolve()
    if not p.is_dir():
        return None, None, "Selected path is not a folder"

    # User picked audio_chunks directly
    if p.name == "audio_chunks" and (p.parent / "text_chunks").is_dir():
        tts = p.parent
        # Prefer book root if structure is Book/TTS/audio_chunks
        log_dir = tts.parent if tts.name == "TTS" else tts
        return tts, log_dir, ""

    # User picked TTS folder
    if (p / "audio_chunks").is_dir() and (p / "text_chunks").is_dir():
        log_dir = p.parent if p.name == "TTS" else p
        return p, log_dir, ""

    # User picked book root with TTS/
    if (p / "TTS" / "audio_chunks").is_dir() and (p / "TTS" / "text_chunks").is_dir():
        return p / "TTS", p, ""

    return None, None, (
        "Folder must contain audio_chunks + text_chunks, "
        "or a TTS/ subfolder with those, or be a book root."
    )


def chunk_id_from_failure_entry(entry: Any) -> Optional[str]:
    """Extract a canonical ``chunk_NNNNN`` id from one failure record.

    Args:
        entry: One JSON object from a list report or a ``records`` row.

    Returns:
        Canonical chunk id, or ``None`` when the row has no usable identity.
    """
    if not isinstance(entry, dict):
        return None

    raw_chunk_id = entry.get("chunk_id")
    if isinstance(raw_chunk_id, str) and raw_chunk_id.startswith("chunk_"):
        return Path(raw_chunk_id).stem

    chunk_index = entry.get("chunk_index")
    if chunk_index is not None:
        try:
            return f"chunk_{int(chunk_index):05d}"
        except Exception:
            pass

    for key in ("filename", "audio_path"):
        value = entry.get(key, "")
        stem = Path(value).stem if value else ""
        if stem.startswith("chunk_"):
            return stem
    return None


def failure_records_from_payload(payload: Any) -> Tuple[List[Any], Optional[str]]:
    """Return failure rows from a list report or a ``{records: [...]}`` object.

    Args:
        payload: Parsed JSON from an ASR failure or verification report.

    Returns:
        Tuple of `(records, error_message)`. ``error_message`` is ``None``
        when the payload is a list or an object containing a records list.
    """
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"], None
    return [], "must be a JSON list or an object containing a records list"


def default_failure_report(search_dirs: List[Path]) -> Optional[Path]:
    """Return the first known failure report found in the given directories.

    Args:
        search_dirs: Directories to search, typically the TTS folder then the
            book-root log directory. Earlier names in
            ``KNOWN_FAILURE_REPORTS`` win.

    Returns:
        Path to the first existing known report, or ``None`` if none exist.
    """
    seen: set[Path] = set()
    for directory in search_dirs:
        if directory is None:
            continue
        resolved = Path(directory)
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        for name in KNOWN_FAILURE_REPORTS:
            candidate = resolved / name
            if candidate.is_file():
                return candidate
    return None


def load_failure_chunk_ids(failure_path: Path) -> Tuple[List[str], Optional[str]]:
    """Load ordered chunk ids from a selected ASR failure JSON file.

    Accepts the legacy Stage 1 list (``asr_failures.json``,
    ``asr_new_failures.json``) and the New-pipeline object report
    (``asr_new_medium_failures.json`` with a ``records`` list).

    Args:
        failure_path: JSON file chosen by the user or auto-detected in the
            TTS / book folder.

    Returns:
        Tuple of `(chunk_ids, error_message)`. `chunk_ids` is de-duplicated and
        ordered as stored in the file. `error_message` is `None` on success.
    """
    report = Path(failure_path)
    if not report.is_file():
        return [], f"No failure report found at {report}"

    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], f"Could not read {report.name}: {exc}"

    records, shape_err = failure_records_from_payload(payload)
    if shape_err:
        return [], f"{report.name} {shape_err}"

    chunk_ids: List[str] = []
    seen = set()
    for entry in records:
        chunk_id = chunk_id_from_failure_entry(entry)
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)

    return chunk_ids, None


def load_gui_config() -> dict:
    """Load last-used GUI settings from asr_gui_config.json."""
    try:
        if CONFIG_FILE.is_file():
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_gui_config(data: dict) -> None:
    """Persist GUI settings to asr_gui_config.json."""
    try:
        existing = load_gui_config()
        existing.update(data)
        CONFIG_FILE.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


class QueueLogHandler(logging.Handler):
    """Push log records into a queue for the Tk UI thread."""

    def __init__(self, q: queue.Queue):
        """Attach handler to the given UI queue."""
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        """Enqueue one formatted log line for the GUI."""
        try:
            self.q.put({"type": "log", "message": self.format(record)})
        except Exception:
            pass


class AsrGuiApp:
    """Tkinter ASR batch validator: pick book, model, device, workers, run."""

    def __init__(self, root: tk.Tk) -> None:
        """Build window, load last settings, wire widgets."""
        self.root = root
        self.root.title("PocketGPU ASR Validator")
        self.root.geometry("900x680")

        cfg = load_gui_config()
        self.selected_folder = Path(cfg.get("last_folder", str(Path.home())))
        self.tts_dir: Optional[Path] = None
        self.log_dir: Optional[Path] = None
        saved_failure = cfg.get("last_failure_file")
        self.failure_report_path: Optional[Path] = (
            Path(saved_failure) if saved_failure else None
        )
        if self.failure_report_path and not self.failure_report_path.is_file():
            self.failure_report_path = None
        self._failure_user_picked = bool(self.failure_report_path)
        self.progress_queue: queue.Queue = queue.Queue()
        self.running = False
        self._stop = threading.Event()

        self._build_ui()
        self._apply_folder(self.selected_folder, silent=True)
        self.root.after(150, self._poll_queue)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        """Create all controls (folder, settings, run, log)."""
        # --- Folder ---
        folder_fr = ttk.LabelFrame(self.root, text="Book / TTS folder", padding=10)
        folder_fr.pack(fill="x", padx=10, pady=6)

        ttk.Label(folder_fr, text="Selected:").grid(row=0, column=0, sticky="w")
        self.folder_var = tk.StringVar(value=str(self.selected_folder))
        self.folder_entry = ttk.Entry(folder_fr, textvariable=self.folder_var, width=70, state="readonly")
        self.folder_entry.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(folder_fr, text="Browse…", command=self._browse).grid(row=0, column=2)
        folder_fr.columnconfigure(1, weight=1)

        self.path_hint = ttk.Label(folder_fr, text="", foreground="#555")
        self.path_hint.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # --- Settings ---
        set_fr = ttk.LabelFrame(self.root, text="ASR settings", padding=10)
        set_fr.pack(fill="x", padx=10, pady=6)

        cfg = load_gui_config()

        ttk.Label(set_fr, text="Backend:").grid(row=0, column=0, sticky="w", pady=2)
        self.backend_var = tk.StringVar(value=cfg.get("backend", "faster_whisper"))
        self.backend_combo = ttk.Combobox(
            set_fr, textvariable=self.backend_var, values=BACKENDS, state="readonly", width=18
        )
        self.backend_combo.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.backend_combo.bind("<<ComboboxSelected>>", self._on_backend_or_device_change)

        ttk.Label(set_fr, text="Model:").grid(row=0, column=2, sticky="w", padx=(15, 0))
        self.model_var = tk.StringVar(value=cfg.get("model", "base"))
        self.model_combo = ttk.Combobox(
            set_fr, textvariable=self.model_var, values=MODELS, state="readonly", width=18
        )
        self.model_combo.grid(row=0, column=3, sticky="w", padx=5, pady=2)

        ttk.Label(set_fr, text="Device:").grid(row=1, column=0, sticky="w", pady=2)
        self.device_var = tk.StringVar(value=cfg.get("device", "cuda"))
        self.device_combo = ttk.Combobox(
            set_fr, textvariable=self.device_var, values=DEVICES, state="readonly", width=18
        )
        self.device_combo.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        self.device_combo.bind("<<ComboboxSelected>>", self._on_backend_or_device_change)
        # Remember last confirmed backend/device so Cancel can restore
        self._prev_backend = self.backend_var.get()
        self._prev_device = self.device_var.get()
        self._cpp_gpu_build_ok = None  # cached probe; None = not checked yet

        ttk.Label(set_fr, text="Workers:").grid(row=1, column=2, sticky="w", padx=(15, 0))
        self.workers_var = tk.IntVar(value=int(cfg.get("workers", 4)))
        self.workers_spin = ttk.Spinbox(
            set_fr, from_=1, to=16, textvariable=self.workers_var, width=6
        )
        self.workers_spin.grid(row=1, column=3, sticky="w", padx=5, pady=2)

        ttk.Label(set_fr, text="Language:").grid(row=2, column=0, sticky="w", pady=2)
        self.lang_var = tk.StringVar(value=cfg.get("language_label", "English"))
        self.lang_combo = ttk.Combobox(
            set_fr,
            textvariable=self.lang_var,
            values=[lab for lab, _ in LANGUAGES],
            state="readonly",
            width=18,
        )
        self.lang_combo.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(set_fr, text="Threshold:").grid(row=2, column=2, sticky="w", padx=(15, 0))
        self.threshold_var = tk.DoubleVar(value=float(cfg.get("threshold", 0.75)))
        self.threshold_spin = ttk.Spinbox(
            set_fr,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.threshold_var,
            width=6,
            format="%.2f",
        )
        self.threshold_spin.grid(row=2, column=3, sticky="w", padx=5, pady=2)

        self.process_fails_var = tk.BooleanVar(
            value=bool(cfg.get("process_asr_fails", False))
        )
        self.process_fails_check = ttk.Checkbutton(
            set_fr,
            text="Process ASR Fails",
            variable=self.process_fails_var,
            command=self._on_process_fails_toggle,
        )
        self.process_fails_check.grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))

        ttk.Label(set_fr, text="Failure report:").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.failure_report_var = tk.StringVar(
            value=str(self.failure_report_path) if self.failure_report_path else ""
        )
        self.failure_report_entry = ttk.Entry(
            set_fr,
            textvariable=self.failure_report_var,
            width=50,
            state="readonly",
        )
        self.failure_report_entry.grid(
            row=4, column=1, columnspan=2, sticky="ew", padx=5, pady=(6, 0)
        )
        self.failure_browse_btn = ttk.Button(
            set_fr, text="Browse…", command=self._browse_failure_report
        )
        self.failure_browse_btn.grid(row=4, column=3, sticky="w", padx=5, pady=(6, 0))
        set_fr.columnconfigure(1, weight=1)

        self.log_filter_var = tk.StringVar(value=cfg.get("log_filter", "Both"))
        ttk.Label(set_fr, text="Log Filter:").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.log_filter_combo = ttk.Combobox(
            set_fr,
            textvariable=self.log_filter_var,
            values=["Both", "Pass Only", "Fail Only"],
            state="readonly",
            width=18,
        )
        self.log_filter_combo.grid(row=5, column=1, sticky="w", padx=5, pady=(6, 0))

        # --- Actions ---
        act = ttk.Frame(self.root, padding=5)
        act.pack(fill="x", padx=10)
        self.run_btn = ttk.Button(act, text="Run Validation", command=self._start_run)
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.stop_btn = ttk.Button(act, text="Stop", command=self._request_stop, state="disabled")
        self.stop_btn.pack(side="left")

        # --- Status ---
        st = ttk.LabelFrame(self.root, text="Status", padding=8)
        st.pack(fill="x", padx=10, pady=4)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(st, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x")
        self.progress = ttk.Progressbar(st, mode="determinate")
        self.progress.pack(fill="x", pady=4)
        self.download_var = tk.StringVar(value="")
        ttk.Label(st, textvariable=self.download_var, relief="sunken", anchor="w").pack(fill="x", pady=2)
        self.download_bar = ttk.Progressbar(st, mode="determinate", maximum=100)
        self.download_bar.pack(fill="x", pady=2)
        self.stats_var = tk.StringVar(value="")
        ttk.Label(st, textvariable=self.stats_var, anchor="w").pack(fill="x")

        # --- Log ---
        log_fr = ttk.LabelFrame(self.root, text="Log", padding=8)
        log_fr.pack(fill="both", expand=True, padx=10, pady=6)
        self.log_text = tk.Text(log_fr, height=18, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(log_fr, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _browse(self) -> None:
        """Pick book or TTS folder and resolve audio/text paths."""
        initial = str(self.selected_folder) if self.selected_folder.is_dir() else str(Path.home())
        path = filedialog.askdirectory(title="Select book or TTS folder", initialdir=initial)
        if path:
            # New book folder: pick a known report there instead of keeping
            # a previous book's auto-detected file.
            self._failure_user_picked = False
            self._apply_folder(Path(path), silent=False)

    def _failure_dialog_initial_dir(self) -> str:
        """Directory to open in the failure-report file dialog.

        Returns:
            Existing TTS folder, selected folder, or the home directory.
        """
        for candidate in (
            self.tts_dir,
            self.failure_report_path.parent if self.failure_report_path else None,
            self.selected_folder,
            Path.home(),
        ):
            if candidate is not None and Path(candidate).is_dir():
                return str(candidate)
        return str(Path.home())

    def _browse_failure_report(self) -> None:
        """Pick any JSON failure report; dialog can switch the filter to *.*."""
        selected = filedialog.askopenfilename(
            title="Select ASR failure report",
            initialdir=self._failure_dialog_initial_dir(),
            initialfile=(
                self.failure_report_path.name if self.failure_report_path else ""
            ),
            filetypes=list(FAILURE_FILE_TYPES),
        )
        if not selected:
            return
        self._apply_failure_report(Path(selected), silent=False)

    def _apply_failure_report(self, path: Path, silent: bool = False) -> None:
        """Use the chosen failure JSON and resolve its book/TTS folder when possible.

        Args:
            path: Failure or verification report selected in the file dialog.
            silent: When True, skip warning dialogs for folder-resolution issues.
        """
        report = path.expanduser().resolve()
        self.failure_report_path = report
        self.failure_report_var.set(str(report))
        self._failure_user_picked = True

        folder_for_audio = report.parent
        tts, _log_dir, err = resolve_paths(folder_for_audio)
        if err:
            tts, _log_dir, err = resolve_paths(folder_for_audio.parent)
            if not err and tts is not None:
                folder_for_audio = folder_for_audio.parent
        if not err and tts is not None:
            self._apply_folder(folder_for_audio, silent=silent)
            return
        if self.tts_dir is not None:
            self._apply_folder(self.selected_folder, silent=True)
            return
        self.path_hint.config(
            text="Selected a failure JSON, but still need a book/TTS folder with audio/text chunks.",
            foreground="#a00",
        )
        self.run_btn.config(state="disabled")

    def _sync_failure_report_for_folder(self) -> None:
        """Keep or auto-select a failure JSON after the book/TTS folder changes."""
        if not self.process_fails_var.get():
            return
        # Keep an explicit file-dialog choice even if it lives outside TTS/.
        if (
            self._failure_user_picked
            and self.failure_report_path is not None
            and self.failure_report_path.is_file()
        ):
            self.failure_report_var.set(str(self.failure_report_path))
            return
        search_dirs = [p for p in (self.tts_dir, self.log_dir) if p is not None]
        found = default_failure_report(search_dirs)
        self.failure_report_path = found
        self.failure_report_var.set(str(found) if found else "")
        self._failure_user_picked = False

    def _set_failure_widgets_enabled(self, enabled: bool) -> None:
        """Enable or disable the failure-report path and its file browser.

        Args:
            enabled: True when Process ASR Fails is checked.
        """
        state = "normal" if enabled else "disabled"
        entry_state = "readonly" if enabled else "disabled"
        self.failure_report_entry.config(state=entry_state)
        self.failure_browse_btn.config(state=state)

    def _persist_folder_settings(self, path: Path) -> None:
        """Save the current folder and failure-report selections.

        Args:
            path: Book or TTS folder shown in the folder field.
        """
        save_gui_config(
            {
                "last_folder": str(path),
                "process_asr_fails": bool(self.process_fails_var.get()),
                "log_filter": self.log_filter_var.get(),
                "last_failure_file": (
                    str(self.failure_report_path) if self.failure_report_path else ""
                ),
            }
        )

    def _apply_folder(self, path: Path, silent: bool = False) -> None:
        """Set selected folder, resolve TTS/log dirs, update UI."""
        self.selected_folder = path
        self.folder_var.set(str(path))
        tts, log_dir, err = resolve_paths(path)
        self.tts_dir = tts
        self.log_dir = log_dir
        if err:
            self.path_hint.config(text=err, foreground="#a00")
            self.run_btn.config(state="disabled")
            if not silent:
                messagebox.showwarning("Folder", err)
            return
        self._set_failure_widgets_enabled(self.process_fails_var.get())
        self._sync_failure_report_for_folder()
        chunk_ids, chunk_err, chunk_warn = self._current_chunk_ids()
        if chunk_err:
            self.path_hint.config(text=chunk_err, foreground="#a00")
            self.run_btn.config(state="disabled")
            self._persist_folder_settings(path)
            return
        n = len(chunk_ids)
        if self.process_fails_var.get() and self.failure_report_path:
            hint_text = (
                f"TTS: {tts}  |  Logs → {log_dir}  |  {n} chunk"
                f"{'s' if n != 1 else ''} from {self.failure_report_path.name}"
            )
        else:
            hint_text = f"TTS: {tts}  |  Logs → {log_dir}  |  {n} chunk pair{'s' if n != 1 else ''}"
        self.path_hint.config(
            text=hint_text,
            foreground="#060",
        )
        self.run_btn.config(state="normal")
        if chunk_warn and not silent:
            self._append_log(f"⚠ {chunk_warn}")
        self._persist_folder_settings(path)

    def _current_chunk_ids(self) -> Tuple[List[str], Optional[str], Optional[str]]:
        """Resolve chunk ids for the current folder and checkbox state.

        Returns:
            A tuple of `(chunk_ids, error_message, warning_message)`. When the
            checkbox is off, chunk ids come from matching audio/text pairs. When
            on, ids come from the selected failure JSON (any ``*.json``, including
            ``asr_new_failures.json`` and ``asr_new_medium_failures.json``) and
            are intersected with available chunk pairs so missing files do not
            crash the run.
        """
        if not self.tts_dir:
            return [], "Select a valid book/TTS folder first.", None

        available = discover_chunks(self.tts_dir)
        if not self.process_fails_var.get():
            return available, None, None

        if not self.failure_report_path or not self.failure_report_path.is_file():
            names = ", ".join(KNOWN_FAILURE_REPORTS)
            return (
                [],
                f"Select a failure JSON (*.json, or *.*). Looked for: {names}.",
                None,
            )

        failure_ids, failure_err = load_failure_chunk_ids(self.failure_report_path)
        if failure_err:
            return [], failure_err, None

        available_set = set(available)
        filtered = [chunk_id for chunk_id in failure_ids if chunk_id in available_set]
        missing = len(failure_ids) - len(filtered)
        if not filtered:
            return [], f"No usable chunk ids found in {self.failure_report_path.name}", None
        if missing:
            return filtered, None, f"{missing} failure chunk(s) missing audio/text pairs and will be skipped"
        return filtered, None, None

    def _lang_code(self) -> str:
        """Map language label to Whisper language code (default en)."""
        label = self.lang_var.get()
        for lab, code in LANGUAGES:
            if lab == label:
                return code or "en"
        return "en"

    def _append_log(self, msg: str) -> None:
        """Append a line to the on-screen log widget."""
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _poll_queue(self) -> None:
        """Drain progress_queue onto the UI (status, progress, log)."""
        try:
            while True:
                item = self.progress_queue.get_nowait()
                kind = item.get("type")
                if kind == "status":
                    self.status_var.set(item.get("message", ""))
                elif kind == "progress":
                    cur = item.get("current", 0)
                    total = max(1, item.get("total", 1))
                    self.progress["maximum"] = total
                    self.progress["value"] = cur
                elif kind == "download":
                    msg = item.get("message", "")
                    pct = item.get("percent")
                    self.download_var.set(msg)
                    if pct is None:
                        self.download_bar.config(mode="indeterminate")
                        try:
                            self.download_bar.start(12)
                        except Exception:
                            pass
                    else:
                        try:
                            self.download_bar.stop()
                        except Exception:
                            pass
                        self.download_bar.config(mode="determinate", maximum=100)
                        self.download_bar["value"] = max(0, min(100, float(pct)))
                    self._append_log(f"⬇ {msg}")
                elif kind == "warning":
                    msg = item.get("message", "")
                    self.download_var.set(msg)
                    self.status_var.set(msg)
                    self._append_log(f"⚠ {msg}")
                elif kind == "stats":
                    self.stats_var.set(item.get("message", ""))
                elif kind == "log":
                    self._append_log(item.get("message", ""))
                elif kind == "done":
                    try:
                        self.download_bar.stop()
                    except Exception:
                        pass
                    self.download_bar.config(mode="determinate")
                    self.download_bar["value"] = 0
                    self.running = False
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    summary = item.get("summary") or {}
                    if summary:
                        self.stats_var.set(
                            f"Done: {summary.get('passed', 0)} pass / "
                            f"{summary.get('failed', 0)} fail / "
                            f"{summary.get('total', 0)} total  "
                            f"wall={summary.get('wall_s', 0):.1f}s  "
                            f"logs → {summary.get('log_dir', '')}"
                        )
                    if item.get("error"):
                        messagebox.showerror("ASR", item["error"])
                    elif summary:
                        messagebox.showinfo(
                            "ASR complete",
                            f"Passed: {summary.get('passed')}\n"
                            f"Failed: {summary.get('failed')}\n"
                            f"Wall: {summary.get('wall_s', 0):.1f}s\n"
                            f"Logs: {summary.get('log_dir')}",
                        )
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    def _request_stop(self) -> None:
        """Signal the worker thread to stop after the current chunk batch."""
        self._stop.set()
        self.progress_queue.put({"type": "status", "message": "Stop requested…"})
        self.progress_queue.put({"type": "log", "message": "⏹ Stop requested — finishing in-flight chunks"})

    def _wants_cpp_gpu(self) -> bool:
        """True when backend is whisper_cpp and device requests GPU."""
        backend = (self.backend_var.get() or "").strip()
        device = (self.device_var.get() or "").strip().lower()
        return backend == "whisper_cpp" and device in ("cuda", "gpu", "auto")

    def _ensure_cpp_gpu_ready(self) -> bool:
        """If CPP+GPU and CUDA build missing, ask to build now (Yes) or cancel (No).

        Returns:
            True if CUDA build is available after this call; False if cancelled/failed.
        """
        if not self._wants_cpp_gpu():
            return True
        if self._cpp_gpu_build_ok is None:
            self._cpp_gpu_build_ok = cpp_cuda_build_available()
        if self._cpp_gpu_build_ok:
            return True
        ok = ask_and_build_cpp_cuda(
            parent=self.root,
            log_fn=self._append_log,
            status_fn=lambda m: self.status_var.set(m),
        )
        self._cpp_gpu_build_ok = ok
        return ok

    def _on_backend_or_device_change(self, _event=None) -> None:
        """If user picks CPP+GPU without CUDA build, offer to build now."""
        if self._wants_cpp_gpu():
            if self._cpp_gpu_build_ok is None:
                self._cpp_gpu_build_ok = cpp_cuda_build_available()
            if not self._cpp_gpu_build_ok:
                if not self._ensure_cpp_gpu_ready():
                    self.backend_var.set(self._prev_backend)
                    self.device_var.set(self._prev_device)
                    self._append_log(
                        "↩ Cancelled whisper_cpp+GPU (no CUDA build)."
                    )
                    return
        self._prev_backend = self.backend_var.get()
        self._prev_device = self.device_var.get()

    def _on_process_fails_toggle(self) -> None:
        """Refresh folder hint after the failure-filter checkbox changes."""
        self._set_failure_widgets_enabled(self.process_fails_var.get())
        self._apply_folder(self.selected_folder, silent=True)

    def _start_run(self) -> None:
        """Validate settings and start background batch validation."""
        if self.running:
            return
        if not self.tts_dir or not self.log_dir:
            messagebox.showerror("ASR", "Select a valid book/TTS folder first.")
            return

        backend = self.backend_var.get().strip()
        model = self.model_var.get().strip()
        device = self.device_var.get().strip()
        workers = max(1, min(16, int(self.workers_var.get())))
        threshold = float(self.threshold_var.get())
        language = self._lang_code()

        # CPP + GPU: build CUDA pywhispercpp now if missing (Yes = build, No = abort)
        if not self._ensure_cpp_gpu_ready():
            self.backend_var.set(self._prev_backend)
            self.device_var.set(self._prev_device)
            return

        if backend == "whisper_cpp":
            try:
                import pywhispercpp  # noqa: F401
                import whisper_cpp_backend  # noqa: F401
            except ImportError as exc:
                messagebox.showerror(
                    "Backend",
                    "whisper_cpp not available.\n\n"
                    f"{exc}\n\n"
                    "Install into the app venv:\n"
                    "  pip install pywhispercpp\n"
                    "For GPU CUDA build (or use the Yes dialog on next run):\n"
                    "  ./ASR/install_pywhispercpp_cuda.sh\n",
                )
                return
        if backend == "faster_whisper" and model.startswith("distil"):
            # First distil load may download weights; warn once in log
            self._append_log(
                f"ℹ distil model '{model}' — first run may download weights from HuggingFace"
            )
        if backend == "whisper_cpp":
            # Rough GPU VRAM: medium ~1.5GB weights + KV; large ~3GB. 4× medium OOMs 8GB.
            rough_gb = {
                "tiny": 0.2, "base": 0.3, "small": 0.7, "medium": 1.8, "large-v3": 3.5,
                "large-v3-turbo": 1.0,
                "distil-small.en": 0.7, "distil-medium.en": 1.8, "distil-large-v3": 3.5,
            }.get(model, 1.0)
            est = rough_gb * workers
            if device in ("cuda", "gpu", "auto") and est > 6.5:
                if not messagebox.askyesno(
                    "VRAM warning",
                    f"whisper_cpp + {model} × {workers} workers ≈ {est:.1f} GB VRAM.\n"
                    f"On an 8 GB card this often OOMs or segfaults.\n\n"
                    f"Recommend workers=1 for medium/large.\n\nContinue anyway?",
                ):
                    return

        chunks, chunk_err, chunk_warn = self._current_chunk_ids()
        if chunk_err:
            messagebox.showerror("ASR", chunk_err)
            return
        if not chunks:
            messagebox.showerror("ASR", "No chunk pairs found for the current folder.")
            return
        if chunk_warn:
            self._append_log(f"⚠ {chunk_warn}")

        save_gui_config(
            {
                "last_folder": str(self.selected_folder),
                "backend": backend,
                "model": model,
                "device": device,
                "workers": workers,
                "threshold": threshold,
                "language_label": self.lang_var.get(),
                "process_asr_fails": bool(self.process_fails_var.get()),
                "log_filter": self.log_filter_var.get(),
                "last_failure_file": (
                    str(self.failure_report_path) if self.failure_report_path else ""
                ),
            }
        )

        self.running = True
        self._stop.clear()
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress["value"] = 0
        self.stats_var.set("")
        self._append_log(
            f"▶ Start backend={backend} model={model} device={device} "
            f"workers={workers} threshold={threshold} lang={language}"
        )
        self._append_log(f"   TTS={self.tts_dir}")
        self._append_log(f"   Logs={self.log_dir}")
        if self.process_fails_var.get():
            report_name = (
                self.failure_report_path.name if self.failure_report_path else "failure report"
            )
            self._append_log(
                f"   Mode=process ASR fails from {report_name} "
                f"({len(chunks)} chunk{'s' if len(chunks) != 1 else ''})"
            )

        thread = threading.Thread(
            target=self._run_batch,
            kwargs={
                "tts_dir": self.tts_dir,
                "log_dir": self.log_dir,
                "chunks": chunks,
                "backend": backend,
                "model": model,
                "device": device,
                "workers": workers,
                "threshold": threshold,
                "language": language,
                "log_filter": self.log_filter_var.get(),
            },
            name="asr-gui-batch",
            daemon=True,
        )
        thread.start()

    # -------------------------------------------------------------- batch
    def _run_batch(
        self,
        tts_dir: Path,
        log_dir: Path,
        chunks: List[str],
        backend: str,
        model: str,
        device: str,
        workers: int,
        threshold: float,
        language: str,
        log_filter: str = "Both",
    ) -> None:
        """Run multi-worker validation; write logs under log_dir."""
        q = self.progress_queue
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)

        run_log = log_dir / f"asr_gui_run_{stamp}.log"
        fail_json = log_dir / f"asr_gui_failures_{stamp}.json"
        summary_json = log_dir / f"asr_gui_summary_{stamp}.json"
        # Stable "latest" copies for easy tail
        latest_run = log_dir / "asr_gui_run.log"
        latest_json = log_dir / "asr_gui_failures.json"
        latest_sum = log_dir / "asr_gui_summary.json"

        file_handler = logging.FileHandler(run_log, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        ui_handler = QueueLogHandler(q)
        ui_handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("asr_gui_batch")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(ui_handler)
        # Also capture root asr_validator logs into the same files
        root_log = logging.getLogger()
        root_log.setLevel(logging.INFO)
        root_log.addHandler(file_handler)

        force = None if device == "auto" else device
        if not chunks:
            q.put({"type": "done", "error": "No matching audio/text chunk pairs found."})
            return

        total = len(chunks)
        # faster_whisper on GPU: Phase 1–2 pipeline (1 model). Else multi-copy path.
        use_pipeline = backend == "faster_whisper" and (
            (force or "").lower() in ("cuda", "gpu", "auto", "")
        )
        if use_pipeline:
            q.put({
                "type": "warning",
                "message": (
                    f"Found {total} chunks. Fast pipeline: 1× {backend}/{model} on GPU, "
                    f"pack=8 + solo-retry on fail, {workers} CPU workers, VAD off. "
                    "First-time model download can take minutes — NOT stuck."
                ),
            })
        else:
            q.put({
                "type": "warning",
                "message": (
                    f"Found {total} chunks. Loading {workers}× {backend}/{model} on {device}. "
                    "First-time model downloads can take minutes — progress below; NOT stuck."
                ),
            })
        q.put({"type": "status", "message": "Loading ASR model…"})
        q.put({"type": "progress", "current": 0, "total": total})
        logger.info(
            "Batch start: chunks=%s backend=%s model=%s device=%s workers=%s pipeline=%s",
            total,
            backend,
            model,
            device,
            workers,
            use_pipeline,
        )

        def _dl_cb(event: dict) -> None:
            """Forward model download events to the UI queue."""
            phase = event.get("phase") or "download"
            msg = event.get("message") or ""
            if phase in ("download", "download_start", "download_done", "cache"):
                q.put({
                    "type": "download",
                    "message": msg,
                    "percent": event.get("percent"),
                })
            elif event.get("level") == "warning" or phase == "warning":
                q.put({"type": "warning", "message": msg})
            else:
                q.put({"type": "log", "message": msg})

        models: List[Any] = []

        def load_one(worker_i: int, n_models: int) -> Tuple[Any, str]:
            """Load one ASR model for a worker slot with download progress."""
            import os

            cores = os.cpu_count() or 4
            threads = max(1, cores // max(1, n_models)) if backend == "whisper_cpp" else 2
            q.put({
                "type": "status",
                "message": f"Loading ASR model worker {worker_i + 1}/{n_models}…",
            })
            q.put({
                "type": "log",
                "message": (
                    f"⏳ Worker {worker_i}: load {backend}/{model} on {device or 'auto'} "
                    f"(download if needed)"
                ),
            })
            if worker_i == 0:
                try:
                    if backend == "whisper_cpp":
                        from model_download_progress import ensure_ggml_model
                        from whisper_cpp_backend import _CPP_MODEL_MAP

                        ggml = _CPP_MODEL_MAP.get(model, model)
                        ensure_ggml_model(ggml, progress_cb=_dl_cb)
                    else:
                        from model_download_progress import ensure_faster_whisper_model

                        ensure_faster_whisper_model(model, progress_cb=_dl_cb)
                except Exception as pre_e:
                    q.put({
                        "type": "warning",
                        "message": f"Pre-download warning: {pre_e}",
                    })

            # Pipeline prefers CUDA; force cuda when auto so we stay on GPU path
            force_dev = force
            if use_pipeline and (not force_dev or force_dev == "auto"):
                force_dev = "cuda"
            m, dev = load_asr_model_adaptive(
                model,
                force_device=force_dev,
                engine=backend,
                n_threads=threads,
            )
            return m, dev or "unknown"

        try:
            n_models = 1 if use_pipeline else workers
            for w in range(n_models):
                if self._stop.is_set():
                    break
                m, dev = load_one(w, n_models)
                if m is None:
                    q.put(
                        {
                            "type": "done",
                            "error": f"Failed to load ASR model on worker {w} "
                            f"(backend={backend} model={model} device={device})",
                        }
                    )
                    return
                models.append(m)
                logger.info("Worker model %s loaded on %s", w, dev)
                q.put({
                    "type": "log",
                    "message": f"✅ Worker {w} ready on {dev}",
                })
            q.put({
                "type": "download",
                "message": "Model load complete",
                "percent": 100,
            })
            if not models:
                q.put({"type": "done", "error": "No models loaded"})
                return

            results: List[Dict[str, Any]] = []
            passed_n = 0
            failed_n = 0
            t0 = time.time()
            progress_lock = threading.Lock()

            def _on_progress(done: int, tot: int, result: Dict[str, Any]) -> None:
                """UI + log update after each scored chunk (pipeline path)."""
                nonlocal passed_n, failed_n
                with progress_lock:
                    if result.get("passed"):
                        passed_n += 1
                    else:
                        failed_n += 1
                    p_n, f_n = passed_n, failed_n
                elapsed = time.time() - t0
                eta = (elapsed / done) * (tot - done) if done else 0
                q.put({"type": "progress", "current": done, "total": tot})
                q.put({
                    "type": "status",
                    "message": (
                        f"Validated {done}/{tot} | pass={p_n} fail={f_n} | "
                        f"elapsed={format_time(elapsed)} ETA={format_time(eta)}"
                    ),
                })
                status = "PASS" if result.get("passed") else "FAIL"
                logger.info(
                    "%s %s score=%.3f class=%s",
                    status,
                    result.get("chunk_num"),
                    float(result.get("score") or 0),
                    result.get("classification"),
                )

            if use_pipeline:
                # Split spinbox budget: half loaders, half scorers (min 2 each when workers>=4)
                load_n = max(2, workers // 2)
                score_n = max(2, workers - load_n)
                # Phase 3: pack clips; fails re-solo so accuracy matches Phase 1–2
                pack_size = 8
                pack_silence_s = 1.0
                q.put({
                    "type": "status",
                    "message": (
                        f"Pipeline running: 1× GPU model, pack={pack_size}, "
                        f"solo-retry on fail, {load_n} loaders, {score_n} scorers…"
                    ),
                })
                logger.info(
                    "Pipeline: load_workers=%s score_workers=%s pack_size=%s "
                    "pack_silence_s=%s vad_filter=False solo_retry_on_fail=True",
                    load_n,
                    score_n,
                    pack_size,
                    pack_silence_s,
                )
                results = run_pipeline_batch_validation(
                    tts_dir,
                    chunks,
                    models[0],
                    threshold,
                    language=language,
                    load_workers=load_n,
                    score_workers=score_n,
                    vad_filter=False,
                    pack_size=pack_size,
                    pack_silence_s=pack_silence_s,
                    stop_event=self._stop,
                    progress_cb=_on_progress,
                )
            else:
                free_models: queue.Queue = queue.Queue()
                for m in models:
                    free_models.put(m)

                def work(chunk_num: str) -> Dict[str, Any]:
                    """Validate one chunk using a borrowed ASR model instance."""
                    if self._stop.is_set():
                        return {
                            "chunk_num": chunk_num,
                            "passed": False,
                            "score": 0.0,
                            "error": "stopped",
                            "classification": "FAIL",
                        }
                    m = free_models.get()
                    try:
                        return validate_single_chunk(
                            chunk_num,
                            tts_dir,
                            threshold,
                            m,
                            language=language,
                            vad_filter=False,
                        )
                    finally:
                        free_models.put(m)

                with ThreadPoolExecutor(max_workers=len(models)) as pool:
                    futures = {pool.submit(work, c): c for c in chunks}
                    done_count = 0
                    for fut in as_completed(futures):
                        if self._stop.is_set():
                            for f in futures:
                                f.cancel()
                        try:
                            result = fut.result()
                        except Exception as e:
                            result = {
                                "chunk_num": futures[fut],
                                "passed": False,
                                "score": 0.0,
                                "error": str(e),
                                "classification": "FAIL",
                            }
                            logger.error("Chunk error %s: %s", futures[fut], e)

                        results.append(result)
                        done_count += 1
                        _on_progress(done_count, total, result)

            wall = time.time() - t0
            stopped = self._stop.is_set()
            # Recompute pass/fail from results (pipeline already tallied; keep consistent)
            passed_n = sum(1 for r in results if r.get("passed"))
            failed_n = len(results) - passed_n

            # Report filenames: backend + model + _Fail## (no timestamp)
            _bm = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{backend}_{model}")
            _fail_tag = f"_Fail{failed_n:02d}"
            validation_log = log_dir / f"asr_gui_validation_{_bm}{_fail_tag}.log"
            fail_log = log_dir / f"asr_gui_failures_{_bm}{_fail_tag}.log"

            self._write_validation_logs(
                validation_log,
                fail_log,
                fail_json,
                results,
                threshold,
                backend,
                model,
                device,
                workers,
                language,
                wall,
                stopped,
                passed_n,
                failed_n,
                log_filter,
            )
            summary = {
                "timestamp": stamp,
                "tts_dir": str(tts_dir),
                "log_dir": str(log_dir),
                "backend": backend,
                "model": model,
                "device": device,
                "workers": workers,
                "pipeline": use_pipeline,
                "threshold": threshold,
                "language": language,
                "total": len(results),
                "passed": passed_n,
                "failed": failed_n,
                "wall_s": round(wall, 2),
                "chunks_per_s": round(len(results) / wall, 3) if wall > 0 else 0,
                "stopped": stopped,
                "files": {
                    "run_log": str(run_log),
                    "validation_log": str(validation_log),
                    "failures_log": str(fail_log),
                    "failures_json": str(fail_json),
                    "summary_json": str(summary_json),
                },
            }
            summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

            for src, dst in [
                (run_log, latest_run),
                (fail_json, latest_json),
                (summary_json, latest_sum),
            ]:
                try:
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                except Exception:
                    pass

            logger.info(
                "Batch done: pass=%s fail=%s wall=%.1fs stopped=%s pipeline=%s logs=%s",
                passed_n,
                failed_n,
                wall,
                stopped,
                use_pipeline,
                log_dir,
            )
            q.put({"type": "done", "summary": summary})

        except Exception as e:
            logger.error("Batch failed: %s\n%s", e, traceback.format_exc())
            q.put({"type": "done", "error": str(e)})
        finally:
            for m in models:
                try:
                    cleanup_asr_model(m)
                except Exception:
                    pass
            try:
                root_log.removeHandler(file_handler)
            except Exception:
                pass
            file_handler.close()

    def _write_validation_logs(
        self,
        validation_log: Path,
        fail_log: Path,
        fail_json: Path,
        results: List[Dict[str, Any]],
        threshold: float,
        backend: str,
        model: str,
        device: str,
        workers: int,
        language: str,
        wall: float,
        stopped: bool,
        passed_n: int,
        failed_n: int,
        log_filter: str = "Both",
    ) -> None:
        """Write human-readable validation/fail logs and failures JSON."""
        header = (
            f"PocketGPU ASR GUI Validation\n"
            f"time={datetime.now().isoformat(timespec='seconds')}\n"
            f"backend={backend} model={model} device={device} workers={workers} "
            f"lang={language} threshold={threshold} wall_s={wall:.1f} stopped={stopped}\n"
            f"passed={passed_n} failed={failed_n} total={len(results)} filter={log_filter}\n"
            f"{'=' * 60}\n\n"
        )
        fails: List[Dict[str, Any]] = []
        with open(validation_log, "w", encoding="utf-8") as vf, open(
            fail_log, "w", encoding="utf-8"
        ) as ff:
            vf.write(header)
            ff.write(header)
            for r in sorted(results, key=lambda x: str(x.get("chunk_num") or "")):
                chunk = r.get("chunk_num", "?")
                score = float(r.get("score") or 0)
                passed = bool(r.get("passed"))
                # Apply log filter: skip chunks that don't match
                if log_filter == "Pass Only" and not passed:
                    continue
                if log_filter == "Fail Only" and passed:
                    continue
                status = "PASSED" if passed else "FAILED"
                # Derive chunk_index from chunk_num (e.g. "chunk_01962" → 1962)
                chunk_index = 0
                if "_" in chunk:
                    try:
                        chunk_index = int(chunk.split("_")[-1])
                    except ValueError:
                        chunk_index = 0
                # Derive decision from classification
                classification = r.get("classification", "FAIL")
                if classification == "PASS":
                    decision = "confirmed_pass"
                elif classification == "REVIEW":
                    decision = "needs_review"
                else:
                    decision = "failed_validation"
                # Build failure entry in requested format
                entry = {
                    "chunk_index": chunk_index,
                    "chunk_id": chunk,
                    "decision": decision,
                    "comparison": {
                        "ref_text_raw": r.get("ref_text_raw", ""),
                        "hyp_text_raw": r.get("hyp_text_raw", ""),
                        "Text to ASR comparison:": "",
                        "ref_normalized": r.get("ref_normalized", ""),
                        "hyp_normalized": r.get("hyp_normalized", ""),
                    },
                    "failure": {
                        "score": score,
                        "classification": classification,
                        "explanation": r.get("explanation") or r.get("error") or "",
                        "coverage_score": float(r.get("coverage_score") or 0),
                        "phonetic_score": float(r.get("phonetic_score") or 0),
                        "prose_score": float(r.get("prose_score") or 0),
                        "id_score": float(r.get("id_score") or 0),
                        "identifier_comparisons": r.get("identifier_comparisons", []),
                        "accepted_list_label_equivalences": r.get("accepted_list_label_equivalences", []),
                        "requires_second_stage_confirmation": bool(r.get("requires_second_stage_confirmation")),
                        "second_stage_confirmation_reason": r.get("second_stage_confirmation_reason", ""),
                        "hallucination_warning": r.get("hallucination_warning", ""),
                        "truncation_warning": r.get("truncation_warning", ""),
                        "minor_mismatch": bool(r.get("minor_mismatch")),
                        "extra_tokens": r.get("extra_tokens", []),
                        "missing_tokens": r.get("missing_tokens", []),
                        "substitutions": r.get("substitutions", []),
                    },
                    "filename": f"{chunk}.wav",
                }
                # Human-readable block for validation log
                block = (
                    f"Chunk: {chunk}\n"
                    f"Status: {status} (Score: {score:.3f}) "
                    f"class={classification}\n"
                    f"Original Text: {r.get('ref_text_raw', '')}\n"
                    f"Transcribed Text: {r.get('hyp_text_raw', '')}\n"
                    f"Explanation: {r.get('explanation') or r.get('error') or ''}\n"
                    f"{'-' * 40}\n"
                )
                vf.write(block)
                # Detailed block for failures log
                if not passed:
                    detail_block = (
                        f"chunk_index: {chunk_index}\n"
                        f"chunk_id: {chunk}\n"
                        f"decision: {decision}\n"
                        f"comparison:\n"
                        f"  ref_text_raw: {r.get('ref_text_raw', '')}\n"
                        f"  hyp_text_raw: {r.get('hyp_text_raw', '')}\n"
                        f"  Text to ASR comparison:\n"
                        f"  ref_normalized: {r.get('ref_normalized', '')}\n"
                        f"  hyp_normalized: {r.get('hyp_normalized', '')}\n"
                        f"failure:\n"
                        f"  score: {score:.4f}\n"
                        f"  classification: {classification}\n"
                        f"  explanation: {r.get('explanation') or r.get('error') or ''}\n"
                        f"  coverage_score: {float(r.get('coverage_score') or 0):.4f}\n"
                        f"  phonetic_score: {float(r.get('phonetic_score') or 0):.4f}\n"
                        f"  prose_score: {float(r.get('prose_score') or 0):.4f}\n"
                        f"  id_score: {float(r.get('id_score') or 0):.4f}\n"
                        f"  identifier_comparisons: {len(r.get('identifier_comparisons') or [])}\n"
                        f"  accepted_list_label_equivalences: {len(r.get('accepted_list_label_equivalences') or [])}\n"
                        f"  requires_second_stage_confirmation: {bool(r.get('requires_second_stage_confirmation'))}\n"
                        f"  hallucination_warning: {r.get('hallucination_warning', '')}\n"
                        f"  truncation_warning: {r.get('truncation_warning', '')}\n"
                        f"  second_stage_confirmation_reason: {r.get('second_stage_confirmation_reason', '')}\n"
                        f"  minor_mismatch: {bool(r.get('minor_mismatch'))}\n"
                        f"  extra_tokens: {r.get('extra_tokens', [])}\n"
                        f"  missing_tokens: {r.get('missing_tokens', [])}\n"
                        f"  substitutions: {r.get('substitutions', [])}\n"
                        f"filename: {chunk}.wav\n"
                        f"{'=' * 40}\n"
                    )
                    ff.write(detail_block)
                    fails.append(entry)
        fail_json.write_text(json.dumps(fails, indent=2) + "\n", encoding="utf-8")


def configure_tk_file_dialogs_hide_hidden(root: tk.Tk) -> None:
    """Hide hidden files/folders in Tk file dialogs by default (Linux/X11).

    Forces tk_getOpenFile to load, enables the Show-hidden checkbox, and leaves
    it unchecked so .git / .cache / etc. stay out of the way unless the user
    opts in.
    """
    try:
        # Load dialog implementation (no-op catch); required before setvar works
        root.tk.eval("catch {tk_getOpenFile -badoption}")
        root.tk.setvar("::tk::dialog::file::showHiddenBtn", 1)
        root.tk.setvar("::tk::dialog::file::showHiddenVar", 0)
    except Exception:
        # Non-X11 or older Tk: ignore; dialogs still open
        pass


def main() -> int:
    """Launch the ASR GUI application."""
    root = tk.Tk()
    configure_tk_file_dialogs_hide_hidden(root)
    # Slightly nicer default fonts on Linux
    try:
        root.option_add("*Font", "TkDefaultFont 10")
    except Exception:
        pass
    AsrGuiApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
