#!/usr/bin/env python3
"""S2S Voice & Accent Studio - Desktop GUI.

A graphical interface for the Speech-to-Speech (S2S) Voice & Accent
Conversion Pipeline:
  Audio Input (.m4a / recording) -> Faster-Whisper (STT) -> Kyutai Pocket TTS (TTS)
"""

from __future__ import annotations

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "DNXS-Spokenword-Pocket-TTS-GPU"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.audio_convert import convert_to_wav
from models.stt.faster_whisper.adapter import FasterWhisperAdapter
import soundfile as sf
import torch
from pocket_tts.models.tts_model import TTSModel

DEFAULT_RECORDINGS_DIR = Path(
    os.getenv("SOUND_RECORDINGS_DIR", r"C:\Users\tripa\Documents\Sound Recordings")
)
DEFAULT_REFERENCE_VOICE = ROOT / "Testinput" / "american accent.wav"
DEFAULT_FALLBACK_AUDIO = ROOT / "Testinput" / "Anoop voice line.wav"
DEFAULT_OUTPUT_DIR = ROOT / "Output"


class S2SApp(tk.Tk):
    """Main Application Window for S2S Voice & Accent Studio."""

    def __init__(self):
        super().__init__()

        self.title("🎙️ S2S Voice & Accent Studio")
        self.geometry("980x760")
        self.minsize(900, 680)

        # Apply dark theme styling
        self._configure_styles()

        self.work_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False

        self.latest_output_file: Optional[Path] = None
        self.recordings_cache: List[Path] = []

        self._build_ui()
        self._refresh_recordings()
        self._check_queue()

    def _configure_styles(self):
        """Set up dark theme palette and TTK widget styles."""
        self.bg_color = "#181825"       # Dark charcoal
        self.card_bg = "#1e1e2e"        # Deep slate
        self.card_border = "#313244"    # Subtle border
        self.fg_color = "#cdd6f4"       # Light text
        self.accent_color = "#89b4fa"   # Soft blue
        self.accent_hover = "#b4befe"
        self.success_color = "#a6e3a1"  # Mint green
        self.warn_color = "#f9e2af"     # Amber
        self.danger_color = "#f38ba8"   # Coral red
        self.entry_bg = "#11111b"       # Inner dark

        self.configure(bg=self.bg_color)

        style = ttk.Style(self)
        style.theme_use("clam")

        # Base configurations
        style.configure(".", background=self.bg_color, foreground=self.fg_color, font=("Segoe UI", 10))
        style.configure("TFrame", background=self.bg_color)
        style.configure("Card.TFrame", background=self.card_bg, relief="solid", borderwidth=1)
        style.configure("TLabelframe", background=self.card_bg, foreground=self.accent_color, font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background=self.card_bg, foreground=self.accent_color, font=("Segoe UI", 10, "bold"))

        # Buttons
        style.configure(
            "Primary.TButton",
            background=self.accent_color,
            foreground="#11111b",
            font=("Segoe UI", 11, "bold"),
            padding=(16, 8),
            borderwidth=0,
        )
        style.map("Primary.TButton", background=[("active", self.accent_hover), ("disabled", "#45475a")])

        style.configure(
            "Secondary.TButton",
            background="#313244",
            foreground=self.fg_color,
            font=("Segoe UI", 9),
            padding=(10, 5),
            borderwidth=0,
        )
        style.map("Secondary.TButton", background=[("active", "#45475a"), ("disabled", "#1e1e2e")])

        style.configure(
            "Success.TButton",
            background=self.success_color,
            foreground="#11111b",
            font=("Segoe UI", 9, "bold"),
            padding=(10, 5),
            borderwidth=0,
        )
        style.map("Success.TButton", background=[("active", "#94e2d5")])

        style.configure("TCombobox", fieldbackground=self.entry_bg, background=self.card_bg, foreground=self.fg_color)
        style.map("TCombobox", fieldbackground=[("readonly", self.entry_bg)])

        style.configure("TProgressbar", background=self.accent_color, troughcolor="#313244")

    def _build_ui(self):
        """Build responsive two-column interface."""
        # Top Header Banner
        header = tk.Frame(self, bg=self.card_bg, height=65)
        header.pack(fill="x", padx=12, pady=(10, 6))

        title_label = tk.Label(
            header,
            text="🎙️ S2S Voice & Accent Studio",
            font=("Segoe UI", 16, "bold"),
            bg=self.card_bg,
            fg=self.accent_color,
        )
        title_label.pack(side="left", padx=16, pady=10)

        gpu_name = "CUDA Available (RTX 3050)" if torch.cuda.is_available() else "CPU Mode"
        badge = tk.Label(
            header,
            text=f"● {gpu_name}",
            font=("Segoe UI", 9, "bold"),
            bg="#313244",
            fg=self.success_color if torch.cuda.is_available() else self.warn_color,
            padx=10,
            pady=4,
        )
        badge.pack(side="right", padx=16)

        # Main Content Layout (Left: Inputs/Config, Right: Text & Outputs)
        main_pane = tk.PanedWindow(self, orient="horizontal", bg=self.bg_color, sashrelief="flat", sashwidth=4)
        main_pane.pack(fill="both", expand=True, padx=12, pady=6)

        left_frame = ttk.Frame(main_pane)
        right_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, minsize=440)
        main_pane.add(right_frame, minsize=440)

        # ==========================================
        # LEFT COLUMN: INPUTS & SETTINGS
        # ==========================================

        # --- Card 1: Input Audio Source ---
        in_group = ttk.LabelFrame(left_frame, text=" 1. Input Audio Source ")
        in_group.pack(fill="x", pady=6)

        self.input_mode = tk.StringVar(value="newest")
        rb_frame = tk.Frame(in_group, bg=self.card_bg)
        rb_frame.pack(fill="x", padx=10, pady=4)

        tk.Radiobutton(
            rb_frame,
            text="Newest Recording",
            variable=self.input_mode,
            value="newest",
            bg=self.card_bg,
            fg=self.fg_color,
            selectcolor="#313244",
            activebackground=self.card_bg,
            activeforeground=self.accent_color,
            command=self._on_input_mode_change,
        ).pack(side="left", padx=4)

        tk.Radiobutton(
            rb_frame,
            text="Pick Recording",
            variable=self.input_mode,
            value="list",
            bg=self.card_bg,
            fg=self.fg_color,
            selectcolor="#313244",
            activebackground=self.card_bg,
            activeforeground=self.accent_color,
            command=self._on_input_mode_change,
        ).pack(side="left", padx=4)

        tk.Radiobutton(
            rb_frame,
            text="Custom File",
            variable=self.input_mode,
            value="custom",
            bg=self.card_bg,
            fg=self.fg_color,
            selectcolor="#313244",
            activebackground=self.card_bg,
            activeforeground=self.accent_color,
            command=self._on_input_mode_change,
        ).pack(side="left", padx=4)

        # Dropdown for Pick Recording
        self.rec_combo_var = tk.StringVar()
        self.rec_combo = ttk.Combobox(in_group, textvariable=self.rec_combo_var, state="readonly", width=42)
        self.rec_combo.pack(fill="x", padx=10, pady=4)

        # Custom File Picker Row
        custom_frame = tk.Frame(in_group, bg=self.card_bg)
        custom_frame.pack(fill="x", padx=10, pady=4)
        self.custom_path_var = tk.StringVar()
        self.custom_entry = tk.Entry(
            custom_frame,
            textvariable=self.custom_path_var,
            bg=self.entry_bg,
            fg=self.fg_color,
            insertbackground=self.fg_color,
            relief="flat",
        )
        self.custom_entry.pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=3)
        self.btn_browse_in = ttk.Button(custom_frame, text="Browse...", style="Secondary.TButton", command=self._browse_custom_audio)
        self.btn_browse_in.pack(side="right")

        # Status & Preview Row
        in_action_row = tk.Frame(in_group, bg=self.card_bg)
        in_action_row.pack(fill="x", padx=10, pady=(2, 8))
        self.lbl_detected_input = tk.Label(
            in_action_row, text="Detected: Scanning...", bg=self.card_bg, fg="#a6adc8", font=("Segoe UI", 9)
        )
        self.lbl_detected_input.pack(side="left")
        ttk.Button(in_action_row, text="🔄 Refresh", style="Secondary.TButton", command=self._refresh_recordings).pack(side="right", padx=(4, 0))
        ttk.Button(in_action_row, text="▶ Preview Input", style="Secondary.TButton", command=self._preview_input_audio).pack(side="right")

        # --- Card 2: STT Options ---
        stt_group = ttk.LabelFrame(left_frame, text=" 2. Speech-to-Text (STT) ")
        stt_group.pack(fill="x", pady=6)

        stt_grid = tk.Frame(stt_group, bg=self.card_bg)
        stt_grid.pack(fill="x", padx=10, pady=6)

        tk.Label(stt_grid, text="Whisper Model:", bg=self.card_bg, fg=self.fg_color).grid(row=0, column=0, sticky="w", pady=4)
        self.stt_model_var = tk.StringVar(value="tiny")
        self.stt_combo = ttk.Combobox(
            stt_grid,
            textvariable=self.stt_model_var,
            values=["tiny (Fastest, ~75MB)", "turbo (High Accuracy, ~1.6GB)"],
            state="readonly",
            width=28,
        )
        self.stt_combo.grid(row=0, column=1, sticky="w", padx=8, pady=4)

        tk.Label(stt_grid, text="Inference Device:", bg=self.card_bg, fg=self.fg_color).grid(row=1, column=0, sticky="w", pady=4)
        self.device_var = tk.StringVar(value="auto")
        self.device_combo = ttk.Combobox(
            stt_grid, textvariable=self.device_var, values=["auto", "cuda", "cpu"], state="readonly", width=12
        )
        self.device_combo.grid(row=1, column=1, sticky="w", padx=8, pady=4)

        # --- Card 3: Target Voice & Accent Cloning ---
        tts_group = ttk.LabelFrame(left_frame, text=" 3. Target Voice & Accent Cloning ")
        tts_group.pack(fill="x", pady=6)

        tts_grid = tk.Frame(tts_group, bg=self.card_bg)
        tts_grid.pack(fill="x", padx=10, pady=6)

        tk.Label(tts_grid, text="Reference Voice:", bg=self.card_bg, fg=self.fg_color).grid(row=0, column=0, sticky="w", pady=4)
        self.voice_preset_var = tk.StringVar(value="american accent.wav (Default)")
        self.voice_presets = [
            "american accent.wav (Default)",
            "Anoop voice line.wav",
            "Built-in: alba",
            "Built-in: cosette",
            "Built-in: marius",
            "Built-in: javert",
            "Built-in: fantine",
            "Custom WAV File...",
        ]
        self.voice_combo = ttk.Combobox(tts_grid, textvariable=self.voice_preset_var, values=self.voice_presets, state="readonly", width=28)
        self.voice_combo.grid(row=0, column=1, sticky="w", padx=8, pady=4)
        self.voice_combo.bind("<<ComboboxSelected>>", self._on_voice_selected)

        self.custom_ref_path = tk.StringVar(value="")
        self.btn_browse_ref = ttk.Button(tts_grid, text="Browse...", style="Secondary.TButton", command=self._browse_custom_ref)
        self.btn_browse_ref.grid(row=0, column=2, padx=4)

        # Tuning Sliders
        tk.Label(tts_grid, text="LSD Steps (Clarity):", bg=self.card_bg, fg=self.fg_color).grid(row=1, column=0, sticky="w", pady=4)
        self.lsd_steps_var = tk.IntVar(value=3)
        self.lsd_spin = tk.Spinbox(
            tts_grid, from_=1, to=6, textvariable=self.lsd_steps_var, width=6, bg=self.entry_bg, fg=self.fg_color, relief="flat"
        )
        self.lsd_spin.grid(row=1, column=1, sticky="w", padx=8, pady=4)

        tk.Label(tts_grid, text="Temperature:", bg=self.card_bg, fg=self.fg_color).grid(row=2, column=0, sticky="w", pady=4)
        self.temp_var = tk.DoubleVar(value=0.7)
        self.temp_spin = tk.Spinbox(
            tts_grid, from_=0.1, to=1.0, increment=0.05, textvariable=self.temp_var, width=6, bg=self.entry_bg, fg=self.fg_color, relief="flat"
        )
        self.temp_spin.grid(row=2, column=1, sticky="w", padx=8, pady=4)

        ttk.Button(tts_grid, text="▶ Preview Voice", style="Secondary.TButton", command=self._preview_ref_audio).grid(row=2, column=2, padx=4)

        # --- Card 4: Action Buttons ---
        act_frame = tk.Frame(left_frame, bg=self.bg_color)
        act_frame.pack(fill="x", pady=10)

        self.btn_run_all = ttk.Button(act_frame, text="🚀 Convert Speech (STT + TTS)", style="Primary.TButton", command=self._start_full_pipeline)
        self.btn_run_all.pack(fill="x", pady=4)

        split_btn_row = tk.Frame(act_frame, bg=self.bg_color)
        split_btn_row.pack(fill="x", pady=2)
        self.btn_stt_only = ttk.Button(split_btn_row, text="📝 Transcribe Only (STT)", style="Secondary.TButton", command=self._start_stt_only)
        self.btn_stt_only.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_tts_only = ttk.Button(split_btn_row, text="🗣️ Synthesize Text (TTS)", style="Secondary.TButton", command=self._start_tts_only)
        self.btn_tts_only.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # ==========================================
        # RIGHT COLUMN: TEXT, LOGS & OUTPUT
        # ==========================================

        # --- Card 5: Transcribed Text Box ---
        text_group = ttk.LabelFrame(right_frame, text=" Transcribed Speech (Editable) ")
        text_group.pack(fill="x", pady=6)

        self.transcribed_text = tk.Text(
            text_group,
            height=6,
            bg=self.entry_bg,
            fg=self.fg_color,
            insertbackground=self.fg_color,
            font=("Segoe UI", 10),
            relief="flat",
            wrap="word",
        )
        self.transcribed_text.pack(fill="both", expand=True, padx=8, pady=8)

        # --- Card 6: Output & Latency Summary ---
        out_group = ttk.LabelFrame(right_frame, text=" Generated Cloned Output ")
        out_group.pack(fill="x", pady=6)

        out_info_frame = tk.Frame(out_group, bg=self.card_bg)
        out_info_frame.pack(fill="x", padx=10, pady=6)

        self.lbl_output_file = tk.Label(
            out_info_frame, text="No output generated yet.", bg=self.card_bg, fg="#a6adc8", anchor="w"
        )
        self.lbl_output_file.pack(fill="x", pady=2)

        self.lbl_metrics = tk.Label(
            out_info_frame, text="Latency Metrics: Ready", bg=self.card_bg, fg=self.accent_color, font=("Segoe UI", 9, "bold"), anchor="w"
        )
        self.lbl_metrics.pack(fill="x", pady=2)

        out_btn_row = tk.Frame(out_group, bg=self.card_bg)
        out_btn_row.pack(fill="x", padx=10, pady=(2, 8))
        self.btn_play_out = ttk.Button(out_btn_row, text="▶ Play Generated Audio", style="Success.TButton", command=self._play_output_audio)
        self.btn_play_out.pack(side="left", padx=(0, 6))
        self.btn_open_folder = ttk.Button(out_btn_row, text="📁 Open Output Folder", style="Secondary.TButton", command=self._open_output_folder)
        self.btn_open_folder.pack(side="left")

        # --- Card 7: Live Execution Logs ---
        log_group = ttk.LabelFrame(right_frame, text=" Pipeline Status & Logs ")
        log_group.pack(fill="both", expand=True, pady=6)

        self.progress_bar = ttk.Progressbar(log_group, mode="indeterminate")
        self.progress_bar.pack(fill="x", padx=8, pady=(6, 4))

        self.log_text = tk.Text(
            log_group,
            bg=self.entry_bg,
            fg="#a6adc8",
            font=("Consolas", 9),
            relief="flat",
            wrap="word",
        )
        self.log_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_text.tag_config("STAGE", foreground=self.accent_color, font=("Consolas", 9, "bold"))
        self.log_text.tag_config("SUCCESS", foreground=self.success_color, font=("Consolas", 9, "bold"))
        self.log_text.tag_config("WARN", foreground=self.warn_color)
        self.log_text.tag_config("ERROR", foreground=self.danger_color, font=("Consolas", 9, "bold"))

    def _log(self, message: str, tag: Optional[str] = None):
        """Append a message safely to the log console."""
        self.log_text.insert("end", message + "\n", tag)
        self.log_text.see("end")

    def _refresh_recordings(self):
        """Scan Windows Sound Recordings folder and update UI."""
        self.recordings_cache = []
        if DEFAULT_RECORDINGS_DIR.exists() and DEFAULT_RECORDINGS_DIR.is_dir():
            valid_exts = {".m4a", ".wav", ".mp3", ".ogg", ".flac", ".aac"}
            files = [f for f in DEFAULT_RECORDINGS_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            self.recordings_cache = files

        if self.recordings_cache:
            newest = self.recordings_cache[0]
            mtime = time.strftime("%I:%M %p", time.localtime(newest.stat().st_mtime))
            size_kb = newest.stat().st_size / 1024
            self.lbl_detected_input.config(
                text=f"Newest: {newest.name} ({size_kb:.0f} KB, {mtime})",
                fg=self.success_color,
            )
            # Update combobox
            display_list = [f"{f.name} ({f.stat().st_size // 1024} KB)" for f in self.recordings_cache[:20]]
            self.rec_combo["values"] = display_list
            self.rec_combo.current(0)
        else:
            self.lbl_detected_input.config(
                text=f"Fallback: {DEFAULT_FALLBACK_AUDIO.name}",
                fg=self.warn_color,
            )
            self.rec_combo["values"] = [DEFAULT_FALLBACK_AUDIO.name]
            self.rec_combo.set(DEFAULT_FALLBACK_AUDIO.name)

        self._on_input_mode_change()

    def _on_input_mode_change(self):
        """Toggle input field visibility depending on radio selection."""
        mode = self.input_mode.get()
        if mode == "newest":
            self.rec_combo.pack_forget()
            self.custom_entry.pack_forget()
            self.btn_browse_in.pack_forget()
        elif mode == "list":
            self.rec_combo.pack(fill="x", padx=10, pady=4)
            self.custom_entry.pack_forget()
            self.btn_browse_in.pack_forget()
        elif mode == "custom":
            self.rec_combo.pack_forget()
            self.custom_entry.pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=3)
            self.btn_browse_in.pack(side="right")

    def _browse_custom_audio(self):
        """Browse file dialog for custom input audio."""
        f = filedialog.askopenfilename(
            title="Select Spoken Audio File",
            filetypes=[("Audio Files", "*.m4a *.wav *.mp3 *.ogg *.flac *.aac"), ("All Files", "*.*")],
        )
        if f:
            self.custom_path_var.set(f)
            self.lbl_detected_input.config(text=f"Selected: {Path(f).name}", fg=self.accent_color)

    def _on_voice_selected(self, event=None):
        """Handle voice preset selection."""
        val = self.voice_preset_var.get()
        if "Custom" in val:
            self.btn_browse_ref.grid()
        else:
            self.btn_browse_ref.grid_remove()

    def _browse_custom_ref(self):
        """Browse file dialog for custom reference voice."""
        f = filedialog.askopenfilename(
            title="Select Reference Voice (.wav)",
            filetypes=[("WAV Audio", "*.wav"), ("All Audio", "*.wav *.m4a *.mp3"), ("All Files", "*.*")],
        )
        if f:
            self.custom_ref_path.set(f)
            self.voice_preset_var.set(f"Custom: {Path(f).name}")

    def _resolve_input_audio(self) -> Path:
        """Resolve current input audio path."""
        mode = self.input_mode.get()
        if mode == "custom":
            path_str = self.custom_path_var.get().strip()
            if not path_str or not Path(path_str).is_file():
                raise FileNotFoundError("Please browse or select a valid custom input audio file.")
            return Path(path_str).resolve()
        elif mode == "list":
            idx = self.rec_combo.current()
            if idx >= 0 and idx < len(self.recordings_cache):
                return self.recordings_cache[idx]
            return DEFAULT_FALLBACK_AUDIO
        else:  # newest
            if self.recordings_cache:
                return self.recordings_cache[0]
            return DEFAULT_FALLBACK_AUDIO

    def _resolve_reference_voice(self) -> tuple[Optional[Path], Optional[str]]:
        """Resolve reference voice path or preset name."""
        val = self.voice_preset_var.get()
        if "american accent" in val:
            return DEFAULT_REFERENCE_VOICE, None
        elif "Anoop voice line" in val:
            return DEFAULT_FALLBACK_AUDIO, None
        elif val.startswith("Built-in:"):
            preset_name = val.split(":")[1].strip()
            return None, preset_name
        elif "Custom" in val:
            ref_str = self.custom_ref_path.get().strip()
            if not ref_str or not Path(ref_str).is_file():
                raise FileNotFoundError("Please browse a valid custom reference voice WAV file.")
            return Path(ref_str).resolve(), None
        return DEFAULT_REFERENCE_VOICE, None

    def _play_audio_file(self, path: Optional[Path]):
        """Open audio file in Windows default player."""
        if not path or not path.exists():
            messagebox.showwarning("File Not Found", "The requested audio file does not exist.")
            return
        try:
            os.startfile(str(path))
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not launch audio player:\n{e}")

    def _preview_input_audio(self):
        try:
            p = self._resolve_input_audio()
            self._play_audio_file(p)
        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    def _preview_ref_audio(self):
        try:
            p, name = self._resolve_reference_voice()
            if p:
                self._play_audio_file(p)
            else:
                messagebox.showinfo("Built-in Voice", f"'{name}' is an embedded Pocket TTS catalog voice.")
        except Exception as e:
            messagebox.showerror("Reference Error", str(e))

    def _play_output_audio(self):
        self._play_audio_file(self.latest_output_file)

    def _open_output_folder(self):
        folder = DEFAULT_OUTPUT_DIR
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ==========================================
    # BACKGROUND PIPELINE WORKERS
    # ==========================================

    def _set_ui_state(self, running: bool):
        """Enable or disable controls during task execution."""
        self.is_running = running
        if running:
            self.progress_bar.start(10)
            self.btn_run_all.config(state="disabled")
            self.btn_stt_only.config(state="disabled")
            self.btn_tts_only.config(state="disabled")
        else:
            self.progress_bar.stop()
            self.btn_run_all.config(state="normal")
            self.btn_stt_only.config(state="normal")
            self.btn_tts_only.config(state="normal")

    def _check_queue(self):
        """Poll the message queue from worker threads."""
        while not self.work_queue.empty():
            msg = self.work_queue.get_nowait()
            mtype = msg.get("type")
            if mtype == "LOG":
                self._log(msg["text"], msg.get("tag"))
            elif mtype == "STT_DONE":
                self.transcribed_text.delete("1.0", "end")
                self.transcribed_text.insert("1.0", msg["text"])
            elif mtype == "TTS_DONE":
                out_path = Path(msg["output_file"])
                self.latest_output_file = out_path
                self.lbl_output_file.config(text=f"File: {out_path.name}", fg=self.fg_color)
                self.lbl_metrics.config(text=msg["metrics"])
            elif mtype == "FINISHED":
                self._set_ui_state(False)
            elif mtype == "ERROR":
                self._set_ui_state(False)
                messagebox.showerror("Pipeline Error", msg["text"])

        self.after(100, self._check_queue)

    def _start_full_pipeline(self):
        """Launch complete STT + TTS pipeline in background thread."""
        try:
            input_file = self._resolve_input_audio()
            ref_path, preset_name = self._resolve_reference_voice()
        except Exception as e:
            messagebox.showerror("Validation Error", str(e))
            return

        stt_model = "tiny" if "tiny" in self.stt_model_var.get() else "turbo"
        device = self.device_var.get()
        lsd_steps = int(self.lsd_steps_var.get())
        temp = float(self.temp_var.get())

        self._set_ui_state(True)
        self.log_text.delete("1.0", "end")
        self._log("=== Starting Speech-to-Speech Conversion ===", "STAGE")

        threading.Thread(
            target=self._worker_run_full,
            args=(input_file, ref_path, preset_name, stt_model, device, lsd_steps, temp),
            daemon=True,
        ).start()

    def _worker_run_full(self, input_file, ref_path, preset_name, stt_model, device, lsd_steps, temp):
        try:
            t_total_start = time.perf_counter()

            # Stage 1: Ingestion
            self.work_queue.put({"type": "LOG", "text": f"[1/4] Ingesting: {input_file.name}", "tag": "STAGE"})

            # Stage 2: Audio Conversion (.m4a -> 16kHz WAV)
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            if input_file.suffix.lower() != ".wav":
                self.work_queue.put({"type": "LOG", "text": f"[2/4] Resampling {input_file.suffix} -> 16kHz mono WAV...", "tag": "STAGE"})
                converted_path = DEFAULT_OUTPUT_DIR / f"converted_{input_file.stem}.wav"
                wav_path = convert_to_wav(input_file, converted_path, target_sr=16000)
            else:
                wav_path = input_file

            # Stage 3: STT (Faster-Whisper)
            self.work_queue.put({"type": "LOG", "text": f"[3/4] Transcribing with Faster-Whisper ('{stt_model}')...", "tag": "STAGE"})
            t_stt_start = time.perf_counter()
            stt_device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"
            stt_compute = "float16" if stt_device == "cuda" else "int8"

            stt = FasterWhisperAdapter(model_path=stt_model, device=stt_device, compute_type=stt_compute)
            stt.load()
            res = stt.transcribe(str(wav_path))
            stt.unload()
            t_stt = time.perf_counter() - t_stt_start

            text = res.get("text", "").strip()
            self.work_queue.put({"type": "STT_DONE", "text": text})
            self.work_queue.put({"type": "LOG", "text": f"Transcribed ({t_stt:.2f}s): \"{text}\"", "tag": "SUCCESS"})

            if not text:
                self.work_queue.put({"type": "ERROR", "text": "No speech detected in audio."})
                return

            # Stage 4: TTS (Pocket TTS)
            self.work_queue.put({"type": "LOG", "text": "[4/4] Synthesizing Voice Clone with Pocket TTS...", "tag": "STAGE"})
            t_tts_start = time.perf_counter()
            tts_device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"

            tts_model = TTSModel.load_model(device=tts_device, temp=temp, lsd_decode_steps=lsd_steps)

            if ref_path and tts_model.has_voice_cloning:
                state = tts_model.get_state_for_audio_prompt(str(ref_path))
                voice_label = ref_path.name
            else:
                fallback = preset_name or "alba"
                state = tts_model.get_state_for_audio_prompt(fallback)
                voice_label = f"Catalog ({fallback})"

            audio = tts_model.generate_audio(state, text_to_generate=text)
            t_tts = time.perf_counter() - t_tts_start

            # Save Output
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_file = DEFAULT_OUTPUT_DIR / f"pipeline_output_{ts}.wav"
            audio_cpu = audio.cpu().squeeze().numpy()
            sf.write(str(out_file), audio_cpu, tts_model.sample_rate)

            t_total = time.perf_counter() - t_total_start
            duration = len(audio_cpu) / tts_model.sample_rate
            rtf = t_total / max(duration, 0.001)

            metrics_str = f"STT: {t_stt:.2f}s | TTS: {t_tts:.2f}s | Total: {t_total:.2f}s | Audio: {duration:.1f}s | RTF: {rtf:.2f}x"
            self.work_queue.put({
                "type": "TTS_DONE",
                "output_file": str(out_file),
                "metrics": metrics_str,
            })
            self.work_queue.put({"type": "LOG", "text": f"Generated successfully -> {out_file.name}", "tag": "SUCCESS"})
            self.work_queue.put({"type": "LOG", "text": f"Metrics: {metrics_str}"})

        except Exception as e:
            self.work_queue.put({"type": "ERROR", "text": str(e)})
        finally:
            self.work_queue.put({"type": "FINISHED"})

    def _start_stt_only(self):
        """Transcribe audio without generating TTS."""
        try:
            input_file = self._resolve_input_audio()
        except Exception as e:
            messagebox.showerror("Validation Error", str(e))
            return

        stt_model = "tiny" if "tiny" in self.stt_model_var.get() else "turbo"
        device = self.device_var.get()

        self._set_ui_state(True)
        self.log_text.delete("1.0", "end")
        self._log("=== Running Faster-Whisper STT Only ===", "STAGE")

        def _work():
            try:
                DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                if input_file.suffix.lower() != ".wav":
                    self.work_queue.put({"type": "LOG", "text": f"Resampling {input_file.suffix} -> WAV...", "tag": "STAGE"})
                    wav_path = convert_to_wav(input_file, DEFAULT_OUTPUT_DIR / f"converted_{input_file.stem}.wav", target_sr=16000)
                else:
                    wav_path = input_file

                stt_device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"
                stt_compute = "float16" if stt_device == "cuda" else "int8"
                stt = FasterWhisperAdapter(model_path=stt_model, device=stt_device, compute_type=stt_compute)
                stt.load()
                res = stt.transcribe(str(wav_path))
                stt.unload()

                text = res.get("text", "").strip()
                self.work_queue.put({"type": "STT_DONE", "text": text})
                self.work_queue.put({"type": "LOG", "text": f"Transcription complete: \"{text}\"", "tag": "SUCCESS"})
            except Exception as e:
                self.work_queue.put({"type": "ERROR", "text": str(e)})
            finally:
                self.work_queue.put({"type": "FINISHED"})

        threading.Thread(target=_work, daemon=True).start()

    def _start_tts_only(self):
        """Synthesize text from the editable text box without STT."""
        text = self.transcribed_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty Text", "Please enter or transcribe some text to synthesize.")
            return

        try:
            ref_path, preset_name = self._resolve_reference_voice()
        except Exception as e:
            messagebox.showerror("Validation Error", str(e))
            return

        device = self.device_var.get()
        lsd_steps = int(self.lsd_steps_var.get())
        temp = float(self.temp_var.get())

        self._set_ui_state(True)
        self._log("=== Running Pocket TTS Synthesis Only ===", "STAGE")

        def _work():
            try:
                DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                t_start = time.perf_counter()
                tts_device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"
                tts_model = TTSModel.load_model(device=tts_device, temp=temp, lsd_decode_steps=lsd_steps)

                if ref_path and tts_model.has_voice_cloning:
                    state = tts_model.get_state_for_audio_prompt(str(ref_path))
                else:
                    state = tts_model.get_state_for_audio_prompt(preset_name or "alba")

                audio = tts_model.generate_audio(state, text_to_generate=text)
                t_tts = time.perf_counter() - t_start

                ts = time.strftime("%Y%m%d_%H%M%S")
                out_file = DEFAULT_OUTPUT_DIR / f"pipeline_output_{ts}.wav"
                audio_cpu = audio.cpu().squeeze().numpy()
                sf.write(str(out_file), audio_cpu, tts_model.sample_rate)

                duration = len(audio_cpu) / tts_model.sample_rate
                metrics_str = f"TTS Time: {t_tts:.2f}s | Audio: {duration:.1f}s | RTF: {t_tts/max(duration, 0.001):.2f}x"
                self.work_queue.put({
                    "type": "TTS_DONE",
                    "output_file": str(out_file),
                    "metrics": metrics_str,
                })
                self.work_queue.put({"type": "LOG", "text": f"Synthesized -> {out_file.name}", "tag": "SUCCESS"})
            except Exception as e:
                self.work_queue.put({"type": "ERROR", "text": str(e)})
            finally:
                self.work_queue.put({"type": "FINISHED"})

        threading.Thread(target=_work, daemon=True).start()


def main():
    app = S2SApp()
    app.mainloop()


if __name__ == "__main__":
    main()
