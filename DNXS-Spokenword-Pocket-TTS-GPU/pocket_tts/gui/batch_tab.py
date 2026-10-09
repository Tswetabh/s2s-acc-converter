"""
Batch Processing Tab for Audiobook Generator
Allows processing multiple text files with progress tracking.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QGroupBox, QTextEdit, QMessageBox, QListWidget,
    QListWidgetItem, QSplitter
)
from qtpy.QtCore import Qt, QThread, Signal


class BatchJob:
    """Represents one immutable Main-tab generation snapshot in the batch queue."""
    
    def __init__(self, file_path: str, voice_path: str, params: dict, config: Any):
        """Store one source file, voice, settings snapshot, and output destination."""
        self.file_path = file_path
        self.voice_path = voice_path
        self.params = params
        self.config = config
        self.output_path = params["output_path"]
        self.status = "queued"  # queued, processing, completed, failed
        self.current_chunk = 0
        self.total_chunks = 0
        self.error_message = None
        self.processing_time = 0
        self.audio_duration = 0
        self.realtime_factor = 0
    
    @property
    def filename(self) -> str:
        """Return source filename used to identify this queued job."""
        return Path(self.file_path).name
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage for batch processing."""
        if self.total_chunks == 0:
            return 0
        return (self.current_chunk / self.total_chunks) * 100
    
    @property
    def status_icon(self) -> str:
        """Return status icon based on current status."""
        icons = {
            "queued": "⏸",
            "processing": "⏳",
            "completed": "✓",
            "failed": "✗"
        }
        return icons.get(self.status, "?")


class BatchGenerationThread(QThread):
    """Thread for batch audiobook generation."""
    
    # Signals
    progress = Signal(dict)  # Overall progress updates
    file_progress = Signal(dict)  # Current file progress
    file_started = Signal(str)  # File path
    file_completed = Signal(str, dict)  # File path, result
    file_failed = Signal(str, str)  # File path, error message
    batch_completed = Signal(dict)  # Overall batch results
    
    def __init__(self, jobs: List[BatchJob]):
        """Initialize a worker thread that executes saved jobs in queue order."""
        super().__init__()
        self.files = jobs
        self._is_paused = False
        self._is_stopped = False
    
    def run(self):
        """Process all files in the batch."""
        start_time = time.time()
        completed = 0
        failed = 0
        
        for i, batch_file in enumerate(self.files):
            if self._is_stopped:
                break
            
            # Wait if paused
            while self._is_paused and not self._is_stopped:
                self.msleep(100)
            
            if self._is_stopped:
                break
            
            # Update status
            batch_file.status = "processing"
            batch_file.current_chunk = 0
            batch_file.total_chunks = 0
            
            self.file_started.emit(batch_file.file_path)
            self.progress.emit({
                'current_file': i + 1,
                'total_files': len(self.files),
                'current_filename': batch_file.filename,
                'status': 'processing'
            })
            
            try:
                result = self._process_file(batch_file)
                
                if result['success']:
                    batch_file.status = "completed"
                    batch_file.output_path = result.get('output_path')
                    batch_file.processing_time = result.get('processing_time', 0)
                    batch_file.audio_duration = result.get('audio_duration', 0)
                    batch_file.realtime_factor = result.get('realtime_factor', 0)
                    completed += 1
                    self.file_completed.emit(batch_file.file_path, result)
                else:
                    batch_file.status = "failed"
                    batch_file.error_message = result.get('reason', 'Unknown error')
                    failed += 1
                    self.file_failed.emit(batch_file.file_path, batch_file.error_message)
            
            except Exception as e:
                batch_file.status = "failed"
                batch_file.error_message = str(e)
                failed += 1
                self.file_failed.emit(batch_file.file_path, str(e))
        
        # Final results
        total_time = time.time() - start_time
        self.batch_completed.emit({
            'total_files': len(self.files),
            'completed': completed,
            'failed': failed,
            'total_time': total_time
        })
    
    def _process_file(self, batch_file: BatchJob) -> Dict[str, Any]:
        """Generate one job using only its saved settings and configuration."""
        from pocket_tts.audiobook.generator import AudiobookGenerator
        from pocket_tts.preprocessing.structure_detector import StructureDetector
        from pocket_tts.preprocessing.chunker import SmartChunker
        from pocket_tts.preprocessing.emotion_analyzer import EmotionAnalyzer
        from pocket_tts.preprocessing.parameter_mapper import ParameterMapper
        from pocket_tts.preprocessing.schema import BoundaryType
        
        # Read text file
        with open(batch_file.file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Periods on bare line ends before structure/chunk merge.
        from pocket_tts.preprocessing.text_normalizer import (
            ensure_terminal_punctuation_on_lines,
        )
        text = ensure_terminal_punctuation_on_lines(text)
        
        run_settings = batch_file.params
        config = batch_file.config
        voice_path = batch_file.voice_path
        if not voice_path:
            return {'success': False, 'reason': 'No voice specified'}
        
        # Preprocess text from the job snapshot (exact UI values at queue time).
        detector = StructureDetector()
        mode = run_settings.get("chunking_mode", "sentence")
        if mode == "smart":
            # Legacy jobs may still say "smart"; treat as sentence soft-floor.
            mode = "sentence"
        chunker = SmartChunker(
            mode=mode,
            min_words=int(run_settings.get("min_words", 35)),
            respect_boundaries=bool(run_settings.get("respect_boundaries", True)),
            target_words=int(run_settings.get("target_words", run_settings.get("min_words", 50))),
        )
        emotion_detection_enabled = bool(
            run_settings.get('emotion_detection_enabled', config.emotion.get('enabled', True))
        )
        analyzer = EmotionAnalyzer() if emotion_detection_enabled else None
        
        # Get pause settings
        pi_enabled = run_settings.get('pause_injection_enabled', False)
        if pi_enabled:
            sentence_pause_ms = 0
            paragraph_pause_ms = 0
            chapter_pause_ms = 0
        else:
            sentence_pause_ms = run_settings.get('sentence_pause_ms', 500)
            paragraph_pause_ms = run_settings.get('paragraph_pause_ms', 1000)
            chapter_pause_ms = run_settings.get('chapter_pause_ms', 2000)
        
        boundary_pauses = {
            BoundaryType.SENTENCE_END: sentence_pause_ms,
            BoundaryType.PARAGRAPH_BREAK: paragraph_pause_ms,
            BoundaryType.CHAPTER_START: chapter_pause_ms
        }
        
        mapper = ParameterMapper(
            config=config,
            boundary_pauses=boundary_pauses,
            base_temperature=run_settings.get('temperature', 0.8),
            base_eos_threshold=run_settings.get('eos_threshold', 0.5),
            base_frames_after_eos=run_settings.get('frames_after_eos', 30)
        )
        
        # Process text
        structure = detector.analyze(text)
        chunks = chunker.chunk(structure)
        
        if not chunks:
            return {'success': False, 'reason': 'No chunks generated from text'}

        # Normalize pause-tag whitespace so Batch matches Main without adding commas.
        from pocket_tts.preprocessing.text_normalizer import flatten_newlines_for_json

        for chunk in chunks:
            chunk.text = flatten_newlines_for_json(chunk.text or "")
        
        # Skip model loading and analysis for jobs queued with detection disabled.
        if emotion_detection_enabled:
            texts_to_analyze = [chunk.text for chunk in chunks]
            emotion_results = analyzer.analyze_batch(texts_to_analyze)
        else:
            emotion_results = [EmotionAnalyzer.neutral_result() for _ in chunks]
        
        # Map emotions to parameters
        for chunk, emotion in zip(chunks, emotion_results):
            mapped_params = mapper.calculate_params(
                emotion=emotion['emotion'],
                punctuation=chunk.punctuation,
                boundary_type=chunk.boundary_type,
                word_count=chunk.word_count,
                emotion_scores=emotion['scores']
            )
            
            silence_duration_ms = mapper.calculate_silence_duration_ms(chunk.boundary_type)
            silence_duration_sec = silence_duration_ms / 1000.0
            
            chunk.tts_params = {
                'temperature': mapped_params.temperature,
                'eos_threshold': mapped_params.eos_threshold,
                'frames_after_eos': mapped_params.frames_after_eos,
                'speed_factor': mapped_params.speed_factor,
                'lsd_decode_steps': mapped_params.lsd_decode_steps
            }
            chunk.post_process = {'silence_duration': silence_duration_sec}
        
        batch_file.total_chunks = len(chunks)
        
        # Create generator
        generator = AudiobookGenerator(config=config)
        generator._pause_injection_enabled = pi_enabled
        generator._pause_durations = run_settings.get('pause_durations', {})
        output_path = batch_file.output_path
        
        # Progress callback for this file (includes phase: starting/tts/asr/regen/done)
        def progress_callback(progress_data):
            """Callback for progress updates during batch generation."""
            batch_file.current_chunk = progress_data.get('current_chunk', 0)
            if progress_data.get('total_chunks'):
                batch_file.total_chunks = progress_data.get('total_chunks', batch_file.total_chunks)
            self.file_progress.emit({
                'file_path': batch_file.file_path,
                'current_chunk': batch_file.current_chunk,
                'total_chunks': batch_file.total_chunks,
                'elapsed_seconds': progress_data.get('elapsed_seconds', 0),
                'eta_seconds': progress_data.get('eta_seconds', 0),
                'phase': progress_data.get('phase'),
                'message': progress_data.get('message'),
                'asr_status': progress_data.get('asr_status'),
                'regeneration': progress_data.get('regeneration'),
            })
        
        # Generate audiobook
        result = generator.generate_audiobook(
            chunks=chunks,
            voice_path=voice_path,
            output_path=output_path,
            progress_callback=progress_callback,
            source_file=batch_file.file_path,
            save_dataset_chunks=True,
            run_settings=run_settings,
        )
        return result
    
    def pause(self):
        """Pause batch processing."""
        self._is_paused = True
    
    def resume(self):
        """Resume batch processing."""
        self._is_paused = False
    
    def stop(self):
        """Stop the queue after the current job completes its generation."""
        self._is_stopped = True


class BatchTab(QWidget):
    """Batch Processing tab for Main-tab generation snapshots."""
    
    def __init__(self, config=None, main_window=None):
        """Initialize BatchTab with configuration and main window."""
        super().__init__()
        self.config = config
        self.main_window = main_window
        self.batch_files: List[BatchJob] = []
        self.generation_thread: Optional[BatchGenerationThread] = None
        self.start_time = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the batch processing UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Create main splitter for file list and details
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: File list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # File list group
        file_group = QGroupBox("Files to Process")
        file_layout = QVBoxLayout(file_group)
        
        # Jobs are added only from Main, so their saved settings cannot change.
        self.file_list_widget = QListWidget()
        file_layout.addWidget(self.file_list_widget)
        
        # File list buttons
        file_buttons_layout = QHBoxLayout()

        self.add_folder_btn = QPushButton("Add Folder")
        self.add_folder_btn.setToolTip(
            "Queue every .txt in a folder (sorted by name), using current Main-tab settings."
        )
        self.add_folder_btn.clicked.connect(self.add_folder)
        file_buttons_layout.addWidget(self.add_folder_btn)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_selected)
        file_buttons_layout.addWidget(self.remove_btn)
        
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        file_buttons_layout.addWidget(self.clear_btn)
        
        file_layout.addLayout(file_buttons_layout)
        
        # Move buttons
        move_layout = QHBoxLayout()
        
        self.move_up_btn = QPushButton("Move Up")
        self.move_up_btn.clicked.connect(self.move_up)
        move_layout.addWidget(self.move_up_btn)
        
        self.move_down_btn = QPushButton("Move Down")
        self.move_down_btn.clicked.connect(self.move_down)
        move_layout.addWidget(self.move_down_btn)
        
        file_layout.addLayout(move_layout)
        
        left_layout.addWidget(file_group)
        
        splitter.addWidget(left_widget)
        
        # Right side: Progress and controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Overall progress
        progress_group = QGroupBox("Overall Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # File progress
        file_progress_layout = QHBoxLayout()
        file_progress_layout.addWidget(QLabel("File:"))
        self.file_progress_label = QLabel("0 / 0")
        file_progress_layout.addWidget(self.file_progress_label)
        progress_layout.addLayout(file_progress_layout)
        
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setValue(0)
        progress_layout.addWidget(self.overall_progress_bar)
        
        # Current file progress
        current_file_layout = QHBoxLayout()
        current_file_layout.addWidget(QLabel("Current:"))
        self.current_file_label = QLabel("None")
        current_file_layout.addWidget(self.current_file_label)
        progress_layout.addLayout(current_file_layout)
        
        self.current_progress_bar = QProgressBar()
        self.current_progress_bar.setValue(0)
        progress_layout.addWidget(self.current_progress_bar)
        
        # Stats
        stats_layout = QHBoxLayout()
        
        stats_layout.addWidget(QLabel("Elapsed:"))
        self.elapsed_label = QLabel("0:00")
        stats_layout.addWidget(self.elapsed_label)
        
        stats_layout.addWidget(QLabel("ETA:"))
        self.eta_label = QLabel("--:--")
        stats_layout.addWidget(self.eta_label)
        
        progress_layout.addLayout(stats_layout)
        
        right_layout.addWidget(progress_group)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Batch")
        self.start_btn.clicked.connect(self.start_batch)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px; }")
        controls_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_batch)
        self.pause_btn.setEnabled(False)
        controls_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_batch)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        controls_layout.addWidget(self.stop_btn)
        
        right_layout.addLayout(controls_layout)
        
        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #00ff00; font-family: Consolas, monospace; }")
        log_layout.addWidget(self.log_text)
        
        right_layout.addWidget(log_group)
        
        splitter.addWidget(right_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
    
    def add_job(self, file_path: str, voice_path: str, params: dict, config: Any) -> BatchJob:
        """Append one Main-tab job and reserve a separate output folder if needed."""
        from pocket_tts.audiobook.generator import AudiobookGenerator

        dataset_paths = AudiobookGenerator.generate_output_paths(file_path, voice_path)
        default_dir = Path(dataset_paths["output_dir"])
        reserved_dirs = {Path(job.output_path).parent for job in self.batch_files}
        output_dir = default_dir
        suffix = 2
        while output_dir in reserved_dirs or output_dir.exists():
            output_dir = default_dir.with_name(f"{default_dir.name}_batch{suffix}")
            suffix += 1

        job_params = dict(params)
        job_params["output_path"] = str(output_dir / dataset_paths["final_audio_filename"])
        job_params["source_file"] = file_path
        job = BatchJob(file_path, voice_path, job_params, config)
        self.batch_files.append(job)
        self._update_list_widget()
        self.log_message(f"Queued: {job.filename} -> {output_dir.name}")
        return job

    def add_folder(self, checked: bool = False) -> None:
        """Pick a folder and queue one job per sorted .txt using Main-tab settings.

        Args:
            checked: Qt passes this from ``clicked(bool)``; it is ignored.
        """
        from pocket_tts.gui.qt_file_dialogs import get_existing_directory

        if self.main_window is None:
            QMessageBox.warning(
                self,
                "No Main Tab",
                "Add Folder needs the Generate tab for voice and settings.",
            )
            return

        if self.generation_thread and self.generation_thread.isRunning():
            QMessageBox.warning(
                self,
                "Batch Running",
                "Stop the current batch before adding more files.",
            )
            return

        start_dir = ""
        if self.batch_files:
            start_dir = str(Path(self.batch_files[-1].file_path).parent)
        elif getattr(self.main_window, "text_file_path", None) is not None:
            current = self.main_window.text_file_path.text()
            if current and current != "No file selected" and Path(current).exists():
                start_dir = str(Path(current).parent)

        folder = get_existing_directory(self, "Select Folder of Text Files", start_dir)
        if not folder:
            return

        txt_files = sorted(Path(folder).glob("*.txt"), key=lambda p: p.name.lower())
        if not txt_files:
            QMessageBox.information(
                self,
                "No Text Files",
                f"No .txt files found in:\n{folder}",
            )
            self.log_message(f"Added 0 file(s) from folder ({Path(folder).name})")
            return

        voice_path, params_template, config_snapshot = (
            self.main_window.build_batch_job_snapshot()
        )
        if not voice_path:
            return

        from copy import deepcopy

        added = 0
        for txt_path in txt_files:
            params = dict(params_template)
            params["source_file"] = str(txt_path)
            # Independent config copy per job (same as repeated Add to Batch).
            self.add_job(str(txt_path), voice_path, params, deepcopy(config_snapshot))
            added += 1

        self.log_message(f"Added {added} file(s) from folder")
    
    def remove_selected(self):
        """Remove selected files from the batch."""
        selected = self.file_list_widget.currentRow()
        if selected >= 0 and selected < len(self.batch_files):
            removed = self.batch_files.pop(selected)
            self._update_list_widget()
            self.log_message(f"Removed: {removed.filename}")
    
    def clear_all(self):
        """Clear all files from the batch."""
        if self.batch_files:
            reply = QMessageBox.question(
                self,
                "Clear All",
                "Remove all files from the batch?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.batch_files.clear()
                self._update_list_widget()
                self.log_message("Cleared all files")
    
    def move_up(self):
        """Move selected file up in the list."""
        selected = self.file_list_widget.currentRow()
        if selected > 0:
            self.batch_files[selected], self.batch_files[selected - 1] = \
                self.batch_files[selected - 1], self.batch_files[selected]
            self._update_list_widget()
            self.file_list_widget.setCurrentRow(selected - 1)
    
    def move_down(self):
        """Move selected file down in the list."""
        selected = self.file_list_widget.currentRow()
        if selected >= 0 and selected < len(self.batch_files) - 1:
            self.batch_files[selected], self.batch_files[selected + 1] = \
                self.batch_files[selected + 1], self.batch_files[selected]
            self._update_list_widget()
            self.file_list_widget.setCurrentRow(selected + 1)
    
    def _update_list_widget(self):
        """Update the list widget to reflect batch_files."""
        self.file_list_widget.clear()
        
        for batch_file in self.batch_files:
            voice_text = f" [{Path(batch_file.voice_path).stem}]"
            output_text = f" -> {Path(batch_file.output_path).parent.name}"
            
            status_text = ""
            if batch_file.status == "processing":
                status_text = f" - {batch_file.current_chunk}/{batch_file.total_chunks}"
            elif batch_file.status == "completed":
                status_text = f" ✓ ({batch_file.realtime_factor:.1f}x)"
            elif batch_file.status == "failed":
                status_text = f" ✗ {batch_file.error_message[:30]}"
            
            display_text = (
                f"{batch_file.status_icon} {batch_file.filename}{voice_text}"
                f"{output_text}{status_text}"
            )
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, batch_file)
            self.file_list_widget.addItem(item)
    
    def start_batch(self):
        """Start batch processing."""
        if not self.batch_files:
            QMessageBox.warning(self, "No Jobs", "Add jobs from the Generate Audiobook tab.")
            return
        
        if self.generation_thread and self.generation_thread.isRunning():
            QMessageBox.warning(self, "Already Running", "Batch processing is already in progress.")
            return
        
        # CPP+GPU source-build consent (same as main Generate)
        if hasattr(self.main_window, "_confirm_cpp_gpu_if_needed"):
            if not self.main_window._confirm_cpp_gpu_if_needed():
                return

        # Reset file statuses
        for batch_file in self.batch_files:
            batch_file.status = "queued"
            batch_file.current_chunk = 0
            batch_file.total_chunks = 0
            batch_file.error_message = None
        
        self._update_list_widget()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.overall_progress_bar.setValue(0)
        self.current_progress_bar.setValue(0)
        self.file_progress_label.setText(f"0 / {len(self.batch_files)}")
        self.current_file_label.setText("None")
        
        self.start_time = time.time()
        
        # Create and start thread
        self.generation_thread = BatchGenerationThread(jobs=self.batch_files)
        
        self.generation_thread.progress.connect(self.on_overall_progress)
        self.generation_thread.file_progress.connect(self.on_file_progress)
        self.generation_thread.file_started.connect(self.on_file_started)
        self.generation_thread.file_completed.connect(self.on_file_completed)
        self.generation_thread.file_failed.connect(self.on_file_failed)
        self.generation_thread.batch_completed.connect(self.on_batch_completed)
        
        self.generation_thread.start()
        
        self.log_message("Started batch processing")
    
    def pause_batch(self):
        """Pause or resume batch processing."""
        if self.generation_thread:
            if self.generation_thread._is_paused:
                self.generation_thread.resume()
                self.pause_btn.setText("Pause")
                self.log_message("Resumed batch processing")
            else:
                self.generation_thread.pause()
                self.pause_btn.setText("Resume")
                self.log_message("Paused batch processing")
    
    def stop_batch(self):
        """Stop batch processing."""
        if self.generation_thread:
            reply = QMessageBox.question(
                self,
                "Stop Batch",
                "Stop batch processing? Current file will complete.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.generation_thread.stop()
                self.log_message("Stopping batch processing...")
    
    def on_overall_progress(self, data: dict):
        """Handle overall progress update."""
        current_file = data.get('current_file', 0)
        total_files = data.get('total_files', 0)
        
        self.file_progress_label.setText(f"{current_file} / {total_files}")
        self.overall_progress_bar.setValue(int((current_file / total_files) * 100) if total_files > 0 else 0)
    
    def on_file_progress(self, data: dict):
        """Handle current file progress update (including ASR/regen phases)."""
        current_chunk = data.get('current_chunk', 0)
        total_chunks = data.get('total_chunks', 0)
        eta_seconds = data.get('eta_seconds', 0)
        phase = (data.get('phase') or '').strip().lower()
        msg = data.get('message') or ''

        phase_icons = {
            'starting': '▶',
            'tts': '🎙',
            'asr': '🔍',
            'regen': '♻',
            'done': '✓',
        }
        if phase in phase_icons:
            key = f"{phase}:{msg}"
            if getattr(self, '_last_batch_phase_msg', None) != key:
                self._last_batch_phase_msg = key
                self.log_message(f"{phase_icons[phase]} [{phase.upper()}] {msg}")
            if phase == 'asr':
                self.current_file_label.setText("ASR check…")
                self.current_progress_bar.setRange(0, 0)
            elif phase == 'regen':
                self.current_file_label.setText("Regeneration…")
            elif phase in ('tts', 'starting') and total_chunks > 0:
                self.current_progress_bar.setRange(0, 100)

        if data.get('asr_status') and msg:
            if getattr(self, '_last_batch_asr_msg', None) != msg:
                self._last_batch_asr_msg = msg
                self.log_message(f"🔍 [ASR] {msg}")
            self.current_file_label.setText("ASR check…")
            self.current_progress_bar.setRange(0, 0)

        if data.get('regeneration'):
            event = data['regeneration']
            self.log_message(
                f"♻ [REGEN] {event.get('chunk_num')} attempt "
                f"{event.get('attempt')}/{event.get('max_retries')}: "
                f"score={event.get('score', 0):.3f}"
            )

        if total_chunks > 0 and phase not in ('asr',):
            self.current_progress_bar.setRange(0, 100)
            self.current_progress_bar.setValue(int((current_chunk / total_chunks) * 100))
            if phase not in ('asr', 'regen'):
                self.current_file_label.setText(f"{current_chunk} / {total_chunks} chunks")
        
        # Update elapsed and ETA
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.elapsed_label.setText(f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
        
        if eta_seconds > 0:
            self.eta_label.setText(f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}")
        
        # Update list widget
        self._update_list_widget()
    
    def on_file_started(self, file_path: str):
        """Handle file started."""
        self.log_message(f"Starting: {Path(file_path).name}")
        self._update_list_widget()
    
    def on_file_completed(self, file_path: str, result: dict):
        """Handle file completed."""
        filename = Path(file_path).name
        duration = result.get('audio_duration', 0)
        rtf = result.get('realtime_factor', 0)
        self.log_message(f"Completed: {filename} ({duration:.1f}s audio, {rtf:.1f}x realtime)")
        self._update_list_widget()
    
    def on_file_failed(self, file_path: str, error: str):
        """Handle file processing failure."""
        filename = Path(file_path).name
        self.log_message(f"Failed: {filename} - {error}")
        self._update_list_widget()

    def on_batch_completed(self, results: dict):
        """Handle batch completed."""
        total = results.get('total_files', 0)
        completed = results.get('completed', 0)
        failed = results.get('failed', 0)
        total_time = results.get('total_time', 0)

        self.log_message(f"Batch completed: {completed}/{total} files in {total_time:.1f}s")

        if failed > 0:
            self.log_message(f"Warning: {failed} file(s) failed")

        # Clean up thread reference safely
        if hasattr(self, 'generation_thread') and self.generation_thread is not None:
            thread = self.generation_thread
            self.generation_thread = None
            thread.wait()
            thread.deleteLater()

        # Update UI
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("Pause")

        self._update_list_widget()

    def closeEvent(self, event):
        """Clean up batch generation thread on widget close."""
        if hasattr(self, 'generation_thread') and self.generation_thread is not None and self.generation_thread.isRunning():
            self.generation_thread.stop()
            self.generation_thread.wait()
        event.accept()

    def log_message(self, message: str):
        """Add a message to the log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
