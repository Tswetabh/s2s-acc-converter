"""Regression coverage for Metadata-dialog values reaching export formats."""

import tempfile
import unittest
import json
import shutil
import subprocess
import wave
from inspect import getsource, signature
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qtpy.QtGui import QColor, QPixmap
from qtpy.QtWidgets import QApplication, QLabel

from pocket_tts.audio.chapter_export import (
    apply_media_metadata,
    build_chapter_export_plan,
    encode_full_mp3_from_chunks,
    export_chapter_mp3s,
    mux_m4b_with_chapters,
    write_ffmetadata,
)
from pocket_tts.audio.metadata import (
    DEFAULT_ARTIST,
    export_metadata_tags,
    metadata_cover_path,
)
from pocket_tts.gui.main_window import AudiobookGenerator


class _Checked:
    """Return one fixed checkbox value through Qt's ``isChecked`` interface."""

    def __init__(self, value: bool) -> None:
        """Store the checkbox state used by the GUI settings test."""
        self.value = value

    def isChecked(self) -> bool:
        """Return the configured test checkbox state."""
        return self.value


class _Value:
    """Return one fixed numeric value through Qt's ``value`` interface."""

    def __init__(self, value: int) -> None:
        """Store the numeric test value."""
        self._value = value

    def value(self) -> int:
        """Return the configured numeric test value."""
        return self._value


class _Text:
    """Return one fixed string through Qt's ``currentText`` interface."""

    def __init__(self, value: str) -> None:
        """Store the combo-box test text."""
        self.value = value

    def currentText(self) -> str:
        """Return the configured combo-box test text."""
        return self.value


class _Data:
    """Return one fixed combo-box data value through ``currentData``."""

    def __init__(self, value: str) -> None:
        """Store the strategy value used by the GUI settings test."""
        self.value = value

    def currentData(self) -> str:
        """Return the configured combo-box strategy value."""
        return self.value


class ExportMetadataTests(unittest.TestCase):
    """Verify user metadata reaches all requested completed-file exporters."""

    def test_tag_mapping_keeps_all_requested_fields(self) -> None:
        """Map dialog fields to portable tags while retaining the artist default."""
        tags = export_metadata_tags(
            {
                "author": "A. Writer",
                "series": "Saga",
                "series_number": "2",
                "composer": "C. Music",
                "year": "2026",
                "genre": "Audiobook",
                "description": "Long description",
            },
            title_fallback="Book",
        )

        self.assertEqual(tags["title"], "Book")
        self.assertEqual(tags["artist"], DEFAULT_ARTIST)
        self.assertEqual(tags["album"], "Saga")
        self.assertEqual(tags["episode_id"], "2")
        self.assertEqual(tags["comment"], "Long description")

    def test_metadata_button_slot_accepts_qt_checked_argument(self) -> None:
        """Prevent clicked(bool) from crashing the Meta Data dialog slot."""
        checked = signature(AudiobookGenerator.open_metadata_dialog).parameters["checked"]
        self.assertIs(checked.default, False)

    def test_cover_browser_slot_accepts_qt_checked_argument(self) -> None:
        """Prevent Browse Image clicked(bool) from crashing its nested handler."""
        source = getsource(AudiobookGenerator.open_metadata_dialog)
        self.assertIn("def choose_cover_image(checked: bool = False)", source)

    def test_cover_path_accepts_only_existing_jpeg_or_png(self) -> None:
        """Keep unsupported or missing artwork out of FFmpeg export commands."""
        with tempfile.TemporaryDirectory() as temp_dir:
            valid = Path(temp_dir) / "cover.png"
            invalid = Path(temp_dir) / "cover.webp"
            valid.write_bytes(b"png")
            invalid.write_bytes(b"webp")

            self.assertEqual(metadata_cover_path({"cover_path": str(valid)}), valid)
            self.assertIsNone(metadata_cover_path({"cover_path": str(invalid)}))
            self.assertIsNone(metadata_cover_path({"cover_path": str(valid) + ".missing"}))

    def test_m4b_metadata_file_contains_global_tags_without_forced_chapters(self) -> None:
        """Write M4B-readable tags even when the user turns Chapterize off."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "book.ffmetadata"
            write_ffmetadata(
                [],
                path,
                metadata={"title": "Book", "artist": "Narrator", "author": "A; B"},
            )
            text = path.read_text(encoding="utf-8")

        self.assertIn("title=Book", text)
        self.assertIn("artist=Narrator", text)
        self.assertIn(r"author=A\; B", text)
        self.assertNotIn("[CHAPTER]", text)

    def test_mp3_chapter_keeps_chapter_title_and_global_tags(self) -> None:
        """Tag chapter MP3s without replacing their per-chapter title."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "chapters"
            commands = []
            with patch("pocket_tts.audio.chapter_export._run", side_effect=commands.append):
                export_chapter_mp3s(
                    [{"chapter_id": 1, "title": "Chapter One", "wav_paths": ["/tmp/chunk.wav"]}],
                    output_dir,
                    metadata={"title": "Whole Book", "artist": "Narrator", "genre": "Audiobook"},
                )

        command = commands[0]
        self.assertIn("title=Chapter One", command)
        self.assertNotIn("title=Whole Book", command)
        self.assertIn("artist=Narrator", command)
        self.assertIn("genre=Audiobook", command)

    def test_cover_is_attached_to_m4b_and_chapter_mp3_commands(self) -> None:
        """Map JPEG/PNG art as an attached picture in each supported output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cover = root / "cover.jpg"
            ffmetadata = root / "chapters.ffmetadata"
            cover.write_bytes(b"jpeg")
            ffmetadata.write_text(";FFMETADATA1\ntitle=Book\n", encoding="utf-8")
            commands = []
            with patch("pocket_tts.audio.chapter_export._run", side_effect=commands.append):
                mux_m4b_with_chapters(
                    root / "chunks.concat.txt",
                    ffmetadata,
                    root / "book.m4b",
                    concat_list=True,
                    cover_path=cover,
                )
                export_chapter_mp3s(
                    [{"chapter_id": 1, "title": "Chapter One", "wav_paths": ["/tmp/chunk.wav"]}],
                    root / "chapters",
                    cover_path=cover,
                )

        for command in commands:
            self.assertIn(str(cover), command)
            self.assertIn("attached_pic", command)
            self.assertIn("-map", command)
        self.assertIn("-map_metadata", commands[0])
        self.assertIn("2:v:0", commands[0])

    def test_cover_picker_starts_in_selected_book_folder(self) -> None:
        """Open cover browsing beside the selected text file rather than home."""
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "Book.txt"
            text_path.write_text("text", encoding="utf-8")
            dummy = SimpleNamespace(
                text_file_path=SimpleNamespace(text=lambda: str(text_path)),
                last_text_dir="/fallback",
            )

            self.assertEqual(AudiobookGenerator._metadata_cover_start_dir(dummy), temp_dir)

    def test_cover_thumbnail_renders_selected_image(self) -> None:
        """Render a bounded thumbnail immediately after choosing a valid cover."""
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "cover.png"
            source = QPixmap(300, 200)
            source.fill(QColor("red"))
            self.assertTrue(source.save(str(image_path)))
            thumbnail = QLabel()

            AudiobookGenerator._update_metadata_cover_thumbnail(str(image_path), thumbnail)

            rendered = thumbnail.pixmap()
            self.assertIsNotNone(rendered)
            self.assertLessEqual(rendered.width(), 140)
            self.assertLessEqual(rendered.height(), 140)
        app.processEvents()

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "FFmpeg and FFprobe are required for actual cover-art export validation.",
    )
    def test_ffmpeg_embeds_cover_art_in_m4b_and_mp3(self) -> None:
        """Verify real M4B and MP3 outputs contain an attached cover stream."""
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wav_path = root / "audio.wav"
            cover_path = root / "cover.png"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(b"\0" * 4800)
            apply_media_metadata(
                wav_path,
                {"title": "Book", "artist": "Narrator", "comment": "Description"},
            )
            cover = QPixmap(100, 100)
            cover.fill(QColor("blue"))
            self.assertTrue(cover.save(str(cover_path)))

            m4b_path = mux_m4b_with_chapters(
                wav_path,
                None,
                root / "book.m4b",
                sample_rate=24000,
                cover_path=cover_path,
            )
            mp3_path = encode_full_mp3_from_chunks(
                [str(wav_path)],
                root / "book.mp3",
                cover_path=cover_path,
            )

            wav_probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(wav_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            wav_tags = json.loads(wav_probe.stdout).get("format", {}).get("tags", {})
            self.assertEqual(wav_tags.get("artist"), "Narrator")
            self.assertEqual(wav_tags.get("title"), "Book")

            for output_path in (m4b_path, mp3_path):
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-show_streams", "-of", "json",
                        str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                streams = json.loads(probe.stdout).get("streams", [])
                self.assertTrue(
                    any(
                        stream.get("codec_type") == "video"
                        and stream.get("disposition", {}).get("attached_pic") == 1
                        for stream in streams
                    ),
                    f"Missing attached cover stream in {output_path.name}",
                )
        app.processEvents()

    def test_wav_remux_uses_copy_and_requested_tags(self) -> None:
        """Apply WAV tags through FFmpeg stream copy rather than re-encoding audio."""
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "book.wav"
            wav_path.write_bytes(b"RIFFtest")
            commands = []

            def fake_run(command):
                """Record FFmpeg command and create its replacement output file."""
                commands.append(command)
                Path(command[-1]).write_bytes(wav_path.read_bytes())

            with patch("pocket_tts.audio.chapter_export._run", side_effect=fake_run):
                apply_media_metadata(wav_path, {"artist": "Narrator", "title": "Book"})

            self.assertTrue(wav_path.exists())
        command = commands[0]
        self.assertIn("-c", command)
        self.assertIn("copy", command)
        self.assertIn("artist=Narrator", command)
        self.assertIn("title=Book", command)

    def test_gui_settings_preserve_metadata_and_prefill_artist(self) -> None:
        """Keep metadata live through export settings without requiring config save."""
        dummy = SimpleNamespace(
            config=SimpleNamespace(m4b={}),
            m4b_write_m4b_check=_Checked(True),
            m4b_write_mp3_check=_Checked(True),
            m4b_write_wav_check=_Checked(True),
            m4b_norm_combo=_Text("peak"),
            m4b_chapterize_check=_Checked(False),
            m4b_chapter_mode_combo=_Data("headings_or_minutes"),
            m4b_chapter_minutes_spin=_Value(0),
        )
        values = AudiobookGenerator._metadata_dialog_values(dummy)
        dummy._metadata_dialog_values = lambda: values

        settings = AudiobookGenerator._m4b_gui_settings(dummy)

        self.assertEqual(settings["metadata"]["artist"], DEFAULT_ARTIST)
        self.assertTrue(settings["write_m4b"])
        self.assertTrue(settings["write_mp3"])
        self.assertTrue(settings["write_wav"])
        self.assertEqual(settings["chapter_mode"], "headings_or_minutes")

    def test_headings_or_minutes_only_splits_when_no_headings_exist(self) -> None:
        """Keep heading chapters intact while untitled books use minute fallback."""
        with tempfile.TemporaryDirectory() as temporary:
            chunk_dir = Path(temporary)
            for index in range(3):
                (chunk_dir / f"chunk_{index:05d}.wav").write_bytes(b"test")

            headings = [
                {"index": 0, "text": "Chapter One"},
                {"index": 1, "text": "First body paragraph."},
                {"index": 2, "text": "Second body paragraph."},
            ]
            untitled = [
                {"index": 0, "text": "First body paragraph."},
                {"index": 1, "text": "Second body paragraph."},
                {"index": 2, "text": "Third body paragraph."},
            ]
            with patch(
                "pocket_tts.audio.chapter_export._wav_duration_seconds",
                return_value=420.0,
            ):
                _, heading_plan, mode, headings_found = build_chapter_export_plan(
                    headings,
                    chunk_dir,
                    chapterize=True,
                    max_chapter_minutes=10,
                    chapter_mode="headings_or_minutes",
                )
                _, fallback_plan, fallback_mode, fallback_headings_found = (
                    build_chapter_export_plan(
                        untitled,
                        chunk_dir,
                        chapterize=True,
                        max_chapter_minutes=10,
                        chapter_mode="headings_or_minutes",
                    )
                )
                _, capped_heading_plan, capped_mode, _ = build_chapter_export_plan(
                    headings,
                    chunk_dir,
                    chapterize=True,
                    max_chapter_minutes=10,
                    chapter_mode="headings_with_max",
                )

        self.assertEqual(mode, "headings_or_minutes")
        self.assertTrue(headings_found)
        self.assertEqual(len(heading_plan), 1)
        self.assertEqual(fallback_mode, "headings_or_minutes")
        self.assertFalse(fallback_headings_found)
        self.assertGreater(len(fallback_plan), 1)
        self.assertEqual(capped_mode, "headings_with_max")
        self.assertGreater(len(capped_heading_plan), 1)
