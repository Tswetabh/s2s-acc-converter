#!/usr/bin/env python3
"""Export chapterized M4B and/or per-chapter MP3s from a finished book folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pocket_tts.audio.chapter_export import export_book_chapters


def main() -> int:
    """CLI entry: chapter-export a book directory after TTS."""
    parser = argparse.ArgumentParser(
        description="Build chapters.json, chapterized M4B, and/or chapter MP3s."
    )
    parser.add_argument(
        "book_dir",
        type=Path,
        help="Book output directory containing TTS/ and the full WAV",
    )
    parser.add_argument("--m4b", action="store_true", help="Write chapterized/full M4B")
    parser.add_argument("--mp3", action="store_true", help="Write chapter or full MP3(s)")
    parser.add_argument(
        "--write-wav",
        action="store_true",
        help="Write full stitched WAV from chunk list",
    )
    parser.add_argument(
        "--mode",
        choices=("m4b", "mp3", "both"),
        default=None,
        help="Legacy: m4b | mp3 | both (prefer --m4b/--mp3/--write-wav)",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=0.0,
        help="Soft max chapter length in minutes (0 = headings only)",
    )
    parser.add_argument(
        "--wav-path",
        type=Path,
        default=None,
        help="Existing full book WAV path (optional)",
    )
    parser.add_argument(
        "--normalization",
        default="peak",
        choices=("none", "peak", "loudness", "simple"),
    )
    parser.add_argument("--title", default="", help="Metadata title")
    parser.add_argument("--artist", default="", help="Metadata artist")
    args = parser.parse_args()

    book_dir = args.book_dir.expanduser().resolve()
    if not book_dir.is_dir():
        print(f"Not a directory: {book_dir}", file=sys.stderr)
        return 1

    write_m4b = bool(args.m4b)
    write_mp3 = bool(args.mp3)
    write_wav = bool(args.write_wav)
    if args.mode:
        write_m4b = args.mode in ("m4b", "both")
        write_mp3 = args.mode in ("mp3", "both")
    if not (write_m4b or write_mp3 or write_wav):
        write_m4b = True
        write_mp3 = True

    result = export_book_chapters(
        book_dir,
        wav_path=args.wav_path,
        chapterize=True,
        max_chapter_minutes=args.max_minutes,
        write_m4b=write_m4b,
        write_mp3=write_mp3,
        write_wav=write_wav,
        m4b_config={"normalization_type": args.normalization, "ffmpeg_path": "ffmpeg"},
        title=args.title,
        artist=args.artist,
    )
    print(f"Chapters: {result.get('chapter_count')}")
    print(f"chapters.json: {result.get('chapters_json')}")
    if result.get("m4b_path"):
        print(f"M4B: {result['m4b_path']}")
    if result.get("mp3_dir"):
        print(f"MP3 dir: {result['mp3_dir']} ({len(result.get('mp3_files') or [])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
