"""Build chapter maps and export chapterized M4B / per-chapter MP3 files."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

_NOCONSOLE = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_AAC_CODEC_CACHE: Dict[str, str] = {}
_EXPORT_WORKER_CAP = 8
_PCM_COPY_FRAMES = 65536

CHAPTER_MODE_HEADINGS_ONLY = "headings_only"
CHAPTER_MODE_HEADINGS_OR_MINUTES = "headings_or_minutes"
CHAPTER_MODE_HEADINGS_WITH_MAX = "headings_with_max"
_CHAPTER_MODES = {
    CHAPTER_MODE_HEADINGS_ONLY,
    CHAPTER_MODE_HEADINGS_OR_MINUTES,
    CHAPTER_MODE_HEADINGS_WITH_MAX,
}


def _wav_duration_seconds(path: Path) -> float:
    """Return duration of a PCM WAV file in seconds."""
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate() or 1
        return float(frames) / float(rate)


def _ffprobe_duration(path: Path, ffmpeg_path: str = "ffmpeg") -> float:
    """Return media duration via ffprobe when available, else WAV reader."""
    probe = "ffprobe"
    ffmpeg = Path(ffmpeg_path)
    if ffmpeg.name.lower().startswith("ffmpeg"):
        candidate = ffmpeg.with_name(ffmpeg.name.replace("ffmpeg", "ffprobe"))
        if candidate.exists():
            probe = str(candidate)
    try:
        result = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            creationflags=_NOCONSOLE,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as exc:
        logger.debug("ffprobe failed for %s: %s", path, exc)
    if path.suffix.lower() == ".wav":
        return _wav_duration_seconds(path)
    raise RuntimeError(f"Could not measure duration for {path}")


def load_chunks_json(chunks_json: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load audiobook.chunks.json records and metadata."""
    data = json.loads(Path(chunks_json).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, {}
    return list(data.get("chunks") or []), dict(data.get("_metadata") or {})


def ensure_chapter_ids(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stamp chapter_id on chunks when missing (0 before first Part/Chapter)."""
    from pocket_tts.preprocessing.chapter_headers import assign_chapter_ids

    if chunks and any(c.get("chapter_id") is not None for c in chunks):
        # Normalize titles for export when only chapter_number exists.
        out = []
        for chunk in chunks:
            row = dict(chunk)
            if row.get("chapter_id") is None:
                num = row.get("chapter_number")
                row["chapter_id"] = int(num) if num is not None else 0
            if not row.get("chapter_title"):
                cid = int(row["chapter_id"])
                row["chapter_title"] = "Front matter" if cid == 0 else f"Chapter {cid}"
            out.append(row)
        return out
    return assign_chapter_ids(chunks)


def resolve_chapter_mode(
    chapter_mode: Optional[str], max_chapter_minutes: float | int
) -> str:
    """Return a supported chapter strategy while preserving legacy minute-cap behavior.

    Older saved configurations did not carry a strategy.  Their positive minute
    value meant "split every long headed chapter," so it maps to the explicit
    maximum mode instead of silently changing a user's existing exports.
    """
    mode = str(chapter_mode or "").strip().lower()
    if mode in _CHAPTER_MODES:
        return mode
    return (
        CHAPTER_MODE_HEADINGS_WITH_MAX
        if float(max_chapter_minutes or 0) > 0
        else CHAPTER_MODE_HEADINGS_ONLY
    )


def build_chunk_timeline(
    chunks: Sequence[Dict[str, Any]],
    audio_chunks_dir: Path,
) -> List[Dict[str, Any]]:
    """Attach start/end seconds to each chunk from per-chunk WAV durations."""
    audio_chunks_dir = Path(audio_chunks_dir)
    timeline: List[Dict[str, Any]] = []
    cursor = 0.0
    for chunk in chunks:
        index = int(chunk.get("index", len(timeline)))
        wav_path = audio_chunks_dir / f"chunk_{index:05d}.wav"
        if not wav_path.exists():
            # Some runs use 1-based or padded differently — try alternate.
            alt = audio_chunks_dir / f"chunk_{index:05d}.wav"
            wav_path = alt if alt.exists() else wav_path
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing audio chunk: {wav_path}")
        duration = _wav_duration_seconds(wav_path)
        row = dict(chunk)
        row["start_s"] = cursor
        row["end_s"] = cursor + duration
        row["duration_s"] = duration
        row["wav_path"] = str(wav_path)
        timeline.append(row)
        cursor += duration
    return timeline


def group_chapters_from_timeline(
    timeline: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse ordered chunks into chapter ranges (id 0, 1, 2, …)."""
    if not timeline:
        return []
    groups: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for row in timeline:
        cid = int(row.get("chapter_id") or 0)
        title = str(row.get("chapter_title") or ("Front matter" if cid == 0 else f"Chapter {cid}"))
        if current is None or current["chapter_id"] != cid:
            current = {
                "chapter_id": cid,
                "title": title if cid > 0 else "Front matter",
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "start_chunk": int(row["index"]),
                "end_chunk": int(row["index"]),
                "chunk_indices": [int(row["index"])],
                "wav_paths": [row["wav_path"]],
                "boundary_types": [row.get("boundary_type") or ""],
            }
            groups.append(current)
        else:
            current["end_s"] = float(row["end_s"])
            current["end_chunk"] = int(row["index"])
            current["chunk_indices"].append(int(row["index"]))
            current["wav_paths"].append(row["wav_path"])
            current["boundary_types"].append(row.get("boundary_type") or "")
    return groups


def build_chapter_export_plan(
    chunks: List[Dict[str, Any]],
    audio_chunks_dir: Path,
    *,
    chapterize: bool,
    max_chapter_minutes: float | int,
    chapter_mode: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, bool]:
    """Build export timeline and chapter groups for the selected user strategy.

    ``headings_or_minutes`` preserves every detected heading chapter.  It uses
    minutes only when no heading exists, which gives untitled books a usable
    TOC without unexpectedly splitting titled chapters.

    Returns:
        Timeline rows, chapter groups, resolved strategy name, and whether one
        or more source headings were detected.
    """
    resolved_mode = resolve_chapter_mode(chapter_mode, max_chapter_minutes)
    planned_chunks = [dict(chunk) for chunk in chunks]
    headings_found = False
    if chapterize:
        planned_chunks = ensure_chapter_ids(planned_chunks)
        headings_found = any(int(chunk.get("chapter_id") or 0) > 0 for chunk in planned_chunks)
    else:
        # Chapterization is intentionally off: retain one full-book deliverable.
        for chunk in planned_chunks:
            chunk["chapter_id"] = 0
            chunk["chapter_title"] = "Full book"
            chunk["chapter_number"] = None

    timeline = build_chunk_timeline(planned_chunks, audio_chunks_dir)
    chapters = group_chapters_from_timeline(timeline)
    should_split = (
        chapterize
        and float(max_chapter_minutes or 0) > 0
        and (
            resolved_mode == CHAPTER_MODE_HEADINGS_WITH_MAX
            or (
                resolved_mode == CHAPTER_MODE_HEADINGS_OR_MINUTES
                and not headings_found
            )
        )
    )
    if should_split:
        chapters = subdivide_chapters_by_minutes(
            timeline, chapters, float(max_chapter_minutes)
        )
    return timeline, chapters, resolved_mode, headings_found


def subdivide_chapters_by_minutes(
    timeline: Sequence[Dict[str, Any]],
    chapters: Sequence[Dict[str, Any]],
    max_minutes: float,
) -> List[Dict[str, Any]]:
    """Split long chapters near paragraph/sentence ends, else hard cut.

    Args:
        timeline: Chunk rows with start_s/end_s/boundary_type.
        chapters: Chapter groups from ``group_chapters_from_timeline``.
        max_minutes: Soft max length; 0 disables subdivision.

    Returns:
        Possibly expanded chapter list with stable titles.
    """
    if not max_minutes or max_minutes <= 0:
        return list(chapters)

    max_s = float(max_minutes) * 60.0
    by_index = {int(row["index"]): row for row in timeline}
    expanded: List[Dict[str, Any]] = []

    for chapter in chapters:
        duration = float(chapter["end_s"]) - float(chapter["start_s"])
        if duration <= max_s + 0.5:
            expanded.append(dict(chapter))
            continue

        indices = list(chapter["chunk_indices"])
        part = 1
        seg_start_idx = 0
        seg_start_s = float(chapter["start_s"])
        target_end = seg_start_s + max_s

        def emit_segment(end_i: int) -> None:
            """Append one artificial segment covering indices [seg_start_idx, end_i]."""
            nonlocal part, seg_start_idx, seg_start_s, target_end
            slice_idx = indices[seg_start_idx : end_i + 1]
            if not slice_idx:
                return
            first = by_index[slice_idx[0]]
            last = by_index[slice_idx[-1]]
            base_title = chapter["title"]
            title = base_title if part == 1 else f"{base_title} (cont. {part})"
            expanded.append(
                {
                    "chapter_id": chapter["chapter_id"],
                    "title": title,
                    "start_s": float(first["start_s"]),
                    "end_s": float(last["end_s"]),
                    "start_chunk": slice_idx[0],
                    "end_chunk": slice_idx[-1],
                    "chunk_indices": slice_idx,
                    "wav_paths": [by_index[i]["wav_path"] for i in slice_idx],
                    "boundary_types": [
                        by_index[i].get("boundary_type") or "" for i in slice_idx
                    ],
                    "artificial": part > 1,
                }
            )
            part += 1
            seg_start_idx = end_i + 1
            if seg_start_idx < len(indices):
                seg_start_s = float(by_index[indices[seg_start_idx]]["start_s"])
                target_end = seg_start_s + max_s

        i = 0
        while i < len(indices):
            row = by_index[indices[i]]
            end_s = float(row["end_s"])
            is_last = i == len(indices) - 1
            if end_s + 0.01 >= target_end or is_last:
                # Prefer paragraph, then sentence, at or after target; else this chunk.
                cut = i
                if not is_last and end_s + 0.01 < target_end:
                    pass
                else:
                    # Search forward a little for a soft boundary after target.
                    for j in range(i, min(len(indices), i + 40)):
                        bt = str(by_index[indices[j]].get("boundary_type") or "")
                        if bt in {"paragraph_break", "chapter_start"}:
                            cut = j
                            break
                        if bt == "sentence_end":
                            cut = j
                            # keep looking briefly for paragraph
                            continue
                    # If we only found sentence later, use first sentence at/after i
                    if cut == i:
                        for j in range(i, min(len(indices), i + 40)):
                            bt = str(by_index[indices[j]].get("boundary_type") or "")
                            if bt in {
                                "paragraph_break",
                                "sentence_end",
                                "chapter_start",
                            }:
                                cut = j
                                break
                emit_segment(cut)
                i = cut + 1
                continue
            i += 1

        # Remainder
        if seg_start_idx < len(indices):
            emit_segment(len(indices) - 1)

    # Re-number display order ids for artificial parts while keeping book chapter_id.
    for order, chapter in enumerate(expanded):
        chapter["order"] = order
    return expanded


def write_ffmetadata(
    chapters: Sequence[Dict[str, Any]],
    path: Path,
    *,
    title: str = "",
    artist: str = "",
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """Write global export tags and chapters as FFMETADATA1 (ms TIMEBASE)."""
    path = Path(path)
    lines = [";FFMETADATA1"]
    tags = {key: str(value) for key, value in (metadata or {}).items() if value}
    if title:
        tags["title"] = title
    if artist:
        tags["artist"] = artist
    for key, value in tags.items():
        lines.append(f"{key}={_ff_escape(value)}")
    for chapter in chapters:
        start_ms = int(round(float(chapter["start_s"]) * 1000))
        end_ms = int(round(float(chapter["end_s"]) * 1000))
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={start_ms}")
        lines.append(f"END={end_ms}")
        lines.append(f"title={_ff_escape(str(chapter.get('title') or 'Chapter'))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ff_escape(value: str) -> str:
    """Escape special characters for FFMETADATA values."""
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace("#", "\\#")
        .replace("=", "\\=")
        .replace("\n", " ")
    )


def _run(cmd: List[str]) -> None:
    """Run a subprocess and raise on failure."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        creationflags=_NOCONSOLE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"{result.stderr[-2000:]}"
        )


def _metadata_args(metadata: Optional[Dict[str, str]]) -> List[str]:
    """Return FFmpeg metadata arguments for nonempty portable export tags."""
    args: List[str] = []
    for key, value in (metadata or {}).items():
        text = str(value or "").strip()
        if text:
            args.extend(["-metadata", f"{key}={text}"])
    return args


def apply_media_metadata(
    media_path: Path,
    metadata: Optional[Dict[str, str]],
    *,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Remux one completed media file with export tags, including WAV RIFF tags.

    FFmpeg copies the audio stream, so this does not re-encode a finished WAV.
    WAV tag support depends on the installed FFmpeg muxer and player.
    """
    media_path = Path(media_path)
    args = _metadata_args(metadata)
    if not args:
        return media_path
    if not media_path.exists():
        raise FileNotFoundError(f"Cannot tag missing audio file: {media_path}")
    temporary = media_path.with_name(f"{media_path.stem}.metadata{media_path.suffix}")
    _run([
        ffmpeg_path,
        "-y",
        "-i",
        str(media_path),
        "-map",
        "0",
        "-c",
        "copy",
        *args,
        str(temporary),
    ])
    temporary.replace(media_path)
    return media_path


def _write_concat_list(wav_paths: Sequence[str], list_path: Path) -> Path:
    """Write an ffmpeg concat demuxer list for the given chunk WAV paths."""
    list_path = Path(list_path)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for wav in wav_paths:
        # Escape single quotes for concat demuxer path syntax.
        safe = str(Path(wav).resolve()).replace("'", r"'\''")
        lines.append(f"file '{safe}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


def _wav_pcm_format(handle: wave.Wave_read) -> Tuple[int, int, int, str]:
    """Return channels, sample width, frame rate, and compression type for one WAV."""
    return (
        handle.getnchannels(),
        handle.getsampwidth(),
        handle.getframerate(),
        handle.getcomptype(),
    )


def stitch_pcm_wavs(wav_paths: Sequence[str], output_wav: Path) -> Path:
    """Concatenate identical PCM WAV files by copying frames after one shared header.

    Args:
        wav_paths: Ordered source WAV paths. All must share channels, width, rate,
            and uncompressed PCM.
        output_wav: Destination WAV path.

    Returns:
        The written ``output_wav`` path.

    Raises:
        ValueError: Empty input, compressed WAV, or a format mismatch.
    """
    if not wav_paths:
        raise ValueError("No WAV files to stitch")
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    expected: Optional[Tuple[int, int, int, str]] = None
    with wave.open(str(output_wav), "wb") as dest:
        for wav in wav_paths:
            with wave.open(str(wav), "rb") as source:
                fmt = _wav_pcm_format(source)
                if expected is None:
                    if fmt[3] not in {"NONE", "not compressed"}:
                        raise ValueError(f"Compressed WAV cannot be stitched: {wav}")
                    expected = fmt
                    dest.setnchannels(fmt[0])
                    dest.setsampwidth(fmt[1])
                    dest.setframerate(fmt[2])
                elif fmt != expected:
                    raise ValueError(
                        f"WAV format mismatch for {wav}: {fmt} != {expected}"
                    )
                while True:
                    payload = source.readframes(_PCM_COPY_FRAMES)
                    if not payload:
                        break
                    dest.writeframes(payload)
    return output_wav


def concat_wavs(wav_paths: Sequence[str], output_wav: Path, ffmpeg_path: str = "ffmpeg") -> Path:
    """Concatenate WAV files in order into one WAV, preferring a native PCM stitch.

    Falls back to the FFmpeg concat demuxer with stream copy when headers differ
    or a source is not uncompressed PCM.
    """
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        return stitch_pcm_wavs(wav_paths, output_wav)
    except Exception as exc:
        logger.info("Native PCM stitch unavailable (%s); using FFmpeg concat", exc)
    list_path = output_wav.with_suffix(".concat.txt")
    _write_concat_list(wav_paths, list_path)
    try:
        _run(
            [
                ffmpeg_path,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_wav),
            ]
        )
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass
    return output_wav


def measure_pcm_peak(wav_paths: Sequence[str]) -> float:
    """Return the maximum absolute PCM sample across files as a 0..1 fraction.

    Args:
        wav_paths: Ordered WAV paths. Non-16-bit files are scanned sample-by-sample
            through the standard-library ``wave`` / ``array`` path.

    Returns:
        Peak magnitude in the range [0, 1]. Silent or empty input returns 0.0.
    """
    peak = 0
    max_abs = 1
    try:
        import numpy as np
    except ImportError:
        np = None
    for wav in wav_paths:
        with wave.open(str(wav), "rb") as source:
            width = source.getsampwidth()
            max_abs = (1 << (8 * width - 1)) - 1
            while True:
                payload = source.readframes(_PCM_COPY_FRAMES)
                if not payload:
                    break
                if np is not None and width == 2:
                    samples = np.frombuffer(payload, dtype="<i2")
                    if samples.size:
                        peak = max(peak, int(np.max(np.abs(samples))))
                    continue
                import array

                typecode = {1: "b", 2: "h", 4: "i"}.get(width)
                if typecode is None:
                    continue
                values = array.array(typecode)
                values.frombytes(payload)
                if values:
                    peak = max(peak, max(abs(int(sample)) for sample in values))
    if peak <= 0 or max_abs <= 0:
        return 0.0
    return min(1.0, float(peak) / float(max_abs))


def peak_volume_filter(peak: float, target_db: float) -> Optional[str]:
    """Return an FFmpeg ``volume=`` filter that places ``peak`` at ``target_db``.

    Args:
        peak: Measured 0..1 peak from ``measure_pcm_peak``.
        target_db: Desired peak level in dBFS (typically ``target_db`` from config).

    Returns:
        An FFmpeg audio-filter string, or None when no gain is needed or the
        source is silent.
    """
    if peak <= 0.0:
        return None
    current_db = 20.0 * math.log10(peak)
    gain_db = float(target_db) - current_db
    if abs(gain_db) < 0.05:
        return None
    gain_db = max(-24.0, min(24.0, gain_db))
    return f"volume={gain_db:.3f}dB"


def _export_worker_count(job_count: int) -> int:
    """Return a conservative encode-worker cap so disk does not thrash."""
    cpu = os.cpu_count() or 2
    return max(1, min(job_count, cpu, _EXPORT_WORKER_CAP))


def _aac_codec(ffmpeg_path: str) -> str:
    """Return ``libfdk_aac`` when this FFmpeg build has it, else native ``aac``."""
    cached = _AAC_CODEC_CACHE.get(ffmpeg_path)
    if cached:
        return cached
    codec = "aac"
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            creationflags=_NOCONSOLE,
            check=False,
        )
        if result.returncode == 0 and "libfdk_aac" in (result.stdout or ""):
            codec = "libfdk_aac"
    except Exception as exc:
        logger.debug("FFmpeg encoder probe failed: %s", exc)
    _AAC_CODEC_CACHE[ffmpeg_path] = codec
    return codec


def _aac_encode_args(ffmpeg_path: str, sample_rate: int, bitrate: str = "128k") -> List[str]:
    """Build AAC encode arguments, keeping the source rate when already correct.

    Args:
        ffmpeg_path: FFmpeg executable used for the encoder probe.
        sample_rate: Requested output rate. Pass 0 to leave the source rate.
        bitrate: CBR AAC bitrate.

    Returns:
        FFmpeg argument list starting at the codec flags.
    """
    codec = _aac_codec(ffmpeg_path)
    args: List[str] = ["-c:a", codec, "-b:a", bitrate]
    if sample_rate > 0:
        args = ["-ar", str(sample_rate), *args]
    if codec == "aac":
        args.extend(["-aac_coder", "fast"])
    return args


def _encode_chapter_mp3(
    chapter: Dict[str, Any],
    mp3_path: Path,
    *,
    order: int,
    chapter_count: int,
    ffmpeg_path: str,
    album: str,
    artist: str,
    chapter_metadata: Dict[str, str],
    cover_path: Optional[Path],
) -> Path:
    """Encode one chapter's ordered chunk WAVs to a tagged MP3 file.

    Args:
        chapter: Chapter dict with ``wav_paths``, ``chapter_id``, and ``title``.
        mp3_path: Destination MP3 path.
        order: Zero-based chapter order used for track numbers and temp names.
        chapter_count: Total chapter count, used only for log text.
        ffmpeg_path: FFmpeg executable.
        album: Album tag, typically the book title.
        artist: Artist tag.
        chapter_metadata: Portable tags except per-file title.
        cover_path: Optional JPEG/PNG attached as cover art.

    Returns:
        The written ``mp3_path``.
    """
    cid = int(chapter.get("chapter_id") or 0)
    title = str(chapter.get("title") or f"Chapter {cid}")
    track = order + 1
    list_path = mp3_path.parent / f".concat_ch_{order:03d}.txt"
    try:
        _write_concat_list(chapter["wav_paths"], list_path)
        cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
        ]
        if cover_path is not None:
            cmd.extend(["-i", str(cover_path), "-map", "0:a:0", "-map", "1:v:0"])
        cmd.extend([
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            "-metadata",
            f"title={title}",
            "-metadata",
            f"track={track}",
        ])
        if cover_path is not None:
            cmd.extend(["-codec:v", "copy", "-disposition:v:0", "attached_pic"])
        cmd.extend(_metadata_args(chapter_metadata))
        if album:
            cmd.extend(["-metadata", f"album={album}"])
        if artist:
            cmd.extend(["-metadata", f"artist={artist}"])
        cmd.append(str(mp3_path))
        _run(cmd)
        logger.info(
            "Chapter MP3 %s/%s: %s (%s chunk WAV(s))",
            order + 1,
            chapter_count,
            mp3_path.name,
            len(chapter["wav_paths"]),
        )
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass
    return mp3_path


def export_chapter_mp3s(
    chapters: Sequence[Dict[str, Any]],
    output_dir: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    album: str = "",
    artist: str = "",
    metadata: Optional[Dict[str, str]] = None,
    cover_path: Optional[Path] = None,
) -> List[Path]:
    """Write one MP3 per chapter by feeding that chapter's chunk WAV list to ffmpeg.

    Uses a tiny concat list only (no intermediate full-chapter temp WAV):
    chunk_N.wav + chunk_N+1.wav + … → chapter.mp3 in one encode pass.
    Independent chapters encode in parallel.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chapter_metadata = dict(metadata or {})
    # Each chapter's title must describe that file, not be replaced by book title.
    chapter_metadata.pop("title", None)
    jobs: List[Tuple[int, Dict[str, Any], Path]] = []
    reserved_names: set[str] = set()
    for order, chapter in enumerate(chapters):
        cid = int(chapter.get("chapter_id") or 0)
        title = str(chapter.get("title") or f"Chapter {cid}")
        safe_title = re.sub(r'[<>:"/\\|?*]', "", title).strip() or f"Chapter_{cid}"
        mp3_path = output_dir / f"{cid:02d} - {safe_title}.mp3"
        # Avoid collisions for artificial cont. parts
        if mp3_path.exists() or str(mp3_path) in reserved_names:
            mp3_path = output_dir / f"{cid:02d}-{order:02d} - {safe_title}.mp3"
        reserved_names.add(str(mp3_path))
        jobs.append((order, chapter, mp3_path))

    def _run_job(job: Tuple[int, Dict[str, Any], Path]) -> Tuple[int, Path]:
        """Encode one reserved chapter job and return its original order."""
        order, chapter, mp3_path = job
        written = _encode_chapter_mp3(
            chapter,
            mp3_path,
            order=order,
            chapter_count=len(jobs),
            ffmpeg_path=ffmpeg_path,
            album=album,
            artist=artist,
            chapter_metadata=chapter_metadata,
            cover_path=cover_path,
        )
        return order, written

    if len(jobs) <= 1:
        return [_run_job(job)[1] for job in jobs]

    written_by_order: Dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=_export_worker_count(len(jobs))) as pool:
        futures = [pool.submit(_run_job, job) for job in jobs]
        for future in as_completed(futures):
            order, path = future.result()
            written_by_order[order] = path
    return [written_by_order[order] for order, _, _ in jobs]


def _normalization_af(
    m4b_config: Dict[str, Any],
    *,
    peak: Optional[float] = None,
) -> Optional[str]:
    """Return ffmpeg -af string for configured normalization, or None.

    ``peak`` applies a constant gain to ``target_db``.  ``loudness`` keeps the
    slower EBU ``loudnorm`` filter.  ``simple`` is a fixed ``volume=`` offset.
    """
    norm = str(m4b_config.get("normalization_type") or "peak")
    target_db = float(m4b_config.get("target_db") or -1.5)
    if norm in {"none", ""}:
        return None
    if norm == "peak":
        return peak_volume_filter(float(peak or 0.0), target_db)
    if norm == "simple":
        return f"volume={target_db}dB"
    if norm == "loudness":
        return "loudnorm=I=-16:TP=-1.5:LRA=11"
    return None


def mux_m4b_with_chapters(
    audio_input: Path,
    chapters_ffmetadata: Optional[Path],
    output_m4b: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    already_aac: bool = False,
    sample_rate: int = 24000,
    normalization_af: Optional[str] = None,
    concat_list: bool = False,
    cover_path: Optional[Path] = None,
    adts: bool = False,
) -> Path:
    """Encode/remux audio with optional FFMETADATA chapters into an M4B file.

    Args:
        audio_input: WAV/M4B path, or concat list when ``concat_list`` is True.
        chapters_ffmetadata: Optional chapter metadata file.
        output_m4b: Destination M4B path.
        concat_list: When True, ``audio_input`` is an ffmpeg concat demuxer list
            of chunk WAVs or already-encoded AAC/M4A parts.
        cover_path: Existing JPEG/PNG file to attach as audiobook cover art.
        adts: When copying concatenated ADTS AAC, apply ``aac_adtstoasc``.
    """
    output_m4b = Path(output_m4b)
    output_m4b.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_path, "-y"]
    if concat_list:
        cmd.extend(["-f", "concat", "-safe", "0", "-i", str(audio_input)])
    else:
        cmd.extend(["-i", str(audio_input)])
    metadata_input_index = None
    if chapters_ffmetadata is not None and Path(chapters_ffmetadata).exists():
        metadata_input_index = 1
        cmd.extend(["-i", str(chapters_ffmetadata)])
    if cover_path is not None:
        cover_input_index = 2 if metadata_input_index is not None else 1
        cmd.extend(["-i", str(cover_path), "-map", "0:a:0", "-map", f"{cover_input_index}:v:0"])
    if metadata_input_index is not None:
        cmd.extend(["-map_metadata", str(metadata_input_index)])
    if already_aac:
        if cover_path is not None:
            cmd.extend(["-c:a", "copy", "-c:v", "copy", "-disposition:v:0", "attached_pic"])
        else:
            cmd.extend(["-c:a", "copy"])
        if adts:
            cmd.extend(["-bsf:a", "aac_adtstoasc"])
        cmd.append(str(output_m4b))
    else:
        if normalization_af:
            cmd.extend(["-af", normalization_af])
        if cover_path is not None:
            cmd.extend(["-c:v", "copy", "-disposition:v:0", "attached_pic"])
        cmd.extend(_aac_encode_args(ffmpeg_path, sample_rate))
        cmd.append(str(output_m4b))
    _run(cmd)
    return output_m4b


def _encode_chapter_aac(
    chapter: Dict[str, Any],
    dest_aac: Path,
    *,
    ffmpeg_path: str,
    sample_rate: int,
    normalization_af: Optional[str],
) -> Path:
    """Encode one chapter's chunk WAV list to ADTS AAC for later stream-copy mux.

    Args:
        chapter: Chapter dict containing ``wav_paths``.
        dest_aac: Destination ``.aac`` path.
        ffmpeg_path: FFmpeg executable.
        sample_rate: Requested AAC sample rate.
        normalization_af: Optional volume/loudnorm filter shared across chapters.

    Returns:
        The written ``dest_aac`` path.
    """
    dest_aac = Path(dest_aac)
    dest_aac.parent.mkdir(parents=True, exist_ok=True)
    list_path = dest_aac.with_suffix(".concat.txt")
    try:
        _write_concat_list(chapter["wav_paths"], list_path)
        cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
        ]
        if normalization_af:
            cmd.extend(["-af", normalization_af])
        cmd.extend(_aac_encode_args(ffmpeg_path, sample_rate))
        cmd.append(str(dest_aac))
        _run(cmd)
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass
    return dest_aac


def mux_m4b_from_parallel_chapters(
    chapters: Sequence[Dict[str, Any]],
    chapters_ffmetadata: Optional[Path],
    output_m4b: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    sample_rate: int = 24000,
    normalization_af: Optional[str] = None,
    cover_path: Optional[Path] = None,
) -> Path:
    """Encode chapters to AAC in parallel, then remux one chapterized M4B.

    Args:
        chapters: Planned chapter groups with ``wav_paths``.
        chapters_ffmetadata: FFMETADATA1 file with chapter timestamps.
        output_m4b: Destination M4B path.
        ffmpeg_path: FFmpeg executable.
        sample_rate: AAC sample rate.
        normalization_af: Shared filter so chapter loudness stays consistent.
        cover_path: Optional JPEG/PNG cover art.

    Returns:
        The written ``output_m4b`` path.
    """
    output_m4b = Path(output_m4b)
    output_m4b.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_m4b.parent / f".{output_m4b.stem}.m4b_parts"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        dests = [work_dir / f"ch_{index:03d}.aac" for index in range(len(chapters))]

        def _job(item: Tuple[Dict[str, Any], Path]) -> Path:
            """Encode one chapter AAC part."""
            chapter, dest = item
            return _encode_chapter_aac(
                chapter,
                dest,
                ffmpeg_path=ffmpeg_path,
                sample_rate=sample_rate,
                normalization_af=normalization_af,
            )

        items = list(zip(chapters, dests))
        if len(items) <= 1:
            for item in items:
                _job(item)
        else:
            with ThreadPoolExecutor(max_workers=_export_worker_count(len(items))) as pool:
                futures = [pool.submit(_job, item) for item in items]
                for future in as_completed(futures):
                    future.result()
        list_path = work_dir / "chapters.concat.txt"
        _write_concat_list([str(path) for path in dests], list_path)
        return mux_m4b_with_chapters(
            list_path,
            chapters_ffmetadata,
            output_m4b,
            ffmpeg_path=ffmpeg_path,
            already_aac=True,
            concat_list=True,
            cover_path=cover_path,
            adts=True,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def encode_full_mp3_from_chunks(
    wav_paths: Sequence[str],
    output_mp3: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    title: str = "",
    album: str = "",
    artist: str = "",
    metadata: Optional[Dict[str, str]] = None,
    cover_path: Optional[Path] = None,
) -> Path:
    """Encode one full-book MP3 by concatenating chunk WAVs in a single pass."""
    output_mp3 = Path(output_mp3)
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_mp3.with_suffix(".concat.txt")
    _write_concat_list(wav_paths, list_path)
    try:
        cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
        ]
        if cover_path is not None:
            cmd.extend(["-i", str(cover_path), "-map", "0:a:0", "-map", "1:v:0"])
        cmd.extend([
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
        ])
        if cover_path is not None:
            cmd.extend(["-codec:v", "copy", "-disposition:v:0", "attached_pic"])
        if title:
            cmd.extend(["-metadata", f"title={title}"])
        if album:
            cmd.extend(["-metadata", f"album={album}"])
        if artist:
            cmd.extend(["-metadata", f"artist={artist}"])
        cmd.extend(_metadata_args(metadata))
        cmd.append(str(output_mp3))
        _run(cmd)
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass
    return output_mp3


def export_book_chapters(
    book_dir: Path,
    *,
    wav_path: Optional[Path] = None,
    chunks_json: Optional[Path] = None,
    audio_chunks_dir: Optional[Path] = None,
    chapterize: bool = True,
    max_chapter_minutes: float | int = 0.0,
    chapter_mode: Optional[str] = None,
    write_m4b: bool = True,
    write_mp3: bool = False,
    write_wav: bool = False,
    m4b_config: Optional[Dict[str, Any]] = None,
    title: str = "",
    artist: str = "",
    metadata: Optional[Dict[str, str]] = None,
    cover_path: Optional[Path] = None,
    output_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Build chapter map and export only the requested formats efficiently.

    M4B is encoded from the ordered chunk WAV list (no intermediate full WAV)
    unless ``wav_path`` already exists and ``write_wav`` was used upstream.
    Multi-chapter M4B encodes chapters in parallel then remuxes. Peak
    normalization is a constant gain, not EBU loudnorm. MP3 chapters use
    per-chapter chunk lists in parallel. Full WAV is only written when
    ``write_wav`` is True.

    Args:
        book_dir: Main book output directory (parent of TTS/).
        wav_path: Optional existing full WAV path / destination for write_wav.
        chunks_json: Path to audiobook.chunks.json.
        audio_chunks_dir: Directory of chunk_XXXXX.wav files.
        chapterize: Enable a chapter TOC rather than one full-book chapter.
        max_chapter_minutes: Minute value for the selected chapter strategy.
        chapter_mode: ``headings_only``, ``headings_or_minutes``, or
            ``headings_with_max``.  Missing legacy values retain prior behavior.
        write_m4b / write_mp3 / write_wav: Which deliverables to create.
        m4b_config: Normalization/ffmpeg settings from config.m4b.
        title: Album/book title metadata.
        artist: Artist metadata.
        metadata: Normalized export tags from the Metadata dialog.
        cover_path: JPEG/PNG artwork embedded in M4B and MP3 outputs.
        output_mode: Legacy ``m4b``/``mp3``/``both`` (overridden by write_*).

    Returns:
        Dict with paths and chapter summary.
    """
    book_dir = Path(book_dir)
    tts_dir = book_dir / "TTS"
    chunks_json = Path(chunks_json or tts_dir / "text_chunks" / "audiobook.chunks.json")
    audio_chunks_dir = Path(audio_chunks_dir or tts_dir / "audio_chunks")
    m4b_config = dict(m4b_config or {})
    ffmpeg_path = str(m4b_config.get("ffmpeg_path") or "ffmpeg")
    if metadata is None:
        from pocket_tts.audio.metadata import export_metadata_tags, metadata_cover_path

        metadata = export_metadata_tags(m4b_config.get("metadata"), title_fallback=title)
        cover_path = cover_path or metadata_cover_path(m4b_config.get("metadata"))

    # Legacy output_mode support for CLI / older callers.
    if output_mode:
        mode = str(output_mode).strip().lower()
        if mode == "m4b":
            write_m4b, write_mp3 = True, False
        elif mode == "mp3":
            write_m4b, write_mp3 = False, True
        elif mode == "both":
            write_m4b, write_mp3 = True, True

    chunks, meta = load_chunks_json(chunks_json)
    if not chunks:
        raise ValueError(f"No chunks in {chunks_json}")
    timeline, chapters, resolved_chapter_mode, headings_found = build_chapter_export_plan(
        chunks,
        audio_chunks_dir,
        chapterize=chapterize,
        max_chapter_minutes=max_chapter_minutes,
        chapter_mode=chapter_mode,
    )

    metadata = {key: str(value) for key, value in (metadata or {}).items() if value}
    book_title = title or metadata.get("title") or meta.get("output_filename") or book_dir.name
    book_title = Path(str(book_title)).stem
    total_duration = timeline[-1]["end_s"] if timeline else 0.0
    all_wav_paths = [row["wav_path"] for row in timeline]

    chapters_json_path = tts_dir / "chapters.json"
    chapters_json_path.write_text(
        json.dumps(
            {
                "title": book_title,
                "total_duration_s": total_duration,
                "chapters": [
                    {
                        "order": i,
                        "chapter_id": c["chapter_id"],
                        "title": c["title"],
                        "start_s": c["start_s"],
                        "end_s": c["end_s"],
                        "start_chunk": c["start_chunk"],
                        "end_chunk": c["end_chunk"],
                    }
                    for i, c in enumerate(chapters)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ffmeta_path = tts_dir / "chapters.ffmetadata"
    write_ffmetadata(
        chapters if chapterize else [],
        ffmeta_path,
        title=book_title,
        artist=artist,
        metadata=metadata,
    )

    result: Dict[str, Any] = {
        "chapters_json": str(chapters_json_path),
        "ffmetadata": str(ffmeta_path),
        "chapter_count": len(chapters),
        "total_duration_s": total_duration,
        "m4b_path": None,
        "mp3_dir": None,
        "mp3_files": [],
        "wav_path": None,
        "chapter_mode": resolved_chapter_mode,
        "headings_found": headings_found,
    }

    # Optional full WAV from chunk list (only when requested).
    if write_wav:
        if wav_path is None:
            wav_path = book_dir / f"{book_title}.wav"
        wav_path = Path(wav_path)
        if not wav_path.exists() or wav_path.stat().st_size < 1000:
            logger.info("Building full WAV from %s chunk files…", len(all_wav_paths))
            concat_wavs(all_wav_paths, wav_path, ffmpeg_path=ffmpeg_path)
        result["wav_path"] = str(wav_path)
        logger.info("Full WAV: %s", wav_path)
        if metadata:
            try:
                apply_media_metadata(wav_path, metadata, ffmpeg_path=ffmpeg_path)
            except Exception as exc:
                # WAV metadata availability differs by FFmpeg build and player.
                logger.warning("Could not add WAV metadata to %s: %s", wav_path, exc)

    sample_rate = int(m4b_config.get("sample_rate") or 24000)
    try:
        with wave.open(str(all_wav_paths[0]), "rb") as probe:
            if probe.getframerate() == sample_rate:
                # Same rate as the chunks: skip a no-op resample in the encoder.
                sample_rate = 0
    except Exception:
        pass
    peak: Optional[float] = None
    if write_m4b and str(m4b_config.get("normalization_type") or "peak") == "peak":
        logger.info("Measuring PCM peak across %s chunk WAV(s)…", len(all_wav_paths))
        peak = measure_pcm_peak(all_wav_paths)
        logger.info("Measured PCM peak=%.4f", peak)
    af = _normalization_af(m4b_config, peak=peak)

    if write_m4b:
        if wav_path is None:
            # Prefer naming beside book dir using title.
            m4b_path = book_dir / f"{book_title}.m4b"
        else:
            m4b_path = Path(wav_path).with_suffix(".m4b")
        if chapterize and len(chapters) > 1:
            mux_m4b_from_parallel_chapters(
                chapters,
                ffmeta_path,
                m4b_path,
                ffmpeg_path=ffmpeg_path,
                sample_rate=sample_rate,
                normalization_af=af,
                cover_path=cover_path,
            )
            logger.info("M4B written from %s parallel chapter encodes: %s", len(chapters), m4b_path)
        else:
            # Single-pass encode from the ordered chunk list (no full WAV required).
            list_path = tts_dir / ".m4b_concat.txt"
            try:
                _write_concat_list(all_wav_paths, list_path)
                mux_m4b_with_chapters(
                    list_path,
                    ffmeta_path,
                    m4b_path,
                    ffmpeg_path=ffmpeg_path,
                    already_aac=False,
                    sample_rate=sample_rate,
                    normalization_af=af,
                    concat_list=True,
                    cover_path=cover_path,
                )
            finally:
                try:
                    list_path.unlink(missing_ok=True)
                except Exception:
                    pass
            logger.info("M4B written from chunk list: %s", m4b_path)
        result["m4b_path"] = str(m4b_path)

    if write_mp3:
        if chapterize and len(chapters) > 1:
            mp3_dir = book_dir / "chapters"
            files = export_chapter_mp3s(
                chapters,
                mp3_dir,
                ffmpeg_path=ffmpeg_path,
                album=book_title,
                artist=artist,
                metadata=metadata,
                cover_path=cover_path,
            )
            result["mp3_dir"] = str(mp3_dir)
            result["mp3_files"] = [str(p) for p in files]
            logger.info("Wrote %s chapter MP3s under %s", len(files), mp3_dir)
        else:
            # One full-book MP3 without per-chapter split.
            mp3_path = book_dir / f"{book_title}.mp3"
            encode_full_mp3_from_chunks(
                all_wav_paths,
                mp3_path,
                ffmpeg_path=ffmpeg_path,
                title=book_title,
                album=book_title,
                artist=artist,
                metadata=metadata,
                cover_path=cover_path,
            )
            result["mp3_files"] = [str(mp3_path)]
            logger.info("Full-book MP3 written: %s", mp3_path)

    return result
