"""Normalize user-entered audiobook fields into portable FFmpeg metadata tags."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

DEFAULT_ARTIST = "Dnxs Spokenword PocketGPU"
COVER_ART_EXTENSIONS = (".jpg", ".jpeg", ".png")
COVER_ART_FILTER = "Cover Images (*.jpg *.jpeg *.png)"


def metadata_cover_path(raw_metadata: Dict[str, Any] | None) -> Path | None:
    """Return an existing JPEG or PNG cover path, otherwise no embeddable art."""
    value = str((raw_metadata or {}).get("cover_path", "") or "").strip()
    path = Path(value) if value else None
    if path is None or path.suffix.lower() not in COVER_ART_EXTENSIONS or not path.is_file():
        return None
    return path


def export_metadata_tags(raw_metadata: Dict[str, Any] | None, *, title_fallback: str = "") -> Dict[str, str]:
    """Map Metadata-dialog values to tags understood by M4B, MP3, and WAV.

    ``album`` and ``show`` make the user-facing series usable in common music
    and audiobook players; the original ``series`` keys remain for containers
    that retain custom tags.
    """
    raw = raw_metadata or {}
    values = {
        key: str(raw.get(key, "") or "").strip()
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
        )
    }
    values["artist"] = values["artist"] or DEFAULT_ARTIST
    values["title"] = values["title"] or str(title_fallback or "").strip()

    tags = {
        "author": values["author"],
        "series": values["series"],
        "show": values["series"],
        "album": values["series"],
        "series_number": values["series_number"],
        "episode_id": values["series_number"],
        "title": values["title"],
        "composer": values["composer"],
        "date": values["year"],
        "genre": values["genre"],
        "artist": values["artist"],
        "album_artist": values["artist"],
        "description": values["description"],
        "comment": values["description"],
    }
    return {key: value for key, value in tags.items() if value}
