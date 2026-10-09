"""Chapter/part header detection for audiobook structure and export.

Rules:
- Chapter 0 is all text before the first real Part/Chapter header (no heading required).
- Part One / Chapter One (word or digit) starts chapter 1.
- A header must begin a sentence (after strip), not appear mid-prose paragraph.
- Prologue/foreword/title before the first Part/Chapter stay in chapter 0.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Word-form ordinals used in fiction headings (extend as needed).
_WORD_NUMBERS: Dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21,
    "twenty one": 21,
    "twenty-two": 22,
    "twenty two": 22,
    "twenty-three": 23,
    "twenty three": 23,
    "twenty-four": 24,
    "twenty four": 24,
    "twenty-five": 25,
    "twenty five": 25,
    "twenty-six": 26,
    "twenty six": 26,
    "twenty-seven": 27,
    "twenty seven": 27,
    "twenty-eight": 28,
    "twenty eight": 28,
    "twenty-nine": 29,
    "twenty nine": 29,
    "thirty": 30,
}

# Word-number body after Chapter/Part (longest forms first via alternation order).
_WORD_BODY = (
    r"Twenty[-\s]?One|Twenty[-\s]?Two|Twenty[-\s]?Three|Twenty[-\s]?Four|"
    r"Twenty[-\s]?Five|Twenty[-\s]?Six|Twenty[-\s]?Seven|Twenty[-\s]?Eight|"
    r"Twenty[-\s]?Nine|Thirty|Twenty|"
    r"Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|"
    r"Eleven|Twelve|Ten|One|Two|Three|Four|Five|Six|Seven|Eight|Nine"
)

# Optional inline pause markers and whitespace between header tokens.
# Allows dramatic titles: "[8s]\nChapter\n[8s] One."
_PAUSE_GAP = r"(?:\[\d+(?:\.\d+)?s\]|\s)*"

# Leading header at sentence/line start only (not mid-prose "this chapter").
_HEADER_RE = re.compile(
    rf"""
    ^\s*
    (?:{_PAUSE_GAP})?
    (?:
        (?P<label_chapter>Chapter|CHAPTER|Ch\.)
        {_PAUSE_GAP}
        (?:
            (?P<ch_digits>\d+)
            | (?P<ch_roman>[IVXLCDM]{{1,6}})
            | (?P<ch_words>{_WORD_BODY})
        )
        |
        (?P<label_part>Part|PART|Book|BOOK)
        {_PAUSE_GAP}
        (?:
            (?P<part_digits>\d+)
            | (?P<part_roman>[IVXLCDM]{{1,6}})
            | (?P<part_words>{_WORD_BODY})
        )
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Mid-prose rejection: "this chapter", "the chapter", etc. already fail ^ match.
# Also reject when header is clearly not at structural start (handled by callers).

_ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20, "XXI": 21,
    "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25, "XXVI": 26, "XXX": 30,
}


def _parse_roman(token: str) -> Optional[int]:
    """Return integer value for a common Roman numeral, or None."""
    return _ROMAN_MAP.get(token.upper())


def _parse_word_number(token: str) -> Optional[int]:
    """Return integer for a word-form number such as twenty-one."""
    key = re.sub(r"\s+", " ", token.strip().lower().replace("-", " "))
    key_hyphen = key.replace(" ", "-")
    if key_hyphen in _WORD_NUMBERS:
        return _WORD_NUMBERS[key_hyphen]
    if key in _WORD_NUMBERS:
        return _WORD_NUMBERS[key]
    # Normalize "twenty one" variants already in map; try compacted hyphen form.
    compact = key.replace(" ", "-")
    return _WORD_NUMBERS.get(compact)


def match_chapter_header(text: str) -> Optional[Dict[str, Any]]:
    """Return header info when ``text`` begins with a Part/Chapter heading.

    Args:
        text: Sentence or chunk text to inspect.

    Returns:
        Dict with keys ``title``, ``kind`` (chapter|part|book), ``number``
        (optional int), or None when this is not a structural header.
    """
    if not text or not text.strip():
        return None
    # Only the first line / sentence start counts — strip leading quotes.
    candidate = text.lstrip()
    candidate = re.sub(r'^[\"\'“”‘’]+', "", candidate)
    match = _HEADER_RE.match(candidate)
    if not match:
        return None

    kind = "chapter"
    if match.group("label_part"):
        label = match.group("label_part")
        kind = "book" if label.lower() == "book" else "part"
        digits = match.group("part_digits")
        roman = match.group("part_roman")
        words = match.group("part_words")
        label_text = label
    else:
        digits = match.group("ch_digits")
        roman = match.group("ch_roman")
        words = match.group("ch_words")
        label_text = match.group("label_chapter")

    number: Optional[int] = None
    number_token = ""
    if digits:
        number = int(digits)
        number_token = digits
    elif roman:
        number = _parse_roman(roman)
        number_token = roman
    elif words:
        number = _parse_word_number(words)
        number_token = words

    # Prefer a clean "Chapter One" / "Part Two" form without pause markers.
    if number_token:
        title = f"{label_text} {number_token}".strip()
        title = re.sub(r"\s+", " ", title)
    else:
        title = candidate[: match.end()].strip()
        title = re.sub(r"\[\d+(?:\.\d+)?s\]", " ", title)
        title = re.sub(r"\s+", " ", title).strip()

    return {
        "title": title,
        "kind": kind,
        "number": number,
        "match_end": match.end(),
    }


def is_chapter_header_sentence(text: str) -> bool:
    """Return True when text starts with a structural chapter/part header."""
    return match_chapter_header(text) is not None


def match_chapter_header_in_chunk(text: str) -> Optional[Dict[str, Any]]:
    """Detect a header at chunk start or after a short sentence-end orphan.

    Used when legacy min_words bundling glued ``Alistair?`` onto
    ``Chapter Eight …``. Only a short lead-in (≤4 words / ≤40 chars) may
    precede the header so title pages like ``Warlord Born. Book One.`` stay
    front matter while ``Alistair? Chapter Eight`` still splits.
    """
    direct = match_chapter_header(text)
    if direct:
        return direct
    if not text:
        return None
    # Only inspect the opening stretch so deep prose "this chapter" is ignored.
    window = text[:200]
    for match in re.finditer(r"[\.!\?][\"'”’)]*\s+", window):
        lead_in = window[: match.start()].strip()
        lead_words = lead_in.split()
        terminator = match.group(0)[0]
        # Short orphans only: "Alistair? Chapter Eight" yes;
        # "Warlord Born. Book One." (title page) no.
        max_words = 3 if terminator == "?" else 1
        if len(lead_words) > max_words or len(lead_in) > 40:
            continue
        header = match_chapter_header(window[match.end() :])
        if header:
            header = dict(header)
            header["offset"] = match.end()
            return header
    return None


def assign_chapter_ids(
    chunks: Sequence[Dict[str, Any]],
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """Stamp sequential chapter_id/title onto chunk dicts (0 before first header).

    Args:
        chunks: Ordered chunk records with text under ``text_key``.
        text_key: Field name holding chunk text.

    Returns:
        New list of chunk dicts with ``chapter_id`` and ``chapter_title`` set.
    """
    out: List[Dict[str, Any]] = []
    chapter_id = 0
    chapter_title = "Front matter"
    for raw in chunks:
        chunk = dict(raw)
        text = str(chunk.get(text_key) or "")
        header = match_chapter_header_in_chunk(text)
        if header:
            chapter_id += 1
            chapter_title = header["title"]
            chunk["boundary_type"] = chunk.get("boundary_type") or "chapter_start"
            chunk["is_chapter_start"] = True
        else:
            chunk["is_chapter_start"] = False
        chunk["chapter_id"] = chapter_id
        chunk["chapter_title"] = chapter_title
        chunk["chapter_number"] = chapter_id if chapter_id > 0 else None
        out.append(chunk)
    return out


def stamp_chunk_metadata_chapters(chunks: Sequence[Any]) -> None:
    """Mutate ChunkMetadata-like objects with chapter_id / chapter_title / number.

    Args:
        chunks: Sequence of objects with ``.text`` and optional chapter fields.
    """
    chapter_id = 0
    chapter_title = "Front matter"
    for chunk in chunks:
        text = str(getattr(chunk, "text", "") or "")
        header = match_chapter_header_in_chunk(text)
        if header:
            chapter_id += 1
            chapter_title = header["title"]
            # Prefer chapter_start boundary when this chunk opens a chapter.
            try:
                from .schema import BoundaryType

                chunk.boundary_type = BoundaryType.CHAPTER_START
            except Exception:
                pass
        chunk.chapter_number = chapter_id if chapter_id > 0 else None
        # Optional attributes used by export
        try:
            chunk.chapter_id = chapter_id
            chunk.chapter_title = chapter_title
        except Exception:
            pass


# Patterns exposed for StructureDetector line-based scanning (multiline).
STRUCTURE_HEADER_LINE_PATTERNS: Tuple[str, ...] = (
    r"^(?:Chapter|CHAPTER|Ch\.)\s+(?:\d+|[IVXLCDM]+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|"
    r"Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|"
    r"Twenty[-\s]?One|Twenty[-\s]?Two|Twenty[-\s]?Three|Twenty[-\s]?Four|Twenty[-\s]?Five|"
    r"Twenty[-\s]?Six|Twenty[-\s]?Seven|Twenty[-\s]?Eight|Twenty[-\s]?Nine|Thirty)\b",
    r"^(?:Part|PART|Book|BOOK)\s+(?:\d+|[IVXLCDM]+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|"
    r"Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|"
    r"Twenty[-\s]?One|Twenty[-\s]?Two|Twenty[-\s]?Three|Twenty[-\s]?Four|Twenty[-\s]?Five|"
    r"Twenty[-\s]?Six|Thirty)\b",
)
