"""Tests for chapter/part header detection and chunk chapter stamping."""

from pocket_tts.preprocessing.chapter_headers import (
    assign_chapter_ids,
    is_chapter_header_sentence,
    match_chapter_header,
)
from pocket_tts.preprocessing.chunker import SmartChunker
from pocket_tts.preprocessing.schema import (
    ParagraphInfo,
    SentenceInfo,
    TextStructure,
)
from pocket_tts.preprocessing.structure_detector import StructureDetector


def test_word_number_and_part_headers() -> None:
    """Chapter One / Part Two at sentence start are headers; mid-prose is not."""
    assert is_chapter_header_sentence('Chapter One "The Titans are the best of us.')
    assert is_chapter_header_sentence("Part Two")
    assert is_chapter_header_sentence("Chapter Twenty-One")
    assert is_chapter_header_sentence("Chapter 12")
    assert not is_chapter_header_sentence(
        "The focus of this chapter is on Alistair Kane's modification."
    )
    assert not is_chapter_header_sentence(
        "Perhaps I write this brief chapter out of selfishness."
    )
    info = match_chapter_header("Chapter Eight \"Demo\"")
    assert info is not None
    assert info["number"] == 8


def test_chapter_header_through_inline_pause_markers() -> None:
    """Leading and mid-title [Xs] markers must not hide Chapter One headers."""
    assert is_chapter_header_sentence("[8s]\nChapter One.")
    assert is_chapter_header_sentence("[8s]\nChapter\n[8s] One.")
    assert is_chapter_header_sentence("Chapter\n[8s] One")
    info = match_chapter_header("[8s]\nChapter\n[8s] One.")
    assert info is not None
    assert info["number"] == 1
    assert info["title"] == "Chapter One"

    rows = assign_chapter_ids(
        [
            {"index": 0, "text": "The last paragraph of the previous chapter."},
            {"index": 1, "text": "[8s]\nChapter\n[8s] One."},
            {"index": 2, "text": "Start of the first paragraph text."},
        ]
    )
    assert [r["chapter_id"] for r in rows] == [0, 1, 1]
    assert rows[1]["chapter_title"] == "Chapter One"


def test_structure_marks_word_chapter_sentences() -> None:
    """Structure detector marks word-form chapter lines as is_chapter."""
    text = (
        "Title page here.\n\n"
        "Chapter One\n"
        "Holmes sat by the fire.\n\n"
        "Part Two\n"
        "Watson entered the room.\n"
    )
    structure = StructureDetector().analyze(text)
    chapter_sentences = [s for s in structure.sentences if s.is_chapter]
    assert len(chapter_sentences) >= 2
    assert any("Chapter One" in s.text for s in chapter_sentences)
    assert any("Part Two" in s.text for s in chapter_sentences)


def test_orphan_before_header_goes_to_previous_chunk() -> None:
    """Short Alistair? before Chapter Eight flushes to the previous chunk."""
    paragraphs = [
        ParagraphInfo(0, 200, "placeholder"),
    ]
    sentences = [
        SentenceInfo("One two three four five.", 0, 24, 5, ".", ends_paragraph=False),
        SentenceInfo("Alistair?", 25, 34, 1, "?", ends_paragraph=False),
        SentenceInfo(
            'Chapter Eight "Demockracy was tried."',
            35,
            80,
            5,
            ".",
            is_chapter=True,
            ends_paragraph=True,
        ),
        SentenceInfo("More words after the header line here.", 81, 120, 7, ".", ends_paragraph=True),
    ]
    structure = TextStructure([], paragraphs, sentences, 18, 120)
    chunks = SmartChunker(mode="sentence", min_words=4).chunk(structure)
    texts = [c.text for c in chunks]
    # Orphan attaches to previous; chapter starts its own chunk.
    assert any(t.startswith("Chapter Eight") or "Chapter Eight" in t and not t.startswith("Alistair") for t in texts)
    orphan_chunk = next(c for c in chunks if "Alistair?" in c.text)
    assert "Chapter Eight" not in orphan_chunk.text
    chapter_chunk = next(c for c in chunks if c.text.strip().startswith("Chapter Eight"))
    assert chapter_chunk.chapter_id == 1
    assert chunks[0].chapter_id == 0


def test_assign_chapter_ids_front_matter_is_zero() -> None:
    """Text before first Part/Chapter is chapter 0."""
    rows = assign_chapter_ids(
        [
            {"index": 0, "text": "Warlord Born. Book One."},
            {"index": 1, "text": "Prologue: history."},
            {"index": 2, "text": 'Chapter One "Hello."'},
            {"index": 3, "text": "Body of chapter one continues."},
            {"index": 4, "text": "Part Two"},
        ]
    )
    assert [r["chapter_id"] for r in rows] == [0, 0, 1, 1, 2]
    assert rows[0]["chapter_title"] == "Front matter"
    assert rows[2]["chapter_title"].startswith("Chapter One")
