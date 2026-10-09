"""Regression tests for shared sentence-boundary detection."""

from pocket_tts.preprocessing.chunker import SmartChunker
from pocket_tts.preprocessing.structure_detector import StructureDetector


def _sentence_texts(text: str) -> list[str]:
    """Return sentence texts produced by the shared detector."""
    return [sentence.text for sentence in StructureDetector().segment_sentences(text)]


def test_abbreviations_initials_and_acronyms_stay_with_sentence() -> None:
    """Common abbreviations, initials, and acronyms should not split sentences."""
    text = "Dr. Smith met J. R. R. Tolkien in the U.S.A. museum. They spoke softly."
    sentences = StructureDetector().segment_sentences(text)

    assert _sentence_texts(text) == [
        "Dr. Smith met J. R. R. Tolkien in the U.S.A. museum.",
        "They spoke softly.",
    ]
    assert sentences[0].start_position == 0
    assert sentences[0].end_position == len("Dr. Smith met J. R. R. Tolkien in the U.S.A. museum.")
    assert sentences[1].start_position == len("Dr. Smith met J. R. R. Tolkien in the U.S.A. museum. ")
    assert sentences[1].end_position == len(text)


def test_decimals_quotes_and_ellipses_keep_sentence_spans() -> None:
    """Decimals, quoted terminal punctuation, and ellipses should stay intact."""
    text = 'She said, "Wait..." Then he wrote 3.14 on the board. "Fine," she answered.'

    assert _sentence_texts(text) == [
        'She said, "Wait..."',
        "Then he wrote 3.14 on the board.",
        '"Fine," she answered.',
    ]


def test_dotted_abbreviation_can_end_before_capitalized_sentence() -> None:
    """A final abbreviation period stays available when a new sentence follows."""
    text = "The alarm rang at 5 p.m. The Hrum advanced."

    assert _sentence_texts(text) == [
        "The alarm rang at 5 p.m.",
        "The Hrum advanced.",
    ]


def test_quoted_terminal_punctuation_stays_with_sentence() -> None:
    """Closing quotes should stay attached to the sentence-ending punctuation."""
    text = 'He said, "Go now." Then he left.'

    assert _sentence_texts(text) == [
        'He said, "Go now."',
        "Then he left.",
    ]


def test_pause_markers_do_not_split_sentences() -> None:
    """Inline pause markers must survive sentence segmentation unchanged."""
    text = "He paused [0.5s] before speaking. Then he left."

    assert _sentence_texts(text) == [
        "He paused [0.5s] before speaking.",
        "Then he left.",
    ]


def test_realistic_passage_one_preserves_offsets() -> None:
    """A realistic book-like passage should keep stable source offsets."""
    text = (
        "Dr. Hale opened the ledger. He noted the U.S.A. coordinates, then turned the page."
    )
    sentences = StructureDetector().segment_sentences(text)

    assert _sentence_texts(text) == [
        "Dr. Hale opened the ledger.",
        "He noted the U.S.A. coordinates, then turned the page.",
    ]
    assert sentences[0].start_position == 0
    assert sentences[0].end_position == len("Dr. Hale opened the ledger.")
    assert sentences[1].start_position == len("Dr. Hale opened the ledger. ")
    assert sentences[1].end_position == len(text)
    assert [text[item.start_position:item.end_position] for item in sentences] == [
        item.text for item in sentences
    ]


def test_realistic_passage_two_keeps_chunker_paragraph_mode_in_sync() -> None:
    """Paragraph chunking should consume the same sentence spans as detection."""
    text = (
        'Captain Vale crossed the room. "Stay alert," she told him.\n\n'
        "The engine hummed [0.5s] under the deck. He checked the 2.5-ton crate and nodded."
    )
    structure = StructureDetector().analyze(text)
    chunks = SmartChunker(mode="paragraph", min_words=1).chunk(structure)

    assert [chunk.text for chunk in chunks] == [
        'Captain Vale crossed the room. "Stay alert," she told him.',
        "The engine hummed [0.5s] under the deck. He checked the 2.5-ton crate and nodded.",
    ]
