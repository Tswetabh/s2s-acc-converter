"""Tests for SmartChunker mode semantics and early period + merge."""

import unittest

from pocket_tts.preprocessing.chunker import SmartChunker
from pocket_tts.preprocessing.structure_detector import StructureDetector
from pocket_tts.preprocessing.text_normalizer import (
    ensure_terminal_punctuation_on_lines,
    flatten_newlines_for_json,
)


class ChunkerModeSemanticsTests(unittest.TestCase):
    """sentence / paragraph / word_target behave as documented."""

    def _structure(self, text: str):
        """Analyze raw text after early line-end periods."""
        text = ensure_terminal_punctuation_on_lines(text)
        return StructureDetector().analyze(text)

    def test_paragraph_mode_ignores_min_words(self) -> None:
        """Two short paragraphs stay two chunks even if min_words is large."""
        raw = "Short one.\n\nAlso short."
        structure = self._structure(raw)
        chunks = SmartChunker(mode="paragraph", min_words=50).chunk(structure)
        self.assertEqual(len(chunks), 2)
        self.assertIn("Short one.", chunks[0].text)
        self.assertIn("Also short.", chunks[1].text)

    def test_sentence_mode_soft_floor_merges(self) -> None:
        """Short sentences pack until min_words floor."""
        raw = "One two. Three four five six."
        structure = self._structure(raw)
        chunks = SmartChunker(mode="sentence", min_words=4).chunk(structure)
        # Both sentences fit one chunk once floor met by second.
        flat = [flatten_newlines_for_json(c.text) for c in chunks]
        joined = " ".join(flat)
        self.assertIn("One two.", joined)
        self.assertIn("Three four five six.", joined)

    def test_word_target_respects_cap(self) -> None:
        """Next sentence is not added when it would exceed target_words."""
        raw = "AAAA BBBB CCCC. DDDD EEEE FFFF GGGG."
        structure = self._structure(raw)
        # First sentence 3 words, second 4; cap 3 → two chunks.
        chunks = SmartChunker(
            mode="word_target",
            min_words=1,
            target_words=3,
        ).chunk(structure)
        self.assertEqual(len(chunks), 2)
        self.assertLessEqual(chunks[0].word_count, 3)
        # Second sentence alone is 4 > cap and stays whole.
        self.assertEqual(chunks[1].word_count, 4)

    def test_word_target_long_sentence_alone(self) -> None:
        """A single sentence longer than the cap is not split."""
        raw = "One two three four five six seven."
        structure = self._structure(raw)
        chunks = SmartChunker(
            mode="word_target",
            min_words=2,
            target_words=3,
        ).chunk(structure)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].word_count, 7)

    def test_early_period_then_sentence_merge(self) -> None:
        """Bare title line keeps period after soft merge with body."""
        raw = "Chapter One\n\nStart of the first paragraph text here."
        text = ensure_terminal_punctuation_on_lines(raw)
        structure = StructureDetector().analyze(text)
        chunks = SmartChunker(mode="sentence", min_words=4).chunk(structure)
        flat = flatten_newlines_for_json(chunks[0].text)
        self.assertIn("Chapter One. Start", flat)
        self.assertNotIn("Chapter One Start", flat)

    def test_word_target_packs_across_blank_lines_to_cap(self) -> None:
        """Double newlines do not force splits; pack toward target_words=50.

        User fixture: six blank-line blocks, ~35 words total → under cap so
        paragraph ends must not yield six chunks.
        """
        raw = (
            "The last paragraph of the previous chapter.\n\n"
            "[2s]\n"
            "Chapter One\n\n"
            "Start of the first paragraph text.\n\n"
            "And chunk it to:\n"
            "The last paragraph of the previous chapter.\n\n"
            "Chapter\n"
            "[2s] One\n\n"
            "Start of the first[2s]paragraph text."
        )
        structure = self._structure(raw)
        total_words = sum(s.word_count for s in structure.sentences)
        self.assertLess(total_words, 50)

        chunks = SmartChunker(
            mode="word_target",
            min_words=4,
            target_words=50,
            respect_boundaries=True,  # ignored: must not re-enable para flushes
        ).chunk(structure)

        # Entire fixture is under 50 words → one packed chunk (not 6 paras).
        self.assertEqual(len(chunks), 1, [c.word_count for c in chunks])
        self.assertEqual(chunks[0].word_count, total_words)
        flat = flatten_newlines_for_json(chunks[0].text)
        self.assertIn("Chapter One.", flat)
        self.assertIn("Start of the first", flat)


if __name__ == "__main__":
    unittest.main()
