"""Tests for auto-period on lines lacking any terminal punctuation."""

import unittest

from pocket_tts.preprocessing.text_normalizer import (
    ensure_terminal_punctuation_on_lines,
    flatten_newlines_for_json,
)


class LineTerminalPunctuationTests(unittest.TestCase):
    """Any punctuation before newline counts; only bare word-ends get ``.``."""

    def test_title_line_without_period_gets_one(self) -> None:
        """Bare title before body must not glue into continuous speech."""
        raw = "Chapter One\n\nStart of the first paragraph text."
        self.assertEqual(
            flatten_newlines_for_json(raw),
            "Chapter One. Start of the first paragraph text.",
        )

    def test_existing_period_not_doubled(self) -> None:
        """Lines that already end with a period stay unchanged."""
        raw = "Chapter One.\n\nStart of the first paragraph text."
        self.assertEqual(
            flatten_newlines_for_json(raw),
            "Chapter One. Start of the first paragraph text.",
        )

    def test_spaces_between_punct_and_newline_still_count(self) -> None:
        """``word.   \\n`` must not get another period."""
        raw = "Chapter One.   \n\nBody here."
        out = ensure_terminal_punctuation_on_lines(raw)
        self.assertEqual(out, "Chapter One.   \n\nBody here.")
        self.assertEqual(
            flatten_newlines_for_json(raw),
            "Chapter One. Body here.",
        )

    def test_any_punctuation_counts_including_comma(self) -> None:
        """Comma (or other punct) at line end is enough — no forced period."""
        raw = "Hello,\n\nWorld"
        out = ensure_terminal_punctuation_on_lines(raw)
        self.assertEqual(out, "Hello,\n\nWorld.")
        self.assertEqual(flatten_newlines_for_json(raw), "Hello, World.")

    def test_pause_marker_closing_bracket_counts_as_punct(self) -> None:
        """``]`` is punctuation, so ``[2s]`` line does not get ``.`` after it."""
        raw = "Intro text\n[2s]\nChapter One."
        flat = flatten_newlines_for_json(raw)
        self.assertNotIn("[2s].", flat)
        self.assertEqual(flat, "Intro text. [2s] Chapter One.")

    def test_line_ending_with_pause_marker_not_given_period(self) -> None:
        """Trailing ``]`` of ``[Xs]`` counts as punct."""
        raw = "Hold here [2s]\nNext sentence."
        flat = flatten_newlines_for_json(raw)
        self.assertEqual(flat, "Hold here [2s] Next sentence.")

    def test_mid_phrase_pause_preserves_user_punctuation(self) -> None:
        """Period injection must not synthesize commas around ``[Xs]`` tags."""
        raw = "Chapter\n[2s] One.\n\nStart of the first[2s]paragraph text."
        flat = flatten_newlines_for_json(raw)
        self.assertEqual(
            flat,
            "Chapter. [2s] One. Start of the first [2s] paragraph text.",
        )

    def test_ensure_preserves_blank_lines(self) -> None:
        """Blank lines remain so paragraph structure is not destroyed early."""
        raw = "Title\n\nBody without end\n"
        out = ensure_terminal_punctuation_on_lines(raw)
        self.assertEqual(out, "Title.\n\nBody without end.\n")

    def test_exclamation_and_question_count(self) -> None:
        """``!`` and ``?`` count as terminal punctuation."""
        raw = "What now?\n\nGo!"
        self.assertEqual(flatten_newlines_for_json(raw), "What now? Go!")

    def test_single_line_title_gets_period(self) -> None:
        """A one-line chunk ending in a letter still gets a period."""
        self.assertEqual(flatten_newlines_for_json("Chapter One"), "Chapter One.")

    def test_colon_counts_as_punctuation(self) -> None:
        """Colon at line end does not get an extra period."""
        raw = "Note:\nNext line"
        self.assertEqual(
            ensure_terminal_punctuation_on_lines(raw),
            "Note:\nNext line.",
        )

    def test_early_period_survives_chunk_merge(self) -> None:
        """Bare title + body must keep a period after chunker joins with space.

        Periods are applied on raw text before structure/chunk so min_words
        merge cannot glue ``Chapter One`` into the next sentence without a cue.
        """
        from pocket_tts.preprocessing.chunker import SmartChunker
        from pocket_tts.preprocessing.structure_detector import StructureDetector

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
        text = ensure_terminal_punctuation_on_lines(raw)
        structure = StructureDetector().analyze(text)
        chunks = SmartChunker(mode="sentence", min_words=4).chunk(structure)
        flat = [flatten_newlines_for_json(c.text or "") for c in chunks]

        # First chapter block: title period must remain after merge.
        chapter_one_chunk = next(t for t in flat if "Chapter One" in t)
        self.assertIn("Chapter One. Start", chapter_one_chunk)
        self.assertNotIn("Chapter One Start", chapter_one_chunk)

        # Second block: bare Chapter / One lines get periods before join.
        mid_pause_chunk = next(t for t in flat if "[2s] One" in t or ", [2s] One" in t)
        self.assertIn("One. Start", mid_pause_chunk)
        self.assertNotIn("One Start", mid_pause_chunk)


if __name__ == "__main__":
    unittest.main()
