"""Shared sentence-boundary detection for audiobook preprocessing.

This module centralizes sentence segmentation so production structure analysis
and any chunking logic consume the same span finder instead of duplicating
regex-only split rules.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .schema import SentenceInfo


class SentenceBoundaryDetector:
    """Detect sentence spans while protecting common non-terminal patterns."""

    _SENTENCE_END_RE = re.compile(r"[.!?]+")
    _PAUSE_MARKER_RE = re.compile(r"\[(\d+\.?\d*)s\]")
    _DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
    _INITIALS_RE = re.compile(r"(?<![A-Za-z.§])\b[A-Za-z]\.(?=\s+[A-Za-z])")
    _ACRONYM_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")
    _SHORT_ABBREVIATION_RE = re.compile(r"\b(?:[A-Za-z]{1,3}\.){2,}")
    _COMMON_ABBREVIATION_RE = re.compile(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|No|Rev|Gen|Gov|Sen|Rep|Lt|Col|Capt|"
        r"Sgt|Ave|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|Fig|Dept|Inc|"
        r"Ltd|Co|Corp|Vol|Ch|vs|etc|al|p\.m|a\.m|e\.g|i\.e)\.",
        re.IGNORECASE,
    )
    _TRAILING_CLOSERS = "\"'”’)]}»›"

    def segment(self, text: str, base_offset: int = 0) -> List[SentenceInfo]:
        """Split text into sentence spans with original offsets preserved.

        Args:
            text: Source text to segment.
            base_offset: Absolute offset to add to each returned sentence span.

        Returns:
            SentenceInfo objects with preserved source text and offsets.
        """
        if not text:
            return []

        protected_text = self._protect_sentence_boundaries(text)
        sentences: List[SentenceInfo] = []

        start = self._skip_whitespace(protected_text, 0)
        cursor = start

        for match in self._SENTENCE_END_RE.finditer(protected_text):
            if match.start() < cursor:
                continue

            if not self._is_sentence_boundary(protected_text, match.start(), match.end()):
                continue

            boundary_end = self._consume_trailing_closers(protected_text, match.end())
            span = self._trim_span(protected_text, start, boundary_end)
            if span is None:
                # Skip punctuation-only noise so the next real sentence can start cleanly.
                start = self._skip_whitespace(protected_text, boundary_end)
                cursor = start
                continue

            span_start, span_end = span
            # Protected text is scanner-only. Slice the original source so an
            # existing sentinel-like character can never be rewritten.
            sentence_text = text[span_start:span_end]
            punctuation = text[match.start():match.end()]

            if sentence_text:
                sentences.append(
                    SentenceInfo(
                        text=sentence_text,
                        start_position=base_offset + span_start,
                        end_position=base_offset + span_end,
                        word_count=len(sentence_text.split()),
                        punctuation=punctuation,
                        ends_paragraph=False,
                    )
                )

            start = self._skip_whitespace(protected_text, boundary_end)
            cursor = start

        if start < len(protected_text):
            span = self._trim_span(protected_text, start, len(protected_text))
            if span is not None:
                span_start, span_end = span
                sentence_text = text[span_start:span_end]
                if sentence_text:
                    sentences.append(
                        SentenceInfo(
                            text=sentence_text,
                            start_position=base_offset + span_start,
                            end_position=base_offset + span_end,
                            word_count=len(sentence_text.split()),
                            punctuation="",
                            ends_paragraph=False,
                        )
                    )

        if sentences:
            sentences[-1].ends_paragraph = True

        return sentences

    def _protect_sentence_boundaries(self, text: str) -> str:
        """Mask periods that should not act as sentence boundaries.

        Args:
            text: Raw source text.

        Returns:
            Text with protected periods swapped to a sentinel character.
        """
        protected = text

        # Apply the most specific patterns first so later regexes do not see
        # punctuation that already belongs to an abbreviation or initialism.
        protected = self._PAUSE_MARKER_RE.sub(
            lambda match: match.group(0).replace(".", "§"),
            protected,
        )
        protected = self._DECIMAL_RE.sub("§", protected)
        protected = self._ACRONYM_RE.sub(self._protect_dotted_term, protected)
        protected = self._SHORT_ABBREVIATION_RE.sub(self._protect_dotted_term, protected)
        protected = self._INITIALS_RE.sub(
            lambda match: match.group(0).replace(".", "§"),
            protected,
        )
        protected = self._COMMON_ABBREVIATION_RE.sub(
            lambda match: match.group(0).replace(".", "§"),
            protected,
        )

        return protected

    def _protect_dotted_term(self, match: re.Match[str]) -> str:
        """Mask internal dotted-term periods while leaving its final dot testable."""
        token = match.group(0)
        return token[:-1].replace(".", "§") + "."

    def _consume_trailing_closers(self, text: str, boundary_end: int) -> int:
        """Include closing quotes or brackets that belong with sentence end."""
        while boundary_end < len(text) and text[boundary_end] in self._TRAILING_CLOSERS:
            boundary_end += 1
        return boundary_end

    def _skip_whitespace(self, text: str, position: int) -> int:
        """Advance past whitespace so sentence spans start on visible text."""
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    def _trim_span(
        self,
        text: str,
        start: int,
        end: int,
    ) -> Optional[Tuple[int, int]]:
        """Trim outer whitespace while keeping offsets usable.

        Args:
            text: Protected source text.
            start: Candidate span start.
            end: Candidate span end.

        Returns:
            A trimmed span, or None when the span is empty after trimming.
        """
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start >= end:
            return None
        return start, end

    def _is_sentence_boundary(self, text: str, start: int, end: int) -> bool:
        """Decide whether one punctuation run closes a sentence.

        Protection rules already mask decimals and abbreviations, so any
        surviving sentence-end run is normally valid. The guard stays narrow to
        keep false splits low while preserving quoted terminal punctuation and
        ellipses.
        """
        boundary = text[start:end]
        if not boundary:
            return False
        if boundary.endswith("..."):
            return True
        if boundary == ".":
            preceding = text[max(0, start - 16):start]
            next_position = self._skip_whitespace(text, end)
            # A dotted abbreviation/acronym followed by lowercase prose stays
            # inside its sentence; uppercase prose starts a new sentence.
            if "§" in preceding and next_position < len(text) and text[next_position].islower():
                return False
        return any(char in ".!?" for char in boundary)
