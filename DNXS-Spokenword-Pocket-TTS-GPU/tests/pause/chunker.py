"""
Smart chunking algorithm for audiobook text preprocessing.
Handles sentence vs paragraph modes with merge/split logic.
"""

import logging
from typing import List
from .schema import ChunkMetadata, BoundaryType, EmotionType, TextStructure

logger = logging.getLogger(__name__)

class SmartChunker:
    """Creates optimal chunks from structured text with configurable modes."""

    def __init__(self,
                 mode: str = "sentence",
                 min_words: int = 5,
                 respect_boundaries: bool = True,
                 target_words: int | None = None):
        """
        Initialize chunker with configuration.

        Args:
            mode: "sentence", "paragraph", or "word_target" - chunking strategy
            min_words: Minimum words per chunk, or target word count for
                ``word_target`` mode.
            respect_boundaries: Whether to split at \n and chapter markers
            target_words: Explicit maximum word target for ``word_target`` mode.
        """
        if mode not in ["sentence", "paragraph", "word_target"]:
            raise ValueError("mode must be 'sentence', 'paragraph', or 'word_target'")
        if min_words < 1:
            raise ValueError("min_words must be at least 1")
        if target_words is not None and target_words < 1:
            raise ValueError("target_words must be at least 1")

        self.mode = mode
        self.min_words = min_words
        self.target_words = target_words if target_words is not None else min_words
        self.respect_boundaries = respect_boundaries

        logger.info(f"SmartChunker initialized: mode={mode}, min_words={min_words}")

    def chunk(self, structure: TextStructure) -> List[ChunkMetadata]:
        """
        Create chunks from analyzed text structure.

        Args:
            structure: Analyzed text structure from StructureDetector

        Returns:
            List of ChunkMetadata objects
        """
        logger.info(f"Starting chunking in {self.mode} mode")

        if self.mode == "sentence":
            chunks = self._chunk_by_sentences(structure)
        elif self.mode == "paragraph":
            chunks = self._chunk_by_paragraphs(structure)
        else:
            chunks = self._chunk_by_word_target(structure)

        logger.info(f"Chunking complete: {len(chunks)} chunks created")

        # Convert to ChunkMetadata with default values
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                index=i,
                text=chunk['text'],
                word_count=chunk['word_count'],
                character_count=len(chunk['text']),
                boundary_type=chunk['boundary_type'],
                punctuation=chunk.get('punctuation', ''),
                start_position=chunk['start_position'],
                end_position=chunk['end_position'],
                emotion=EmotionType.NEUTRAL,  # Placeholder - will be analyzed later
                emotion_scores={'neutral': 1.0},
                emotion_confidence=1.0,
                tts_params={},
                post_process={}
            )
            chunk_metadata.append(metadata)

        return chunk_metadata

    def _chunk_by_sentences(self, structure: TextStructure) -> List[dict]:
        """Group adjacent sentences until each chunk reaches minimum words.

        Groups never cross paragraph boundaries. No external maximum is applied;
        the TTS model performs its own token-window splitting during generation.
        """
        chunks = []

        current_sentences = []
        current_word_count = 0
        for sentence in structure.sentences:
            # Chapter starts and paragraph ends are hard boundaries for grouping.
            if sentence.is_chapter and current_sentences:
                chunks.append(self._create_sentence_chunk(current_sentences))
                current_sentences = []
                current_word_count = 0
            elif current_sentences and current_sentences[-1].ends_paragraph:
                chunks.append(self._create_sentence_chunk(current_sentences))
                current_sentences = []
                current_word_count = 0

            current_sentences.append(sentence)
            current_word_count += sentence.word_count
            if current_word_count >= self.min_words:
                chunks.append(self._create_sentence_chunk(current_sentences))
                current_sentences = []
                current_word_count = 0

        if current_sentences:
            chunks.append(self._create_sentence_chunk(current_sentences))

        return chunks

    def _create_sentence_chunk(self, sentences) -> dict:
        """Create one chunk from grouped sentences while preserving boundaries."""
        last_sentence = sentences[-1]
        return {
            'text': ' '.join(sentence.text for sentence in sentences),
            'word_count': sum(sentence.word_count for sentence in sentences),
            'start_position': sentences[0].start_position,
            'end_position': last_sentence.end_position,
            'boundary_type': (
                BoundaryType.PARAGRAPH_BREAK
                if last_sentence.ends_paragraph
                else BoundaryType.SENTENCE_END
            ),
            'punctuation': last_sentence.punctuation,
        }

    def _chunk_by_paragraphs(self, structure: TextStructure) -> List[dict]:
        """Create chunks by grouping adjacent paragraphs until min_words is reached."""
        chunks = []
        buffer = []          # list of (sentences, paragraph) tuples
        buffer_words = 0

        for paragraph in structure.paragraphs:
            para_sentences = self._get_sentences_in_paragraph(paragraph, structure.sentences)
            if not para_sentences:
                continue

            buffer.append((para_sentences, paragraph))
            buffer_words += sum(s.word_count for s in para_sentences)

            if buffer_words >= self.min_words:
                chunks.append(self._merge_paragraphs(buffer))
                buffer = []
                buffer_words = 0

        if buffer:
            chunks.append(self._merge_paragraphs(buffer))

        return chunks

    def _get_sentences_in_paragraph(self, paragraph, sentences) -> List:
        """Return analyzed sentences whose positions fall inside a paragraph."""
        return [
            sentence
            for sentence in sentences
            if sentence.start_position >= paragraph.start_position
            and sentence.end_position <= paragraph.end_position
        ]

    def _chunk_by_word_target(self, structure: TextStructure) -> List[dict]:
        """Group complete sentences without exceeding the requested word target.

        The method stops before adding a sentence that would exceed the target,
        so chunks are intentionally shorter rather than padded with another
        sentence. A single sentence longer than the target remains whole.
        """
        chunks = []
        current_sentences = []
        current_word_count = 0

        for sentence in structure.sentences:
            crosses_hard_boundary = bool(
                current_sentences
                and (sentence.is_chapter or current_sentences[-1].ends_paragraph)
            )
            would_exceed = (
                current_sentences
                and current_word_count + sentence.word_count > self.target_words
            )
            if crosses_hard_boundary or would_exceed:
                chunks.append(self._create_sentence_chunk(current_sentences))
                current_sentences = []
                current_word_count = 0

            current_sentences.append(sentence)
            current_word_count += sentence.word_count

        if current_sentences:
            chunks.append(self._create_sentence_chunk(current_sentences))

        return chunks

    def _merge_paragraphs(self, para_groups) -> dict:
        """Merge a list of (sentences, paragraph) tuples into one chunk."""
        all_sentences = [s for group, _ in para_groups for s in group]
        first_para = para_groups[0][1]
        last_para = para_groups[-1][1]
        return {
            'text': ' '.join(s.text for s in all_sentences),
            'word_count': sum(s.word_count for s in all_sentences),
            'start_position': first_para.start_position,
            'end_position': last_para.end_position,
            'boundary_type': BoundaryType.PARAGRAPH_BREAK,
            'punctuation': all_sentences[-1].punctuation if all_sentences else '',
        }

    def get_statistics(self, chunks: List[ChunkMetadata]) -> dict:
        """Generate statistics about chunking results."""
        if not chunks:
            return {}

        word_counts = [c.word_count for c in chunks]
        boundary_types = [c.boundary_type.value for c in chunks]

        return {
            'total_chunks': len(chunks),
            'total_words': sum(word_counts),
            'avg_words_per_chunk': sum(word_counts) / len(chunks),
            'min_words_per_chunk': min(word_counts),
            'max_words_per_chunk': max(word_counts),
            'boundary_type_distribution': {
                bt: boundary_types.count(bt) for bt in set(boundary_types)
            }
        }
