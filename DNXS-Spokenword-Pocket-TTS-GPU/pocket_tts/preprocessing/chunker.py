"""
Smart chunking algorithm for audiobook text preprocessing.
Handles sentence vs paragraph modes with merge/split logic.
"""

import logging
from typing import List
from .schema import ChunkMetadata, BoundaryType, EmotionType, TextStructure

logger = logging.getLogger(__name__)

class SmartChunker:
    """Creates chunks from structured text with explicit mode semantics.

    Mode semantics
    --------------
    sentence
        ``min_words`` is a soft merge floor. Full sentences are packed until
        the floor is reached, then the chunk is flushed. A chunk may exceed
        ``min_words`` when the last sentence overshoots; never split mid-sentence.
    paragraph
        ``min_words`` and ``target_words`` are ignored. One structure paragraph
        becomes one chunk (split only on paragraph breaks / ``\\n\\n``).
    word_target
        ``min_words`` is a soft lower floor; ``target_words`` is a hard upper
        cap. Pack full sentences up to the cap without exceeding it unless a
        single sentence alone is longer than the cap. Double newlines /
        paragraph breaks do **not** force a new chunk — only the word cap
        (and chapter headers starting a new structural unit) do. Blank-line
        separated blocks under the cap become one TTS chunk.
    """

    def __init__(self,
                 mode: str = "sentence",
                 min_words: int = 5,
                 respect_boundaries: bool = True,
                 target_words: int | None = None):
        """
        Initialize chunker with configuration.

        Args:
            mode: ``sentence``, ``paragraph``, or ``word_target``.
            min_words: Soft floor for sentence mode; soft floor for word_target;
                ignored in paragraph mode.
            respect_boundaries: Prefer hard breaks at paragraph ends where the
                mode already consults them (word_target); chapter headers always
                hard-split.
            target_words: Hard upper bound for word_target mode only.
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

        logger.info(
            "SmartChunker initialized: mode=%s, min_words=%s, target_words=%s",
            mode,
            min_words,
            self.target_words,
        )

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
            boundary = chunk['boundary_type']
            if chunk.get('is_chapter_start'):
                boundary = BoundaryType.CHAPTER_START
            metadata = ChunkMetadata(
                index=i,
                text=chunk['text'],
                word_count=chunk['word_count'],
                character_count=len(chunk['text']),
                boundary_type=boundary,
                punctuation=chunk.get('punctuation', ''),
                start_position=chunk['start_position'],
                end_position=chunk['end_position'],
                emotion=EmotionType.NEUTRAL,  # Placeholder - will be analyzed later
                emotion_scores={'neutral': 1.0},
                emotion_confidence=1.0,
                tts_params={},
                post_process={},
                chapter_number=chunk.get('chapter_number'),
                chapter_id=chunk.get('chapter_id'),
                chapter_title=chunk.get('chapter_title'),
            )
            chunk_metadata.append(metadata)

        # Stamp sequential chapter ids (0 = all text before first Part/Chapter).
        from .chapter_headers import stamp_chunk_metadata_chapters

        stamp_chunk_metadata_chapters(chunk_metadata)
        return chunk_metadata

    def _chunk_by_sentences(self, structure: TextStructure) -> List[dict]:
        """Group full sentences until ``min_words`` soft floor is reached.

        Chunks may exceed the floor when the last sentence overshoots. Never
        split mid-sentence. Chapter/part headers always start a new chunk;
        short orphans before a header flush to the previous chunk first.
        """
        chunks = []

        current_sentences = []
        current_word_count = 0
        for sentence in structure.sentences:
            # Hard split before a chapter header: orphans stay on previous chunk.
            if sentence.is_chapter:
                if current_sentences:
                    chunks.append(self._create_sentence_chunk(current_sentences))
                    current_sentences = []
                    current_word_count = 0
                current_sentences = [sentence]
                current_word_count = sentence.word_count
                # Header chunks may be short; flush when min_words met or alone later.
                if current_word_count >= self.min_words:
                    chunks.append(
                        self._create_sentence_chunk(
                            current_sentences,
                            is_chapter_start=True,
                        )
                    )
                    current_sentences = []
                    current_word_count = 0
                continue

            current_sentences.append(sentence)
            current_word_count += sentence.word_count
            if current_word_count >= self.min_words:
                # If buffer starts with a chapter header, mark the chunk.
                is_start = bool(current_sentences and current_sentences[0].is_chapter)
                chunks.append(
                    self._create_sentence_chunk(
                        current_sentences,
                        is_chapter_start=is_start,
                    )
                )
                current_sentences = []
                current_word_count = 0

        if current_sentences:
            is_start = bool(current_sentences and current_sentences[0].is_chapter)
            chunks.append(
                self._create_sentence_chunk(
                    current_sentences,
                    is_chapter_start=is_start,
                )
            )

        return chunks

    def _create_sentence_chunk(self, sentences, is_chapter_start: bool = False) -> dict:
        """Create one chunk from grouped sentences while preserving boundaries."""
        last_sentence = sentences[-1]
        first_is_chapter = bool(sentences and sentences[0].is_chapter)
        chapter_start = is_chapter_start or first_is_chapter
        if chapter_start:
            boundary = BoundaryType.CHAPTER_START
        elif last_sentence.ends_paragraph:
            boundary = BoundaryType.PARAGRAPH_BREAK
        else:
            boundary = BoundaryType.SENTENCE_END
        return {
            'text': ' '.join(sentence.text for sentence in sentences),
            'word_count': sum(sentence.word_count for sentence in sentences),
            'start_position': sentences[0].start_position,
            'end_position': last_sentence.end_position,
            'boundary_type': boundary,
            'punctuation': last_sentence.punctuation,
            'is_chapter_start': chapter_start,
        }

    def _chunk_by_paragraphs(self, structure: TextStructure) -> List[dict]:
        """One chunk per structure paragraph; ignore min_words / target_words.

        Paragraph breaks come from the structure detector (``\\n\\n``). This
        mode never packs multiple paragraphs together for a word floor.
        """
        chunks = []
        for paragraph in structure.paragraphs:
            para_sentences = self._get_sentences_in_paragraph(
                paragraph, structure.sentences
            )
            if not para_sentences:
                # Structure paragraph with no sentence hits: keep raw text.
                text = (paragraph.text or "").strip()
                if not text:
                    continue
                chunks.append({
                    "text": text,
                    "word_count": len(text.split()),
                    "start_position": paragraph.start_position,
                    "end_position": paragraph.end_position,
                    "boundary_type": BoundaryType.PARAGRAPH_BREAK,
                    "punctuation": text[-1] if text else "",
                    "is_chapter_start": False,
                })
                continue
            chunks.append(self._merge_paragraphs([(para_sentences, paragraph)]))
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
        """Pack full sentences under ``target_words`` across paragraph blanks.

        Paragraph breaks (``\\n\\n``) are **not** chunk boundaries here. Six
        short blocks separated by blank lines under the cap become **one**
        chunk so TTS sees them as a single packed unit (after flatten).

        Hard cap: do not add the next sentence if it would push the chunk over
        ``target_words``, unless the buffer is empty (one long sentence alone
        may exceed the cap and remains whole).

        Soft floor: packing continues past ``min_words`` toward the cap.

        Blank lines, paragraph ends, and chapter labels do **not** force a
        new chunk. Only the word cap splits (or end of book). Six short
        blocks separated by ``\\n\\n`` under the cap → one TTS chunk.
        ``respect_boundaries`` is ignored here.
        """
        chunks = []
        current_sentences = []
        current_word_count = 0
        cap = max(self.min_words, self.target_words)

        def flush() -> None:
            """Emit current buffer as one chunk and clear it."""
            nonlocal current_sentences, current_word_count
            if not current_sentences:
                return
            is_start = bool(current_sentences[0].is_chapter)
            chunks.append(
                self._create_sentence_chunk(
                    current_sentences,
                    is_chapter_start=is_start,
                )
            )
            current_sentences = []
            current_word_count = 0

        for sentence in structure.sentences:
            # Cap only — no paragraph flush, no chapter flush.
            would_exceed_cap = (
                current_sentences
                and current_word_count + sentence.word_count > cap
            )
            if would_exceed_cap:
                flush()

            current_sentences.append(sentence)
            current_word_count += sentence.word_count

        flush()
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
