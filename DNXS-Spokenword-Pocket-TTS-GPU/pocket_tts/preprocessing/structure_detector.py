"""
Structure detection for audiobook text preprocessing.
Detects chapters, paragraphs, and sentence boundaries.
"""

import re
import logging
from typing import List
from .schema import ChapterInfo, ParagraphInfo, SentenceInfo, TextStructure
from .sentence_boundary import SentenceBoundaryDetector

logger = logging.getLogger(__name__)

class StructureDetector:
    """Detects chapters, paragraphs, and sentences in text."""

    # Chapter/part line patterns (digit, Roman, and word-number forms).
    # Prologue/foreword stay in chapter 0 — they are not structural chapter 1.
    CHAPTER_PATTERNS = [
        r'^(?:Chapter|CHAPTER|Ch\.)\s+(?:\d+|[IVXLCDM]+|'
        r'Twenty[-\s]?One|Twenty[-\s]?Two|Twenty[-\s]?Three|Twenty[-\s]?Four|'
        r'Twenty[-\s]?Five|Twenty[-\s]?Six|Twenty[-\s]?Seven|Twenty[-\s]?Eight|'
        r'Twenty[-\s]?Nine|Thirty|Twenty|'
        r'Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|'
        r'Eleven|Twelve|Ten|One|Two|Three|Four|Five|Six|Seven|Eight|Nine)\b',
        r'^(?:Part|PART|Book|BOOK)\s+(?:\d+|[IVXLCDM]+|'
        r'Twenty[-\s]?One|Twenty[-\s]?Two|Twenty[-\s]?Three|Twenty[-\s]?Four|'
        r'Twenty[-\s]?Five|Twenty[-\s]?Six|Thirty|Twenty|'
        r'Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|'
        r'Eleven|Twelve|Ten|One|Two|Three|Four|Five|Six|Seven|Eight|Nine)\b',
        r'^\d+\.\s*$',              # "1." alone on line
    ]

    # Sentence ending punctuation
    SENTENCE_ENDERS = r'[.!?]+'

    def __init__(self):
        """Build chapter and sentence detectors used during analysis.

        The sentence detector is shared by full-text and paragraph-local
        segmentation so boundary rules live in one place.
        """
        self.chapter_regexes = [re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                               for pattern in self.CHAPTER_PATTERNS]
        self.sentence_boundary_detector = SentenceBoundaryDetector()

    def analyze(self, text: str) -> TextStructure:
        """
        Perform complete structural analysis of input text.

        Args:
            text: Raw input text

        Returns:
            TextStructure: Complete analysis results
        """
        logger.info("Starting structural analysis of text (%d characters)", len(text))
        logger.info("Raw text sample: %r", text[:200])

        # FIRST: Normalize text formatting
        text = self.normalize_formatting(text)

        # Detect chapters
        chapters = self.detect_chapters(text)
        logger.info("Detected %d chapters", len(chapters))

        # Detect paragraphs
        paragraphs = self.detect_paragraphs(text)
        logger.info("Detected %d paragraphs", len(paragraphs))

        # Segment sentences within each paragraph (prevents spanning)
        sentences = []
        for paragraph in paragraphs:
            para_sentences = self.segment_sentences_in_paragraph(paragraph, text)
            sentences.extend(para_sentences)
        logger.info("Detected %d sentences", len(sentences))

        # Calculate statistics
        total_words = sum(len(s.text.split()) for s in sentences)
        total_characters = len(text)

        # Mark chapter and paragraph relationships
        sentences = self._mark_relationships(sentences, chapters, paragraphs)

        structure = TextStructure(
            chapters=chapters,
            paragraphs=paragraphs,
            sentences=sentences,
            total_words=total_words,
            total_characters=total_characters
        )

        logger.info("Structural analysis complete: %d words, %d sentences",
                   total_words, len(sentences))
        return structure

    def detect_chapters(self, text: str) -> List[ChapterInfo]:
        """
        Detect all chapter markers in the text.

        Returns list of ChapterInfo objects sorted by position.
        """
        chapters = []

        for i, regex in enumerate(self.chapter_regexes):
            matches = regex.finditer(text)
            for match in matches:
                # Extract chapter number if present
                chapter_num = None
                title = match.group(0).strip()

                # Try to extract number
                num_match = re.search(r'\d+', title)
                if num_match:
                    chapter_num = int(num_match.group(0))

                # Check if Roman numeral
                roman_match = re.search(r'[IVXLCDM]+', title, re.IGNORECASE)
                is_roman = roman_match is not None and chapter_num is None

                chapter_info = ChapterInfo(
                    title=title,
                    start_position=match.start(),
                    chapter_number=chapter_num,
                    is_roman_numeral=is_roman
                )
                chapters.append(chapter_info)

        # Sort by position and remove duplicates (if multiple patterns match same location)
        chapters.sort(key=lambda c: c.start_position)
        unique_chapters = []
        seen_positions = set()

        for chapter in chapters:
            if chapter.start_position not in seen_positions:
                unique_chapters.append(chapter)
                seen_positions.add(chapter.start_position)

        return unique_chapters

    def normalize_formatting(self, text: str) -> str:
        """Normalize text formatting for consistent processing."""
        # FIRST: Normalize Unicode punctuation to ASCII (critical for contractions)
        try:
            from .text_normalizer import normalize_unicode_punctuation
            text = normalize_unicode_punctuation(text)
        except ImportError:
            # Fallback if text_normalizer not available
            pass

        # Normalize excessive newlines to exactly 2
        text = self._normalize_newlines(text)
        # Placeholder for future formatting rules
        text = self._normalize_other_formatting(text)
        return text

    def _normalize_newlines(self, text: str) -> str:
        """Normalize 2+ consecutive newlines to exactly 2."""
        import re
        # Replace any sequence of 2+ newlines (with optional whitespace) with exactly \n\n
        return re.sub(r'\n\s*\n+', '\n\n', text)

    def _normalize_other_formatting(self, text: str) -> str:
        """Placeholder for additional formatting normalizations."""
        # Future extensions can be added here
        return text

    def detect_paragraphs(self, text: str) -> List[ParagraphInfo]:
        """
        Detect paragraph boundaries using double newlines.
        
        Returns list of ParagraphInfo objects.
        """
        paragraphs = []
        
        # Split by double newlines (paragraph breaks)
        para_texts = re.split(r'\n\s*\n', text.strip())
        
        current_pos = 0
        for para_text in para_texts:
            if not para_text.strip():  # Skip empty paragraphs
                continue
            
            start_pos = text.find(para_text.strip(), current_pos)
            if start_pos == -1:
                continue
            
            end_pos = start_pos + len(para_text)
            
            para_info = ParagraphInfo(
                start_position=start_pos,
                end_position=end_pos,
                text=para_text.strip()
            )
            paragraphs.append(para_info)
            
            current_pos = end_pos
        
        return paragraphs

    def save_paragraph_structure(self, paragraphs: List[ParagraphInfo], sentences: List[SentenceInfo], output_path: str):
        """
        Save paragraph structure to JSON file for debugging and verification.
        
        Args:
            paragraphs: List of detected paragraphs
            sentences: List of all sentences in text
            output_path: Path to save JSON file
        """
        try:
            import json
            from datetime import datetime
            
            # Build paragraph data with associated sentences
            paragraph_data = []
            for i, paragraph in enumerate(paragraphs):
                # Find sentences that belong to this paragraph
                para_sentences = [s for s in sentences 
                                if paragraph.start_position <= s.start_position < paragraph.end_position]
                
                paragraph_dict = {
                    'index': i,
                    'text': paragraph.text,
                    'start_position': paragraph.start_position,
                    'end_position': paragraph.end_position,
                    'sentence_count': len(para_sentences),
                    'sentences': [
                        {
                            'text': s.text,
                            'start_position': s.start_position,
                            'end_position': s.end_position,
                            'punctuation': s.punctuation,
                            'ends_paragraph': s.ends_paragraph
                        }
                        for s in para_sentences
                    ]
                }
                paragraph_data.append(paragraph_dict)
            
            # Build metadata
            metadata = {
                'source_file': output_path,
                'total_paragraphs': len(paragraphs),
                'total_sentences': len(sentences),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Build complete structure
            structure_data = {
                '_metadata': metadata,
                'paragraphs': paragraph_data
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structure_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Paragraph structure saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save paragraph structure: {e}")
            raise

    def segment_sentences(self, text: str) -> List[SentenceInfo]:
        """
        Segment text into sentences using the shared boundary detector.

        The detector preserves abbreviations, initials, decimals, pause
        markers, ellipses, and closing quotes while keeping source offsets
        usable for downstream chunking.
        """
        return self.sentence_boundary_detector.segment(text)

    def segment_sentences_in_paragraph(self, paragraph: ParagraphInfo, full_text: str) -> List[SentenceInfo]:
        """
        Segment sentences within a single paragraph's boundaries.

        This delegates to the shared sentence detector so paragraph-local
        segmentation and full-text segmentation always agree on boundary rules.
        """
        para_start = paragraph.start_position
        para_end = paragraph.end_position
        para_text = full_text[para_start:para_end]
        if not para_text.strip():
            return []
        return self.sentence_boundary_detector.segment(para_text, base_offset=para_start)

    def _check_ends_paragraph(self, text: str, position: int) -> bool:
        """
        Check if text at given position is followed by paragraph-ending newlines.

        A paragraph ends if followed by:
        - \n\n (double newline)
        - \n\n\n+ (multiple newlines)
        - End of text
        """
        if position >= len(text):
            return True  # End of text always ends paragraph

        # Look ahead for newlines, allowing for whitespace
        remaining = text[position:]
        # Match \n\n or more, with optional whitespace
        newline_match = re.match(r'\s*\n\s*\n+', remaining)

        return newline_match is not None

    def _mark_relationships(self, sentences: List[SentenceInfo],
                          chapters: List[ChapterInfo],
                          paragraphs: List[ParagraphInfo]) -> List[SentenceInfo]:
        """
        Mark which sentences start chapters and end paragraphs.

        A sentence is a chapter header when its text begins with a Part/Chapter
        form, or when its start position matches a line-level chapter hit.
        Mid-prose uses of the word \"chapter\" are not headers.
        """
        from .chapter_headers import is_chapter_header_sentence

        # Create lookup sets for quick checking
        chapter_starts = {c.start_position for c in chapters}
        para_ends = {p.end_position for p in paragraphs}

        for sentence in sentences:
            # Structural header at sentence start (Chapter One / Part Two / …).
            if is_chapter_header_sentence(sentence.text):
                sentence.is_chapter = True
            # Line-level detector hit (position align).
            elif sentence.start_position in chapter_starts:
                sentence.is_chapter = True

            # Check if this sentence ends a paragraph
            # Allow some tolerance for text processing differences
            for para_end in para_ends:
                if abs(sentence.end_position - para_end) <= 5:  # 5 char tolerance
                    sentence.ends_paragraph = True
                    break

        return sentences

    def get_statistics(self, structure: TextStructure) -> dict:
        """
        Generate statistics about the structural analysis.
        """
        return {
            'total_characters': structure.total_characters,
            'total_words': structure.total_words,
            'total_sentences': len(structure.sentences),
            'total_paragraphs': len(structure.paragraphs),
            'total_chapters': len(structure.chapters),
            'avg_words_per_sentence': structure.total_words / len(structure.sentences) if structure.sentences else 0,
            'avg_sentences_per_paragraph': len(structure.sentences) / len(structure.paragraphs) if structure.paragraphs else 0,
        }

    def _protect_pause_markers(self, text: str) -> str:
        """
        Protect [Xs] pause markers from being split by sentence segmentation.

        Replaces periods inside pause markers with a placeholder that won't trigger
        sentence splitting, e.g., "[0.5s]" becomes "[0§5s]".

        Args:
            text: Original text with potential pause markers

        Returns:
            Text with protected pause markers
        """
        # Replace any [number.s] with placeholder (using § as it's unlikely in text)
        # This handles [0.5s], [10s], [1.25s], etc.
        return re.sub(r'\[(\d+\.?\d*)s\]', lambda m: m.group(0).replace('.', '§'), text)

    def _restore_pause_markers(self, text: str) -> str:
        """
        Restore [Xs] pause markers after sentence segmentation.

        Reverts the placeholder back to the original period, e.g., "[0§5s]" becomes "[0.5s]".

        Args:
            text: Text with protected pause markers

        Returns:
            Text with restored pause markers
        """
        # Restore § to .
        return text.replace('§', '.')
