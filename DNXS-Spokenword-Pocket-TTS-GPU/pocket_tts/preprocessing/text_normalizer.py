"""Unicode punctuation normalization for consistent TTS text processing.

This module provides text normalization that converts Unicode punctuation characters
to their ASCII equivalents. This ensures consistent tokenization and pronunciation
regardless of text source (PDFs, Word docs, web scraping, copy-paste).

The SentencePiece tokenizer and TTS model work best with standard ASCII characters.
Unicode smart quotes, dashes, and other punctuation can cause:
- Contraction mispronunciation ("I'd" -> "ID")
- Word splitting ("don't" -> "don' + t")
- Tokenization errors with unknown Unicode characters

Usage:
    from pocket_tts.preprocessing.text_normalizer import normalize_unicode_punctuation

    text = "I'd like to go, but I can't. He said, \"Hello!\""
    normalized = normalize_unicode_punctuation(text)
    # Result: 'I\'d like to go, but I can\'t. He said, "Hello!"'
"""

import re
from typing import Dict


UNICODE_TO_ASCII: Dict[str, str] = {
    # Smart single quotes and apostrophes -> ASCII apostrophe

    "\u2019": "'",  # Right single quotation mark (the one in he’s)
    "\u2018": "'",  # Left single quotation mark
    "\u02bc": "'",  # Modifier letter apostrophe
    "\u201a": "'",  # Single low-9 quotation mark
    "\u0060": "'",  # Grave accent
#    "'": "'",  # U+2019 right single quotation mark
#    "'": "'",  # U+2018 left single quotation mark
#    "'": "'",  # U+02BC modifier letter apostrophe
#    "'": "'",  # U+201A single low-9 quotation mark
#    "`": "'",  # U+0060 grave accent (often misused as apostrophe)

    # Smart double quotes -> ASCII double quote
    '"': '"',  # U+201D right double quotation mark
    '"': '"',  # U+201C left double quotation mark
    '"': '"',  # U+201F double high-reversed-9 quotation mark
    '"': '"',  # U+201E double low-9 quotation mark

    # Dashes -> ASCII hyphen/minus
    "–": "-",  # U+2013 en dash
    "—": "-",  # U+2014 em dash
    "―": "-",  # U+2015 horizontal bar
    "‐": "-",  # U+2010 hyphen
    "‑": "-",  # U+2011 non-breaking hyphen

    # Ellipsis -> Three periods
    "…": "...",  # U+2026 horizontal ellipsis
    "‥": "...",  # U+2025 two-dot leader

    # Non-breaking spaces -> Regular space
    "\u00A0": " ",  # Non-breaking space
    "\u2009": " ",  # Thin space
    "\u200A": " ",  # Hair space
    "\u2008": " ",  # Punctuation space
    "\u202F": " ",  # Narrow no-break space
    "\u205F": " ",  # Medium mathematical space

    # Mathematical symbols -> ASCII equivalents
    "×": "x",  # U+00D7 multiplication sign
    "÷": "/",  # U+00F7 division sign
    "±": "+/-",  # U+00B1 plus-minus sign
    "≠": "!=",  # U+226D not equal to
    "≤": "<=",  # U+226C less-than or equal to
    "≥": ">=",  # U+226D greater-than or equal to
    "∞": "infinity",  # U+221E infinity

    # Fractions -> ASCII representation
    "½": "1/2",  # U+00BD vulgar fraction one half
    "¼": "1/4",  # U+00BC vulgar fraction one quarter
    "¾": "3/4",  # U+00BE vulgar fraction three quarters
    "⅓": "1/3",  # U+2153 vulgar fraction one third
    "⅔": "2/3",  # U+2154 vulgar fraction two thirds
    "⅛": "1/8",  # U+215B vulgar fraction one eighth
    "⅜": "3/8",  # U+215C vulgar fraction three eighths
    "⅝": "5/8",  # U+215D vulgar fraction five eighths
    "⅞": "7/8",  # U+215E vulgar fraction seven eighths

    # Ligatures -> Separate letters
    "æ": "ae",  # U+00E6 Latin small letter ae
    "Æ": "AE",  # U+00C6 Latin capital letter ae
    "œ": "oe",  # U+0153 Latin small ligature oe
    "Œ": "OE",  # U+0152 Latin capital ligature oe
    "ß": "ss",  # U+00DF Latin small letter sharp s (German)
    "ƀ": "l",  # U+0180 Latin small letter b with stroke
    "Ɓ": "B",  # U+0181 Latin capital letter b with stroke

    # Currency symbols -> Text equivalents
    "€": "euro",  # U+20AC Euro sign
    "£": "pound",  # U+00A3 Pound sign
    "¥": "yen",  # U+00A5 Yen sign
    "₹": "rupee",  # U+20B9 Indian rupee sign
    "₩": "won",  # U+20A9 Korean won sign
    "₽": "ruble",  # U+20BD Russian ruble sign

    # Common symbols -> Text equivalents
    "©": "(c)",  # U+00A9 copyright sign
    "®": "(R)",  # U+00AE registered sign
    "™": "(TM)",  # U+2122 trademark sign
    "§": "section",  # U+00A7 section sign
    "¶": "paragraph",  # U+00B6 pilcrow sign
    "•": "-",  # U+2022 bullet
    "‣": "-",  # U+2023 triangular bullet
    "※": "*",  # U+203B reference mark

    # Ordinal indicators -> Text equivalents
    "º": "o",  # U+00BA masculine ordinal indicator
    "ª": "a",  # U+00AA feminine ordinal indicator
    "¹": "1",  # U+00B9 superscript one
    "²": "2",  # U+00B2 superscript two
    "³": "3",  # U+00B3 superscript three

    # Arrow characters -> Text equivalents
    "→": "->",  # U+2192 rightward arrow
    "←": "<-",  # U+2190 leftward arrow
    "↑": "^",  # U+2191 upward arrow
    "↓": "v",  # U+2193 downward arrow
    "↔": "<->",  # U+2194 left-right arrow

    # Miscellaneous Unicode punctuation -> ASCII equivalents
    "«": "<<",  # U+00AB left-pointing double angle quotation mark
    "»": ">>",  # U+00BB right-pointing double angle quotation mark
    "‹": "<",  # U+2039 single left-pointing angle quotation mark
    "›": ">",  # U+203A single right-pointing angle quotation mark
    "„": ',,',  # U+201E double low-9 quotation mark (German opening)
    "‚": "',",  # U+201A single low-9 quotation mark
    "❛": "'",  # U+275B heavy single quotation mark
    "❜": "'",  # U+275C heavy single quotation mark
    "❝": '"',  # U+275D heavy double quotation mark
    "❞": '"',  # U+275E heavy double quotation mark
}


def normalize_unicode_punctuation(text: str) -> str:
    """Convert Unicode punctuation characters to ASCII equivalents.

    This function normalizes text by converting smart quotes, dashes,
    ellipsis, and other Unicode punctuation to standard ASCII characters.
    This ensures consistent tokenization and pronunciation regardless
    of text source.

    Args:
        text: Input text that may contain Unicode punctuation.

    Returns:
        Text with all problematic Unicode punctuation converted to ASCII.

    Examples:
        >>> normalize_unicode_punctuation("I'd like to go.")
        "I'd like to go."

        >>> normalize_unicode_punctuation("He said, \\"Hello!\\"")
        'He said, "Hello!"'

        >>> normalize_unicode_punctuation("This is a long dash—used here.")
        "This is a long dash-used here."

        >>> normalize_unicode_punctuation("Wait... what?")
        "Wait... what?"
    """
    if not text:
        return text

    # Use str.translate() for efficient bulk character replacement
    translation_table = str.maketrans(UNICODE_TO_ASCII)
    result = text.translate(translation_table)

    # Post-processing for common patterns that need special handling

    # Handle multiple spaces that may result from space conversions
    result = re.sub(r'  +', ' ', result)

    # Clean up any leftover smart quote remnants (edge cases)
    # Remove any remaining high Unicode quote characters not in table
    result = re.sub(r'[\u0080-\u009F\u2000-\u200F\u2010-\u202F\u2050-\u205F\u3000-\u303F]', '', result)

    return result


def normalize_unicode_punctuation_preserve_newlines(text: str) -> str:
    """Convert Unicode punctuation to ASCII, preserving newlines.

    Same as normalize_unicode_punctuation() but preserves paragraph
    structure by handling newlines specially.

    Args:
        text: Input text that may contain Unicode punctuation.

    Returns:
        Text with Unicode punctuation normalized, newlines preserved.

    Examples:
        >>> normalize_unicode_punctuation_preserve_newlines("First line.\\nI'd go.")
        "First line.\\nI'd go."
    """
    if not text:
        return text

    # Preserve newlines by temporarily replacing them
    temp_marker = '\x00TEMP_NEWLINE\x00'
    preserved_text = text.replace('\n', temp_marker)

    # Normalize the rest
    normalized = normalize_unicode_punctuation(preserved_text)

    # Restore newlines
    result = normalized.replace(temp_marker, '\n')

    return result


def is_normalized(text: str) -> bool:
    """Check if text contains any problematic Unicode punctuation.

    Args:
        text: Text to check for Unicode issues.

    Returns:
        True if text contains only ASCII punctuation, False otherwise.

    Examples:
        >>> is_normalized("Hello, world!")
        True

        >>> is_normalized("I'd like to go.")
        False  # Has smart apostrophe
    """
    if not text:
        return True

    # Check if any Unicode characters exist outside ASCII range
    for char in text:
        code = ord(char)
        # Check for Unicode punctuation outside ASCII (0-127)
        if code > 127:
            return False

    return True


def get_normalization_report(text: str) -> dict:
    """Generate a report of Unicode characters that would be normalized.

    Args:
        text: Text to analyze.

    Returns:
        Dictionary with:
        - 'needs_normalization': bool
        - 'unicode_chars_found': list of (char, count) tuples
        - 'normalized_version': str with Unicode chars replaced
    """
    if not text:
        return {
            'needs_normalization': False,
            'unicode_chars_found': [],
            'normalized_version': text
        }

    # Count Unicode characters
    unicode_counts: Dict[str, int] = {}
    for char in text:
        if ord(char) > 127:
            unicode_counts[char] = unicode_counts.get(char, 0) + 1

    # Sort by frequency
    unicode_chars = sorted(unicode_counts.items(), key=lambda x: -x[1])

    return {
        'needs_normalization': len(unicode_chars) > 0,
        'unicode_chars_found': unicode_chars,
        'normalized_version': normalize_unicode_punctuation(text)
    }


# TTS pause metadata tokens like [2s] / [0.50s] (JSON + generation text).
_JSON_PAUSE_MARKER_RE = re.compile(r"\[\d+(?:\.\d+)?s\]", re.IGNORECASE)


# Sentence/clause punctuation that already cues a break before a pause tag.
_PAUSE_LEAD_PUNCT = set(".,;:!?…—–-")


def _line_needs_terminal_period(line: str) -> bool:
    """True when a non-empty line has no punctuation before its end.

    Trailing whitespace is ignored, so ``word.  \\n`` still counts as punctuated.
    Any non-letter/non-digit last character counts as punctuation (``.``, ``,``,
    ``!``, ``]``, quotes, etc.) — only a bare word/number end gets a period.

    Args:
        line: One source line body (newline already stripped).

    Returns:
        True if a period should be appended before newline flatten.
    """
    # Spaces between last glyph and newline do not clear punctuation.
    stripped = line.rstrip(" \t")
    if not stripped:
        return False
    last = stripped[-1]
    # Letter or digit → needs period; any other glyph is terminal punctuation.
    return last.isalnum()


def ensure_terminal_punctuation_on_lines(text: str) -> str:
    """Append ``.`` only on lines with no punctuation before the newline.

    Rule: if the last non-space character before ``\\n`` is not punctuation,
    append ``.``. Spaces between punctuation and the newline still count as
    punctuated (``Hello.   \\n`` is fine). Any punctuation character counts,
    not only ``.!?``.

    Blank lines are preserved. Used before newline→space flatten so line
    breaks without punct do not become continuous TTS speech.

    Args:
        text: Multi-line chunk or book fragment.

    Returns:
        Same text with terminal periods added where needed.
    """
    if not text:
        return text
    if "\n" not in text and "\r" not in text:
        # Single line with no newline: same rule on the whole string.
        if _line_needs_terminal_period(text):
            return text.rstrip(" \t") + "."
        return text

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        # Separate content from the newline ending (if any).
        if line.endswith("\r\n"):
            body, nl = line[:-2], "\r\n"
        elif line.endswith("\n") or line.endswith("\r"):
            body, nl = line[:-1], line[-1]
        else:
            body, nl = line, ""
        if _line_needs_terminal_period(body):
            body = body.rstrip(" \t") + "."
        out.append(body + nl)
    return "".join(out)


def _format_json_pause_markers(text: str) -> str:
    """Normalize whitespace around ``[Xs]`` markers without adding punctuation.

    Inline pauses split TTS into independent text segments, so a marker must
    remain a neutral boundary: ``Chapter[4s]One`` becomes ``Chapter [4s] One``
    rather than a synthetic comma. Existing user punctuation is preserved.
    """
    if not text or not _JSON_PAUSE_MARKER_RE.search(text):
        return text

    pieces: list[str] = []
    cursor = 0
    for match in _JSON_PAUSE_MARKER_RE.finditer(text):
        before = text[cursor : match.start()]
        # Preserve original tag casing/digits; normalize only surrounding form.
        marker = match.group(0)
        head = before.rstrip()
        if head:
            pieces.append(head)
            pieces.append(" ")
        pieces.append(marker)
        if match.end() < len(text):
            pieces.append(" ")
        cursor = match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    pieces.append(text[cursor:])
    return "".join(pieces)


def flatten_newlines_for_json(text: str) -> str:
    """Canonical chunk text for JSON, TTS plan, and pause insertion.

    Same string is written to audiobook.chunks.json and used when generating
    audio / inserting pauses. Mid-string markers always become
    ``, [Xs] `` (comma space tag space) so the model hears a comma break
    after tags are stripped (``Chapter, One``), matching hand-authored input.

    - add ``.`` on lines that lack terminal sentence punctuation (so a bare
      title line does not run into the next line after newline flatten)
    - replace newlines with spaces
    - force ``, [Xs] `` (or ``{punct} [Xs] ``) around every mid-string tag
    - leading ``[Xs] `` stays without a leading comma

    Call this when saving JSON, after chunking, and at every generate/insert
    entry point. Do not use during structure/paragraph detection on the raw
    book file.

    Args:
        text: Chunk text that may contain newlines or unspaced pause markers.

    Returns:
        Normalized chunk text for JSON + generation.
    """
    if not text:
        return text
    # Periods first so line-end boundaries survive newline → space collapse.
    with_terminals = ensure_terminal_punctuation_on_lines(text)
    flattened = re.sub(r"[\r\n]+", " ", with_terminals)
    flattened = re.sub(r" +", " ", flattened).strip()
    flattened = _format_json_pause_markers(flattened)
    flattened = re.sub(r" +", " ", flattened).strip()
    # Avoid doubled commas if source was already odd.
    flattened = re.sub(r",\s*,+", ",", flattened)
    flattened = re.sub(r" +", " ", flattened).strip()
    return flattened


def _single_quote_edges(text: str) -> tuple[bool, bool]:
    """Identify unmatched dialogue-quote edges while ignoring contractions.

    Returns:
        A pair of booleans for an unmatched opening quote and an unmatched
        closing quote respectively.
    """
    opening_quote_seen = False
    unmatched_closing_quote = False
    for index, character in enumerate(text):
        if character != "'":
            continue
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if previous.isalnum() and following.isalnum():
            # Apostrophes inside words, such as can't and won't, are never quotes.
            continue
        is_opening = following.isalnum() and not previous.isalnum()
        is_closing = previous and not following.isalnum()
        if is_opening:
            opening_quote_seen = True
        elif is_closing:
            if opening_quote_seen:
                opening_quote_seen = False
            elif previous.lower() != "s":
                # A punctuation-ended quote like ?' is a closing dialogue quote.
                unmatched_closing_quote = True
    return opening_quote_seen, unmatched_closing_quote


def _balance_dialogue_quotes(text: str) -> str:
    """Add model-only matching quote edges without changing word apostrophes.

    The source text remains untouched. This function only makes a standalone
    prompt self-contained when chunking divides one span of dialogue.
    """
    # A previous storage repair could have added an extra leading quote. Source
    # dialogue never needs two identical opening apostrophes for one chunk.
    if text.startswith("''") and (len(text) == 2 or text[2] != "'"):
        text = text[1:]

    single_open, single_close = _single_quote_edges(text)
    if single_close and not text.startswith("'"):
        text = "'" + text
    if single_open and not text.endswith("'"):
        text = text + "'"

    double_quote_positions = [index for index, character in enumerate(text) if character == '"']
    if len(double_quote_positions) % 2:
        quote_index = double_quote_positions[0]
        following = text[quote_index + 1] if quote_index + 1 < len(text) else ""
        if following.isalnum():
            text = text + '"'
        else:
            text = '"' + text
    return text


def build_tts_prompt(source_text: str) -> str:
    """Build a normalized, quote-balanced prompt without modifying source text.

    Args:
        source_text: Authored chunk text retained for sidecars, JSON, and ASR.

    Returns:
        A model-only prompt with normalized Unicode and balanced dialogue edges.
    """
    # Keep punctuation and pause-segment boundaries intact; source canonicalization
    # happens before this model-only layer in the audiobook pipeline.
    prompt = re.sub(r"[\r\n]+", " ", source_text)
    prompt = re.sub(r" +", " ", prompt).strip()
    if not prompt:
        return prompt
    prompt = normalize_unicode_punctuation(prompt)
    prompt = _balance_dialogue_quotes(prompt)
    return prompt


def normalize_text_chunk_for_storage(text: str) -> str:
    """Return canonical stored chunk text with self-contained dialogue quotes.

    Args:
        text: Chunk text produced by the structure detector or an older run.

    Returns:
        Canonical text for JSON and sidecars, with matching quote edges added
        when a sentence chunk divides a dialogue span.
    """
    canonical = flatten_newlines_for_json(text)
    if not canonical:
        return canonical
    return _balance_dialogue_quotes(normalize_unicode_punctuation(canonical))


if __name__ == "__main__":
    test_texts = [
        "I'd like to go, but I can't.",
        "He said, \"Hello!\"",
        "This is an em dash—like this.",
        "Wait... what?",
        "Cost: €100 or £80.",
        "¼ cup of sugar + ½ tsp salt.",
        "Sign up for our newsletter→click here",
        "Testing—multiple—dashes—and…ellipsis...",
    ]

    print("=" * 70)
    print("UNICODE PUNCTUATION NORMALIZATION TEST")
    print("=" * 70)

    for text in test_texts:
        report = get_normalization_report(text)
        print(f"\nOriginal:  {text}")
        print(f"Normalized: {report['normalized_version']}")
        if report['unicode_chars_found']:
            print(f"Unicode found: {report['unicode_chars_found']}")
        print("-" * 70)
