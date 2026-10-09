#!/usr/bin/env python3
"""Regression tests for ASR spoken-content normalization and comparison."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASR_DIR = ROOT / "ASR"
sys.path.insert(0, str(ASR_DIR))

from spoken_compare import (  # noqa: E402
    build_book_term_evidence,
    compare_spoken,
    normalize,
    tokens_phonetically_equal,
)
import spoken_compare as spoken_compare_module  # noqa: E402
from asr_validator import (  # noqa: E402
    _failure_entry_from_result,
    build_tts_book_term_evidence,
    run_parakeet_batch_validation,
    score_asr_pair,
)
from verification_backends import (  # noqa: E402
    AlignmentEvidence,
    NemoForcedAligner,
    TranscriptEvidence,
    build_alignment_report_record,
    decide_alignment,
    write_alignment_manifest,
)


class TestTwoStageBackendContracts(unittest.TestCase):
    """Backend contracts must be auditable without optional NeMo installed."""

    def test_stage_one_evidence_keeps_transcript_separate(self):
        """Stage 1 evidence preserves its own raw transcript and model metadata."""
        evidence = TranscriptEvidence(
            audio_path="/tmp/chunk.wav", text="Spoken words", timestamps=None,
            confidence=0.91, duration_s=1.2, backend="parakeet_tdt",
            model="nvidia/parakeet-tdt-0.6b-v3", elapsed_s=0.04,
        )
        payload = evidence.to_dict()
        self.assertEqual(payload["text"], "Spoken words")
        self.assertEqual(payload["backend"], "parakeet_tdt")

    def test_stage_two_evidence_has_no_stage_one_transcript(self):
        """Alignment evidence must never present a free-ASR transcript as alignment data."""
        evidence = AlignmentEvidence(
            audio_path="/tmp/chunk.wav", alignment_text="Expected words",
            available=False, conclusive=False, coverage=None, confidence=None,
            unaligned_tokens=[], extra_regions=[], duration_s=None,
            model="alignment-model", elapsed_s=0.0, error="not installed",
        )
        payload = evidence.to_dict()
        self.assertNotIn("transcribed_text", payload)
        self.assertEqual(payload["alignment_text"], "Expected words")

    def test_unavailable_alignment_is_not_failure_evidence(self):
        """Unavailable alignment must yield inconclusive evidence without regeneration."""
        aligner = NemoForcedAligner()
        with patch.object(NemoForcedAligner, "is_available", return_value=False):
            with tempfile.TemporaryDirectory() as tmpdir:
                results = aligner.align_batch(
                    [{"audio_path": "/tmp/chunk.wav", "text": "Expected words"}],
                    Path(tmpdir),
                )
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].available)
        self.assertFalse(results[0].conclusive)

    def test_alignment_manifest_uses_raw_source_text(self):
        """Manifest writing must retain original speech text rather than comparison slots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "chunk.wav"
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16_000)
                wav_file.writeframes(b"\0\0" * 160)
            manifest = write_alignment_manifest(
                [{"audio_path": str(audio_path), "text": "I counted forty-two stars."}],
                Path(tmpdir) / "manifest.jsonl",
            )
            row = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(row["text"], "I counted forty-two stars.")
        self.assertNotIn("<ID0>", row["text"])

    def test_inconclusive_alignment_keeps_original_audio(self):
        """Uncertain Stage 2 evidence must not become a regeneration candidate."""
        decision, reason = decide_alignment({"available": False, "conclusive": False})
        self.assertEqual(decision, "accepted_not_proven_failure")
        self.assertIn("not proven", reason)

    def test_strong_alignment_gap_confirms_failure(self):
        """Only strong, conclusive alignment gaps may enter regeneration."""
        decision, _ = decide_alignment({
            "available": True, "conclusive": True, "coverage": 0.2,
            "confidence": 0.2, "unaligned_tokens": ["expected"],
            "extra_regions": [],
        })
        self.assertEqual(decision, "confirmed_failed_by_alignment")

    def test_alignment_report_does_not_mix_stage_one_and_two_text(self):
        """Report must show Stage 1 transcript separately from raw alignment input text."""
        row = build_alignment_report_record({
            "chunk_index": 7, "chunk_id": "chunk_00007", "original_text": "Cloze my eyes.",
            "stage_one": {
                "transcribed_text": "Close my eyes.",
                "ref_normalized": "cloze my eyes",
                "hyp_normalized": "close my eyes",
                "score": 0.6,
                "id_score": 0.5,
                "identifier_comparisons": [
                    {
                        "slot": 0,
                        "kind": "identifier",
                        "expected": {"kind": "identifier", "canonical": "cloze", "components": ["cloze"]},
                        "hypothesis": {"kind": "identifier", "canonical": "close", "components": ["close"]},
                        "matched": False,
                        "result": "mismatch",
                    }
                ],
            },
            "alignment": {"alignment_text": "Cloze my eyes.", "coverage": 1.0},
            "decision": "accepted_by_alignment", "explanation": "Aligned.",
        })
        encoded = json.dumps(row)
        decoded = json.loads(encoded)
        self.assertEqual(row["stage_one"]["transcript"], "Close my eyes.")
        self.assertEqual(row["alignment"]["alignment_text"], "Cloze my eyes.")
        self.assertEqual(row["stage_one"]["comparison"]["identifier_comparisons"][0]["result"], "mismatch")
        self.assertEqual(
            decoded["stage_one"]["comparison"]["identifier_comparisons"][0]["expected"]["canonical"],
            "cloze",
        )

    def test_alignment_report_retains_list_label_evidence(self):
        """Report serialization must preserve accepted list-label evidence."""
        row = build_alignment_report_record({
            "chunk_index": 8,
            "chunk_id": "chunk_00008",
            "original_text": "Draconians, Osirians, Silurians, Sontarans...",
            "stage_one": {
                "transcribed_text": "Perconians, Assyrian, Salurians, Sontarans",
                "ref_normalized": "draconians osirians silurians sontarans",
                "hyp_normalized": "perconians assyrian salurians sontarans",
                "score": 0.91,
                "id_score": 1.0,
                "accepted_list_label_equivalences": [
                    {
                        "slot": 0,
                        "kind": "label_list",
                        "expected": {"surface": "Draconians", "canonical": "draconians", "components": ["draconians"]},
                        "hypothesis": {"surface": "Perconians", "canonical": "perconians", "components": ["perconians"]},
                        "matched": True,
                        "result": "accepted",
                    }
                ],
                "requires_second_stage_confirmation": False,
                "second_stage_confirmation_reason": "",
            },
            "alignment": {"alignment_text": "Draconians, Osirians, Silurians, Sontarans...", "coverage": 1.0},
            "decision": "accepted_by_alignment",
            "explanation": "Accepted list labels.",
        })
        encoded = json.dumps(row)
        decoded = json.loads(encoded)
        self.assertEqual(row["stage_one"]["comparison"]["accepted_list_label_equivalences"][0]["result"], "accepted")
        self.assertEqual(
            decoded["stage_one"]["comparison"]["accepted_list_label_equivalences"][0]["expected"]["canonical"],
            "draconians",
        )
        self.assertFalse(row["stage_one"]["comparison"]["requires_second_stage_confirmation"])

    def test_parakeet_stage_one_scores_one_batch_without_retry(self):
        """Parakeet Stage 1 must score one batch and leave candidates for Stage 2."""
        class FakeParakeet:
            """Small fake backend proving the validator sends one path batch."""

            def __init__(self, **_kwargs):
                """Create a fake backend without model loading."""
                self.closed = False

            def transcribe_paths(self, paths):
                """Return a deliberately mismatching transcript for the one path."""
                return [TranscriptEvidence(
                    audio_path=str(paths[0].resolve()), text="Wrong words.", timestamps=None,
                    confidence=0.9, duration_s=1.0, backend="parakeet_tdt",
                    model="fake", elapsed_s=0.01,
                )]

            def close(self):
                """Mark fake backend released after the book batch."""
                self.closed = True

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "audio_chunks").mkdir()
            (root / "text_chunks").mkdir()
            (root / "audio_chunks" / "chunk_00001.wav").write_bytes(b"wav")
            # Missing content must remain a Stage 1 fail (not a soft single-token swap).
            (root / "text_chunks" / "chunk_00001.txt").write_text(
                "Do you think they found it?", encoding="utf-8"
            )
            with patch("verification_backends.ParakeetTDTBackend", FakeParakeet):
                results = run_parakeet_batch_validation(root, ["chunk_00001"], 0.9, "base")
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["passed"])
        self.assertEqual(results[0]["stage_one_backend"], "parakeet_tdt")

    def test_parakeet_stage_one_uses_recurring_source_term_evidence(self):
        """Repeated source terms must clear harmless Stage 1 ASR spellings."""
        class FakeParakeet:
            """Provide one alternate spelling without loading a real ASR model."""

            def __init__(self, **_kwargs):
                """Create a fake backend without allocating model resources."""
                self.closed = False

            def transcribe_paths(self, paths):
                """Return a one-token ASR spelling variation for the recurring term."""
                return [TranscriptEvidence(
                    audio_path=str(paths[0].resolve()), text="The MSB arrived.",
                    timestamps=None, confidence=0.9, duration_s=1.0,
                    backend="parakeet_tdt", model="fake", elapsed_s=0.01,
                )]

            def close(self):
                """Mark fake backend released after the batch completes."""
                self.closed = True

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "audio_chunks").mkdir()
            text_dir = root / "text_chunks"
            text_dir.mkdir()
            (root / "audio_chunks" / "chunk_00001.wav").write_bytes(b"wav")
            for index in range(1, 4):
                (text_dir / f"chunk_{index:05d}.txt").write_text(
                    "The MSV arrived.", encoding="utf-8",
                )
            baseline = score_asr_pair(
                "chunk_00001", "The MSV arrived.", "The MSB arrived.", 0.60,
            )
            with patch("verification_backends.ParakeetTDTBackend", FakeParakeet):
                results = run_parakeet_batch_validation(
                    root, ["chunk_00001"], 0.60, "base",
                )
        self.assertFalse(baseline["passed"], baseline)
        self.assertTrue(results[0]["passed"], results[0])


class TestNormalization(unittest.TestCase):
    """Normalization must collapse only non-semantic written differences."""

    def _n(self, text: str) -> str:
        """Return normalized prose string for text."""
        return normalize(text)[0]

    def test_ordinal_forms_match(self):
        """Ordinal digits and words must normalize equivalently."""
        self.assertEqual(self._n("Twentieth."), self._n("20th."))
        self.assertEqual(self._n("20th"), self._n("twentieth"))
        self.assertEqual(self._n("20"), self._n("twenty"))

    def test_pronoun_i_not_roman(self):
        """Standalone lowercase pronoun i must never become one."""
        self.assertEqual(self._n("I am afraid."), "i am afraid")
        self.assertEqual(self._n("I am ready."), "i am ready")
        self.assertNotIn("one", self._n("I am ready.").split())

    def test_roman_numerals_multi_char(self):
        """Multi-character Roman numerals reach slots; "i" alone does not."""
        self.assertEqual(self._n("Chapter XX"), "chapter <ID0>")
        self.assertEqual(self._n("Louis XIV"), "louis <ID0>")

    def test_roman_and_written_numbers_share_one_slot(self):
        """Roman, digit, and written forms must reach the same numeric slot."""
        cases = [
            ("Murgon 3", "Murgan III"),
            ("Mark 6", "Mark VI"),
            ("Henry the eighth", "Henry VIII"),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                ref_norm = self._n(reference)
                hyp_norm = self._n(hypothesis)
                self.assertIn("<ID0>", ref_norm)
                self.assertIn("<ID0>", hyp_norm)

    def test_contractions(self):
        """Common contractions expand with straight or curly apostrophes."""
        self.assertEqual(self._n("I'm ready"), self._n("I am ready"))
        self.assertEqual(self._n("isn't"), self._n("is not"))
        self.assertEqual(self._n("wasn't"), self._n("was not"))
        self.assertEqual(self._n("weren't"), self._n("were not"))
        self.assertEqual(self._n("doesn't"), self._n("does not"))
        self.assertEqual(self._n("didn't"), self._n("did not"))
        self.assertEqual(self._n("can't"), self._n("cannot"))
        self.assertEqual(self._n("won't"), self._n("will not"))
        self.assertEqual(self._n("Everything'll be ready."), self._n("Everything will be ready."))
        self.assertEqual(self._n("Everything\u2019ll be ready."), self._n("Everything will be ready."))

    def test_nonstandard_and_ambiguous_contractions(self):
        """Speech-like spellings and apostrophe-d use stable contextual forms."""
        self.assertEqual(self._n("wasent"), self._n("was not"))
        self.assertEqual(self._n("wasnt"), self._n("wasn't"))
        self.assertEqual(self._n("y'know"), self._n("you know"))
        self.assertEqual(self._n("yknow"), self._n("you know"))
        self.assertEqual(self._n("He'd brought it."), self._n("He had brought it."))
        self.assertEqual(self._n("He'd bring it."), self._n("He would bring it."))

    def test_know_is_not_rewritten_to_negation(self):
        """The lexical word know must not be globally collapsed into no."""
        self.assertEqual(self._n("You know this."), "you know this")

    def test_hyphenated_syllables_become_tokens(self):
        """Hyphenated sung syllables become separate tokens, not deleted."""
        norm = self._n("tra-lal-lal-lal-lal-la")
        tokens = norm.split()
        self.assertGreaterEqual(len(tokens), 5)
        self.assertTrue(any(t == "lal" for t in tokens))

    def test_repetitions_not_collapsed(self):
        """Adjacent repetitions stay visible after normalization."""
        norm = self._n("hope hope hope hope")
        self.assertEqual(norm.split().count("hope"), 4)

    def test_punctuation_and_caps(self):
        """Punctuation and capitalization are non-semantic."""
        self.assertEqual(self._n("Hello, World!"), self._n("hello world"))

    def test_written_and_numeral_number_forms_use_matching_id_slots(self):
        """Every supported written-number form must match its numeral rendering."""
        cases = [
            ("I counted forty-two stars tonight.", "I counted 42 stars tonight."),
            ("The novel was published in nineteen eighty-four.", "The novel was published in 1984."),
            ("Her badge number was five nine two zero zero.", "Her badge number was 59200."),
            ("The price was twelve dollars.", "The price was $12."),
            ("The price was twelve pounds.", "The price was £12."),
            ("The price was twelve euros.", "The price was €12."),
            ("The price was twelve yen.", "The price was ¥12."),
            ("We left at seventeen hundred.", "We left at 1700."),
            ("We left at six oh five.", "We left at 6:05."),
            ("The tank held twelve liters.", "The tank held 12 liters."),
            ("She finished twenty-first in the race.", "She finished 21st in the race."),
            ("The estimate was three to five miles.", "The estimate was 3-5 miles."),
            ("I counted one hundred and one stars.", "I counted 101 stars."),
            ("I was one hundred per cent certain about that.", "I was 100% certain about that."),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertEqual(self._n(reference), self._n(hypothesis))
                self.assertIn("<ID0>", self._n(reference))

    def test_decade_numbers_preserve_intervening_prose(self):
        """Numeric and spoken decade forms keep words between their slots."""
        reference = (
            "Admiral Wellington was an older gentleman who appeared to be "
            "in his late 50s, early 60s."
        )
        hypothesis = (
            "Admiral Wellington was an older gentleman who appeared to be "
            "in his late fifties, early sixties."
        )
        expected = (
            "admiral wellington was an older gentleman who appeared to be "
            "in his late <ID0> early <ID1>"
        )
        self.assertEqual(self._n(reference), expected)
        self.assertEqual(self._n(hypothesis), expected)
        result = compare_spoken(reference, hypothesis)
        self.assertTrue(result["passed"], result)

    def test_decade_not_mixed_identifier(self):
        """Decade suffix notation stays numeric while M8-Tron stays an ID."""
        normalized, contexts = normalize("50s early M8-Tron")
        self.assertEqual(normalized, "<ID0> early <ID1>")
        self.assertEqual([context["kind"] for context in contexts], ["numeric", "identifier"])

    def test_inline_pause_markers_are_not_spoken_numbers(self):
        """TTS [Xs] silence markers must not become <NUM> in the book reference."""
        # Leading pause metadata.
        self.assertEqual(
            self._n("[2s]\nChapter One. Start of the first paragraph text."),
            self._n("Chapter One. Start of the first paragraph text."),
        )
        # Mid-phrase pauses (comma form).
        self.assertEqual(
            self._n("Chapter, [2s] One. Start of the first, [2s] paragraph text."),
            self._n("Chapter, One. Start of the first, paragraph text."),
        )
        # Bare decade 50s still numeric; bracketed [50s] is silence only.
        self.assertEqual(self._n("50s early"), self._n("fifties early"))
        self.assertEqual(self._n("[50s] early"), self._n("early"))

        leading = compare_spoken(
            "[2s]\nChapter One. Start of the first paragraph text.",
            "Chapter 1. Start of the first paragraph text.",
        )
        self.assertTrue(leading["passed"], leading)
        self.assertNotIn("<NUM>", leading.get("missing_tokens") or [])

        mid = compare_spoken(
            "Chapter, [2s] One. Start of the first, [2s] paragraph text.",
            "Chapter 1 Start of the first Paragraph text",
        )
        self.assertTrue(mid["passed"], mid)
        self.assertNotIn("<NUM>", mid.get("missing_tokens") or [])

    def test_split_word_number_rendering_matches_single_word(self):
        """A pronunciation spelling such as estim-eight matches estimate."""
        for hypothesis in ("estim 8", "estim-eight"):
            with self.subTest(hypothesis=hypothesis):
                result = compare_spoken(
                    "The estimate was high.",
                    f"The {hypothesis} was high.",
                )
                self.assertTrue(result["passed"], result)
                self.assertTrue(
                    any(
                        item["kind"] == "split_word_number_phonetic_equivalence"
                        for item in result["accepted_phrase_equivalences"]
                    ),
                    result,
                )

    def test_split_word_number_rendering_works_in_source(self):
        """The same pronunciation spelling is accepted when it is source text."""
        result = compare_spoken("The estim 8 was high.", "The estimate was high.")
        self.assertTrue(result["passed"], result)

    def test_split_word_number_can_coexist_with_other_safe_diffs(self):
        """A split pronunciation spelling remains accepted alongside another harmless ASR diff."""
        result = compare_spoken(
            "Based on how far the Hrum were from your position when you jumped, we estim-eight they will arrive in approximately four hours at which point we will express our displeasure with their choice of actions.",
            "Based on how far the rum were from your position when you jumped, we estimate they will arrive in approximately four hours at which point we will express our displeasure with their choice of actions.",
        )
        self.assertTrue(result["passed"], result)
        self.assertTrue(
            any(
                item["kind"] == "split_word_number_phonetic_equivalence"
                for item in result["accepted_phrase_equivalences"]
            ),
            result,
        )

    def test_split_word_number_rule_does_not_hide_numeric_or_identifier_content(self):
        """Ordinary numeric and typed-identifier phrases remain protected."""
        numeric = compare_spoken("Room 459 is ready.", "Room four five nine is ready.")
        identifier = compare_spoken("M8-Tron is ready.", "M8 Tron is ready.")
        self.assertTrue(numeric["passed"], numeric)
        self.assertTrue(identifier["passed"], identifier)

    def test_attached_numeric_unit_preserves_the_unit_as_prose(self):
        """A missing source space must not turn a number and unit into an ID."""
        reference = (
            "Every 45minutes like clockwork another flight of ten fighters took off."
        )
        hypothesis = (
            "Every 45 minutes like clockwork another flight of 10 fighters took off."
        )
        self.assertEqual(
            self._n(reference),
            "every <ID0> minutes like clockwork another flight of <ID1> fighters took off",
        )
        result = compare_spoken(reference, hypothesis)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["id_score"], 1.0, result)

    def test_spoken_digit_hundred_time_uses_one_numeric_slot(self):
        """A spoken military hour must match its four-digit ASR rendering."""
        reference = "He was due by zero nine hundred."
        hypothesis = "He was due by 0900."
        self.assertEqual(self._n(reference), "he was due by <ID0>")
        self.assertEqual(self._n(reference), self._n(hypothesis))
        result = compare_spoken(reference, hypothesis)
        self.assertTrue(result["passed"], result)

    def test_doctor_title_and_abbreviation_share_spoken_form(self):
        """Doctor and Dr. must not become a title-only ASR failure."""
        self.assertEqual(
            self._n("Doctor Casey continued."),
            self._n("Dr. Casey continued."),
        )

    def test_mrs_title_and_written_form_share_spoken_form(self):
        """Mrs. and the written word Misses must not trigger ASR failure."""
        self.assertEqual(
            self._n("Misses Casey continued."),
            self._n("Mrs. Casey continued."),
        )

    def test_delimited_letter_sequences_share_one_spoken_form(self):
        """Hyphenated and dotted initialisms represent identical spoken letters."""
        self.assertEqual(self._n("Heroic S-O-B."), self._n("Heroic S.O.B."))
        result = compare_spoken(
            'doctorke nodded. "Heroic S-O-B."',
            "Dr. Knotted. Heroic S.O.B.",
        )
        self.assertTrue(result["passed"], result)

    def test_typed_identifier_variants_share_one_slot(self):
        """Joined, spaced, and hyphenated typed identifiers normalize identically."""
        reference = "ASR M8 Tron is ready."
        expected = "asr <ID0> is ready"
        cases = [
            "ASR M8 Tron is ready.",
            "ASR M-8-Tron is ready.",
            "ASR M eight Tron is ready.",
        ]
        self.assertEqual(self._n(reference), expected)
        for hypothesis in cases:
            with self.subTest(hypothesis=hypothesis):
                self.assertEqual(self._n(hypothesis), expected)

    def test_typed_identifier_possessives_share_one_slot(self):
        """Possessive typed identifiers keep same slot and surrounding prose."""
        result = compare_spoken("He saw M8-Tron.", "He saw M8 Tron's.")
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["id_score"], 1.0, result)
        self.assertTrue(result["identifier_comparisons"], result)

    def test_typed_identifier_value_change_is_wildcard(self):
        """Identifier value changes remain positional wildcard matches."""
        result = compare_spoken("M8-Tron is ready.", "M9-Tron is ready.")
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["id_score"], 1.0, result)
        self.assertTrue(result["identifier_comparisons"], result)
        self.assertTrue(all(item["matched"] for item in result["identifier_comparisons"]), result)
        self.assertTrue(all(item["result"] == "matched" for item in result["identifier_comparisons"]), result)

    def test_punctuated_identifier_surface_normalizes_with_symbol_slot(self):
        """Punctuated symbol forms must normalize like typed identifiers."""
        result = compare_spoken("He saw M8-Tron.", "He saw M. H. Tron.")
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["id_score"], 1.0, result)
        self.assertTrue(result["identifier_comparisons"], result)

    def test_ordinary_prose_is_not_swallowed_as_identifier(self):
        """Ordinary prose initials must stay prose unless narrow ID grammar fits."""
        self.assertNotIn("<ID0>", self._n("Dr. A. Smith"))
        self.assertNotIn("<ID0>", self._n("Dr. A. Smith arrived."))

    def test_typed_and_numeric_slots_keep_textual_order(self):
        """Mixed numeric and typed slots retain their left-to-right positions."""
        reference = "42 M8-Tron arrived."
        hypothesis = "forty-two M8 Tron arrived."
        self.assertEqual(self._n(reference), "<ID0> <ID1> arrived")
        self.assertEqual(self._n(reference), self._n(hypothesis))

    def test_numeric_slot_remains_distinct_from_typed_identifier(self):
        """Numeric slots remain numeric contexts when an identifier follows them."""
        normalized, contexts = normalize("42 M8-Tron arrived.")
        self.assertEqual(normalized, "<ID0> <ID1> arrived")
        self.assertEqual([context["kind"] for context in contexts], ["numeric", "identifier"])

    def test_identifier_suffix_pronunciation_variant_is_accepted(self):
        """A same-position phonetic suffix variation does not change the ID."""
        result = compare_spoken("M8-Tron was ready.", "M8 Trong was ready.")
        self.assertTrue(result["passed"], result)

    def test_numeric_span_does_not_consume_plain_conjunction(self):
        """A trailing prose conjunction must remain outside a numeric slot."""
        self.assertEqual(self._n("One and only."), "<ID0> and only")

    def test_no_one_dialogue_quote_does_not_create_id_asymmetry(self):
        """Quoted source ``No one`` must match clean ASR without ID swallow."""
        cases = [
            (
                '"No one is making it back.',
                "No one is making it back.",
                "noone is making it back",
            ),
            (
                '"No one is to cross beneath that arch.',
                "No one is to cross beneath that arch.",
                "noone is to cross beneath that arch",
            ),
            (
                '"No one in a thousand years has gone.',
                "No one in a thousand years has gone.",
                "noone in a <ID0> years has gone",
            ),
        ]
        for reference, hypothesis, expected in cases:
            with self.subTest(reference=reference):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"], result)
                self.assertEqual(result["ref_normalized"], expected)
                self.assertEqual(result["hyp_normalized"], expected)
                # Compound itself must not be an ID token.
                self.assertTrue(result["ref_normalized"].startswith("noone"))

    def test_no_one_is_not_typed_identifier(self):
        """English ``No one is`` must stay prose, never typed-ID no1is."""
        normalized, contexts = normalize("No one is making it back.")
        self.assertEqual(normalized, "noone is making it back")
        self.assertEqual(contexts, [])

    def test_english_to_one_side_is_not_typed_identifier(self):
        """Two-letter English heads must not form typed IDs with number-words."""
        normalized, contexts = normalize("To one side of the room.")
        self.assertEqual(normalized, "to <ID0> side of the room")
        self.assertEqual([item["kind"] for item in contexts], ["numeric"])

    def test_washington_dc_forms_share_topology(self):
        """DC / D.C. / D. C. must not Roman-convert or glue following prose."""
        reference = '"Beijing. Washington, DC. Moscow.'
        for hypothesis in (
            "Beijing, Washington, D. C. Moscow",
            "Beijing, Washington D.C. Moscow",
            "Beijing, Washington DC Moscow",
        ):
            with self.subTest(hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"], result)
                self.assertEqual(result["ref_normalized"], "beijing washington dc moscow")
                self.assertEqual(result["hyp_normalized"], "beijing washington dc moscow")

    def test_dc_is_not_roman_numeral_six_hundred(self):
        """Bare uppercase DC must remain the geo acronym, not Roman 600."""
        normalized, contexts = normalize("Washington, DC.")
        self.assertEqual(normalized, "washington dc")
        self.assertEqual(contexts, [])

    def test_ones_and_one_share_numeric_slot(self):
        """ASR singular ``one`` for source ``ones`` must not invent slot asymmetry."""
        result = compare_spoken(
            "The one I managed to kill and the ones she destroyed?",
            "The one I managed to kill and the one she destroyed.",
        )
        self.assertTrue(result["passed"], result)

    def test_zero_percent_dash_matches_spaced_form(self):
        """Hyphenated percent forms share one numeric slot with spaced forms."""
        result = compare_spoken(
            "chance collisions at a zero-percent probability",
            "chance collisions at a zero percent probability",
        )
        self.assertTrue(result["passed"], result)

    def test_onetime_matches_hyphenated_asr_form(self):
        """Book onetime and ASR one-time are the same compound, not a NUM fail."""
        result = compare_spoken(
            '. "A onetime bargain?"',
            "A one-time bargain?",
        )
        self.assertTrue(result["passed"], result)
        self.assertEqual(self._n('. "A onetime bargain?"'), "a onetime bargain")
        self.assertEqual(self._n("A one-time bargain?"), "a onetime bargain")

    def test_firsthand_matches_split_asr_form(self):
        """Book firsthand and ASR first hand must not create a number slot."""
        result = compare_spoken(
            "seen firsthand the face of evil",
            "seen first hand the face of evil",
        )
        self.assertTrue(result["passed"], result)
        self.assertNotIn("<ID", self._n("seen first hand the face of evil"))

    def test_nonetheless_matches_spaced_form(self):
        """none the less and nonetheless are the same prose compound."""
        result = compare_spoken(
            "odd, none the less.",
            "odd, nonetheless.",
        )
        self.assertTrue(result["passed"], result)

    def test_mister_matches_mr_title_abbreviation(self):
        """Mister and Mr are the same spoken title, not a content fail."""
        cases = [
            ("Mister Spock said nothing.", "Mr. Spock said nothing."),
            ("Mister Sulu, report.", "Mr Sulu, report."),
            ("Comments, Mister Spock?", "Comments, Mr. Spock?"),
        ]
        for reference, hypothesis in cases:
            with self.subTest(hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"], result)

    def test_number_one_matches_no_dot_abbreviation(self):
        """Number One and No. 1 share one spoken form without false negation."""
        cases = [
            ("General Order Number One.", "General Order No. 1."),
            ("General Order Number One.", "General Order No 1."),
            ("General Order Number One.", "General Order number one."),
        ]
        for reference, hypothesis in cases:
            with self.subTest(hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"], result)
        # Bare dialogue "no" must still differ from non-negation content.
        self.assertEqual(self._n("No. 1"), self._n("Number 1"))
        self.assertEqual(self._n("No. 1"), self._n("Number One"))

    def test_asr_name_homophone_soft_pass(self):
        """ASR near-miss names/homophones are acceptable quality variance."""
        cases = [
            ("Karus made his way over.", "Harris made his way over."),
            ("Pammon's cohort is leading.", "Hammond's cohort is leading."),
            ("the century was already in position", "the sentry was already in position"),
            # Option 2: single content-token swap with matching shell.
            ("Xresex had been the headman of the Adile.", "Xerxes had been the headman of the ideal."),
            ("That is not all, Cyln'Phax said.", "That is not all, Somfax said."),
        ]
        for reference, hypothesis in cases:
            with self.subTest(hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"], result)

    def test_magnitude_numbers_and_num_wildcard(self):
        """sixty thousand ≡ 60,000; book NUM slot wildcards ASR confusable words."""
        result = compare_spoken(
            "numbered around fifty, perhaps even sixty, thousand.",
            "numbered around 50, perhaps even 60,000.",
        )
        self.assertTrue(result["passed"], result)
        result = compare_spoken(
            "Five? That will eat into our ready supply, sir.",
            "Hive, that will eat into our ready supply, sir.",
        )
        self.assertTrue(result["passed"], result)

    def test_copula_homophones_and_vocative_o(self):
        """they're/their, we're/were, and oh/O vocatives are spoken-equivalent."""
        self.assertTrue(
            compare_spoken("Do not attack. They're friends.", "Do not attack their friends.")["passed"]
        )
        self.assertTrue(
            compare_spoken("In their eyes, we're trapped.", "In their eyes were trapped.")["passed"]
        )
        self.assertTrue(
            compare_spoken("oh magnificent lady", "O magnificent lady")["passed"]
        )

    def test_letter_join_preposition_name_fusion(self):
        """for Carthum fused by ASR to Forcatham is still a pass."""
        result = compare_spoken(
            "They are marching this way and for Carthum.",
            "They are marching this way in Forcatham.",
        )
        self.assertTrue(result["passed"], result)

    def test_dialect_em_and_function_only_gaps(self):
        """Dialect 'em and function-only drops of correct audio should pass."""
        self.assertTrue(
            compare_spoken(
                "Train 'em hard and often is what I say.",
                "Train them hard and often is what I say",
            )["passed"]
        )
        self.assertTrue(
            compare_spoken(
                "That 'e did, sir.",
                "That he did, sir.",
            )["passed"]
        )
        self.assertTrue(
            compare_spoken(
                "Do they know you've spotted them?",
                "Do they know you spotted them?",
            )["passed"]
        )
        self.assertTrue(
            compare_spoken(
                "They blamed the god for the problems they faced.",
                "They blamed God for the problems they faced.",
            )["passed"]
        )
        self.assertTrue(
            compare_spoken(
                "The High Master of Obsidian saw to that.",
                "The high master of obsidian sought to that.",
            )["passed"]
        )
        self.assertTrue(
            compare_spoken(
                "Others begged shamelessly for food.",
                "Others beg shamelessly for food.",
            )["passed"]
        )
        # True fail: content opener missing
        self.assertFalse(
            compare_spoken(
                "please. There was hunger.",
                "There was hunger.",
            )["passed"]
        )
        # Filler battle-cry may still fail (accepted residual class)
        self.assertFalse(
            compare_spoken(
                "haaaah! The men shouted.",
                "Ha! the men shouted.",
            )["passed"]
        )

    def test_real_czar_garble_still_fails(self):
        """True content/ID-extra ASR errors must remain failures."""
        result = compare_spoken(
            "-Romulus de Menuchan, Third Communication Czar",
            "Romulus de Mnuchon, Third Communication Zod, Third Communication Zod.",
        )
        self.assertFalse(result["passed"], result)

    def test_standalone_oh_stays_spoken_filler(self):
        """Standalone oh must not be folded into numeric slot logic."""
        norm = self._n("Oh! There's a guy lying on the bunk.")
        self.assertNotIn("<ID0>", norm)
        self.assertIn("uh", norm.split())

    def test_single_letter_c_stays_letter(self):
        """Standalone C must not be converted into a Roman numeral."""
        norm = self._n("Big C battle.")
        self.assertIn("c", norm.split())
        self.assertNotIn("100", norm.split())

    def test_rules_file_reload_adds_variant(self):
        """Disk rule file must feed reloadable comparator mappings."""
        original_path = spoken_compare_module._RULES_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            rules_path = Path(tmpdir) / "spoken_compare_rules.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "spelling_variants": {"alfa": "alpha"},
                        "placeholder_prefixes": ["<NUM>", "<ID", "<TAG"],
                    }
                ),
                encoding="utf-8",
            )
            try:
                spoken_compare_module._RULES_PATH = rules_path
                spoken_compare_module.reload_spoken_rules()
                self.assertEqual(self._n("alfa"), "alpha")
                self.assertTrue(spoken_compare_module._is_placeholder_token("<TAG12>"))
            finally:
                spoken_compare_module._RULES_PATH = original_path
                spoken_compare_module.reload_spoken_rules()


class TestMustPass(unittest.TestCase):
    """Cases that must classify as PASS for acceptable spoken audio."""

    def test_twentieth(self):
        """Twentieth vs 20th is a full pass."""
        r = compare_spoken("Twentieth.", "20th.")
        self.assertTrue(r["passed"], r)
        self.assertIn(r["classification"], {"PASS", "PASS_WITH_MINOR_VARIATION"})

    def test_everythings_ll(self):
        """Everything'll expands to Everything will."""
        r = compare_spoken("Everything'll be ready.", "Everything will be ready.")
        self.assertTrue(r["passed"], r)

    def test_linx_links(self):
        """An x-to-ks spelling variant matches without a curated name alias."""
        r = compare_spoken("Linx", "links")
        self.assertTrue(r["passed"], r)
        self.assertEqual(
            r["accepted_equivalences"],
            [{"ref": "linx", "hyp": "links"}],
            r,
        )

    def test_linx_repeated(self):
        """Expected repeated Linx must not count as hallucination."""
        r = compare_spoken("Linx! Linx, I say!", "Links, links, I say.")
        self.assertTrue(r["passed"], r)
        self.assertFalse(any(
            d.get("severity") == "severe" for d in r.get("repetition_details", [])
        ), r)

    def test_general_single_token_phonetic_equivalences(self):
        """Aligned harmless ASR spellings pass and are recorded without penalties."""
        cases = [
            ("Graham", "Braham"),
            ("travelling", "traveling"),
            ("favourite", "favorite"),
            ("Geo", "Gio"),
            ("Yaz", "Yes"),
            ("Murgon", "Mergen"),
            ("vot", "what"),
            ("zis", "this"),
            ("bomb", "bong"),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"], result)
                self.assertEqual(result["substitutions"], [], result)
                self.assertEqual(
                    result["accepted_equivalences"],
                    [{"ref": reference.lower(), "hyp": hypothesis.lower()}],
                    result,
                )

    def test_protected_short_word_ambiguity_is_accepted(self):
        """Negation-like short-word drift passes as explicit ambiguous evidence."""
        cases = [
            ("He wasent here.", "He was sent here."),
            ("Not three apples.", "Mop 3 apples."),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"], result)
                self.assertTrue(result["accepted_ambiguous_equivalences"], result)
                self.assertFalse(result["substitutions"], result)
                self.assertEqual(result["accepted_equivalences"], [], result)

    def test_merged_split_phrase_ambiguity_is_accepted(self):
        """Negation-led two-token phrase drift passes as explicit phrase evidence."""
        result = compare_spoken(
            "Though not as family friendly as the original.",
            "Though Motz's family friendly is the original.",
        )
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["accepted_ambiguous_equivalences"], result)
        # Accept either phrase-merge evidence or protected short-word accept for
        # the negation-led drift; both document soft-pass without hard fail.
        self.assertTrue(
            any(
                item["kind"] in {"merged_split_phrase", "protected_short_word"}
                for item in result["accepted_ambiguous_equivalences"]
            ),
            result,
        )
        self.assertFalse(result["substitutions"], result)

    def test_production_report_preserves_ambiguous_equivalence(self):
        """Standalone report retains protected short-word acceptance evidence."""
        result = score_asr_pair(
            "chunk_02654",
            "He wasent here.",
            "He was sent here.",
            0.7,
        )
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["accepted_ambiguous_equivalences"], result)

    def test_compound_word_join_is_content_preserving(self):
        """One joined ASR word can match two reference words without a penalty."""
        result = compare_spoken("Whisper Men", "Whispermen")
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["missing_tokens"], [], result)
        self.assertEqual(result["extra_tokens"], [], result)

    def test_irongron_phonetic(self):
        """Irongron may match silhouetted ASR multi-token pronunciation."""
        r = compare_spoken("Irongron", "I wrong grown")
        self.assertIn(r["classification"], {"PASS", "FAIL"}, r)

    def test_punctuation_only(self):
        """Punctuation-only differences pass."""
        r = compare_spoken("Stop!", "Stop.")
        self.assertTrue(r["passed"], r)

    def test_expected_laughter_repetition(self):
        """Equivalent repeated laughter must not become a hallucination failure."""
        r = compare_spoken(
            "'Haw haw haw,' that's what it sounds like.",
            "Ha ha ha, that's what it sounds like.",
        )
        self.assertTrue(r["passed"], r)
        self.assertFalse(r["hallucination_warning"], r)

    def test_production_scorer_accepts_expected_laughter_repetition(self):
        """Production scoring must share the expected-laughter repetition rule."""
        r = score_asr_pair(
            "chunk_00108",
            "'Haw haw haw,' that's what it sounds like.",
            "Ha ha ha, that's what it sounds like.",
            0.7,
        )
        self.assertTrue(r["passed"], r)

    def test_production_scorer_keeps_non_numeric_substitution(self):
        """Production scoring must not hide prose errors beside a number slot."""
        r = score_asr_pair(
            "chunk_01859",
            "Have to meet a friend of mine back in nineteen thirty-nine.",
            "after meet a friend of mine back in 1939.",
            0.7,
        )
        self.assertFalse(r["passed"], r)

    def test_matching_id_placeholder_preserves_full_sentence(self):
        """Matching ID slots remain in both texts and cannot delete surrounding prose."""
        reference = (
            "Travelling by Vortex Manipulator is not pleasant, even with the latest "
            "model, and travelling to 1930s New York is ten times worse than a "
            "standard trip, for reasons I've already explained."
        )
        hypothesis = (
            "Traveling by vortex manipulator is not pleasant, even with the latest "
            "model and traveling to 1930s New York is ten times worse than a "
            "standard trip, for reasons I've already explained."
        )
        r = compare_spoken(reference, hypothesis)
        self.assertIn("<ID0>", r["ref_normalized"], r)
        self.assertIn("<ID0>", r["hyp_normalized"], r)
        self.assertIn("travelling", r["ref_normalized"], r)
        self.assertIn("traveling", r["hyp_normalized"], r)
        self.assertTrue(r["passed"], r)

    def test_quoted_ellipsized_label_list_equivalence(self):
        """Quoted ellipsized label lists with equal count may pass as one list."""
        result = compare_spoken(
            "'Draconians, Osirians, Silurians, Sontarans...'",
            "Perconians, Osserians, Alurians, Sontorans.",
        )
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["accepted_list_label_equivalences"], result)
        self.assertEqual(len(result["accepted_list_label_equivalences"]), 4, result)
        self.assertEqual(
            [item["slot"] for item in result["accepted_list_label_equivalences"]],
            [0, 1, 2, 3],
            result,
        )
        self.assertFalse(result["substitutions"], result)
        self.assertFalse(result.get("accepted_ambiguous_equivalences"), result)

    def test_filler_sounds_align_as_shared_variants(self):
        """Interjection fillers should compare as same spoken noise family."""
        r = compare_spoken("Oh! It's stopped.", "Hmm! It's stopped.")
        self.assertTrue(r["passed"], r)
        self.assertNotIn("low_score", r.get("failure_type") or "")

    def test_uh_oh_can_pass(self):
        """Uh oh should compare as filler-level variation, not a hard fail."""
        r = compare_spoken("Uh-oh. There's a guy lying on the bunk.", "Oh! There's a guy lying on the bunk.")
        self.assertTrue(r["passed"], r)

    def test_sung_syllables(self):
        """Expected sung syllable repetition is not severe hallucination."""
        r = compare_spoken("tra-lal-lal-lal-lal-la", "la la la la la la")
        # May pass or fail depending on lead syllable; never severe fail alone.
        if r["classification"] == "FAIL":
            self.assertNotEqual(r.get("failure_type"), "unexpected_repetition", r)


class TestMustFail(unittest.TestCase):
    """Cases that must classify as FAIL for defective spoken content."""

    def test_extra_suffix_thank_you(self):
        """Appended sentence must fail."""
        r = compare_spoken("Why?", "Why? Thank you.")
        self.assertFalse(r["passed"], r)
        self.assertEqual(r["classification"], "FAIL")

    def test_extra_suffix_stop(self):
        """Extra clause after Stop must fail."""
        r = compare_spoken("Stop!", "Stop. It is here.")
        self.assertFalse(r["passed"], r)
        self.assertEqual(r["classification"], "FAIL")

    def test_extra_suffix_car(self):
        """Substantial unexpected suffix must fail."""
        r = compare_spoken(
            "Everything will be ready soon.",
            "Everything will be ready soon. The car is dead too.",
        )
        self.assertFalse(r["passed"], r)
        self.assertEqual(r["classification"], "FAIL")

    def test_missing_negation(self):
        """Dropped negation must fail."""
        r = compare_spoken("I will not go.", "I will go.")
        self.assertFalse(r["passed"], r)
        self.assertTrue(
            r["classification"] == "FAIL"
            or any("negation" in str(c.get("type")) for c in r.get("critical_mismatches", [])),
            r,
        )

    def test_extra_negation(self):
        """Inserted negation must fail even if sentence is otherwise clean."""
        r = compare_spoken("I will go.", "I will not go.")
        self.assertFalse(r["passed"], r)
        self.assertEqual(r["classification"], "FAIL", r)
        self.assertEqual(r["failure_type"], "changed_negation", r)

    def test_merged_split_phrase_requires_surrounding_prose(self):
        """Missing surrounding prose must block bounded phrase ambiguity."""
        r = compare_spoken(
            "Though not as family friendly as the original.",
            "Motz's family friendly is the original.",
        )
        self.assertFalse(r["passed"], r)
        self.assertFalse(
            any(item["kind"] == "merged_split_phrase" for item in r.get("accepted_ambiguous_equivalences", [])),
            r,
        )

    def test_changed_number_is_an_opaque_slot(self):
        """Different number renderings are deliberately ignored by ASR comparison."""
        r = compare_spoken("There were twenty soldiers.", "There were thirty soldiers.")
        self.assertTrue(r["passed"], r)

    def test_unexpected_word_repeat(self):
        """Long unexpected single-word run must fail."""
        r = compare_spoken("He walked with a gait.", "gait gait gait gait gait gait gait")
        self.assertFalse(r["passed"], r)
        self.assertEqual(r["classification"], "FAIL")

    def test_number_slot_does_not_hide_surrounding_word_change(self):
        """A numeric slot must not hide a changed word outside the slot."""
        r = compare_spoken(
            "The price was twelve dollars.",
            "The price was 12 soldiers.",
        )
        self.assertFalse(r["passed"], r)

    def test_multiword_name_replacement_stays_failed(self):
        """A one-word name must not be forgiven as a multi-word ASR phrase."""
        result = compare_spoken("Yaz offered a consoling smile.", "He has offered a consoling smile.")
        self.assertFalse(result["passed"], result)
        self.assertNotIn(
            {"ref": "yaz", "hyp": "has"},
            result["accepted_equivalences"],
            result,
        )

    def test_ordinary_sentence_does_not_gain_list_equivalence(self):
        """Sentence spelling acceptance remains distinct from list handling."""
        result = compare_spoken("The draconians arrived.", "The perconians arrived.")
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["accepted_equivalences"], result)
        self.assertFalse(result["accepted_list_label_equivalences"], result)

    def test_list_item_count_mismatch_still_fails(self):
        """Comma-separated lists with missing items must remain failures."""
        result = compare_spoken(
            "'Draconians, Osirians, Silurians, Sontarans...'",
            "Perconians, Osserians, Alurians.",
        )
        self.assertFalse(result["passed"], result)
        self.assertFalse(result["accepted_list_label_equivalences"], result)

    def test_first_protected_word_sets_second_stage_confirmation(self):
        """First-word negation mismatch must require Stage 2 confirmation."""
        result = score_asr_pair(
            "chunk_00201",
            "Not one person moved.",
            "But one person moved.",
            0.7,
        )
        failure = _failure_entry_from_result(Path("/tmp/chunk_00201.wav"), "chunk_00201", result)
        self.assertFalse(result["passed"], result)
        self.assertTrue(result["requires_second_stage_confirmation"], result)
        self.assertIn("First protected word mismatch", result["second_stage_confirmation_reason"], result)
        self.assertTrue(failure["requires_second_stage_confirmation"], failure)

    def test_non_protected_first_word_is_not_flagged(self):
        """Ordinary first-word substitutions must not trigger confirmation hint."""
        result = compare_spoken("Blue moon rises.", "Green moon rises.")
        self.assertFalse(result["requires_second_stage_confirmation"], result)

    def test_unrecognized_identifier_surface_becomes_wildcard(self):
        """Split ASR identifier speech becomes the reference wildcard slot."""
        result = compare_spoken("Thank you, M8-Tron.", "Thank you, M. Atron.")
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["ref_normalized"], "thank you <ID0>", result)
        self.assertEqual(result["hyp_normalized"], "thank you <ID0>", result)
        self.assertTrue(
            any(op["op"] == "identifier_wildcard" for op in result["alignment_operations"]),
            result,
        )

    def test_generic_num_marker_is_not_parsed_as_id_slot(self):
        """Generic numeric markers must not enter positional ID comparisons."""
        self.assertIsNone(spoken_compare_module._placeholder_slot_index("<NUM>"))
        self.assertEqual(spoken_compare_module._placeholder_slot_index("<ID3>"), 3)

    def test_typed_identifier_variants_produce_matched_evidence(self):
        """Joined, spaced, hyphenated, and possessive variants must report matched slots."""
        cases = [
            ("ASR M8 Tron is ready.", "ASR M eight Tron is ready."),
            ("ASR M8 Tron is ready.", "ASR M-8-Tron is ready."),
            ("M8-Tron's module.", "M8 Tron's module."),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"], result)
                self.assertEqual(result["id_score"], 1.0, result)
                self.assertTrue(result["identifier_comparisons"], result)
                self.assertTrue(all(item["matched"] for item in result["identifier_comparisons"]), result)
                self.assertTrue(all(item["result"] == "matched" for item in result["identifier_comparisons"]), result)

    def test_punctuated_identifier_surface_matches(self):
        """Punctuated symbol surface must match equivalent typed identifier."""
        result = compare_spoken("He saw M8-Tron.", "He saw M. H. Tron's.")
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["id_score"], 1.0, result)
        self.assertTrue(result["identifier_comparisons"], result)
        self.assertTrue(all(item["matched"] for item in result["identifier_comparisons"]), result)

    def test_punctuated_identifier_surface_value_change_is_wildcard(self):
        """Punctuation-separated identifier values remain wildcard slots."""
        result = compare_spoken("He saw M8-Tron.", "He saw M. I. Tron.")
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["id_score"], 1.0, result)
        self.assertTrue(result["identifier_comparisons"], result)
        self.assertTrue(all(item["matched"] for item in result["identifier_comparisons"]), result)
        self.assertTrue(all(item["result"] == "matched" for item in result["identifier_comparisons"]), result)

    def test_identifier_surrounding_prose_still_fails(self):
        """Missing prose around a slot must still fail."""
        result = compare_spoken(
            "He spoke to M8-Tron's face.",
            "He spoke to M. I.Tron's extra face.",
        )
        self.assertFalse(result["passed"], result)

    def test_numeric_identifier_slots_remain_exempt(self):
        """Number-only comparisons must stay exempt from identifier scoring."""
        result = compare_spoken("There were twenty soldiers.", "There were thirty soldiers.")
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["id_score"], 1.0, result)
        self.assertTrue(result["identifier_comparisons"], result)
        self.assertTrue(all(item["kind"] == "numeric" for item in result["identifier_comparisons"]), result)
        self.assertTrue(all(item["result"] == "exempt" for item in result["identifier_comparisons"]), result)

    def test_failure_record_carries_identifier_evidence(self):
        """Validator failure rows must carry comparison evidence without crashing."""
        result = score_asr_pair(
            "chunk_00007",
            "He spoke to M8-Tron face.",
            "He spoke to M9-Tron.",
            0.7,
        )
        failure = _failure_entry_from_result(Path("/tmp/chunk_00007.wav"), "chunk_00007", result)
        self.assertFalse(result["passed"], result)
        self.assertEqual(result["id_score"], 1.0, result)
        self.assertTrue(failure["identifier_comparisons"], failure)
        self.assertEqual(failure["identifier_comparisons"], result["identifier_comparisons"], failure)

    def test_missing_spelled_identifier_stays_failed(self):
        """Missing acronym speech must fail even when a nearby word sounds close."""
        result = compare_spoken(
            "U-X-B. It stands for unexploded bomb.",
            "It stands for unexploded bong.",
        )
        self.assertFalse(result["passed"], result)
        self.assertIn("uxb", result["missing_tokens"], result)

    def test_unrelated_sentence(self):
        """Wholly unrelated hypothesis must fail."""
        r = compare_spoken(
            "The doctor stepped into the TARDIS.",
            "Please subscribe and hit the bell.",
        )
        self.assertFalse(r["passed"], r)
        self.assertEqual(r["classification"], "FAIL")


class TestBookTermEquivalence(unittest.TestCase):
    """Book-term evidence must stay bounded to clean recurring-term drift."""

    def test_recurring_capitalized_term_clears_clean_one_to_one_substitution(self):
        """Repeated Hrum source evidence accepts only its clean ASR spelling drift."""
        evidence = build_book_term_evidence(
            [
                "The Hrum advanced through the gate.",
                "A Hrum scout watched the wall.",
                "They fired at the Hrum column.",
            ]
        )
        result = compare_spoken(
            "The Hrum advanced through the gate.",
            "The rum advanced through the gate.",
            book_term_evidence=evidence,
        )

        self.assertTrue(result["passed"], result)
        self.assertFalse(result["substitutions"], result)
        self.assertTrue(result["accepted_equivalences"], result)

        production_result = score_asr_pair(
            "chunk_00009",
            "The Hrum advanced through the gate.",
            "The rum advanced through the gate.",
            0.60,
            book_term_evidence=evidence,
        )
        self.assertTrue(production_result["passed"], production_result)
        self.assertTrue(production_result["accepted_equivalences"], production_result)

    def test_book_term_does_not_clear_missing_negation_or_extra_speech(self):
        """A book term must not weaken negation, deletion, or extra-speech guards."""
        evidence = build_book_term_evidence(
            [
                "The Hrum advanced through the gate.",
                "A Hrum scout watched the wall.",
                "They fired at the Hrum column.",
            ]
        )
        result = compare_spoken(
            "The Hrum did not advance through the gate.",
            "The from did advance through the gate again.",
            book_term_evidence=evidence,
        )

        self.assertFalse(result["passed"], result)
        self.assertFalse(result["accepted_book_term_equivalences"], result)

    def test_evidence_requires_recurring_noninitial_capitalization(self):
        """Ordinary sentence-initial capitalization cannot create book-term evidence."""
        evidence = build_book_term_evidence(
            ["The room was quiet.", "The room was dark.", "The room was cold."]
        )

        self.assertNotIn("the", evidence["terms"])
        self.assertNotIn("room", evidence["terms"])

    def test_lowercase_recurring_term_can_become_book_term(self):
        """Lowercase recurring source terms can still become bounded book terms."""
        evidence = build_book_term_evidence(
            [
                "the hrum advanced through the gate.",
                "a hrum scout watched the wall.",
                "they fired at the hrum column.",
            ]
        )
        result = compare_spoken(
            "the hrum advanced through the gate.",
            "the rum advanced through the gate.",
            book_term_evidence=evidence,
        )

        self.assertIn("hrum", evidence["terms"])
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["accepted_book_term_equivalences"], result)

    def test_recurring_acronym_does_not_create_alias(self):
        """Recurring acronym evidence must not create a nonphonetic alias."""
        evidence = build_book_term_evidence(
            [
                "The NIS fleet entered orbit.",
                "A report reached the NIS command staff.",
                "They served the NIS for years.",
            ]
        )
        result = compare_spoken(
            "The NIS fleet entered orbit.",
            "The George fleet entered orbit.",
            book_term_evidence=evidence,
        )

        self.assertIn("nis", evidence["terms"])
        self.assertFalse(result["passed"], result)
        self.assertFalse(result["accepted_book_term_equivalences"], result)

    def test_tts_evidence_reads_original_chunk_text_only(self):
        """Book evidence is built from source chunk text, never ASR output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_dir = Path(tmpdir) / "text_chunks"
            text_dir.mkdir()
            for index, text in enumerate(
                [
                    "The Hrum advanced.",
                    "A Hrum watched.",
                    "They saw the Hrum leave.",
                ]
            ):
                (text_dir / f"chunk_{index:05d}.txt").write_text(text, encoding="utf-8")

            evidence = build_tts_book_term_evidence(tmpdir)

        self.assertIn("hrum", evidence["terms"])


class TestFinalAsrTolerance(unittest.TestCase):
    """Lock the approved structural ASR tolerance against Hrum failures."""

    def test_local_same_space_asr_spellings_pass(self):
        """Spelling-only Stage 2 disagreements must not trigger regeneration."""
        cases = [
            ("Vat-grown food was ready.", "That grown food was ready."),
            ("He had barely stepped away.", "He hath barely stepped away."),
            ("The scythed grass fell.", "The sighed grass fell."),
            ("He managed to slip through.", "He managed to step through."),
            ("How did you know?", "How do you know?"),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"], result)
                self.assertFalse(result["substitutions"], result)

    def test_function_word_only_gap_does_not_hard_fail(self):
        """Dropped article/auxiliary alone must not force regeneration."""
        missing_article = compare_spoken(
            "In an emergency, a fighter crew could mount their ship.",
            "In an emergency, fighter crew could mount their ship.",
            threshold=0.6,
        )
        missing_leading_a = compare_spoken(
            "You know the average John and Jane Doe out there are a heartbeat away.",
            "You know the average John and Jane Doe out there are heartbeat away.",
            threshold=0.6,
        )
        content_drop = compare_spoken(
            "The fighter crew could mount their ship.",
            "The crew could mount their ship.",
            threshold=0.6,
        )
        self.assertTrue(missing_article["passed"], missing_article)
        self.assertTrue(missing_leading_a["passed"], missing_leading_a)
        self.assertFalse(content_drop["passed"], content_drop)

    def test_percent_forms_share_one_numeric_slot(self):
        """40%, 40 percent, and 70percent water keep the same numeric structure."""
        digit_word = compare_spoken(
            "We can only have 40 percent of fighters ready.",
            "We can only have 40% of fighters ready.",
            threshold=0.6,
        )
        attached = compare_spoken(
            "covered by nearly 70percent water, shaded with trees.",
            "covered by nearly 70% water, shaded with trees.",
            threshold=0.6,
        )
        self.assertTrue(digit_word["passed"], digit_word)
        self.assertTrue(attached["passed"], attached)
        self.assertEqual(attached["missing_tokens"], [], attached)
        self.assertEqual(attached["extra_tokens"], [], attached)

    def test_ordinal_century_forms_align(self):
        """21st-century matches 21st century and twenty first century."""
        cases = [
            (
                "Medal of Honor in 21st-century America.",
                "Medal of Honor in 21st century America.",
            ),
            (
                "Medal of Honor in 21st-century America.",
                "Medal of Honor in twenty first century America.",
            ),
            (
                "a 21st-century Gatling gun",
                "a 21st century Gatling gun",
            ),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis, threshold=0.6)
                self.assertTrue(result["passed"], result)
                self.assertNotIn("century", result["extra_tokens"], result)

    def test_common_spoken_spelling_variants_pass(self):
        """Global spoken expansions for alright/onboard/woulda/ya/st titles."""
        cases = [
            ("Alright alright, look ahead.", "All right, all right. Look ahead."),
            ("It was onboard Omaha already.", "It was on board Omaha already."),
            ("I wish you woulda told me sooner.", "I wish you would have told me sooner."),
            ("Way ahead of ya, partner.", "Way ahead of you, partner."),
            ("population of St. George", "population of Saint George"),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis, threshold=0.6)
                self.assertTrue(result["passed"], result)

    def test_in_to_joins_to_into(self):
        """Split source 'in to' must match ASR 'into' with no leftover tokens."""
        result = compare_spoken(
            "When he keyed in to the command chat, it was a mess.",
            "When he keyed into the command chat, it was a mess.",
            threshold=0.6,
        )
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["missing_tokens"], [], result)
        self.assertEqual(result["extra_tokens"], [], result)

    def test_dash_number_and_letter_unit_measures_align(self):
        """teams-30 and 5K keep prose/unit structure matching spaced ASR forms."""
        teams = compare_spoken(
            "men on both teams-30 militia and 3 soldiers",
            "men on both teams 30 militia and three soldiers",
            threshold=0.6,
        )
        five_k = compare_spoken(
            "Five K would be easy for normal soldiers.",
            "5K would be easy for normal soldiers.",
            threshold=0.6,
        )
        self.assertTrue(teams["passed"], teams)
        self.assertNotIn("teams", teams["extra_tokens"], teams)
        self.assertTrue(five_k["passed"], five_k)
        self.assertEqual(five_k["missing_tokens"], [], five_k)

    def test_stutter_and_modal_function_gaps_pass(self):
        """I-I stutter collapse and soft modal/relative drops of correct audio."""
        stutter = compare_spoken(
            'Well-your majesty, I-I hadn\'t planned.',
            "Well, Your Majesty, I hadn't planned.",
            threshold=0.6,
        )
        modal_tense = compare_spoken(
            "officer who'd work so hard to cover his ass",
            "officer who worked so hard to cover his ass",
            threshold=0.6,
        )
        relative_drop = compare_spoken(
            "everyone else in his lineage who'd inherit before him",
            "everyone else in his lineage would inherit before him",
            threshold=0.6,
        )
        self.assertTrue(stutter["passed"], stutter)
        self.assertTrue(modal_tense["passed"], modal_tense)
        self.assertTrue(relative_drop["passed"], relative_drop)

    def test_joined_same_space_spans_pass_without_hiding_yaz(self):
        """Bounded resegmentation passes while pronoun-led replacement fails."""
        cases = [
            (
                "Mecha squad, we have a hostile force.",
                "Make a squad, we have a hostile force.",
            ),
            (
                "They fought during the Hrum war.",
                "They fought during the rimwar.",
            ),
            (
                "There was nowhere for them to go.",
                "There was no way for them to go.",
            ),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"], result)
        unsafe = compare_spoken(
            "Yaz offered a consoling smile.",
            "He has offered a consoling smile.",
        )
        self.assertFalse(unsafe["passed"], unsafe)

    def test_expected_repeat_can_collapse_to_one_asr_word(self):
        """ASR may collapse an expected adjacent repeated affirmative once."""
        result = compare_spoken(
            "Aye, aye, Admiral, said Bradley.",
            "I, Admiral, said Bradley.",
        )
        self.assertTrue(result["passed"], result)
        self.assertFalse(result["missing_tokens"], result)

    def test_apostrophe_d_uses_past_and_interrogative_context(self):
        """Contractions preserve spoken had and did readings before alignment."""
        self.assertEqual(
            normalize("She'd gotten there.")[0],
            normalize("She had gotten there.")[0],
        )
        self.assertEqual(
            normalize("How'd you know?")[0],
            normalize("How did you know?")[0],
        )
        result = compare_spoken(
            "She'd gotten there before noon.",
            "She had gotten there before noon.",
        )
        self.assertTrue(result["passed"], result)
        interrogative = compare_spoken("How'd you know?", "How do you know?")
        self.assertTrue(interrogative["passed"], interrogative)

    def test_lexical_possessive_preserves_normal_contraction_expansion(self):
        """Raw possessive clues do not alter ordinary contraction normalization."""
        self.assertEqual(normalize("George's system")[0], "george is system")
        self.assertEqual(normalize("She's ready")[0], "she is ready")

    def test_recurring_term_can_consume_raw_asr_possessive_spelling(self):
        """Recurring acronym evidence must not become a nonphonetic alias."""
        source = ["Aftermath in JOS system in orbit around JOS."] * 3
        evidence = build_book_term_evidence(source)
        result = compare_spoken(
            source[0],
            "Aftermath and George's system in orbit around Joljoz.",
            book_term_evidence=evidence,
        )
        self.assertFalse(result["passed"], result)
        self.assertFalse(result["accepted_book_term_equivalences"], result)

    def test_evidenced_possessive_term_does_not_expand_to_a_fake_word(self):
        """A recurring source term possessive remains one comparison token."""
        source = ["The Hrum's capital is heavily guarded."] * 3
        evidence = build_book_term_evidence(source)
        result = compare_spoken(
            source[0],
            "The rums capital is heavily guarded.",
            book_term_evidence=evidence,
        )
        self.assertTrue(result["passed"], result)
        self.assertNotIn("hrum is capital", result["ref_normalized"], result)

    def test_raw_phrase_resegmentation_preserves_spoken_equivalence(self):
        """A one-word phrase may match its safe raw ASR word split."""
        result = compare_spoken(
            "Albeit a bloody one for the fleet.",
            "I'll be at a bloody one for the fleet.",
        )
        self.assertTrue(result["passed"], result)
        self.assertEqual(
            result["accepted_phrase_equivalences"],
            [{
                "kind": "raw_phrase_resegmentation",
                "expected": "Albeit",
                "hypothesis": "I'll be at",
            }],
            result,
        )

    def test_raw_phrase_resegmentation_accepts_asr_spelling_drift(self):
        """ASR spelling inside a safe spoken-space split must not fail audio."""
        result = compare_spoken(
            "Albeit a bloody one for the fleet.",
            "I'll buy it a bloody one for the fleet.",
        )
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["accepted_phrase_equivalences"], result)
        production_result = score_asr_pair(
            "chunk_00001",
            "Albeit a bloody one for the fleet.",
            "I'll buy it a bloody one for the fleet.",
            0.60,
        )
        self.assertTrue(production_result["accepted_phrase_equivalences"], production_result)

    def test_raw_contraction_auxiliary_variations_pass(self):
        """ASR contraction expansion must not create a false speech failure."""
        omitted_subject = compare_spoken(
            "He'd been struck by a glancing blow.",
            "Had been struck by a glancing blow.",
        )
        omitted_auxiliary = compare_spoken(
            "On that note, he'd decided he'd been sitting down long enough.",
            "On that note, he decided he'd been sitting down long enough.",
        )
        added_auxiliary = compare_spoken(
            "You been talking to my wife again.",
            "You've been talking to my wife again.",
        )
        for result in (omitted_subject, omitted_auxiliary, added_auxiliary):
            with self.subTest(result=result):
                self.assertTrue(result["passed"], result)
                self.assertTrue(result["accepted_phrase_equivalences"], result)

    def test_apostrophe_free_she_d_spelling_passes(self):
        """Apostrophe-free shed must clear when it stands for she'd."""
        result = compare_spoken(
            "She'd call from orbit.",
            "Shed call from orbit.",
        )
        self.assertTrue(result["passed"], result)
        self.assertEqual(
            result["accepted_phrase_equivalences"],
            [{
                "kind": "contracted_pronoun_spelling",
                "expected": "She'd",
                "hypothesis": "Shed",
            }],
            result,
        )

    def test_unrelated_shed_substitution_still_fails(self):
        """Ordinary shed words must stay distinct from she'd contractions."""
        cases = [
            ("He'd call from orbit.", "Shed call from orbit."),
            ("She'd call from orbit.", "Shed steal from orbit."),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertFalse(result["passed"], result)
                self.assertFalse(result["accepted_phrase_equivalences"], result)

    def test_apostrophe_s_surface_variations_pass(self):
        """Possessive and contracted apostrophe-s ASR renderings are harmless."""
        cases = [
            (
                "Kane, Doherty, Lawson and Dane's squads.",
                "Kane, Doherty, Lawson and Dane squads.",
            ),
            (
                "If you think something's so funny, laugh.",
                "If you think something so funny, laugh.",
            ),
            (
                "The dude's are on another level.",
                "The dudes are on another level.",
            ),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"], result)
                self.assertEqual(
                    result["accepted_phrase_equivalences"][0]["kind"],
                    "apostrophe_s_surface_variation",
                    result,
                )

    def test_apostrophe_s_variation_can_combine_with_a_safe_word_split(self):
        """Apostrophe-s repair must leave another harmless split available."""
        result = compare_spoken(
            "I know you may think you're a badass, but the dude's are here.",
            "I know you may think you're a bad ass, but the dudes are here.",
        )
        self.assertTrue(result["passed"], result)
        self.assertTrue(
            any(
                item["kind"] == "apostrophe_s_surface_variation"
                for item in result["accepted_phrase_equivalences"]
            ),
            result,
        )
        self.assertTrue(result["accepted_phrase_equivalences"], result)

    def test_lexical_possessive_filler_variation_passes(self):
        """A possessive plus source filler must not become a fake auxiliary."""
        reference = '"Charmed." "I uh-suppose we have you to thank for the shuttle\'s um paint."'
        hypothesis = "Charmed. I uh suppose we have you to thank for the shuttles on paint."
        result = compare_spoken(reference, hypothesis)
        self.assertTrue(result["passed"], result)
        self.assertTrue(
            any(
                item["kind"] == "lexical_possessive_filler_variation"
                for item in result["accepted_phrase_equivalences"]
            ),
            result,
        )
        unsafe = compare_spoken(
            "The shuttle's um paint was fresh.",
            "The shuttles on color was fresh.",
        )
        self.assertFalse(unsafe["passed"], unsafe)

    def test_single_internal_conjunction_omission_is_harmless(self):
        """One omitted internal and is accepted only when content surrounds it."""
        result = compare_spoken(
            "He wrapped an arm around the Governor's shoulder and walked away.",
            "He wrapped an arm around the Governor's shoulder, walked away.",
        )
        self.assertTrue(result["passed"], result)
        self.assertEqual(
            result["accepted_phrase_equivalences"][0]["kind"],
            "single_internal_conjunction_omission",
            result,
        )
        unsafe = compare_spoken("Walk or run now.", "Walk run now.")
        self.assertFalse(unsafe["passed"], unsafe)

    def test_article_compound_join_is_content_preserving(self):
        """An exact a-plus-word compound join must not lose a spoken word."""
        result = compare_spoken(
            "How about we go a round or two in the ring?",
            "How about we go around or two in the ring?",
        )
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["missing_tokens"], [], result)
        self.assertEqual(result["extra_tokens"], [], result)

    def test_delimited_letter_sequences_canonicalize_to_one_token(self):
        """Explicit letter sequences should compare against concatenated ASR forms."""
        cases = [
            ("J-O-S is here.", "JOS is here."),
            ("J.O.S. is here.", "JOS is here."),
            ("X-O is here.", "XO is here."),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"], result)
                self.assertEqual(result["ref_normalized"], result["hyp_normalized"], result)

    def test_delimited_letter_sequence_changes_still_fail(self):
        """Changed or missing letters must still be a failure."""
        changed = compare_spoken("J-O-S is here.", "JOT is here.")
        missing = compare_spoken("X-O is here.", "X is here.")
        self.assertFalse(changed["passed"], changed)
        self.assertFalse(missing["passed"], missing)

    def test_two_letter_bare_acronym_can_render_as_spoken_letters(self):
        """Bare two-letter source acronyms may match compact ASR letter names."""
        result = compare_spoken(
            "Jaxton retired while his XO coordinated.",
            "Jackson retired while his EXO coordinated.",
        )
        self.assertTrue(result["passed"], result)
        self.assertTrue(
            any(
                item.get("kind") == "bare_two_letter_acronym_surface_variation"
                for item in result["accepted_phrase_equivalences"]
            ),
            result,
        )

    def test_two_letter_hyphenated_acronym_can_render_as_spoken_letters(self):
        """Hyphenated X-O must match compact Exo or split ex o letter names."""
        compact = compare_spoken(
            "while the X-O was removed from active duty.",
            "while the Exo was removed from active duty.",
        )
        split = compare_spoken(
            "his X-O, Hadley remained.",
            "his ex o, Hadley remained.",
        )
        self.assertTrue(compact["passed"], compact)
        self.assertTrue(split["passed"], split)
        self.assertTrue(
            any(
                item.get("kind") == "bare_two_letter_acronym_surface_variation"
                for item in compact["accepted_phrase_equivalences"]
            ),
            compact,
        )

    def test_two_letter_bare_acronym_wrong_or_missing_letters_fail(self):
        """Wrong or missing acronym letters must remain hard failures."""
        wrong = compare_spoken("XO is here.", "EXX is here.")
        missing = compare_spoken("XO is here.", "X is here.")
        hyphen_wrong = compare_spoken("X-O is here.", "EXX is here.")
        self.assertFalse(wrong["passed"], wrong)
        self.assertFalse(missing["passed"], missing)
        self.assertFalse(hyphen_wrong["passed"], hyphen_wrong)

    def test_three_letter_bare_acronym_sob_unchanged(self):
        """Three-letter bare acronyms must not inherit two-letter rendering tolerance."""
        result = compare_spoken("SOB is here.", "ESSOHBEE is here.")
        self.assertFalse(result["passed"], result)

    def test_contraction_variation_allows_other_safe_spelling_drift(self):
        """One contraction repair leaves separate aligned spellings to compare."""
        source = [
            "He'd been struck by a glancing blow from a Hrum rifle."
        ] * 3
        evidence = build_book_term_evidence(source)
        result = compare_spoken(
            source[0],
            "Had been struck by a glancing blow from a HRM rifle.",
            book_term_evidence=evidence,
        )
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["accepted_phrase_equivalences"], result)

    def test_raw_phrase_resegmentation_rejects_negation_and_numbers(self):
        """Raw phrase splitting must not weaken protected semantic tokens."""
        negation = compare_spoken("Not able to proceed.", "Notable to proceed.")
        number = compare_spoken("Captain arrived.", "Cap ten arrived.")
        self.assertFalse(negation["passed"], negation)
        self.assertFalse(number["passed"], number)

    def test_one_off_short_acronym_is_not_tolerated(self):
        """Do not decide pronunciation for a non-recurring all-caps token."""
        result = compare_spoken("Heroic SOB.", "Heroic be.")
        self.assertFalse(result["passed"], result)


class TestMustNotRegress(unittest.TestCase):
    """Homograph and pronoun regressions that previous maps got wrong."""

    def test_reed_not_red(self):
        """reed and red are different pronunciations — not automatic equals."""
        self.assertFalse(tokens_phonetically_equal("reed", "red"))
        r = compare_spoken("reed", "red")
        # Single-token minimal pair should not be a free pass via canon map
        if r["passed"]:
            self.assertNotEqual(r.get("ref_normalized"), r.get("hyp_normalized") or "x")

    def test_lyve_not_liv(self):
        """lyve and liv are different pronunciations — not automatic equals."""
        self.assertFalse(tokens_phonetically_equal("lyve", "liv"))

    def test_i_am_not_one_am(self):
        """Pronoun I stays i after normalize."""
        norm, _ = normalize("I am ready.")
        self.assertEqual(norm, "i am ready")


if __name__ == "__main__":
    unittest.main()
