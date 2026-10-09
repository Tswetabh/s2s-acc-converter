"""Regression coverage for model-only prompt normalization."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from pocket_tts.models.tts_model import TTSModel, prepare_text_prompt
from pocket_tts.preprocessing.text_normalizer import (
    build_tts_prompt,
    normalize_text_chunk_for_storage,
)


class TtsPromptBuilderTests(unittest.TestCase):
    """Verify quote balancing never rewrites retained source text."""

    def test_closing_quote_only_gets_model_only_opening_quote(self) -> None:
        """A chunk ending dialogue must receive an opening quote for TTS only."""
        source = "But, Whitechurch! I can't believe it, can you?'"
        self.assertEqual(
            build_tts_prompt(source),
            "'But, Whitechurch! I can't believe it, can you?'",
        )
        self.assertEqual(source, "But, Whitechurch! I can't believe it, can you?'")

    def test_opening_quote_only_gets_model_only_closing_quote(self) -> None:
        """A chunk starting dialogue must receive a matching model-only close."""
        self.assertEqual(
            build_tts_prompt("'But, Whitechurch! I can't believe it, can you?"),
            "'But, Whitechurch! I can't believe it, can you?'",
        )

    def test_contractions_and_possessives_are_not_quote_edges(self) -> None:
        """Apostrophes inside words and possessive endings must remain intact."""
        source = "James' book says we can't and won't leave."
        self.assertEqual(build_tts_prompt(source), source)

    def test_unicode_dialogue_quotes_are_normalized_and_balanced(self) -> None:
        """Curly closing dialogue quote must become an ASCII balanced prompt."""
        self.assertEqual(build_tts_prompt("But, Whitechurch! Can you?\u2019"), "'But, Whitechurch! Can you?'")

    def test_scalar_model_generation_uses_the_model_only_prompt(self) -> None:
        """Scalar generation must send the balanced prompt, not raw chunk text."""
        model = TTSModel.__new__(TTSModel)
        torch.nn.Module.__init__(model)
        model.max_tokens_per_chunk = 50
        model.flow_lm = SimpleNamespace(conditioner=SimpleNamespace(tokenizer=object()))
        captured = []

        def capture_prompt(**kwargs):
            """Record the short prompt while yielding no audio for this unit test."""
            captured.append(kwargs["text_to_generate"])
            return iter(())

        model._generate_audio_stream_short_text = capture_prompt
        source = "But, Whitechurch! I can't believe it, can you?'"
        with patch(
            "pocket_tts.models.tts_model.split_into_best_sentences",
            return_value=[source],
        ):
            list(TTSModel.generate_audio_stream(model, {}, source))
        self.assertEqual(captured, ["'But, Whitechurch! I can't believe it, can you?'"])

    def test_prompt_preparation_does_not_invent_pause_segment_punctuation(self) -> None:
        """Shared preparation must keep a pause-separated segment exactly spoken."""
        prompt, _frames_after_eos = prepare_text_prompt("Chapter")
        self.assertEqual(prompt, "Chapter")

    def test_storage_normalization_balances_the_actual_chunk_text(self) -> None:
        """Stored chunks must gain the dialogue edge required by Regenerate."""
        self.assertEqual(
            normalize_text_chunk_for_storage(
                "In fact, I'm sure I heard a thud or thump here.'"
            ),
            "'In fact, I'm sure I heard a thud or thump here.'",
        )
        self.assertEqual(
            normalize_text_chunk_for_storage(
                "'Yes,' said Watson Major, 'what with the dark."
            ),
            "'Yes,' said Watson Major, 'what with the dark.'",
        )

    def test_storage_normalization_is_idempotent_for_repaired_quote_edges(self) -> None:
        """Reloading a repaired folder must not add another leading quote."""
        repaired = "'In fact, I'm sure I heard a thud or thump here.'"
        self.assertEqual(normalize_text_chunk_for_storage(repaired), repaired)
        self.assertEqual(
            normalize_text_chunk_for_storage("''Well, Your Grace,' he said."),
            "'Well, Your Grace,' he said.",
        )


if __name__ == "__main__":
    unittest.main()
