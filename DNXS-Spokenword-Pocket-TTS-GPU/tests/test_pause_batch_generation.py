"""Regression coverage for batched split-and-concatenate pause plans."""

import logging
import unittest
from types import SimpleNamespace

import torch

from pocket_tts.audiobook.generator import AudiobookGenerator, _generate_batch_chunk_audio
from pocket_tts.preprocessing.schema import BoundaryType, ChunkMetadata, EmotionType


class _Tokenizer:
    """Return a stable token count for the batch grouping key."""

    def __call__(self, text: str) -> SimpleNamespace:
        """Model the tokenizer object used by batched generation."""
        return SimpleNamespace(tokens=torch.ones(max(1, len(text.split())), dtype=torch.int64))


class _BatchTTS:
    """Record batch requests and return unique one-sample speech tensors."""

    sample_rate = 10
    device = "cpu"

    def __init__(self) -> None:
        """Create model-like fields used by the production batch helper."""
        self.flow_lm = SimpleNamespace(conditioner=SimpleNamespace(tokenizer=_Tokenizer()))
        self.requests: list[str] = []

    def generate_audio_batch(self, voice_state, requests):
        """Record complete segment requests and return one waveform per request."""
        self.requests.extend(request["text"] for request in requests)
        return [torch.tensor([float(index + 1)]) for index, _ in enumerate(requests)]


class _ScalarTTS:
    """Record serial calls made by the Main generation path."""

    sample_rate = 10

    def __init__(self) -> None:
        """Create mutable model settings and an empty request ledger."""
        self.requests: list[str] = []
        self.temp = 0.7
        self.eos_threshold = -4.0
        self.lsd_decode_steps = 1

    def generate_audio(self, voice_state, text, frames_after_eos=2):
        """Record a complete text segment and return deterministic speech."""
        self.requests.append(text)
        return torch.tensor([float(len(self.requests))])


def _chunk(text: str) -> SimpleNamespace:
    """Build minimal chunk metadata consumed by the production batch helper."""
    return SimpleNamespace(
        text=text,
        tts_params={
            "temperature": 0.7,
            "frames_after_eos": 2,
            "eos_threshold": -4.0,
            "lsd_decode_steps": 1,
            "speed_factor": 1.0,
        },
        pause_events=None,
        post_process={"silence_duration": 0.0},
    )


def _serial_chunk(text: str) -> ChunkMetadata:
    """Build real typed metadata for the decorated serial generator method."""
    return ChunkMetadata(
        index=0,
        text=text,
        word_count=2,
        character_count=len(text),
        boundary_type=BoundaryType.SENTENCE_END,
        punctuation=".",
        start_position=0,
        end_position=len(text),
        emotion=EmotionType.NEUTRAL,
        emotion_scores={"neutral": 1.0},
        emotion_confidence=1.0,
        tts_params={
            "temperature": 0.7,
            "frames_after_eos": 2,
            "eos_threshold": -4.0,
            "speed_factor": 1.0,
        },
        post_process={"silence_duration": 0.0},
    )


class PauseBatchGenerationTests(unittest.TestCase):
    """Verify batch reassembly uses the same pause-event behavior as Main."""

    def test_manual_pause_uses_segment_requests_when_punctuation_is_off(self) -> None:
        """Manual `[Xs]` plans must split in batch mode without auto punctuation."""
        model = _BatchTTS()
        chunk = _chunk("Chapter[0.2s]One.")

        output = _generate_batch_chunk_audio(
            model, "voice", [(7, chunk)], batch_size=4,
            pause_injection_enabled=False, pause_durations={},
            worker_logger=logging.getLogger(__name__), cleanup_enabled=False,
        )

        self.assertEqual(model.requests, ["Chapter", "One."])
        self.assertTrue(torch.equal(output[7], torch.tensor([1.0, 0.0, 0.0, 2.0])))
        self.assertTrue(chunk._inline_pause_preprocessed)
        self.assertEqual(chunk.pause_events, [
            {"kind": "text", "text": "Chapter", "samples": 1},
            {"kind": "pause", "seconds": 0.2, "samples": 2},
            {"kind": "text", "text": "One.", "samples": 1},
        ])

    def test_automatic_punctuation_uses_live_duration_values(self) -> None:
        """Enabled punctuation settings must become split pauses in batch mode."""
        model = _BatchTTS()
        chunk = _chunk("Wait. Go.")

        output = _generate_batch_chunk_audio(
            model, "voice", [(8, chunk)], batch_size=4,
            pause_injection_enabled=True, pause_durations={".": 0.2},
            worker_logger=logging.getLogger(__name__), cleanup_enabled=False,
        )

        self.assertEqual(model.requests, ["Wait.", "Go."])
        self.assertTrue(torch.equal(output[8], torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])))
        self.assertEqual(
            [event["samples"] for event in chunk.pause_events if event["kind"] == "pause"],
            [2, 2],
        )

    def test_batch_drops_quote_only_tail_after_manual_pause(self) -> None:
        """Batch generation must skip a standalone closing quote request."""
        model = _BatchTTS()
        chunk = _chunk("'Yes, an impression that all was not above board? [0.20s] '")

        output = _generate_batch_chunk_audio(
            model, "voice", [(9, chunk)], batch_size=4,
            pause_injection_enabled=False, pause_durations={},
            worker_logger=logging.getLogger(__name__), cleanup_enabled=False,
        )

        self.assertEqual(model.requests, ["'Yes, an impression that all was not above board?'"])
        self.assertTrue(torch.equal(output[9], torch.tensor([1.0, 0.0, 0.0])))

    def test_serial_main_path_keeps_manual_pause_when_punctuation_is_off(self) -> None:
        """Main generation must split manual plans and retain boundary silence."""
        model = _ScalarTTS()
        generator = SimpleNamespace(
            tts_model=model,
            config=SimpleNamespace(quality={"lsd_steps": 1}, audio_cleanup={"enabled": False}),
            _pause_injection_enabled=False,
            _pause_durations={},
            _ffmpeg_atempo=lambda audio, speed: audio,
        )
        chunk = _serial_chunk("Alpha[0.2s]Beta")
        chunk.post_process = {"silence_duration": 0.3}

        audio = AudiobookGenerator._generate_chunk_audio(generator, chunk, "voice")

        self.assertEqual(model.requests, ["Alpha", "Beta."])
        self.assertTrue(torch.equal(audio, torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])))
        self.assertEqual(chunk.pause_events[1]["samples"], 2)
