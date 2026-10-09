"""Regression coverage for split-and-concatenate inline pause rendering."""

import unittest

import torch

from pocket_tts.preprocessing.pause_injector import (
    generate_audio_with_pauses,
    has_inline_pause_markers,
    inject_pauses_for_punctuation,
    is_speech_text_segment,
    parse_text_with_pauses,
    render_text_with_native_pauses,
)
from pocket_tts.preprocessing.text_normalizer import flatten_newlines_for_json


class _RecordingTTS:
    """Minimal TTS double that identifies each complete generation request."""

    sample_rate = 10
    device = "cpu"

    def __init__(self) -> None:
        """Create a request ledger for the current test."""
        self.requests: list[tuple[object, str]] = []

    def generate_audio(self, voice_state: object, text: str) -> torch.Tensor:
        """Record one text segment and return its deterministic waveform."""
        self.requests.append((voice_state, text))
        value = float(len(self.requests)) / 10.0
        return torch.tensor([value, -value])


class PauseInjectorTests(unittest.TestCase):
    """Verify manual and automatic events never cut a continuous waveform."""

    def test_manual_marker_splits_glued_text_without_synthetic_comma(self) -> None:
        """Glued markers must create two complete TTS calls and exact silence."""
        tts = _RecordingTTS()

        audio, events = generate_audio_with_pauses(tts, "voice", "Chapter[4s]One.")

        self.assertEqual(tts.requests, [("voice", "Chapter"), ("voice", "One.")])
        self.assertEqual(audio.numel(), 44)
        self.assertTrue(torch.equal(audio[:2], torch.tensor([0.1, -0.1])))
        self.assertTrue(torch.equal(audio[2:42], torch.zeros(40)))
        self.assertTrue(torch.equal(audio[42:], torch.tensor([0.2, -0.2])))
        self.assertEqual(events[1], {"kind": "pause", "seconds": 4.0, "samples": 40})

    def test_decimal_marker_preserves_exact_sample_duration(self) -> None:
        """A decimal manual duration must not be rounded to a 100 ms step."""
        tts = _RecordingTTS()

        audio, events = generate_audio_with_pauses(tts, "voice", "Alpha[0.15s]Beta")

        self.assertEqual(tts.requests, [("voice", "Alpha"), ("voice", "Beta.")])
        self.assertEqual(events[1]["samples"], 2)  # round(0.15 * 10)
        self.assertTrue(torch.equal(audio[2:4], torch.zeros(2)))

    def test_leading_trailing_consecutive_and_marker_only_plans(self) -> None:
        """Every pause topology must avoid empty TTS requests and retain zeros."""
        tts = _RecordingTTS()
        audio, events = generate_audio_with_pauses(
            tts, "voice", "[0.2s] Alpha[0.1s][0.2s] Beta[0.3s]"
        )

        self.assertEqual(tts.requests, [("voice", "Alpha"), ("voice", "Beta")])
        self.assertEqual(audio.numel(), 12)
        self.assertTrue(torch.equal(audio[:2], torch.zeros(2)))
        self.assertTrue(torch.equal(audio[4:7], torch.zeros(3)))
        self.assertTrue(torch.equal(audio[-3:], torch.zeros(3)))
        self.assertEqual([event["kind"] for event in events], ["pause", "text", "pause", "pause", "text", "pause"])

        marker_only, marker_events = generate_audio_with_pauses(tts, "voice", "[0.2s][0.3s]")
        self.assertEqual(marker_only.numel(), 5)
        self.assertEqual(len(tts.requests), 2)
        self.assertTrue(torch.equal(marker_only, torch.zeros(5)))
        self.assertEqual([event["samples"] for event in marker_events], [2, 3])

    def test_punctuation_annotation_does_not_duplicate_manual_pause(self) -> None:
        """Automatic punctuation events must defer to a manual marker."""
        annotated = inject_pauses_for_punctuation(
            "Wait.[2s] Continue.", {".": 0.5}
        )

        self.assertEqual(annotated, "Wait.[2s] Continue.[0.50s]")
        self.assertTrue(has_inline_pause_markers("Manual[0.15s]pause"))
        self.assertFalse(has_inline_pause_markers("[pause:150ms]"))

    def test_punctuation_annotation_does_not_rewrite_marker_decimals(self) -> None:
        """Existing decimal markers must remain atomic during auto injection."""
        annotated = inject_pauses_for_punctuation("Wait.[0.15s] Continue.", {".": 0.5})

        self.assertEqual(annotated, "Wait.[0.15s] Continue.[0.50s]")

    def test_quote_only_tail_is_not_sent_to_tts(self) -> None:
        """A closing dialogue quote after a pause must not synthesize alone."""
        tts = _RecordingTTS()

        audio, events = generate_audio_with_pauses(
            tts,
            "voice",
            "'Yes, an impression that all was not above board? [0.50s] '",
        )

        self.assertEqual(
            tts.requests,
            [("voice", "'Yes, an impression that all was not above board?")],
        )
        self.assertEqual([event["kind"] for event in events], ["text", "pause"])
        self.assertTrue(is_speech_text_segment("said Holmes"))
        self.assertFalse(is_speech_text_segment("'"))

    def test_parser_and_metadata_keep_user_punctuation(self) -> None:
        """Pause parsing must preserve punctuation instead of adding comma cues."""
        plan = flatten_newlines_for_json("Chapter[4s]One. Next [0.15s] word.")
        events, pauses = parse_text_with_pauses(plan)

        self.assertEqual(plan, "Chapter [4s] One. Next [0.15s] word.")
        self.assertEqual(events, [
            ("text", "Chapter "),
            ("pause", 4.0),
            ("text", " One. Next "),
            ("pause", 0.15),
            ("text", " word."),
        ])
        self.assertEqual(len(pauses), 2)
        self.assertEqual(render_text_with_native_pauses(plan), "Chapter One. Next word.")

    def test_custom_segment_postprocess_runs_before_zero_assembly(self) -> None:
        """Renderer callbacks must finish speech before exact silence is added."""
        tts = _RecordingTTS()
        calls: list[str] = []

        def render_segment(text: str) -> torch.Tensor:
            """Return a postprocessed waveform that proves callback ordering."""
            calls.append(text)
            return torch.tensor([float(len(calls))])

        audio, events = generate_audio_with_pauses(
            tts, "voice", "One[0.2s]Two", generate_segment=render_segment
        )

        self.assertEqual(calls, ["One", "Two."])
        self.assertTrue(torch.equal(audio, torch.tensor([1.0, 0.0, 0.0, 2.0])))
        self.assertEqual(events[1]["samples"], 2)


if __name__ == "__main__":
    unittest.main()
