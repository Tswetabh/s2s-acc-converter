#!/usr/bin/env python3
"""Apply the Pocket TTS ghosting/echo fixes to a project tree.

Usage:
    python3 apply_pocket_tts_fixes.py            # run inside the project root
    python3 apply_pocket_tts_fixes.py /path/to/project

Safe: asserts every anchor, applies each edit once, skips already-patched
files. Same code runs on Linux and Windows.
"""
import sys
from pathlib import Path

def main() -> None:
    """Entry point for the script; processes command-line arguments and checks project root directory."""
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = Path.cwd()
    assert (root / "pocket_tts").is_dir(), f"not a project root: {root}"

    edits = {}
    edits['pocket_tts/audiobook/generator.py'] = [
        ('    from pocket_tts.preprocessing.pause_injector import (\n        has_inline_pause_markers,\n        inject_pauses_for_punctuation,\n',
         '    from pocket_tts.preprocessing.pause_injector import (\n        ensure_terminal_punctuation,\n        has_inline_pause_markers,\n        inject_pauses_for_punctuation,\n',
         None),
        ('            events = [("text", render_text_with_native_pauses(raw_text))]',
         '            events = [("text", ensure_terminal_punctuation(render_text_with_native_pauses(raw_text)))]',
         None),
        ('            events = [("text", raw_text)]',
         '            events = [("text", ensure_terminal_punctuation(raw_text))]',
         None),
        ('                logger.debug("Calling self.tts_model.generate_audio...")\n                audio = self.tts_model.generate_audio(\n                    voice_state,\n                    chunk.text,\n',
         '                logger.debug("Calling self.tts_model.generate_audio...")\n                from ..preprocessing.pause_injector import (\n                    ensure_terminal_punctuation,\n                )\n                audio = self.tts_model.generate_audio(\n                    voice_state,\n                    ensure_terminal_punctuation(chunk.text),\n',
         None),
        ('                else:\n                    chunk._inline_pause_text = None\n                    audio = tts_model.generate_audio(\n                        voice_state,\n                        chunk.text,\n',
         '                else:\n                    chunk._inline_pause_text = None\n                    from pocket_tts.preprocessing.pause_injector import (\n                        ensure_terminal_punctuation,\n                    )\n                    audio = tts_model.generate_audio(\n                        voice_state,\n                        ensure_terminal_punctuation(chunk.text),\n',
         None),
    ]
    edits['pocket_tts/preprocessing/pause_injector.py'] = [
        ('def _text_anchor_ratio(raw_text: str, marker_start: int) -> float:',
         '_TERMINAL_PUNCTUATION = ".!?\\u2026"\n\n\ndef ensure_terminal_punctuation(text: str) -> str:\n    """Guarantee a stop cue so the model does not loop on punctuation-less text.\n\n    Short prompts without terminal punctuation make the autoregressive model\n    repeat the utterance (e.g. "Chapter One Chapter One" ghosting).  Appends a\n    period only when the spoken text has no final punctuation.\n    """\n    stripped = text.strip()\n    if not stripped:\n        return text\n    if stripped[-1] in _TERMINAL_PUNCTUATION:\n        return text\n    return stripped + "."\n\n\ndef _text_anchor_ratio(raw_text: str, marker_start: int) -> float:',
         None),
        ('    spoken_text = render_text_with_native_pauses(raw_text).strip()\n    if not spoken_text:\n        return torch.zeros(0, dtype=torch.float32), pauses\n    logger.debug(f"Generating one TTS prompt with native pauses: {repr(spoken_text[:50])}")\n',
         '    spoken_text = render_text_with_native_pauses(raw_text).strip()\n    if not spoken_text:\n        return torch.zeros(0, dtype=torch.float32), pauses\n    spoken_text = ensure_terminal_punctuation(spoken_text)\n    logger.debug(f"Generating one TTS prompt with native pauses: {repr(spoken_text[:50])}")\n',
         '    spoken_text = ensure_terminal_punctuation(spoken_text)'),
    ]
    edits['pocket_tts/models/flow_lm.py'] = [
        ('        std = temp_value.clamp_min(0).sqrt()\n        noise = torch.empty(noise_shape, dtype=dtype, device=device)\n        # N(0,1) on device then scale by std (scalar or batched). Avoids\n        # torch.nn.init.normal_(..., std=float(std.item())) host sync.\n        noise.normal_()\n',
         '        std = temp_value.clamp_min(0).sqrt()\n        noise = torch.empty(noise_shape, dtype=dtype, device=device)\n        # N(0,1) on device then scale by std (scalar or batched). Avoids\n        # torch.nn.init.normal_(..., std=float(std.item())) host sync.\n        rng = getattr(self, "_rng", None)\n        if rng is None:\n            # Per-model-instance CUDA generator: worker threads must not share\n            # the process-global RNG, or concurrent generations can draw\n            # identical noise (duplicate utterances).\n            import os as _os\n\n            rng = torch.Generator(device=device)\n            rng.manual_seed(int.from_bytes(_os.urandom(8), "little") & 0x7FFFFFFF)\n            self._rng = rng\n        noise.normal_(generator=rng)\n',
         None),
    ]
    edits['pocket_tts/models/cuda_graph_flow_lm.py'] = [
        ('            self.graph = torch.cuda.CUDAGraph()\n            with torch.inference_mode(), torch.cuda.graph(self.graph):\n',
         '            self.graph = torch.cuda.CUDAGraph()\n            rng = getattr(flow, "_rng", None)\n            if rng is not None:\n                self.graph.register_generator_state(rng)\n            with torch.inference_mode(), torch.cuda.graph(self.graph):\n',
         None),
    ]
    edits['pocket_tts/models/tts_model.py'] = [
        ('                    batch_graph = None\n            lat, eos = self._run_flow_lm_and_increment_step(\n',
         '                    batch_graph = None\n                    self.flow_lm._rng = None  # release graph-registered generator\n            lat, eos = self._run_flow_lm_and_increment_step(\n',
         None),
        ('        if not audio_frames:\n            return [torch.zeros(0, device=device) for _ in requests]\n\n',
         '        if not audio_frames:\n            return [torch.zeros(0, device=device) for _ in requests]\n\n        self.last_batch_eos_steps = list(eos_steps)\n        self.last_batch_gen_len = max_gen_len\n\n',
         None),
        ('                logger.warning("CUDA graph AR step failed; falling back to eager: %s", exc)\n',
         '                logger.warning("CUDA graph AR step failed; falling back to eager: %s", exc)\n                if hasattr(self.flow_lm, "_rng"):\n                    self.flow_lm._rng = None  # release graph-registered generator\n',
         None),
    ]

    applied, skipped = [], []
    for rel, pairs in edits.items():
        p = root / rel
        s = p.read_text(encoding="utf-8")
        modified = False
        for before, after, loose in pairs:
            if after in s or (loose and loose in s):
                continue
            assert before in s, f"missing anchor in {rel}: {before[:60]!r}"
            assert s.count(before) == 1, f"ambiguous anchor in {rel}: {before[:60]!r}"
            s = s.replace(before, after, 1)
            modified = True
        if modified:
            p.write_text(s, encoding="utf-8", newline="\n")
            applied.append(rel)
        else:
            skipped.append(rel)
    for rel in applied:
        print(f"PATCHED  {rel}")
    for rel in skipped:
        print(f"SKIPPED  {rel} (already patched)")
    print("ALL_OK" if applied else "NOTHING_TO_DO")

if __name__ == "__main__":
    main()
