"""Help text for main GUI parameter sections (shown via ? buttons)."""

from __future__ import annotations

from typing import Dict

# Keys must match create_param_section(..., help_key=...) in main_window.
PARAMETER_SECTION_HELP: Dict[str, str] = {
    "tts_core": """\
What this section does
Controls how the neural TTS model generates speech for each text chunk: randomness, when it decides a phrase is finished, and a little tail after the end.

Settings
• Temperature (0.0–1.5)
  How creative / random sampling is.
  Lower (e.g. 0.5–0.7) = more stable, consistent voice; fewer weird glitches.
  Higher (e.g. 0.9–1.2) = more variety, can sound livelier but less reliable.
  Default around 0.7 is a good starting point.

• EOS Threshold (negative, e.g. −4.0)
  How strongly the model must “see” an end-of-sequence before stopping.
  More negative (e.g. −6) = tends to keep talking longer (can overshoot).
  Closer to zero (e.g. −2) = stops sooner (can cut off early).
  Tune if clips end too early or trail off oddly.

• Frames After EOS (0–10)
  Extra audio frames generated after EOS is detected, so the ending doesn’t clip mid-breath.
  Higher = slightly longer tails after sentences; usually 1–3 is enough.

Tips
Change temperature first if speech sounds robotic or unstable. Only tweak EOS if endings sound wrong.
""",
    "chunking": """\
What this section does
How the book text is split into pieces before TTS. Smaller chunks are easier for the model; too small can sound choppy and slow the run.

Settings
• Mode
  – sentence: one (or merged) sentence per chunk — usual choice for natural pacing and ASR checks.
  – paragraph: whole paragraphs — fewer chunks, longer audio per piece; can stress quality on very long paragraphs.

• Min Words
  Soft floor on chunk length. Short fragments below this can be merged with a neighbor so you don’t get tiny “Yes.” / “No.” clips alone.
  Higher min words = fewer, longer chunks. Lower = more, shorter chunks.

Tips
Sentence mode + min words around the config default is recommended for audiobooks with ASR quality control.
""",
    "pauses": """\
What this section does
Silence inserted between structural units when the full audiobook is assembled (after each chunk is generated). Values are milliseconds.

Settings
• Sentence End
  Pause after a sentence boundary when joining audio.

• Paragraph Break
  Longer pause between paragraphs.

• Chapter Start
  Pause used at chapter (or similar large) boundaries.

Tips
These do not change how the model speaks inside a chunk; they only space joined segments. Increase for a more “read aloud / dramatic” pace; decrease for denser narration.
""",
    "quality": """\
What this section does
Trade-offs between generation quality and speed, plus optional natural speed variation.

Settings
• LSD Steps (latent steps, typically 1–30)
  How many refinement steps the model uses per generation.
  Higher = often cleaner audio, slower and more GPU work.
  Lower = faster; can sound slightly rougher.
  Note on the panel: mid-range values (about 5–10) are a common quality/speed balance; the project default may be lower for speed.

• Enable speed variation
  When on, slight speaking-rate changes can be applied (emotion / style related) so long books sound less metronomic.
  When off, pacing stays more uniform.

• Enable emotion detection
  When off, Pocket TTS does not load or run the emotion model. Every chunk uses neutral TTS parameters.

Tips
If generation is too slow, lower LSD steps before cutting workers. If audio quality is the problem, raise LSD a few steps and re-test a short sample.
""",
    "pause_injection": """\
What this section does
Converts configured punctuation into pause events. Each text fragment is generated as a complete TTS segment, then exact digital silence is placed between fragments. Manual [Xs] markers always work, even when this switch is off.

Settings
• Enable punctuation pauses
  Master switch for automatic punctuation pauses only. It never disables manually written [Xs] pauses.

• Period / Exclamation / Question / Comma / Ellipsis / Em Dash / Semicolon / Colon
  Exact digital-silence duration used for that mark. Manual markers are not rounded.
  Commas are usually shorter; sentence-ending marks longer.
  Set to 0 to skip that mark.

Tips
Use normal punctuation for automatic rhythm or write [0.15s] and [4s] when exact placement is needed. Text beside a marker is never cut or aligned inside a completed waveform.
""",
    "m4b": """\
What this section does
Optional conversion of the finished WAV audiobook into M4B (chapter-friendly audiobook container), with optional loudness processing.

Settings
• Convert to M4B
  When checked, builds an M4B after generation (requires ffmpeg available to the app).

• Normalization
  – none: no volume processing.
  – peak: scale so peaks approach full scale (simple, common).
  – loudness: aim for more consistent perceived loudness (heavier processing).
  – simple: lightweight level adjust.

Tips
Leave off if you only want WAV. Turn on for players that prefer M4B. Prefer peak or loudness if chapters vary a lot in volume.
""",
    "performance": """\
What this section does
Where TTS runs and how many parallel generation workers are used. This is the main knob for throughput vs VRAM/CPU load.

Settings
• Device
  – auto: prefer GPU if CUDA is available, else CPU.
  – cuda: force GPU (needs working NVIDIA stack).
  – cpu: force CPU (slower; frees GPU for other tasks).

• Max Workers
  How many TTS worker processes/threads generate chunks in parallel.
  More workers = faster book, more RAM/VRAM.
  The spinner max is capped from your hardware estimate (GPU VRAM and/or CPU cores).
  If you hit out-of-memory or crashes, lower this.

Tips
On a mid-range GPU (e.g. 8 GB), a few workers with batching is often better than maxing the spinner. ASR after gen also needs free VRAM — leave headroom if ASR is enabled.
""",
    "asr": """\
What this section does
After TTS finishes, re-listens to each chunk with speech recognition and compares it to the original text. Bad chunks can be regenerated. ASR does not run during TTS generation (avoids fighting the GPU).

Settings
• Enable ASR Quality Control
  Master switch. Off = skip all post-gen ASR and regen from this path.

• ASR Backend
  – faster_whisper: recommended; GPU-friendly CTranslate2 path; supports real distil models. Uses a fast pipeline (one GPU model, packed clips, solo-retry if a pack fails the score).
  – whisper_cpp: ggml via pywhispercpp; GPU needs a CUDA build of the library. Distil names map to nearest full-size ggml.

• ASR Model
  Size / variant (tiny → large, plus distil-* for faster_whisper). Larger = usually more accurate, slower, more VRAM. base is a good default.

• GPU ASR workers (after gen)
  – 0: ASR off even if enabled (no post-gen workers).
  – faster_whisper: CPU budget for loaders/scorers around one GPU model (try ~8). Not “N full models.”
  – whisper_cpp: number of concurrent model copies on GPU (use 1–3 on 8 GB for larger models).

• Language
  Language code pinned for recognition (should match the book language).

• Threshold (0–1)
  Minimum similarity score for a chunk to pass. Higher = stricter (more regenerations). Lower = more permissive. ~0.75–0.85 is typical.

• Max Retries
  How many times a failing chunk may be regenerated with adjusted settings before giving up. Set to 0 to run ASR and create reports without regenerating any chunks.

• Temp Decrement
  How much to lower TTS temperature on each regen attempt (more conservative speech). Larger steps = faster move toward stable but flatter delivery.

Tips
Keep ASR on for quality; use faster_whisper + base + workers ~8 for speed/quality balance. Raise threshold only if you accept more regen time. whisper_cpp medium/large with many workers can OOM on 8 GB cards.
""",
}


def help_for(section_key: str) -> str:
    """Return help body for a section key, or a short fallback."""
    return PARAMETER_SECTION_HELP.get(
        section_key,
        "No help is defined for this section yet.",
    )
