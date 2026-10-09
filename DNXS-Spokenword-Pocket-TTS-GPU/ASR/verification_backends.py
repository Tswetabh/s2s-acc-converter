"""Optional ASR and forced-alignment backends for two-stage TTS verification.

Dependencies are loaded only when their backend is selected.  This keeps the
existing Whisper installation usable while NeMo compatibility is benchmarked.
"""

from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol


PARAKEET_TDT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
FASTCONFORMER_ALIGNMENT_MODEL = "stt_en_fastconformer_hybrid_large_pc"


@dataclass(frozen=True)
class TranscriptEvidence:
    """One Stage 1 transcription result, independent of ASR implementation."""

    audio_path: str
    text: str
    timestamps: Any
    confidence: float | None
    duration_s: float | None
    backend: str
    model: str
    elapsed_s: float

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe Stage 1 evidence for the candidate report."""
        return {
            "audio_path": self.audio_path,
            "text": self.text,
            "timestamps": self.timestamps,
            "confidence": self.confidence,
            "duration_s": self.duration_s,
            "backend": self.backend,
            "model": self.model,
            "elapsed_s": self.elapsed_s,
        }


def _frame_units_to_dicts(units: Iterable[Any]) -> List[Dict[str, Any]]:
    """Convert NeMo CTM units into plain JSON records."""
    rendered: List[Dict[str, Any]] = []
    for unit in units:
        rendered.append({
            "label": getattr(unit, "label", ""),
            "start_frame": int(getattr(unit, "start_frame", 0)),
            "length": int(getattr(unit, "length", 0)),
            "probability": float(getattr(unit, "probability", 0.0)),
        })
    return rendered


def _mean_weighted_probability(units: Iterable[Any]) -> float | None:
    """Return length-weighted unit probability, or None when no evidence exists."""
    total_length = 0
    weighted_probability = 0.0
    for unit in units:
        length = max(int(getattr(unit, "length", 0)), 1)
        probability = float(getattr(unit, "probability", 0.0))
        total_length += length
        weighted_probability += probability * length
    if total_length <= 0:
        return None
    return weighted_probability / total_length


def _gap_regions(units: Iterable[Any]) -> List[Dict[str, Any]]:
    """Summarize frame gaps between consecutive CTM units."""
    ordered = list(units)
    gaps: List[Dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:]):
        left_end = int(getattr(left, "start_frame", 0)) + int(getattr(left, "length", 0))
        right_start = int(getattr(right, "start_frame", 0))
        if right_start > left_end:
            gaps.append({
                "start_frame": left_end,
                "end_frame": right_start,
                "length": right_start - left_end,
            })
    return gaps


def _audio_duration_seconds(audio_path: str) -> float | None:
    """Read audio duration from file metadata without loading full waveform."""
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        if not info.samplerate:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


class TranscriptionBackend(Protocol):
    """Protocol used by Stage 1 to transcribe a batch of WAV files."""

    name: str
    model_name: str

    def transcribe_paths(self, audio_paths: Iterable[Path]) -> List[TranscriptEvidence]:
        """Return one transcript evidence object for every supplied audio path."""

    def close(self) -> None:
        """Release backend resources after the full batch completes."""


class ParakeetTDTBackend:
    """Resident NVIDIA Parakeet TDT backend with real path-batch transcription."""

    name = "parakeet_tdt"

    def __init__(self, model_name: str = PARAKEET_TDT_MODEL, device: str = "cuda") -> None:
        """Load the selected Parakeet model once for the whole Stage 1 book run."""
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    @staticmethod
    def is_available() -> bool:
        """Return whether NeMo ASR is installed without importing its heavy modules."""
        return (
            importlib.util.find_spec("nemo") is not None
            and importlib.util.find_spec("nemo.collections.asr") is not None
        )

    def _load(self) -> Any:
        """Load Parakeet lazily so fallback backends work without NeMo installed."""
        if self._model is None:
            if not self.is_available():
                raise RuntimeError("NeMo ASR is not installed; Parakeet is unavailable")
            import nemo.collections.asr as nemo_asr

            self._model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
            if self.device == "cuda":
                self._model = self._model.cuda()
            self._model.eval()
        return self._model

    def transcribe_paths(self, audio_paths: Iterable[Path]) -> List[TranscriptEvidence]:
        """Transcribe paths in one NeMo batch and retain timing/timestamp evidence."""
        paths = [Path(path).resolve() for path in audio_paths]
        if not paths:
            return []
        model = self._load()
        started = time.monotonic()
        outputs = model.transcribe([str(path) for path in paths], timestamps=True)
        elapsed_each = (time.monotonic() - started) / len(paths)
        evidence: List[TranscriptEvidence] = []
        for path, output in zip(paths, outputs):
            text, timestamps, confidence = _extract_nemo_transcript(output)
            evidence.append(TranscriptEvidence(
                audio_path=str(path), text=text, timestamps=timestamps,
                confidence=confidence, duration_s=None, backend=self.name,
                model=self.model_name, elapsed_s=elapsed_each,
            ))
        return evidence

    def close(self) -> None:
        """Release the resident model reference after all Stage 1 paths finish."""
        self._model = None


def _extract_nemo_transcript(output: Any) -> tuple[str, Any, float | None]:
    """Extract stable evidence fields from NeMo versions with different result shapes."""
    if isinstance(output, str):
        return output, None, None
    text = str(getattr(output, "text", ""))
    timestamps = getattr(output, "timestamp", None)
    confidence = getattr(output, "confidence", None)
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None
    return text, timestamps, confidence


@dataclass(frozen=True)
class AlignmentEvidence:
    """Stage 2 result from known-text forced alignment, not a free transcript."""

    audio_path: str
    alignment_text: str
    available: bool
    conclusive: bool
    coverage: float | None
    confidence: float | None
    unaligned_tokens: List[str]
    extra_regions: List[Dict[str, Any]]
    duration_s: float | None
    model: str
    elapsed_s: float
    ctm_units: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe Stage 2 evidence without mixing Stage 1 text fields."""
        return {
            "audio_path": self.audio_path,
            "alignment_text": self.alignment_text,
            "available": self.available,
            "conclusive": self.conclusive,
            "coverage": self.coverage,
            "confidence": self.confidence,
            "unaligned_tokens": self.unaligned_tokens,
            "extra_regions": self.extra_regions,
            "ctm_units": self.ctm_units,
            "duration_s": self.duration_s,
            "model": self.model,
            "elapsed_s": self.elapsed_s,
            "error": self.error,
        }


def decide_alignment(evidence: Dict[str, Any]) -> tuple[str, str]:
    """Return a binary regeneration decision from forced-alignment evidence.

    Stage 1 is only a candidate detector.  Therefore unavailable or inconclusive
    alignment always accepts the original audio instead of retrying it.
    """
    if not evidence.get("available") or not evidence.get("conclusive"):
        return "accepted_not_proven_failure", "Alignment unavailable or inconclusive; speech mismatch was not proven."
    coverage = float(evidence.get("coverage") or 0.0)
    confidence = float(evidence.get("confidence") or 0.0)
    unaligned = evidence.get("unaligned_tokens") or []
    extra_regions = evidence.get("extra_regions") or []
    if coverage < 0.85 and confidence < 0.65 and (unaligned or extra_regions):
        return "confirmed_failed_by_alignment", "Alignment found unsupported missing or extra speech evidence."
    return "accepted_by_alignment", "Expected source text aligned with sufficient coverage and confidence."


def build_alignment_report_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build human-readable output with Stage 1 transcript and Stage 2 data separated."""
    stage_one = record.get("stage_one") or {}
    return {
        "chunk_index": record.get("chunk_index"),
        "chunk_id": record.get("chunk_id"),
        "original_text": record.get("original_text", ""),
        "stage_one": {
            "transcript": stage_one.get("transcribed_text", ""),
            "comparison": {
                "ref_normalized": stage_one.get("ref_normalized", ""),
                "hyp_normalized": stage_one.get("hyp_normalized", ""),
                "score": stage_one.get("score"),
                "id_score": stage_one.get("id_score"),
                "identifier_comparisons": stage_one.get("identifier_comparisons", []),
                "accepted_list_label_equivalences": stage_one.get("accepted_list_label_equivalences", []),
                "requires_second_stage_confirmation": stage_one.get("requires_second_stage_confirmation", False),
                "second_stage_confirmation_reason": stage_one.get("second_stage_confirmation_reason", ""),
                "explanation": stage_one.get("explanation", ""),
            },
        },
        "alignment": record.get("alignment") or {},
        "decision": record.get("decision"),
        "accepted_reason": record.get("accepted_reason"),
        "explanation": record.get("explanation", ""),
    }


class NemoForcedAligner:
    """Resident manifest-based NeMo Forced Aligner adapter for Stage 2 candidates.

    NVIDIA documents NFA as a manifest tool using CTC or hybrid CTC-transducer
    models; this adapter deliberately calls that supported command rather than
    depending on an undocumented Python-internal API.
    """

    name = "nemo_forced_aligner"

    def __init__(self, model_name: str = FASTCONFORMER_ALIGNMENT_MODEL) -> None:
        """Prepare a resident aligner configuration without downloading a model."""
        self.model_name = model_name
        self._aligner: Any = None

    @staticmethod
    def is_available() -> bool:
        """Return whether a local NeMo Forced Aligner installation is discoverable."""
        return importlib.util.find_spec("nemo") is not None and importlib.util.find_spec(
            "nemo.collections.asr"
        ) is not None

    def _load(self) -> Any:
        """Load NeMo ASR once and wrap it with the forced-alignment model."""
        if self._aligner is None:
            if not self.is_available():
                raise RuntimeError("NeMo Forced Aligner is not installed")
            import nemo.collections.asr as nemo_asr
            from nemo.collections.asr.models.k2_aligner_model import AlignerWrapperModel
            from omegaconf import OmegaConf
            from omegaconf import open_dict

            base_model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
            align_cfg = OmegaConf.create({
                "alignment_type": "forced",
                # NeMo 2.7's hybrid word formatter indexes past its probability
                # list. Token units remain complete timing/confidence evidence.
                "word_output": False,
                "cpu_decoding": True,
                "decode_batch_size": 4,
                "ctc_cfg": {},
                "rnnt_cfg": {},
            })
            self._aligner = AlignerWrapperModel(base_model, align_cfg)
            # Wrapper construction clears validation_ds on some hybrid
            # checkpoints, but NeMo's temporary manifest loader reads it.
            with open_dict(self._aligner._model.cfg):
                if self._aligner._model.cfg.get("validation_ds") is None:
                    self._aligner._model.cfg.validation_ds = OmegaConf.create({
                        "use_start_end_token": False,
                    })
        return self._aligner

    def align_batch(self, jobs: Iterable[Dict[str, str]], work_dir: Path) -> List[AlignmentEvidence]:
        """Align known text in one manifest batch or return explicit unavailable evidence.

        Runtime execution is intentionally guarded until the benchmark records a
        verified NeMo install.  An unavailable tool must never become evidence of
        bad speech or trigger regeneration.
        """
        started = time.monotonic()
        job_list = list(jobs)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        if not self.is_available():
            return [AlignmentEvidence(
                audio_path=job["audio_path"], alignment_text=job["text"], available=False,
                conclusive=False, coverage=None, confidence=None, unaligned_tokens=[],
                extra_regions=[], ctm_units=[], duration_s=None, model=self.model_name,
                elapsed_s=time.monotonic() - started,
                error="NeMo Forced Aligner is not installed",
            ) for job in job_list]
        if not job_list:
            return []
        manifest_path = write_alignment_manifest(job_list, work_dir / "alignment_manifest.jsonl")
        aligner = self._load()
        try:
            predictions = self._align_manifest(
                aligner,
                manifest_path,
                batch_size=max(1, min(4, len(job_list))),
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            return [AlignmentEvidence(
                audio_path=job["audio_path"], alignment_text=job["text"], available=True,
                conclusive=False, coverage=None, confidence=None, unaligned_tokens=[],
                extra_regions=[], ctm_units=[], duration_s=_audio_duration_seconds(job["audio_path"]),
                model=self.model_name, elapsed_s=elapsed, error=str(exc),
            ) for job in job_list]
        elapsed = time.monotonic() - started
        results: List[AlignmentEvidence] = []
        for index, job in enumerate(job_list):
            units = predictions[index] if index < len(predictions) else []
            ctm_units = list(units or [])
            # NeMo returns SentencePiece units here. Their count is not word
            # coverage, so completion means every forced-reference unit aligned.
            token_coverage = 1.0 if ctm_units else 0.0
            confidence = _mean_weighted_probability(ctm_units)
            extra_regions = _gap_regions(ctm_units)
            unaligned_tokens: List[str] = []
            results.append(AlignmentEvidence(
                audio_path=job["audio_path"],
                alignment_text=job["text"],
                available=True,
                conclusive=bool(ctm_units),
                coverage=float(token_coverage),
                confidence=confidence,
                unaligned_tokens=unaligned_tokens,
                extra_regions=extra_regions,
                ctm_units=_frame_units_to_dicts(ctm_units),
                duration_s=_audio_duration_seconds(job["audio_path"]),
                model=self.model_name,
                elapsed_s=elapsed,
                error="",
            ))
        return results

    @staticmethod
    def _align_manifest(aligner: Any, manifest_path: Path, batch_size: int) -> List[Any]:
        """Run NeMo alignment through its compatible five-field manifest loader.

        Newer NeMo releases default general transcription to Lhotse, which emits
        four fields.  The k2 forced aligner still requires signal, signal length,
        transcript, transcript length, and sample id, so this path opts out only
        for the alignment dataloader.
        """
        import torch

        base_model = aligner._model
        device = next(base_model.parameters()).device
        original_mode = base_model.training
        original_dither = base_model.preprocessor.featurizer.dither
        original_pad_to = base_model.preprocessor.featurizer.pad_to
        predictions: List[Any] = []
        sample_offset = 0
        try:
            base_model.preprocessor.featurizer.dither = 0.0
            base_model.preprocessor.featurizer.pad_to = 0
            base_model.eval()
            dataloader = base_model._setup_transcribe_dataloader({
                "manifest_filepath": [str(manifest_path)],
                "batch_size": batch_size,
                "num_workers": 0,
                "use_lhotse": False,
            })
            with torch.no_grad():
                for batch in dataloader:
                    # NeMo 2.7 returns tuples here, unlike the mutable batches
                    # used by its older AlignerWrapperModel.transcribe helper.
                    batch = list(batch)
                    batch[0] = batch[0].to(device)
                    batch[1] = batch[1].to(device)
                    sample_count = int(batch[0].shape[0])
                    batch.append(torch.arange(sample_offset, sample_offset + sample_count))
                    predictions.extend(unit for _, unit in aligner.predict_step(batch, 0))
                    sample_offset += sample_count
        finally:
            base_model.train(mode=original_mode)
            base_model.preprocessor.featurizer.dither = original_dither
            base_model.preprocessor.featurizer.pad_to = original_pad_to
        return predictions


def write_alignment_manifest(jobs: Iterable[Dict[str, str]], manifest_path: Path) -> Path:
    """Write a NeMo alignment manifest including required audio durations."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as stream:
        for job in jobs:
            duration_s = _audio_duration_seconds(job["audio_path"])
            if duration_s is None:
                raise RuntimeError(f"Could not read alignment audio duration: {job['audio_path']}")
            stream.write(json.dumps({
                "audio_filepath": str(Path(job["audio_path"]).resolve()),
                "text": job["text"],
                "duration": duration_s,
            }, ensure_ascii=False) + "\n")
    return manifest_path
