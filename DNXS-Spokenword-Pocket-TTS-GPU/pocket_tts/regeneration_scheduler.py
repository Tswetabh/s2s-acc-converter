"""Pick failed-chunk recovery strategy from benchmark evidence.

The scheduler keeps two jobs separate:

* Decide whether failed chunks need a Medium verification pass before regen.
* Pick the lowest-cost safe recovery plan from measured benchmark evidence.

This module is intentionally data-driven. It reads JSON evidence files already
produced by the repo's benchmark and probe tools, then emits one normalized
plan that the GUI and generator can surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable
import json
import re


STAGE_TWO_DISABLED = "disabled"

_MODEL_RANKS = {
    "tiny": 0,
    "base": 1,
    "small": 2,
    "distil-small.en": 2,
    "medium": 3,
    "distil-medium.en": 3,
    "large": 4,
    "large-v2": 4,
    "large-v3": 4,
    "large-v3-turbo": 5,
    "distil-large-v3": 5,
}

_NEXT_HIGHER = {
    "tiny": "base",
    "base": "small",
    "small": "medium",
    "medium": "large-v3",
    "large": "large-v3-turbo",
    "large-v2": "large-v3-turbo",
    "large-v3": "large-v3-turbo",
    "distil-small.en": "distil-medium.en",
    "distil-medium.en": "medium",
}


@dataclass
class StrategyCandidate:
    """Benchmark row for one recovery strategy candidate."""

    name: str
    wall_s: float
    source: str
    workload: str = "regeneration"
    tts_workers: int | None = None
    tts_batch_size: int | None = None
    asr_workers: int | None = None
    asr_batch_size: int | None = None
    keep_tts_resident: bool = False
    concurrent: bool = False
    accumulate_failures: bool = False
    peak_vram_mb: float | None = None
    rejected_reason: str | None = None


@dataclass
class RegenerationPlan:
    """Normalized recovery plan used by GUI, generator, and reporting tools."""

    gui_asr_model: str
    verification_required: bool
    verification_model: str
    regen_asr_model: str
    tts_workers: int
    tts_batch_size: int
    asr_workers: int
    asr_batch_size: int
    keep_tts_resident: bool
    concurrent: bool
    accumulate_failures: bool
    peak_vram_mb: float | None
    chosen_strategy: str
    benchmark_source: str
    benchmark_rows: list[StrategyCandidate] = field(default_factory=list)
    rejected_strategies: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert plan to a JSON-safe dictionary for logs and GUI display."""
        payload = asdict(self)
        payload["benchmark_rows"] = [asdict(row) for row in self.benchmark_rows]
        return payload


def normalize_model_name(model_name: str) -> str:
    """Return a normalized ASR model token for comparisons and reporting."""
    return (model_name or "").strip().lower().replace("_", "-")


def model_rank(model_name: str) -> int:
    """Rank ASR model size on the Stage 1/2 ladder."""
    normalized = normalize_model_name(model_name)
    if is_stage_two_disabled(normalized):
        return -1
    return _MODEL_RANKS.get(normalized, 3 if "medium" in normalized or "large" in normalized else 1)


def is_stage_two_disabled(model_name: str | None) -> bool:
    """Return True when Stage 2 is the Disabled sentinel or an empty value."""
    token = normalize_model_name(model_name or "")
    return token in {"", "disabled", "none", "off"}


def next_higher(model_name: str) -> str | None:
    """Return the next-larger model on the locked ladder, or None at the top.

    Args:
        model_name: Stage 1 or Stage 2 model token.

    Returns:
        The next model name, or None when no larger verifier exists.
    """
    key = normalize_model_name(model_name)
    if key in {"large-v3-turbo", "distil-large-v3"}:
        return None
    return _NEXT_HIGHER.get(key)


def model_is_higher(candidate: str, baseline: str) -> bool:
    """Return True when ``candidate`` is strictly larger than ``baseline``."""
    if is_stage_two_disabled(candidate) or is_stage_two_disabled(baseline):
        return False
    return model_rank(candidate) > model_rank(baseline)


def recommended_stage_two_model(stage_one_model: str, *, engine: str = "") -> str:
    """Return the default Stage 2 model for the current Stage 1 engine/model.

    Parakeet always uses medium. Whisper tiny/base/small jump to medium.
    Stage 1 at medium or above takes one ladder step, or Disabled at the top.

    Args:
        stage_one_model: Selected Stage 1 Whisper model (ignored for Parakeet).
        engine: Stage 1 engine name.

    Returns:
        A Whisper model name, or ``disabled`` when no larger Stage 2 exists.
    """
    if str(engine or "").strip().lower() == "parakeet":
        return "medium"
    key = normalize_model_name(stage_one_model)
    if key in {"large-v3-turbo", "distil-large-v3"}:
        return STAGE_TWO_DISABLED
    if key == "distil-small.en":
        return "distil-medium.en"
    if model_rank(key) < _MODEL_RANKS["medium"]:
        return "medium"
    nxt = next_higher(key)
    if nxt is None:
        return STAGE_TWO_DISABLED
    # Stage 2 must be strictly larger than Stage 1 (distil-medium.en → large-v3).
    if not model_is_higher(nxt, key):
        return "large-v3"
    return nxt


def should_verify_failed_chunks(
    gui_model: str, stage_two_model: str | None = None
) -> bool:
    """Return True when an independent Stage 2 pass should run.

    With no Stage 2 argument, keep the historical floor: verify when Stage 1
    is smaller than medium. With a Stage 2 argument, verify only when that
    model is present and strictly larger than Stage 1.
    """
    if stage_two_model is None:
        return model_rank(gui_model) < _MODEL_RANKS["medium"]
    if is_stage_two_disabled(stage_two_model):
        return False
    return model_is_higher(stage_two_model, gui_model)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a possibly-missing benchmark field to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Convert a possibly-missing benchmark field to int."""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _candidate_from_probe(source: str, payload: dict[str, Any]) -> StrategyCandidate | None:
    """Translate a recovery probe summary into one candidate row."""
    if "total_s" not in payload:
        return None
    rows = payload.get("rows") or []
    regen_rows = payload.get("regenerated_rows")
    if regen_rows is None:
        regen_rows = sum(1 for row in rows if row.get("attempts"))
    keep_tts_resident = bool(payload.get("keep_tts_resident", True))
    concurrent = bool(payload.get("concurrent", False))
    return StrategyCandidate(
        name="resident_medium_recovery",
        wall_s=_safe_float(payload.get("total_s")),
        source=source,
        # The current probe generates and validates one candidate at a time.
        # Its summary predates explicit worker fields, so these are known values,
        # not inferred from the GUI's current original-generation configuration.
        tts_workers=_safe_int(payload.get("tts_workers"), 1),
        tts_batch_size=_safe_int(payload.get("tts_batch_size"), 1),
        asr_workers=_safe_int(payload.get("asr_workers"), 1),
        asr_batch_size=_safe_int(payload.get("asr_batch_size"), 1),
        keep_tts_resident=keep_tts_resident,
        concurrent=concurrent,
        accumulate_failures=bool(payload.get("accumulate_failures", False)),
        peak_vram_mb=_safe_float(payload.get("peak_vram_mb"), default=0.0) or None,
        rejected_reason=None if regen_rows is None or regen_rows >= 0 else "invalid_probe",
    )


def _candidate_from_scheduler_report(source: str, payload: dict[str, Any]) -> StrategyCandidate | None:
    """Translate an already-generated scheduler report into a candidate row."""
    row = payload.get("chosen")
    if not isinstance(row, dict):
        return None
    return StrategyCandidate(
        name=str(row.get("name", "scheduled_recovery")),
        wall_s=_safe_float(row.get("wall_s")),
        source=source,
        tts_workers=_safe_int(row.get("tts_workers")),
        tts_batch_size=_safe_int(row.get("tts_batch_size")),
        asr_workers=_safe_int(row.get("asr_workers")),
        asr_batch_size=_safe_int(row.get("asr_batch_size")),
        keep_tts_resident=bool(row.get("keep_tts_resident", False)),
        concurrent=bool(row.get("concurrent", False)),
        accumulate_failures=bool(row.get("accumulate_failures", False)),
        peak_vram_mb=_safe_float(row.get("peak_vram_mb"), default=0.0) or None,
        rejected_reason=row.get("rejected_reason"),
    )


def load_benchmark_matrix(results_root: Path | None = None) -> list[StrategyCandidate]:
    """Load strategy rows from benchmark/probe JSON under ``tests/results``.

    The loader accepts only recovery probes. Full-book ASR-overlap timings are
    deliberately excluded: their wall time includes original generation and a
    different ASR workload, so comparing them to failed-chunk recovery would
    produce a false recommendation.
    """
    root = (results_root or Path(__file__).resolve().parents[1] / "tests" / "results").resolve()
    candidates: list[StrategyCandidate] = []
    if not root.is_dir():
        return candidates

    for summary_path in sorted(root.rglob("summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source = str(summary_path.relative_to(root))
        if "rows" in payload and "total_s" in payload:
            candidate = _candidate_from_probe(source, payload)
            if candidate is not None:
                candidates.append(candidate)
            continue
        candidate = _candidate_from_scheduler_report(source, payload)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _pick_best_candidate(
    candidates: Iterable[StrategyCandidate],
    safe_vram_mb: float | None = None,
) -> tuple[StrategyCandidate | None, list[StrategyCandidate]]:
    """Choose the lowest-wall-time safe strategy and return rejected rows too."""
    accepted: list[StrategyCandidate] = []
    rejected: list[StrategyCandidate] = []
    for candidate in candidates:
        if candidate.rejected_reason:
            rejected.append(candidate)
            continue
        if candidate.keep_tts_resident and candidate.peak_vram_mb is None:
            # Keeping TTS loaded while Medium ASR runs is safe only after a
            # measured peak exists for this exact recovery strategy.
            rejected.append(
                StrategyCandidate(
                    **{**candidate.__dict__, "rejected_reason": "missing_peak_vram"}
                )
            )
            continue
        if safe_vram_mb is not None and candidate.peak_vram_mb is not None and candidate.peak_vram_mb > safe_vram_mb:
            rejected.append(
                StrategyCandidate(
                    **{**candidate.__dict__, "rejected_reason": "vram_limit"}
                )
            )
            continue
        accepted.append(candidate)
    if not accepted:
        return None, rejected
    best = min(accepted, key=lambda row: row.wall_s)
    rejected.extend(row for row in accepted if row is not best)
    return best, rejected


def choose_regeneration_plan(
    gui_model: str,
    *,
    stage_two_model: str | None = None,
    results_root: Path | None = None,
    requested_tts_workers: int | None = None,
    requested_tts_batch_size: int | None = None,
    requested_asr_workers: int | None = None,
    requested_asr_batch_size: int | None = None,
    safe_vram_mb: float | None = None,
) -> RegenerationPlan:
    """Build a recovery plan from benchmark matrix plus current GUI settings.

    Args:
        gui_model: Stage 1 model shown in the GUI.
        stage_two_model: Independent Stage 2 model, or Disabled. When omitted,
            Stage 2 defaults to medium so older callers keep verifying small
            Stage 1 models.
    """
    matrix = load_benchmark_matrix(results_root)
    best, rejected = _pick_best_candidate(matrix, safe_vram_mb=safe_vram_mb)

    resolved_stage_two = (
        "medium" if stage_two_model is None else normalize_model_name(stage_two_model)
    )
    verification_required = should_verify_failed_chunks(gui_model, resolved_stage_two)
    verification_model = (
        normalize_model_name(resolved_stage_two) if verification_required else "none"
    )
    regen_asr_model = (
        verification_model if verification_required else "medium"
    )

    if best is None:
        chosen_strategy = "fallback_sequential"
        # This path generates one candidate then checks it with one Medium ASR
        # process. More TTS workers would start multiple Medium subprocesses and
        # violate the conservative no-contention fallback contract.
        tts_workers = 1
        tts_batch_size = 1
        asr_workers = 1
        asr_batch_size = 1
        keep_tts_resident = False
        concurrent = False
        accumulate_failures = False
        peak_vram_mb = None
        benchmark_source = "fallback"
        notes = [
            "No safe comparable recovery benchmark row is available; using conservative staged recovery.",
            "TTS unloads before initial GUI ASR, then one worker is reloaded for regeneration.",
        ]
    else:
        chosen_strategy = best.name
        tts_workers = max(1, int(best.tts_workers or requested_tts_workers or 1))
        tts_batch_size = max(1, int(best.tts_batch_size or requested_tts_batch_size or 1))
        asr_workers = max(1, int(best.asr_workers or requested_asr_workers or 1))
        asr_batch_size = max(1, int(best.asr_batch_size or requested_asr_batch_size or 1))
        keep_tts_resident = bool(best.keep_tts_resident)
        concurrent = bool(best.concurrent)
        accumulate_failures = bool(best.accumulate_failures)
        peak_vram_mb = best.peak_vram_mb
        benchmark_source = best.source
        notes = [f"Selected {best.name} from benchmark matrix: {best.wall_s:.3f}s."]
        if verification_required:
            notes.append(
                f"Stage 2 {verification_model} will verify Stage 1 {normalize_model_name(gui_model)} failures."
            )
        else:
            notes.append("Stage 2 is Disabled or not higher than Stage 1; verification is skipped.")

    rejected_names = [
        f"{row.name} ({row.rejected_reason or 'slower'})"
        for row in rejected
    ]
    return RegenerationPlan(
        gui_asr_model=normalize_model_name(gui_model),
        verification_required=verification_required,
        verification_model=verification_model,
        regen_asr_model=regen_asr_model,
        tts_workers=tts_workers,
        tts_batch_size=tts_batch_size,
        asr_workers=asr_workers,
        asr_batch_size=asr_batch_size,
        keep_tts_resident=keep_tts_resident,
        concurrent=concurrent,
        accumulate_failures=accumulate_failures,
        peak_vram_mb=peak_vram_mb,
        chosen_strategy=chosen_strategy,
        benchmark_source=benchmark_source,
        benchmark_rows=list(matrix),
        rejected_strategies=rejected_names,
        notes=notes,
    )


def format_plan_summary(plan: RegenerationPlan) -> list[str]:
    """Return GUI-ready summary lines for the selected regeneration plan."""
    peak_vram = "unknown"
    if plan.peak_vram_mb is not None:
        peak_vram = f"{plan.peak_vram_mb:.0f} MB"
    lines = [
        f"Strategy: {plan.chosen_strategy}",
        f"GUI ASR model: {plan.gui_asr_model} -> verification: {plan.verification_model if plan.verification_required else 'none'}",
        f"Regen ASR model: {plan.regen_asr_model}",
        f"TTS workers/batch: {plan.tts_workers}/{plan.tts_batch_size}",
        f"Stage-2 ASR workers/batch: {plan.asr_workers}/{plan.asr_batch_size}",
        f"TTS resident during ASR: {'yes' if plan.keep_tts_resident else 'no'}",
        f"Concurrent TTS+ASR: {'yes' if plan.concurrent else 'no'}",
        f"Peak VRAM: {peak_vram}",
    ]
    if plan.rejected_strategies:
        lines.append("Rejected: " + ", ".join(plan.rejected_strategies))
    lines.extend(plan.notes)
    return lines
