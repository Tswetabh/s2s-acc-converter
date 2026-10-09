"""Shared ASR failure-report selection and manifest helpers.

The ASR pipelines retain their existing artifact names, but Regenerate needs
one stable way to identify the report that contains actionable failures.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


FAILURE_MANIFEST_FILENAME = "asr_failure_manifest.json"


@dataclass(frozen=True)
class FailureReportSelection:
    """Describe one report selected for manual chunk regeneration."""

    path: Path
    label: str
    kind: str
    source: str


_KNOWN_REPORTS = (
    (
        "asr_new_medium_failures.json",
        "New Medium-confirmed failures",
        "medium_confirmed_failures",
    ),
    (
        "asr_medium_validation_failed.json",
        "Legacy Medium-confirmed failures",
        "medium_confirmed_failures",
    ),
    (
        "asr_new_failures.json",
        "New Stage 1 candidates (not Medium-confirmed)",
        "stage_one_candidates",
    ),
    (
        "asr_failures.json",
        "Legacy Stage 1 candidates (not Medium-confirmed)",
        "stage_one_candidates",
    ),
)


def write_failure_manifest(
    tts_dir: Path,
    *,
    pipeline: str,
    authoritative_filename: str,
    label: str,
    kind: str,
    fallback_filenames: Iterable[str] = (),
    threshold: Optional[float] = None,
    language: Optional[str] = None,
) -> Path:
    """Write the current run's authoritative manual-regeneration report.

    Args:
        tts_dir: TTS artifact directory receiving the manifest.
        pipeline: ``legacy`` or ``new`` ASR pipeline that produced the report.
        authoritative_filename: Basename of the chosen failure-only report.
        label: User-facing provenance text for the Regenerate tab.
        kind: Whether rows are confirmed Medium failures or Stage 1 candidates.
        fallback_filenames: Earlier-stage reports retained for diagnostics.
        threshold: Optional ASR comparison threshold saved for inspection.
        language: Optional language identifier saved for inspection.

    Returns:
        Path to the completed manifest file.
    """
    tts_dir = Path(tts_dir)
    authoritative_name = Path(authoritative_filename).name
    fallback_names = [Path(name).name for name in fallback_filenames]
    manifest = {
        "report_type": "asr_regeneration_failure_manifest_v1",
        "run_id": str(uuid.uuid4()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completion": "complete",
        "pipeline": str(pipeline),
        "threshold": threshold,
        "language": language,
        "authoritative_report": {
            "path": authoritative_name,
            "label": str(label),
            "kind": str(kind),
        },
        "fallback_reports": fallback_names,
    }
    manifest_path = tts_dir / FAILURE_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def select_failure_report(tts_dir: Path) -> Optional[FailureReportSelection]:
    """Select manifest-owned failures first, then support historical reports.

    A valid manifest prevents an old report left in a folder from overriding a
    newer run. A present but unusable manifest fails closed, because falling
    back to stale filenames would select a report from a different run.
    Folders produced before manifests use documented filename priority.
    """
    tts_dir = Path(tts_dir)
    manifest_path = tts_dir / FAILURE_MANIFEST_FILENAME
    if manifest_path.exists():
        return _selection_from_manifest(tts_dir)

    for filename, label, kind in _KNOWN_REPORTS:
        report_path = tts_dir / filename
        if report_path.is_file():
            return FailureReportSelection(report_path, label, kind, "filename discovery")
    return None


def parse_failure_report(failure_path: Path) -> Dict[str, Dict[str, Any]]:
    """Normalize legacy and New failure rows for Regenerate's detail panel.

    Failure-only Medium reports use different nested layouts in the legacy and
    New pipelines. Stage 1 list reports remain useful when no Medium report
    was produced, but are explicitly labelled as unconfirmed candidates.
    """
    payload = json.loads(Path(failure_path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    else:
        raise ValueError(
            "Failure report must be a JSON list or an object containing a records list."
        )

    failures: Dict[str, Dict[str, Any]] = {}
    for row_number, row in enumerate(records, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Failure row {row_number} is not a JSON object.")
        try:
            chunk_index = int(row["chunk_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Failure row {row_number} has no valid chunk_index."
            ) from exc

        chunk_id = f"chunk_{chunk_index:05d}"
        reference = _as_mapping(row.get("reference"))
        stage_one = _as_mapping(row.get("stage_one"))
        medium = _as_mapping(row.get("medium"))
        comparison = _as_mapping(row.get("comparison"))
        original_failure = _as_mapping(row.get("original_failure"))
        medium_result = _as_mapping(row.get("medium_result"))
        final_result = medium or medium_result or stage_one or original_failure
        score = final_result.get("score", row.get("score"))
        classification = str(
            final_result.get("classification", row.get("classification", "FAIL")) or "FAIL"
        )
        score_text = f" | Score: {float(score):.3f}" if score is not None else ""
        failures[chunk_id] = {
            "chunk_id": chunk_id,
            "status": f"{classification}{score_text}",
            "original_text": str(
                reference.get("text_raw")
                or comparison.get("ref_text_raw")
                or row.get("original_text", "")
            ),
            "transcribed_text": str(
                final_result.get("transcript_raw")
                or comparison.get("hyp_text_raw")
                or row.get("transcribed_text", "")
            ),
            "hallucination": str(row.get("hallucination_warning", "")),
            "truncation": str(row.get("truncation_warning", "")),
            "explanation": str(final_result.get("explanation", row.get("explanation", ""))),
            "error": str(final_result.get("error", row.get("error", ""))),
        }
    return failures


def _selection_from_manifest(tts_dir: Path) -> Optional[FailureReportSelection]:
    """Return a usable manifest report while rejecting malformed path data."""
    manifest_path = tts_dir / FAILURE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("completion") != "complete":
        return None
    report = _as_mapping(manifest.get("authoritative_report"))
    report_name = Path(str(report.get("path") or ""))
    # The manifest is book-local metadata; never let it point outside its TTS folder.
    if not report_name.name or report_name.parent != Path("."):
        return None
    report_path = tts_dir / report_name.name
    if not report_path.is_file():
        return None
    return FailureReportSelection(
        report_path,
        str(report.get("label") or report_name.name),
        str(report.get("kind") or "unknown"),
        "run manifest",
    )


def _as_mapping(value: Any) -> Dict[str, Any]:
    """Return a dict value or an empty mapping for optional report sections."""
    return value if isinstance(value, dict) else {}
