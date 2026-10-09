#!/usr/bin/env python3
"""Verify Pocket TTS FlowLM/T3 text-token round trips on real book chunks.

Pocket TTS feeds SentencePiece text IDs into ``flow_lm.conditioner`` before
the autoregressive FlowLM stage (called T3 in this investigation).  FlowLM
then emits continuous audio latents, not text-token IDs, so those generated
latents cannot be decoded back into prose.  This probe therefore verifies the
last text-decodable boundary: original T3 input text -> SentencePiece IDs ->
decoded text.

It writes one readable log for each run. Every chunk is represented as:

    Chunk: 00042
    Original: Text supplied to FlowLM/T3.
    Decoded token text: Text reconstructed from FlowLM input IDs.
    Result: PASS

Three independently constructed tokenizer passes expose input-tokenization
drift. The report separates byte-exact differences from normalized-text
differences, and also checks decoded text stability across all runs.

Example
-------
``venv/bin/python tests/tools/probe_t3_text_token_roundtrip.py \\
  --input 'input/His (A Dark Erotic Romance Novel) - Dark, Aubrey_output_part25.txt'``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_VARIANT_CONFIG = ROOT / "pocket_tts" / "config" / "b6369a24.yaml"
DEFAULT_RESULTS_ROOT = ROOT / "tests" / "results" / "t3_text_token_roundtrip"


def normalize_for_comparison(text: str) -> str:
    """Normalize Unicode and whitespace without changing lexical content."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def load_t3_tokenizer(config_path: Path):
    """Build the exact SentencePiece tokenizer used by the FlowLM conditioner."""
    from pocket_tts.conditioners.text import SentencePieceTokenizer

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    lookup_table = config.get("flow_lm", {}).get("lookup_table", {})
    n_bins = int(lookup_table["n_bins"])
    tokenizer_path = str(lookup_table["tokenizer_path"])
    return SentencePieceTokenizer(nbins=n_bins, tokenizer_path=tokenizer_path)


def make_t3_input_chunks(input_path: Path, min_words: int) -> List[str]:
    """Return the normalized text chunks that Pocket TTS supplies to FlowLM."""
    from pocket_tts.preprocessing.chunker import SmartChunker
    from pocket_tts.preprocessing.structure_detector import StructureDetector

    raw_text = input_path.read_text(encoding="utf-8")
    structure = StructureDetector().analyze(raw_text)
    chunks = SmartChunker(mode="sentence", min_words=min_words).chunk(structure)
    return [chunk.text for chunk in chunks]


@dataclass
class ChunkResult:
    """Record one original/decode comparison for report and log rendering."""

    chunk_number: int
    original_text: str
    decoded_text: str
    token_count: int
    exact_match: bool
    normalized_match: bool


def probe_one_run(chunks: Iterable[str], config_path: Path) -> List[ChunkResult]:
    """Encode/decode every T3 input chunk with a freshly constructed tokenizer."""
    tokenizer = load_t3_tokenizer(config_path)
    results: List[ChunkResult] = []
    for index, original in enumerate(chunks, start=1):
        ids = tokenizer(original).tokens[0].tolist()
        decoded = tokenizer.sp.decode(ids)
        results.append(
            ChunkResult(
                chunk_number=index,
                original_text=original,
                decoded_text=decoded,
                token_count=len(ids),
                exact_match=original == decoded,
                normalized_match=(
                    normalize_for_comparison(original)
                    == normalize_for_comparison(decoded)
                ),
            )
        )
    return results


def write_run_log(path: Path, run_number: int, rows: Iterable[ChunkResult]) -> None:
    """Write the requested chunk/original/decoded text audit log for one run."""
    rows = list(rows)
    normalized_failures = sum(not row.normalized_match for row in rows)
    exact_differences = sum(not row.exact_match for row in rows)
    with path.open("w", encoding="utf-8") as log_file:
        log_file.write("Pocket TTS T3 Text-Token Round-Trip Log\n")
        log_file.write(f"Run: {run_number}\n")
        log_file.write(f"Chunks: {len(rows)}\n")
        log_file.write(f"Exact differences: {exact_differences}\n")
        log_file.write(f"Normalized failures: {normalized_failures}\n\n")
        for row in rows:
            result = "PASS" if row.normalized_match else "FAIL"
            log_file.write(f"Chunk: {row.chunk_number:05d}\n")
            log_file.write(f"Original: {row.original_text}\n")
            log_file.write(f"Decoded token text: {row.decoded_text}\n")
            log_file.write(
                f"Result: {result} | exact_match={row.exact_match} "
                f"| token_count={row.token_count}\n\n"
            )


def collect_stability_failures(run_rows: List[List[ChunkResult]]) -> List[int]:
    """Return chunk numbers whose decoded text differs across repeated runs."""
    unstable = []
    for row_group in zip(*run_rows):
        decoded_values = {row.decoded_text for row in row_group}
        if len(decoded_values) != 1:
            unstable.append(row_group[0].chunk_number)
    return unstable


def write_report(
    path: Path,
    input_path: Path,
    config_path: Path,
    run_rows: List[List[ChunkResult]],
    unstable_chunks: List[int],
) -> None:
    """Write a concise pass/fail report covering all repeated tokenizer runs."""
    total_chunks = len(run_rows[0]) if run_rows else 0
    per_run = []
    for index, rows in enumerate(run_rows, start=1):
        per_run.append(
            {
                "run": index,
                "chunks": len(rows),
                "exact_matches": sum(row.exact_match for row in rows),
                "exact_differences": sum(not row.exact_match for row in rows),
                "normalized_passes": sum(row.normalized_match for row in rows),
                "normalized_failures": sum(not row.normalized_match for row in rows),
            }
        )
    normalized_failures = sum(
        row["normalized_failures"] for row in per_run
    )
    verdict = (
        "PASS: all decoded token text matches normalized original T3 input and "
        "is stable across runs."
        if normalized_failures == 0 and not unstable_chunks
        else "FAIL: one or more T3 text-token round-trip comparisons changed text."
    )
    with path.open("w", encoding="utf-8") as report_file:
        report_file.write("Pocket TTS T3 Text-Token Round-Trip Report\n")
        report_file.write("=" * 48 + "\n")
        report_file.write(f"Input: {input_path}\n")
        report_file.write(f"FlowLM config: {config_path}\n")
        report_file.write(f"Chunks per run: {total_chunks}\n")
        report_file.write(f"Runs: {len(run_rows)}\n\n")
        for row in per_run:
            report_file.write(
                f"Run {row['run']}: normalized pass={row['normalized_passes']}/"
                f"{row['chunks']}; normalized fail={row['normalized_failures']}; "
                f"exact differences={row['exact_differences']}\n"
            )
        report_file.write(f"\nCross-run decoded-text instability: {len(unstable_chunks)}\n")
        if unstable_chunks:
            report_file.write(
                "Unstable chunk numbers: "
                + ", ".join(f"{number:05d}" for number in unstable_chunks)
                + "\n"
            )
        report_file.write(f"\nVerdict: {verdict}\n\n")
        report_file.write(
            "Scope: this verifies SentencePiece text IDs supplied to the FlowLM/T3 "
            "conditioner. Pocket TTS FlowLM output is continuous audio latents, so "
            "this test cannot decode or validate those generated latents as text. "
            "If this passes while generated speech is wrong, investigate FlowLM "
            "sampling, EOS behavior, or audio decoding with audio/ASR evidence.\n"
        )


def write_machine_summary(
    path: Path, run_rows: List[List[ChunkResult]], unstable_chunks: List[int]
) -> None:
    """Persist structured results alongside readable logs for automated follow-up."""
    payload: Dict[str, object] = {
        "runs": [[asdict(row) for row in rows] for rows in run_rows],
        "unstable_chunks": unstable_chunks,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Run three independent text-token round trips and create logs plus report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input book text file")
    parser.add_argument(
        "--variant-config",
        type=Path,
        default=DEFAULT_VARIANT_CONFIG,
        help="Pocket TTS FlowLM model configuration YAML",
    )
    parser.add_argument("--min-words", type=int, default=5, help="Sentence chunk merge floor")
    parser.add_argument("--runs", type=int, default=3, help="Independent tokenizer passes")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for logs/report")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.variant_config.is_file():
        raise FileNotFoundError(args.variant_config)
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or DEFAULT_RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = make_t3_input_chunks(args.input, args.min_words)
    run_rows = []
    for run_number in range(1, args.runs + 1):
        rows = probe_one_run(chunks, args.variant_config)
        write_run_log(output_dir / f"t3_token_roundtrip_run_{run_number}.log", run_number, rows)
        run_rows.append(rows)
    unstable_chunks = collect_stability_failures(run_rows)
    write_report(
        output_dir / "t3_token_roundtrip_report.txt",
        args.input.resolve(),
        args.variant_config.resolve(),
        run_rows,
        unstable_chunks,
    )
    write_machine_summary(output_dir / "t3_token_roundtrip_summary.json", run_rows, unstable_chunks)
    print(f"Wrote {args.runs} logs and report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
