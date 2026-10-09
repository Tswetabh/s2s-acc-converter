#!/usr/bin/env python3
"""Review tool for ASR failure/verification JSON and matching audio chunks.

Loads an ASR failure report (``asr_failures.json``), a medium verification
report (``asr_medium_verification.json``), or a plain-text final failure
investigation report (``asr_investigation.log``) from a TTS directory, matches
each entry to audio chunks in ``audio_chunks/`` and ``audio_chunks/Failed/``,
opens a Tk review window, and writes ``fail_check.json`` with rows still failed.
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


COMMON_FAILURE_FILENAMES = (
    "asr_failures.json",
    "fail_check.json",
    "failures.json",
    "asr_medium_verification.json",
    "asr_gui_failures.json",
    "asr_new_failures.json",
    "asr_new_medium_failures.json",
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for target folder and server settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Open a browser UI for reviewing ASR failures in a TTS directory."
        )
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help=(
            "TTS directory or failure/verification JSON or investigation log. "
            "Examples: 'Output/Book/TTS' or 'Output/Book/TTS/asr_failures.json' "
            "or 'Output/Book/asr_investigation.log'. "
            "If omitted, a file dialog opens."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the local web server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to bind. Use 0 for an auto-selected free port.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Start server without opening a browser tab.",
    )
    parser.add_argument(
        "--include-accepted",
        action="store_true",
        help=(
            "Include records the medium model accepted when loading a medium "
            "verification report (default: only remains_failed_after_medium)."
        ),
    )
    return parser.parse_args()


def _configure_tk_file_dialogs_hide_hidden(root: Any) -> None:
    """Enable Tk hidden-file checkbox and leave it unchecked by default.

    The ``tk_getOpenFile -badoption`` call forces Tk's native file dialog
    implementation to load first, which makes the later ``setvar`` calls work.
    """
    try:
        root.tk.eval("catch {tk_getOpenFile -badoption}")
        root.tk.setvar("::tk::dialog::file::showHiddenBtn", 1)
        root.tk.setvar("::tk::dialog::file::showHiddenVar", 0)
    except Exception:
        pass


def _pick_failure_file(initial_dir: Path | None = None) -> Path | None:
    """Open Tk file dialog and return selected failure JSON file, if any."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("tkinter is required for file picking") from exc

    root = tk.Tk()
    root.withdraw()
    _configure_tk_file_dialogs_hide_hidden(root)
    try:
        selected = filedialog.askopenfilename(
            parent=root,
            title="Select ASR failure or verification report",
            initialdir=str(initial_dir or Path.cwd()),
            filetypes=[
                (
                    "ASR reports",
                    "asr_failures.json asr_medium_verification.json",
                ),
                ("JSON files", "*.json"),
                ("All files", "*"),
            ],
        )
        if not selected:
            return None
        return Path(selected).expanduser().resolve()
    finally:
        root.destroy()


def _find_failure_file(tts_dir: Path) -> Path:
    """Return first known failure JSON file inside the TTS directory."""
    for name in COMMON_FAILURE_FILENAMES:
        candidate = tts_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No failure JSON found in {tts_dir}. Tried: {', '.join(COMMON_FAILURE_FILENAMES)}"
    )


def _path_from_target(target: str) -> Path:
    """Convert a filesystem path or ``file://`` URL into an absolute Path."""
    if target.startswith("file://"):
        rest = target[len("file://") :]
        rest = urllib.parse.unquote(rest)
        if rest.startswith("localhost/"):
            rest = rest[len("localhost") :]
        path = Path(rest)
    else:
        path = Path(target)
    return path.expanduser().resolve()


def _resolve_target(target: str) -> tuple[Path, Path]:
    """Resolve CLI target into TTS directory and failure JSON path."""
    path = _path_from_target(target)
    if path.is_dir():
        return path, _find_failure_file(path)
    if path.is_file():
        tts_dir = path.parent
        # If audio_chunks/ is not here, look for a TTS subdirectory
        if not (tts_dir / "audio_chunks").is_dir():
            for child in sorted(tts_dir.iterdir()):
                if child.is_dir() and (child / "audio_chunks").is_dir():
                    tts_dir = child
                    break
        return tts_dir, path
    raise FileNotFoundError(f"Target path does not exist: {path}")


def _safe_text(value: Any) -> str:
    """Convert arbitrary JSON value into a displayable string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    """Remove duplicate paths while preserving order."""
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in paths:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _audio_candidates(entry: dict[str, Any], tts_dir: Path) -> tuple[list[Path], list[Path]]:
    """Build (main, failed) audio candidate paths for one failure entry.

    Main candidates live in ``audio_chunks/``, failed candidates in
    ``audio_chunks/Failed/``; both may hold a version of the same chunk.
    """
    main_candidates: list[Path] = []
    failed_candidates: list[Path] = []
    filename = _safe_text(entry.get("filename")).strip()
    chunk_index = entry.get("chunk_index")

    if filename:
        source = Path(filename)
        if source.is_absolute():
            main_candidates.append(source)
        else:
            main_candidates.extend(
                [
                    tts_dir / source,
                    tts_dir / source.name,
                ]
            )
            if source.parts and source.parts[0] == "audio_chunks":
                main_candidates.append(tts_dir / source)

    if chunk_index is not None:
        try:
            idx = int(chunk_index)
        except (TypeError, ValueError):
            idx = None
        if idx is not None:
            names = [
                f"chunk_{idx:05d}.wav",
                f"chunk_{idx}.wav",
                f"{idx}.wav",
            ]
            for name in names:
                main_candidates.append(tts_dir / "audio_chunks" / name)
                failed_candidates.append(tts_dir / "audio_chunks" / "Failed" / name)

    return _dedupe_paths(main_candidates), _dedupe_paths(failed_candidates)


def _row_display_fields(entry: dict[str, Any]) -> dict[str, Any]:
    """Map a legacy failure or medium verification entry to display fields."""
    if "reference" in entry and "stage_one" in entry:
        ref = entry.get("reference") or {}
        stage = entry.get("stage_one") or {}
        medium = entry.get("medium") or {}
        return {
            "decision": _safe_text(entry.get("decision")),
            "original_text": _safe_text(ref.get("text_raw") or ref.get("normalized")),
            "transcribed_text": _safe_text(medium.get("transcript_raw") or stage.get("transcript_raw")),
            "ref_normalized": _safe_text(ref.get("normalized")),
            "hyp_normalized": _safe_text(medium.get("normalized") or stage.get("normalized")),
            "score": _safe_text(medium.get("score", stage.get("score", ""))),
            "classification": _safe_text(medium.get("classification", stage.get("classification", ""))),
            "original_score": _safe_text(stage.get("score")),
            "original_explanation": _safe_text(stage.get("explanation")),
            "medium_score": _safe_text(medium.get("score")),
            "medium_explanation": _safe_text(medium.get("explanation")),
        }
    if "comparison" in entry:
        comparison = entry.get("comparison", {})
        # New format: single 'failure' key; legacy: 'original_failure' + 'medium_result'
        failure = entry.get("failure") or entry.get("original_failure") or {}
        medium_result = entry.get("medium_result") or {}
        return {
            "decision": _safe_text(entry.get("decision")),
            "original_text": _safe_text(
                comparison.get("ref_text_raw") or comparison.get("ref_normalized")
            ),
            "transcribed_text": _safe_text(
                comparison.get("hyp_text_raw") or comparison.get("hyp_normalized")
            ),
            "ref_normalized": _safe_text(comparison.get("ref_normalized")),
            "hyp_normalized": _safe_text(comparison.get("hyp_normalized")),
            "score": _safe_text(
                medium_result.get("score", failure.get("score", ""))
            ),
            "classification": _safe_text(
                medium_result.get(
                    "classification", failure.get("classification", "")
                )
            ),
            "original_score": _safe_text(failure.get("score")),
            "original_explanation": _safe_text(failure.get("explanation")),
            "medium_score": _safe_text(medium_result.get("score")),
            "medium_explanation": _safe_text(medium_result.get("explanation")),
        }
    return {
        "decision": "",
        "original_text": _safe_text(entry.get("original_text")),
        "transcribed_text": _safe_text(entry.get("transcribed_text")),
        "ref_normalized": "",
        "hyp_normalized": "",
        "score": _safe_text(entry.get("score")),
        "classification": _safe_text(entry.get("classification")),
        "original_score": "",
        "original_explanation": "",
        "medium_score": "",
        "medium_explanation": "",
    }


def _extract_failure_entries(raw: Any, failure_path: Path) -> list[dict[str, Any]]:
    """Extract failure dictionaries from array or common wrapped JSON formats."""
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        entries = None
        for key in ("failures", "failed_chunks", "chunks", "results", "records"):
            value = raw.get(key)
            if isinstance(value, list):
                entries = value
                break
        if entries is None:
            raise ValueError(
                f"Failure JSON object has no failures array: {failure_path}"
            )
    else:
        raise ValueError(f"Failure JSON must contain an array or object: {failure_path}")

    return [entry for entry in entries if isinstance(entry, dict)]


def _parse_investigation_log(log_path: Path) -> list[dict[str, Any]]:
    """Parse an ASR final failure investigation report into failure entries.

    The report is a plain-text log with one ``CHUNK:`` block per failed chunk,
    each holding top-level ``Status:``, ``Original Text:``, ``Transcribed
    Text:`` and ``Explanation:`` lines at column zero. Indented regeneration
    attempt blocks are ignored.
    """
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    chunk_index_pattern = re.compile(r"CHUNK:\s*chunk_(\d+)")
    score_pattern = re.compile(r"Score:\s*([\d.]+)")

    for line in log_path.read_text(encoding="utf-8").splitlines():
        chunk_match = chunk_index_pattern.match(line)
        if chunk_match:
            current = {"chunk_index": int(chunk_match.group(1))}
            entries.append(current)
            continue
        if current is None:
            continue
        if line.startswith("Status:"):
            score_match = score_pattern.search(line)
            current["score"] = float(score_match.group(1)) if score_match else None
            current["classification"] = "FAILED"
        elif line.startswith("Original Text: "):
            current["original_text"] = line[len("Original Text: "):].strip()
        elif line.startswith("Transcribed Text: "):
            current["transcribed_text"] = line[len("Transcribed Text: "):].strip()
        elif line.startswith("Explanation: "):
            current["explanation"] = line[len("Explanation: "):].strip()
            current["original_explanation"] = current["explanation"]

    return entries


def _build_rows(
    failure_path: Path,
    tts_dir: Path,
    include_accepted: bool = False,
) -> list[dict[str, Any]]:
    """Load failures and enrich them with audio path and display fields."""
    if failure_path.suffix.lower() == ".log":
        entries = _parse_investigation_log(failure_path)
    else:
        raw = json.loads(failure_path.read_text(encoding="utf-8"))
        entries = _extract_failure_entries(raw, failure_path)

    if not include_accepted:
        entries = [
            entry
            for entry in entries
            if entry.get("decision") != "accepted_by_medium"
        ]

    rows: list[dict[str, Any]] = []
    for row_id, entry in enumerate(entries):
        main_candidates, failed_candidates = _audio_candidates(entry, tts_dir)
        audio_main = next((c for c in main_candidates if c.is_file()), None)
        audio_failed = next((c for c in failed_candidates if c.is_file()), None)

        def _rel(path: Path | None) -> str:
            """Return the audio path relative to the TTS dir, or empty string."""
            return str(path.relative_to(tts_dir)).replace(os.sep, "/") if path else ""

        rows.append(
            {
                "row_id": row_id,
                "chunk_index": entry.get("chunk_index"),
                "filename": entry.get("filename", ""),
                **_row_display_fields(entry),
                "audio_rel": _rel(audio_main),
                "audio_main_rel": _rel(audio_main),
                "audio_failed_rel": _rel(audio_failed),
                "audio_exists": audio_main is not None or audio_failed is not None,
                "source_entry": entry,
            }
        )

    def _sort_key(item: dict[str, Any]) -> tuple[int, int]:
        """Sort failures by chunk index while keeping stable fallback order."""
        try:
            return (int(item.get("chunk_index", 10**12)), int(item["row_id"]))
        except (TypeError, ValueError):
            return (10**12, int(item["row_id"]))

    return sorted(rows, key=_sort_key)


def _build_state(
    tts_dir: Path,
    failure_path: Path,
    include_accepted: bool = False,
) -> dict[str, Any]:
    """Build immutable server state for the browser app."""
    rows = _build_rows(failure_path, tts_dir, include_accepted=include_accepted)
    return {
        "tts_dir": str(tts_dir),
        "failure_path": str(failure_path),
        "report_path": str(tts_dir / "fail_check.json"),
        "pass_path": str(tts_dir / "pass_check.json"),
        "rows": rows,
    }


def _partition_entries(
    state: dict[str, Any],
    passed_row_ids: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split source entries into (passed, failed) by checked row ids."""
    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for row in state["rows"]:
        if int(row["row_id"]) in passed_row_ids:
            passed.append(row["source_entry"])
        else:
            failed.append(row["source_entry"])
    return passed, failed


def _write_report(state: dict[str, Any], passed_row_ids: set[int]) -> Path:
    """Write ``fail_check.json`` with rows still marked as failed."""
    _, failed = _partition_entries(state, passed_row_ids)
    report_path = Path(state["report_path"])
    report_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
    return report_path


def _write_reports(state: dict[str, Any], passed_row_ids: set[int]) -> tuple[Path, Path]:
    """Write pass and fail reports, returning (fail_path, pass_path)."""
    passed, _ = _partition_entries(state, passed_row_ids)
    fail_path = _write_report(state, passed_row_ids)
    pass_path = Path(state["pass_path"])
    pass_path.write_text(json.dumps(passed, indent=2), encoding="utf-8")
    return fail_path, pass_path


def _render_html(state: dict[str, Any]) -> str:
    """Render the browser UI as a single self-contained HTML document."""
    title = html.escape(Path(state["failure_path"]).name)
    payload = json.dumps(
        {
            "tts_dir": state["tts_dir"],
            "failure_path": state["failure_path"],
            "report_path": state["report_path"],
            "pass_path": state["pass_path"],
            "rows": state["rows"],
        }
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ASR Failure Review - {title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111418;
      --panel: #171b21;
      --panel-2: #1f242c;
      --text: #e7edf5;
      --muted: #97a3b6;
      --accent: #7ec8ff;
      --good: #4fd1a5;
      --bad: #ff8a80;
      --border: #2a313b;
    }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: linear-gradient(180deg, #0d1014 0%, #161b22 100%);
      color: var(--text);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      padding: 16px 20px;
      background: rgba(17, 20, 24, 0.96);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(10px);
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 18px;
      font-weight: 700;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 12px;
      align-items: center;
    }}
    button {{
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      padding: 8px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }}
    button:hover {{
      border-color: var(--accent);
    }}
    button:disabled {{
      opacity: 0.5;
      cursor: not-allowed;
    }}
    main {{
      padding: 16px 20px 28px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(23, 27, 33, 0.92);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    thead th {{
      text-align: left;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      padding: 12px 10px;
      background: #20262f;
      border-bottom: 1px solid var(--border);
      color: var(--muted);
      position: sticky;
      top: 108px;
      z-index: 3;
    }}
    tbody td {{
      vertical-align: top;
      padding: 10px;
      border-bottom: 1px solid var(--border);
    }}
    tbody tr:hover {{
      background: rgba(255, 255, 255, 0.02);
    }}
    .col-index {{
      width: 72px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .col-controls {{
      width: 130px;
      white-space: nowrap;
    }}
    .col-score {{
      width: 120px;
      white-space: nowrap;
      color: var(--muted);
    }}
    .text-cell {{
      white-space: pre-wrap;
      line-height: 1.45;
    }}
    .text-small {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .file-cell {{
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }}
    .status-good {{
      color: var(--good);
      font-weight: 700;
    }}
    .status-bad {{
      color: var(--bad);
      font-weight: 700;
    }}
    .footer {{
      position: sticky;
      bottom: 0;
      padding: 12px 20px;
      background: rgba(17, 20, 24, 0.96);
      border-top: 1px solid var(--border);
      backdrop-filter: blur(10px);
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
    }}
    .summary {{
      color: var(--muted);
      font-size: 13px;
    }}
    .notice {{
      margin-top: 12px;
      padding: 10px 12px;
      border: 1px solid var(--border);
      background: rgba(31, 36, 44, 0.75);
      border-radius: 10px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>ASR Failure Review</h1>
    <div class="meta">
      TTS dir: <span id="ttsDir"></span><br>
      Failure file: <span id="failurePath"></span><br>
      Fail file: <span id="reportPath"></span><br>
      Pass file: <span id="passPath"></span>
    </div>
    <div class="toolbar">
      <button id="reportBtn" type="button">Report</button>
      <button id="markAllBtn" type="button">Mark all pass</button>
      <button id="markNoneBtn" type="button">Mark none pass</button>
      <span id="statusLine" class="summary"></span>
    </div>
  </header>
  <main>
    <div id="notice" class="notice">Loading chunks...</div>
    <table>
      <thead>
        <tr>
          <th class="col-index">Chunk</th>
          <th class="col-controls">Play / Pass</th>
          <th>Original Text</th>
          <th>Transcribed Text</th>
          <th class="col-score">Score</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </main>
  <audio id="player" preload="none"></audio>
  <div class="footer">
    <div class="summary" id="footerSummary"></div>
    <div class="summary">Passed rows go to <code>pass_check.json</code>; unchecked stay in <code>fail_check.json</code>.</div>
  </div>
  <script>
    const STATE = {payload};
    const rows = STATE.rows.map((row) => ({{...row, passed: false}}));
    const tbody = document.getElementById('rows');
    const notice = document.getElementById('notice');
    const statusLine = document.getElementById('statusLine');
    const footerSummary = document.getElementById('footerSummary');
    const player = document.getElementById('player');

    function escapeHtml(text) {{
      return String(text ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function formatChunkIndex(value) {{
      const n = Number(value);
      return Number.isFinite(n) ? String(n).padStart(5, '0') : 'n/a';
    }}

    function counts() {{
      const total = rows.length;
      const passed = rows.filter((row) => row.passed).length;
      const failed = total - passed;
      return {{ total, passed, failed }};
    }}

    function updateSummary() {{
      const c = counts();
      statusLine.textContent = `Passed: ${{c.passed}} / ${{c.total}} | Remaining: ${{c.failed}}`;
      footerSummary.textContent = `Rows: ${{c.total}} | Passed: ${{c.passed}} | Remaining to report: ${{c.failed}}`;
    }}

    function playAudio(url) {{
      player.pause();
      player.src = url;
      player.currentTime = 0;
      player.play().catch((err) => {{
        notice.textContent = `Audio play failed: ${{err}}`;
      }});
    }}

    function render() {{
      tbody.innerHTML = rows.map((row) => {{
        const audioDisabled = row.audio_exists ? '' : 'disabled';
        const audioLabel = row.audio_exists ? 'Play' : 'Missing';
        const statusClass = row.passed ? 'status-good' : 'status-bad';
        const statusText = row.passed ? 'pass' : 'keep';
        const audioUrl = row.audio_exists ? `/api/audio?path=${{encodeURIComponent(row.audio_rel)}}` : '';
        return `
          <tr data-row-id="${{row.row_id}}">
            <td class="col-index">
              #${{formatChunkIndex(row.chunk_index)}}
              <div class="file-cell">${{escapeHtml(row.filename || row.audio_rel || '')}}</div>
            </td>
            <td class="col-controls">
              <button type="button" class="play-btn" ${{audioDisabled}} data-url="${{escapeHtml(audioUrl)}}">${{audioLabel}}</button>
              <div style="margin-top:8px;">
                <label>
                  <input type="checkbox" class="pass-box">
                  <span class="${{statusClass}}">${{statusText}}</span>
                </label>
              </div>
            </td>
            <td class="text-cell">${{escapeHtml(row.original_text)}}</td>
            <td class="text-cell">${{escapeHtml(row.transcribed_text)}}</td>
            <td class="col-score">
              <div>${{escapeHtml(row.score)}}</div>
              <div class="text-small">${{escapeHtml(row.classification)}}</div>
            </td>
          </tr>
        `;
      }}).join('');

      tbody.querySelectorAll('tr').forEach((tr) => {{
        const rowId = Number(tr.getAttribute('data-row-id'));
        const row = rows.find((item) => Number(item.row_id) === rowId);
        const checkbox = tr.querySelector('.pass-box');
        const button = tr.querySelector('.play-btn');
        checkbox.checked = Boolean(row.passed);
        checkbox.addEventListener('change', () => {{
          row.passed = checkbox.checked;
          renderStatusOnly();
        }});
        button.addEventListener('click', () => playAudio(button.dataset.url));
      }});

      updateSummary();
    }}

    function renderStatusOnly() {{
      tbody.querySelectorAll('tr').forEach((tr) => {{
        const rowId = Number(tr.getAttribute('data-row-id'));
        const row = rows.find((item) => Number(item.row_id) === rowId);
        const status = tr.querySelector('.pass-box + span');
        status.textContent = row.passed ? 'pass' : 'keep';
        status.className = row.passed ? 'status-good' : 'status-bad';
      }});
      updateSummary();
    }}

    async function report() {{
      const passed_row_ids = rows.filter((row) => row.passed).map((row) => row.row_id);
      const response = await fetch('/api/report', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ passed_row_ids }})
      }});
      const payload = await response.json();
      if (!response.ok) {{
        throw new Error(payload.error || 'Report failed');
      }}
      notice.textContent = `Saved ${{payload.passed_count}} passes to ${{payload.pass_path}} and ${{payload.failed_count}} failures to ${{payload.report_path}}`;
    }}

    document.getElementById('reportBtn').addEventListener('click', async () => {{
      try {{
        await report();
      }} catch (err) {{
        notice.textContent = String(err);
      }}
    }});

    document.getElementById('markAllBtn').addEventListener('click', () => {{
      rows.forEach((row) => row.passed = true);
      render();
      notice.textContent = 'Marked all rows pass.';
    }});

    document.getElementById('markNoneBtn').addEventListener('click', () => {{
      rows.forEach((row) => row.passed = false);
      render();
      notice.textContent = 'Marked all rows keep.';
    }});

    document.getElementById('ttsDir').textContent = STATE.tts_dir.replace(/\\/g, '/');
    document.getElementById('failurePath').textContent = STATE.failure_path.replace(/\\/g, '/');
    document.getElementById('reportPath').textContent = STATE.report_path.replace(/\\/g, '/');
    document.getElementById('passPath').textContent = STATE.pass_path.replace(/\\/g, '/');
    render();
    notice.textContent = `Loaded ${{rows.length}} failure rows. Check pass boxes, then click Report.`;
  </script>
</body>
</html>
"""


class _ReviewHTTPServer(ThreadingHTTPServer):
    """HTTP server that carries review state for request handlers."""

    def __init__(self, server_address: tuple[str, int], state: dict[str, Any]):
        """Store review state before serving requests."""
        super().__init__(server_address, _ReviewRequestHandler)
        self.state = state


class _ReviewRequestHandler(BaseHTTPRequestHandler):
    """Serve the review page, audio files, and export endpoint."""

    server_version = "ASRFailureReviewBrowser/1.0"

    def do_GET(self) -> None:
        """Dispatch GET requests for page, data, and audio assets."""
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send_html()
            return
        if parsed.path == "/api/failures":
            self._send_json(self.server.state)
            return
        if parsed.path == "/api/audio":
            self._send_audio(parsed)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        """Handle report generation requests from the browser UI."""
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/report":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send_json({"error": f"Invalid JSON payload: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        passed_ids = payload.get("passed_row_ids", [])
        try:
            passed_row_ids = {int(item) for item in passed_ids}
        except Exception:
            self._send_json({"error": "passed_row_ids must be a list of integers."}, status=HTTPStatus.BAD_REQUEST)
            return

        fail_path, pass_path = _write_reports(self.server.state, passed_row_ids)
        failed_count = len(self.server.state["rows"]) - len(passed_row_ids)
        self._send_json(
            {
                "ok": True,
                "report_path": str(fail_path),
                "pass_path": str(pass_path),
                "failed_count": failed_count,
                "passed_count": len(passed_row_ids),
            }
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        """Suppress noisy access logs for the review browser."""
        return

    def _send_html(self) -> None:
        """Return the HTML user interface."""
        body = _render_html(self.server.state).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        """Return JSON with a standard content type."""
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_audio(self, parsed: urllib.parse.ParseResult) -> None:
        """Stream one WAV file from the TTS directory."""
        rel_path = urllib.parse.parse_qs(parsed.query).get("path", [""])[0]
        if not rel_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing audio path")
            return

        tts_dir = Path(self.server.state["tts_dir"])
        candidate = (tts_dir / Path(rel_path)).resolve()
        try:
            candidate.relative_to(tts_dir.resolve())
        except Exception:
            self.send_error(HTTPStatus.FORBIDDEN, "Audio path escapes TTS directory")
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Audio file not found")
            return

        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        body = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _open_browser(url: str) -> None:
    """Open the default browser in a background-safe way."""
    webbrowser.open(url, new=2, autoraise=True)


def _play_audio_file(audio_path: Path) -> None:
    """Open one matched audio chunk with the system audio player."""
    opener = shutil.which("xdg-open")
    if opener is None:
        raise RuntimeError("xdg-open is required to play audio chunks")
    subprocess.Popen(
        [opener, str(audio_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _run_tk_review(state: dict[str, Any]) -> None:
    """Show the ASR failure review directly in a Tkinter window."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("ASR Failure Review")
    root.geometry("1400x800")

    header = ttk.Frame(root, padding=8)
    header.pack(fill="x")
    ttk.Label(header, text=f"Failure file: {state['failure_path']}").pack(anchor="w")
    ttk.Label(
        header,
        text="Check Pass for chunks that sound correct. Passed go to pass_check.json, unchecked go to fail_check.json.",
    ).pack(anchor="w")
    toolbar = ttk.Frame(header)
    toolbar.pack(anchor="w", pady=(8, 0))
    report_button = ttk.Button(toolbar, text="Report")
    report_button.pack(side="left")
    count_label = ttk.Label(toolbar, text="")
    count_label.pack(side="left", padx=12)

    outer = ttk.Frame(root)
    outer.pack(fill="both", expand=True, padx=8, pady=(0, 8))
    canvas = tk.Canvas(outer, highlightthickness=0)
    scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    rows_frame = ttk.Frame(canvas)
    canvas_window = canvas.create_window((0, 0), window=rows_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def resize_rows(_event: Any = None) -> None:
        """Keep review rows as wide as the scrolling canvas."""
        canvas.itemconfigure(canvas_window, width=canvas.winfo_width())
        canvas.configure(scrollregion=canvas.bbox("all"))

    rows_frame.bind("<Configure>", resize_rows)
    canvas.bind("<Configure>", resize_rows)

    def _on_wheel(event: Any) -> str:
        """Scroll the review canvas for mouse-wheel events on any widget."""
        if getattr(event, "num", None) == 4:
            canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(3, "units")
        else:
            delta = getattr(event, "delta", 0) or 0
            canvas.yview_scroll(-int(delta / 120), "units")
        return "break"

    def _on_nav_key(event: Any) -> str | None:
        """Scroll the review canvas for navigation keyboard events."""
        keysym = event.keysym
        if keysym in ("Up", "Down"):
            canvas.yview_scroll(-1 if keysym == "Up" else 1, "units")
        elif keysym in ("Prior", "Next"):
            canvas.yview_scroll(-1 if keysym == "Prior" else 1, "pages")
        elif keysym == "Home":
            canvas.yview_moveto(0.0)
        elif keysym == "End":
            canvas.yview_moveto(1.0)
        else:
            return None
        return "break"

    for keysym in ("Button-4", "Button-5", "MouseWheel"):
        root.bind_all(f"<{keysym}>", _on_wheel)
    for keysym in ("Up", "Down", "Prior", "Next", "Home", "End"):
        root.bind_all(f"<{keysym}>", _on_nav_key)
    canvas.focus_set()

    ttk.Label(rows_frame, text="Chunk", width=10).grid(row=0, column=0, sticky="nw", padx=4, pady=4)
    ttk.Label(rows_frame, text="Audio / Pass", width=16).grid(row=0, column=1, sticky="nw", padx=4, pady=4)
    ttk.Label(rows_frame, text="Original text").grid(row=0, column=2, sticky="nw", padx=4, pady=4)
    ttk.Label(rows_frame, text="Transcribed text").grid(row=0, column=3, sticky="nw", padx=4, pady=4)
    rows_frame.columnconfigure(2, weight=1)
    rows_frame.columnconfigure(3, weight=1)

    pass_vars: dict[int, tk.BooleanVar] = {}

    def update_counts() -> None:
        """Refresh the live pass/keep counter in the header toolbar."""
        passed = sum(1 for variable in pass_vars.values() if variable.get())
        remaining = len(state["rows"]) - passed
        count_label.configure(
            text=(
                f"Passed: {passed} / {len(state['rows'])} | "
                f"Remaining to report: {remaining}"
            )
        )

    for row_number, row in enumerate(state["rows"], start=1):
        row_id = int(row["row_id"])
        pass_var = tk.BooleanVar(value=False)
        pass_vars[row_id] = pass_var
        index_cell = ttk.Frame(rows_frame)
        index_cell.grid(row=row_number, column=0, sticky="nw", padx=4, pady=6)
        ttk.Label(index_cell, text=f"{int(row['chunk_index']):05d}").pack(anchor="w")
        if row.get("decision"):
            color = "#4fd1a5" if row["decision"] == "accepted_by_medium" else "#ff8a80"
            ttk.Label(index_cell, text=row["decision"], foreground=color).pack(anchor="w")
        controls = ttk.Frame(rows_frame)
        controls.grid(row=row_number, column=1, sticky="nw", padx=4, pady=6)
        tts_dir = Path(state["tts_dir"])
        main_path = tts_dir / row["audio_main_rel"] if row["audio_main_rel"] else None
        failed_path = (
            tts_dir / row["audio_failed_rel"] if row["audio_failed_rel"] else None
        )
        main_button = ttk.Button(controls, text="Original")
        main_button.pack(anchor="w")
        if main_path is None:
            main_button.configure(state="disabled")
        else:
            main_button.configure(
                command=lambda path=main_path: _play_audio_file(path)
            )
        failed_button = ttk.Button(controls, text="Failed")
        failed_button.pack(anchor="w", pady=(4, 0))
        if failed_path is None:
            failed_button.configure(state="disabled")
        else:
            failed_button.configure(
                command=lambda path=failed_path: _play_audio_file(path)
            )
        ttk.Checkbutton(
            controls,
            text="Pass",
            variable=pass_var,
            command=update_counts,
        ).pack(anchor="w", pady=(4, 0))
        original_cell = ttk.Frame(rows_frame)
        original_cell.grid(row=row_number, column=2, sticky="nw", padx=4, pady=6)
        ttk.Label(
            original_cell,
            text=row["original_text"],
            wraplength=480,
            justify="left",
        ).pack(anchor="w")
        if row.get("original_explanation"):
            ttk.Label(
                original_cell,
                text=f"orig fail: {row['original_explanation']}",
                foreground="#97a3b6",
                wraplength=480,
                justify="left",
            ).pack(anchor="w")
        transcribed_cell = ttk.Frame(rows_frame)
        transcribed_cell.grid(row=row_number, column=3, sticky="nw", padx=4, pady=6)
        ttk.Label(
            transcribed_cell,
            text=row["transcribed_text"],
            wraplength=480,
            justify="left",
        ).pack(anchor="w")
        if row.get("medium_explanation"):
            ttk.Label(
                transcribed_cell,
                text=f"medium: {row['medium_explanation']}",
                foreground="#97a3b6",
                wraplength=480,
                justify="left",
            ).pack(anchor="w")

    status = ttk.Label(root, text=f"Loaded {len(state['rows'])} failure chunks")
    status.pack(fill="x", padx=8)
    root.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

    def write_report() -> None:
        """Write pass and fail reports, then show the saved paths."""
        passed_ids = {row_id for row_id, variable in pass_vars.items() if variable.get()}
        fail_path, pass_path = _write_reports(state, passed_ids)
        remaining = len(state["rows"]) - len(passed_ids)
        status.configure(
            text=(
                f"Saved {len(passed_ids)} passes to {pass_path} "
                f"and {remaining} failures to {fail_path}"
            )
        )
        messagebox.showinfo(
            "Report saved",
            f"Saved {len(passed_ids)} passes to:\n{pass_path}\n\n"
            f"Saved {remaining} failures to:\n{fail_path}",
        )

    ttk.Button(root, text="Report", command=write_report).pack(pady=8)
    report_button.configure(command=write_report)
    update_counts()
    root.mainloop()


def main() -> int:
    """Open Tkinter review window for one ASR failure JSON file."""
    args = _parse_args()
    if args.target is None:
        selected = _pick_failure_file()
        if selected is None:
            print("No failure JSON selected.")
            return 1
        tts_dir, failure_path = _resolve_target(str(selected))
    else:
        tts_dir, failure_path = _resolve_target(args.target)
    state = _build_state(tts_dir, failure_path, include_accepted=args.include_accepted)
    matched_audio = sum(1 for row in state["rows"] if row["audio_exists"])
    matched_failed = sum(1 for row in state["rows"] if row["audio_failed_rel"])
    print(f"Failure file:   {failure_path}")
    print(f"Loaded rows:    {len(state['rows'])} ({matched_audio} audio files matched, {matched_failed} failed-copy)")
    print(f"Report file:    {state['report_path']}")
    print(f"Pass file:      {state['pass_path']}")
    _run_tk_review(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
