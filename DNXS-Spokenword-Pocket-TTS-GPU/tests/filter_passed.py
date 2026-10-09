#!/usr/bin/env python3
"""Filter a medium ASR verification report to keep only records whose
``medium_result.passed`` is false, writing a filtered copy alongside the
source and leaving the original untouched.
"""
import argparse
import json
from pathlib import Path


def _configure_tk_file_dialogs_hide_hidden(root: object) -> None:
    """Enable Tk hidden-file checkbox and leave it unchecked by default."""
    try:
        root.tk.eval("catch {tk_getOpenFile -badoption}")
        root.tk.setvar("::tk::dialog::file::showHiddenBtn", 1)
        root.tk.setvar("::tk::dialog::file::showHiddenVar", 0)
    except Exception:
        pass


def _output_dir() -> Path:
    """Return the project's ``Output`` directory next to the tests folder."""
    candidate = Path(__file__).resolve().parent.parent / "Output"
    return candidate if candidate.is_dir() else Path.cwd()


def _pick_report_file(initial_dir: Path) -> Path | None:
    """Open a Tk file browser rooted in the Output directory."""
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
            title="Select asr_medium_verification.json",
            initialdir=str(initial_dir),
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


def _parse_args() -> argparse.Namespace:
    """Parse source report path and optional destination path."""
    parser = argparse.ArgumentParser(
        description=(
            "Keep only passed:false records from a medium verification report."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help=(
            "Path to asr_medium_verification.json. If omitted, a file "
            "browser opens in the project Output directory."
        ),
    )
    parser.add_argument(
        "destination",
        nargs="?",
        default=None,
        help=(
            "Output path. Defaults to <source stem>.passed_false.json "
            "beside the source."
        ),
    )
    return parser.parse_args()


def _filter_failed_report(source: Path, destination: Path) -> dict:
    """Return a copy of the report with only passed:false records."""
    report = json.loads(source.read_text(encoding="utf-8"))
    all_records = report.get("records", [])
    kept = [
        record
        for record in all_records
        if record.get("medium_result", {}).get("passed") is False
    ]
    summary = dict(report.get("summary", {}))
    summary["attempted"] = len(kept)
    summary["verified_pass"] = 0
    summary["verified_fail"] = len(kept)
    filtered = dict(report)
    filtered["summary"] = summary
    filtered["records"] = kept
    print(
        f"total: {len(all_records)}  kept (passed:false): {len(kept)}  "
        f"dropped: {len(all_records) - len(kept)}"
    )
    print(f"wrote: {destination}")
    return filtered


def main() -> int:
    """Run the filter and write the passed:false-only report."""
    args = _parse_args()
    if args.source is None:
        picked = _pick_report_file(_output_dir())
        if picked is None:
            print("No report file selected.")
            return 1
        source = picked
    else:
        source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Report file does not exist: {source}")
    destination = (
        Path(args.destination).expanduser().resolve()
        if args.destination
        else source.with_name(source.stem + ".passed_false.json")
    )
    filtered = _filter_failed_report(source, destination)
    destination.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
