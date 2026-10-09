#!/usr/bin/env python3
"""Create a filtered, timestamped copy of the PocketGPU project tree."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


COMPRESSED_SUFFIXES = frozenset(
    {
        ".7z",
        ".bz2",
        ".cab",
        ".gz",
        ".iso",
        ".lz",
        ".lz4",
        ".lzma",
        ".rar",
        ".tar",
        ".tbz",
        ".tbz2",
        ".tgz",
        ".txz",
        ".xz",
        ".zip",
        ".zst",
    }
)
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        "output",
        "venv",
        ".venv",
        "env",
        ".env",
        "virtualenv",
        ".virtualenv",
        "__pypackages__",
        "docdna",
        "input",
        "backup",
        "backups",
    }
)
ROOT_EXCLUDED_FILENAMES = frozenset(
    {
        "AGENTS.md",
        "opencode.json",
        "pocket_tts.log",
        "plan.txt",
        "prompt.txt",
        "project_state.json",
        "restore.bat",
        "session-ses_0926.md",
    }
)
DOCS_EXCLUDED_SUFFIXES = frozenset(
    {
        "_fix.md",
        "_guide.md",
        "_investigation.md",
        "_plan.md",
        "_results.md",
    }
)
BACKUP_MARKER_RE = re.compile(
    r"(?:\.backup(?:[._].+)?|\.back|\.old|\.orig|\bcopy\b|\.\d+(?:_\d+)?$)",
    re.IGNORECASE,
)
DEFAULT_SOURCE = Path(__file__).resolve().parents[1]


@dataclass
class BackupSummary:
    """Counts produced while copying a filtered project tree."""

    copied: int = 0
    skipped: int = 0
    failed: int = 0


def _is_excluded_name(name: str) -> bool:
    """Return whether a filename matches one of the backup exclusion rules."""
    lowered = name.lower()
    return (
        name.startswith(".")
        or name.endswith("~")
        or lowered.endswith(".bak")
        or Path(lowered).suffix in COMPRESSED_SUFFIXES
    )


def _is_excluded_path(path: Path) -> bool:
    """Return whether a relative path should stay out of the backup tree."""
    name = path.name
    lowered = name.lower()
    if len(path.parts) == 1 and name in ROOT_EXCLUDED_FILENAMES:
        return True
    if path.parts[:1] == ("tests",) and name == "test_backup_project.py":
        return True
    if (
        path.parts[:2] == ("tests", "__pycache__")
        and name.startswith("test_backup_project.")
        and lowered.endswith(".pyc")
    ):
        return True
    if path.parts[:2] == ("tests", "results"):
        return True
    if path.parts and path.parts[0] == "docs":
        if "copy" in lowered:
            return True
        if any(lowered.endswith(suffix) for suffix in DOCS_EXCLUDED_SUFFIXES):
            return True
    return BACKUP_MARKER_RE.search(name) is not None and name != "backup_project.py"


def _is_excluded_directory(name: str) -> bool:
    """Return whether a directory is generated data or a local environment."""
    return name.lower() in EXCLUDED_DIRECTORY_NAMES or _is_excluded_name(name)


def _iter_files(source: Path, summary: BackupSummary) -> Iterable[Path]:
    """Yield eligible regular files while pruning hidden and archive directories."""
    for current_dir, dir_names, file_names in os.walk(source):
        current = Path(current_dir)
        relative_current = current.relative_to(source)
        kept_dirs = []
        for name in sorted(dir_names):
            child = current / name
            if (
                (relative_current == Path("tests") and name == "results")
                or _is_excluded_directory(name)
                or child.is_symlink()
            ):
                summary.skipped += 1
            else:
                kept_dirs.append(name)
        dir_names[:] = kept_dirs
        for name in sorted(file_names):
            path = current / name
            relative_path = path.relative_to(source)
            if (
                _is_excluded_name(name)
                or _is_excluded_path(relative_path)
                or path.is_symlink()
                or not path.is_file()
            ):
                summary.skipped += 1
                continue
            yield path


def _default_destination(source: Path) -> Path:
    """Return a timestamped sibling directory that cannot recurse into source."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return source.parent / f"{source.name}_backup_{stamp}"


def backup_project(source: Path, destination: Path, dry_run: bool = False) -> BackupSummary:
    """Copy eligible project files into a timestamped backup tree.

    Args:
        source: Existing project directory to copy.
        destination: New directory receiving the filtered tree.
        dry_run: Report eligible files without creating or copying anything.

    Returns:
        Counts of copied, skipped, and failed entries.

    Raises:
        ValueError: If source and destination overlap.
        FileNotFoundError: If source is not an existing directory.
        FileExistsError: If destination already exists for a real backup.
    """
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Destination must be outside the source directory")
    if destination.exists() and not dry_run:
        raise FileExistsError(f"Destination already exists: {destination}")

    summary = BackupSummary()
    if not dry_run:
        destination.mkdir(parents=True)

    for source_file in _iter_files(source, summary):
        relative = source_file.relative_to(source)
        target = destination / relative
        if dry_run:
            summary.copied += 1
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
            summary.copied += 1
        except OSError as exc:
            summary.failed += 1
            print(f"WARNING: could not copy {relative}: {exc}")

    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the backup utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy a project tree while excluding hidden entries, compressed files, "
            "editor backups, scratch files, docs notes, and tests/results."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Project directory to back up (default: current PocketGPU project).",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="Backup directory; default is a timestamped sibling of source.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count eligible files without creating a backup.",
    )
    return parser


def main() -> int:
    """Parse arguments, create the backup, and return a process exit code."""
    args = _build_parser().parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination or _default_destination(source)
    try:
        summary = backup_project(source, destination, dry_run=args.dry_run)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    action = "would copy" if args.dry_run else "created"
    print(f"Backup {action}: {destination}")
    print(f"Eligible files: {summary.copied}")
    print(f"Skipped by rules/symlink policy: {summary.skipped}")
    print(f"Copy failures: {summary.failed}")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
