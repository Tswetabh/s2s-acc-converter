"""First-run GPU ASR worker calibration (runs once, then gated by config flag).

On first program launch (``worker_calibration.done: false``), finds a finished
book TTS directory with enough chunks, runs ``tests/tools/calibrate_asr_workers.py``,
writes ``gpu_workers_after_tts`` + sets ``done: true`` so it never auto-runs again.

GUI spinboxes remain authoritative for each generation run: calibration only
seeds the default; the user can change the GUI for a run without saving.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CALIBRATE_SCRIPT = ROOT / "tests" / "tools" / "calibrate_asr_workers.py"
# Single venv: same interpreter as the main app
ASR_PYTHON = Path(sys.executable)
DEFAULT_CONFIG = ROOT / "pocket_tts" / "config" / "default_config.yaml"
MIN_SAMPLE_CHUNKS = 24


def is_calibration_done(asr_cfg: Optional[Dict[str, Any]]) -> bool:
    """Return True if first-run ASR worker calibration already completed.

    Args:
        asr_cfg: ``asr_quality_control`` dict (or None).

    Returns:
        Whether ``worker_calibration.done`` is truthy.
    """
    if not isinstance(asr_cfg, dict):
        return False
    cal = asr_cfg.get("worker_calibration") or {}
    if not isinstance(cal, dict):
        return False
    return bool(cal.get("done"))


def discover_chunk_count(tts_dir: Path) -> int:
    """Count matching audio/text chunk pairs under a TTS directory.

    Args:
        tts_dir: Path containing audio_chunks and text_chunks.

    Returns:
        Number of paired chunk stems.
    """
    audio = tts_dir / "audio_chunks"
    text = tts_dir / "text_chunks"
    if not audio.is_dir() or not text.is_dir():
        return 0
    n = 0
    for wav in audio.glob("chunk_*.wav"):
        if (text / f"{wav.stem}.txt").is_file():
            n += 1
    return n


def find_sample_tts_dir(
    project_root: Optional[Path] = None,
    min_chunks: int = MIN_SAMPLE_CHUNKS,
) -> Optional[Path]:
    """Find a TTS directory with enough chunks for calibration.

    Prefers known Edge book path, then any ``Output/*/TTS`` with enough pairs.

    Args:
        project_root: Repo root (default: package parent).
        min_chunks: Minimum paired chunks required.

    Returns:
        Path to TTS dir, or None if nothing suitable exists yet.
    """
    root = project_root or ROOT
    preferred = [
        root / "Output" / "Doctor Who_ The Edge of Destruction" / "TTS",
    ]
    for p in preferred:
        if discover_chunk_count(p) >= min_chunks:
            return p

    output = root / "Output"
    if not output.is_dir():
        return None
    candidates: List[tuple] = []
    for tts in output.glob("*/TTS"):
        n = discover_chunk_count(tts)
        if n >= min_chunks:
            candidates.append((n, tts))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def mark_calibration_done(
    config_path: Path,
    recommended: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write calibration result into YAML: done=true + gpu_workers_after_tts.

    Args:
        config_path: Path to default_config.yaml.
        recommended: Worker count to store as gpu_workers_after_tts.
        extra: Optional metadata (gpu name, thruput) under worker_calibration.
    """
    import time

    import yaml

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    text = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    asr = data.setdefault("asr_quality_control", {})
    if not isinstance(asr, dict):
        asr = {}
        data["asr_quality_control"] = asr
    parallel = asr.setdefault("parallel", {})
    if not isinstance(parallel, dict):
        parallel = {}
        asr["parallel"] = parallel
    parallel["gpu_workers_after_tts"] = int(recommended)

    cal = asr.setdefault("worker_calibration", {})
    if not isinstance(cal, dict):
        cal = {}
        asr["worker_calibration"] = cal
    cal["done"] = True
    cal["recommended"] = int(recommended)
    cal["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if extra:
        cal.update(extra)

    # Prefer surgical line edits for gpu_workers + inject calibration block
    lines = text.splitlines(keepends=True)
    out_lines: List[str] = []
    i = 0
    saw_gpu = False
    saw_cal = False
    in_asr = False
    asr_indent = ""
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        if stripped.startswith("asr_quality_control:"):
            in_asr = True
            asr_indent = indent
            out_lines.append(line)
            i += 1
            continue

        # Left ASR section when a top-level key appears
        if in_asr and stripped and not stripped.startswith("#"):
            if len(indent) <= len(asr_indent) and ":" in stripped and not stripped.startswith(" "):
                # top-level sibling — close cal injection before it if needed
                if not saw_cal:
                    base = asr_indent + "  "
                    out_lines.append(f"{base}worker_calibration:\n")
                    out_lines.append(f"{base}  done: true\n")
                    out_lines.append(f"{base}  recommended: {int(recommended)}\n")
                    out_lines.append(
                        f"{base}  last_run: \"{cal['last_run']}\"\n"
                    )
                    saw_cal = True
                in_asr = False

        if stripped.startswith("gpu_workers_after_tts:"):
            comment = ""
            if "#" in line:
                comment = "  #" + line.split("#", 1)[1].rstrip("\n")
            out_lines.append(f"{indent}gpu_workers_after_tts: {int(recommended)}{comment}\n")
            saw_gpu = True
            i += 1
            continue

        if stripped.startswith("worker_calibration:"):
            # Replace whole nested block
            base = indent
            out_lines.append(f"{base}worker_calibration:\n")
            out_lines.append(f"{base}  done: true\n")
            out_lines.append(f"{base}  recommended: {int(recommended)}\n")
            out_lines.append(f"{base}  last_run: \"{cal['last_run']}\"\n")
            saw_cal = True
            i += 1
            # skip old nested keys
            while i < len(lines):
                nxt = lines[i]
                ns = nxt.lstrip()
                ni = nxt[: len(nxt) - len(ns)]
                if ns and len(ni) <= len(base) and not ns.startswith("#"):
                    break
                if ns.startswith("#") or not ns:
                    # keep blank? skip nested content
                    i += 1
                    continue
                if len(ni) > len(base):
                    i += 1
                    continue
                break
            continue

        out_lines.append(line)
        i += 1

    if in_asr and not saw_cal:
        base = asr_indent + "  "
        out_lines.append(f"{base}worker_calibration:\n")
        out_lines.append(f"{base}  done: true\n")
        out_lines.append(f"{base}  recommended: {int(recommended)}\n")
        out_lines.append(f"{base}  last_run: \"{cal['last_run']}\"\n")

    if not saw_gpu:
        # Fallback full dump
        config_path.write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        return

    config_path.write_text("".join(out_lines), encoding="utf-8")


def run_first_calibration(
    *,
    tts_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    model: str = "base",
    workers: str = "1,2,4,6",
    sample_chunks: int = 48,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the ASR worker calibrator once and mark config done.

    Args:
        tts_dir: Sample TTS dir; auto-discovered if None.
        config_path: YAML to update; default default_config.yaml.
        model: faster-whisper model name.
        workers: Worker list string for the calibrator.
        sample_chunks: Chunks per trial.
        project_root: Repo root for discovery.

    Returns:
        Result dict with ok, recommended, message, and optional error.
    """
    root = project_root or ROOT
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG
    sample = Path(tts_dir) if tts_dir else find_sample_tts_dir(root)
    if sample is None:
        return {
            "ok": False,
            "deferred": True,
            "recommended": None,
            "message": (
                "No finished book with enough chunks for calibration yet; "
                "will retry when sample audio exists"
            ),
        }

    asr_py = ASR_PYTHON if ASR_PYTHON.is_file() else Path(sys.executable)
    if not CALIBRATE_SCRIPT.is_file():
        return {
            "ok": False,
            "deferred": False,
            "recommended": None,
            "message": f"Calibrator missing: {CALIBRATE_SCRIPT}",
        }

    cmd = [
        str(asr_py),
        str(CALIBRATE_SCRIPT),
        "--tts-dir",
        str(sample),
        "--model",
        model,
        "--device",
        "cuda",
        "--workers",
        workers,
        "--sample-chunks",
        str(sample_chunks),
        "--write-config",
        str(cfg_path),
    ]
    logger.info("First-run ASR worker calibration: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "deferred": False,
            "recommended": None,
            "message": "Calibration timed out",
        }
    except Exception as e:
        return {
            "ok": False,
            "deferred": False,
            "recommended": None,
            "message": f"Calibration failed: {e}",
        }

    # Parse RECOMMENDED line
    recommended = None
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in out.splitlines():
        if "RECOMMENDED gpu_workers_after_tts" in line:
            try:
                recommended = int(line.rsplit("=", 1)[-1].strip())
            except ValueError:
                pass

    if recommended is None:
        # try latest.json
        latest = root / "tests" / "results" / "asr_worker_calibrate" / "latest.json"
        if latest.is_file():
            try:
                import json

                data = json.loads(latest.read_text(encoding="utf-8"))
                recommended = int(data.get("pick", {}).get("recommended") or 0) or None
            except Exception:
                pass

    if recommended is None or proc.returncode not in (0,):
        # still try to mark done only on clear success
        if recommended is None:
            return {
                "ok": False,
                "deferred": False,
                "recommended": None,
                "message": f"Calibration did not produce a recommendation (rc={proc.returncode})",
                "stdout_tail": out[-2000:],
            }

    try:
        mark_calibration_done(
            cfg_path,
            int(recommended),
            extra={"sample_tts": str(sample)},
        )
    except Exception as e:
        logger.warning("Could not mark calibration done in config: %s", e)

    return {
        "ok": True,
        "deferred": False,
        "recommended": int(recommended),
        "sample_tts": str(sample),
        "message": f"Calibrated gpu_workers_after_tts={recommended} from {sample}",
        "returncode": proc.returncode,
    }
