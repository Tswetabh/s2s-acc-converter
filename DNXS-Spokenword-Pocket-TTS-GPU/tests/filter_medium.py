#!/usr/bin/env python3
"""Filter asr_medium_verification.json to keep only records whose decision is
'remains_failed_after_medium', dropping 'accepted_by_medium' entries.

Writes a new file alongside the original, leaving the source untouched.
"""
import json
from pathlib import Path

KEEP_DECISION = "remains_failed_after_medium"
DROP_DECISION = "accepted_by_medium"
SRC = Path(__file__).parent / "asr_medium_verification.json"
DST = SRC.with_name(SRC.stem + ".remains_failed.json")


def main() -> None:
    """Load the report, keep only failed-after-medium records, write filtered copy."""
    report = json.loads(SRC.read_text(encoding="utf-8"))
    kept = [r for r in report["records"] if r.get("decision") == KEEP_DECISION]
    dropped = [r for r in report["records"] if r.get("decision") == DROP_DECISION]

    summary = report.get("summary", {})
    summary["attempted"] = len(kept)
    summary["verified_pass"] = 0
    summary["verified_fail"] = len(kept)

    filtered = dict(report)
    filtered["summary"] = summary
    filtered["records"] = kept

    DST.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
    print(f"total: {len(report['records'])}  kept: {len(kept)}  dropped: {len(dropped)}")
    print(f"wrote: {DST}")


if __name__ == "__main__":
    main()
