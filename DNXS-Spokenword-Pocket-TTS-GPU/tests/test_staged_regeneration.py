"""Regression tests for staged regeneration strategy dispatch."""

import json
from pathlib import Path
from unittest.mock import Mock

from pocket_tts.audiobook.generator import AudiobookGenerator


def test_staged_strategy_dispatches_before_legacy_regeneration(tmp_path: Path) -> None:
    """The explicit staged strategy must bypass first-pass legacy regeneration."""
    failure_log = tmp_path / "asr_failures.json"
    failure_log.write_text(json.dumps([{"chunk_index": 4, "score": 0.2}]))
    generator = AudiobookGenerator.__new__(AudiobookGenerator)
    staged = Mock(return_value=[{"chunk_index": 4}])
    generator._reprocess_failed_chunks_staged = staged
    dataset_paths = {
        "audio_chunks_dir": str(tmp_path / "audio_chunks"),
        "text_chunks_dir": str(tmp_path / "text_chunks"),
        "tts_dir": str(tmp_path),
    }

    result = generator._reprocess_failed_chunks(
        str(failure_log),
        object(),
        dataset_paths,
        {"regeneration_strategy": "staged_three_candidates", "max_retries": 3},
    )

    assert result == [{"chunk_index": 4}]
    staged.assert_called_once()
