"""Regression coverage for final ASR investigation-log persistence."""

from pocket_tts.audiobook.generator import AudiobookGenerator


def test_investigation_log_keeps_high_score_policy_failure(tmp_path):
    """Write a final failure even when its numeric score exceeds the threshold."""
    entry = {
        "chunk_index": 7,
        "best_score": 0.9,
        "threshold": 0.6,
        "text": "Protected source text.",
        "original_asr": {"original_text": "Protected source text."},
        "best_attempt": {
            "attempt": 1,
            "temp": 0.5,
            "score": 0.9,
            "is_original": False,
            "asr_result": {
                "passed": False,
                "score": 0.9,
                "hyp_text_raw": "Changed protected content.",
                "explanation": "Protected content mismatch.",
            },
        },
        "all_attempts": [
            {
                "attempt": 1,
                "temp": 0.5,
                "score": 0.9,
                "is_original": False,
                "asr_result": {"passed": False},
            }
        ],
    }

    generator = AudiobookGenerator.__new__(AudiobookGenerator)
    generator._save_investigation_log([entry], {"tts_dir": tmp_path})

    report = (tmp_path / "asr_investigation.log").read_text(encoding="utf-8")
    assert "Final failed chunks: 1" in report
    assert "CHUNK: chunk_00007" in report
    assert "Status: FAILED (Score: 0.900)" in report
    assert "✗ Attempt 1" in report
