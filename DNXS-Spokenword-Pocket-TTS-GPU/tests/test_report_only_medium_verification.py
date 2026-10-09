"""Regression coverage for report-only ASR verification order."""

from pathlib import Path
import unittest


class ReportOnlyMediumVerificationOrderTests(unittest.TestCase):
    """Protect Medium verification from the zero-retry report-only exit."""

    def test_report_only_exit_follows_medium_verification(self) -> None:
        """Require report-only mode to audit failures before it returns."""
        generator_source = (
            Path(__file__).resolve().parents[1]
            / "pocket_tts"
            / "audiobook"
            / "generator.py"
        ).read_text(encoding="utf-8")

        medium_call = generator_source.index(
            "failures, regen_verification_summary = self._verify_failed_chunks_with_medium_asr("
        )
        report_only_guard = generator_source.index(
            "ASR report-only mode enabled (max_retries=0); "
        )

        self.assertLess(
            medium_call,
            report_only_guard,
            "Report-only mode must run Medium verification before returning.",
        )


if __name__ == "__main__":
    unittest.main()
