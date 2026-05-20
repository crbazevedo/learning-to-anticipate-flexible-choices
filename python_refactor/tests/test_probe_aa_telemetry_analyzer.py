"""W22 Probe AA regression tests for the AMFC telemetry analyzer.

Covers (from the Probe AA contract):
  - test_summarize_empty_telemetry
  - test_summarize_single_record
  - test_summarize_multiple_records_agreement_rate
  - test_summarize_tie_break_rate
  - test_amfc_picks_more_certain_when_lower_variance
  - test_format_summary_markdown_returns_string_with_keys
  - test_save_and_load_telemetry_roundtrip
  - test_compare_summaries_returns_diff_markdown
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes.probe_aa_amfc_telemetry_analyzer import (  # noqa: E402
    SUMMARY_KEYS,
    compare_telemetry_summaries,
    format_summary_markdown,
    load_telemetry_from_json,
    save_telemetry_to_json,
    summarize_telemetry,
)


def _make_record(
    *,
    amfc_idx: int = 0,
    hv_dm_idx: int = 0,
    amfc_pick_roi: float = 0.001,
    hv_dm_pick_roi: float = 0.001,
    amfc_pick_risk: float = 0.01,
    hv_dm_pick_risk: float = 0.01,
    mean_forecast_variance: float = 1e-4,
    amfc_pick_forecast_variance: float = 1e-4,
    tie_break_fired: bool = False,
    n_candidates: int = 5,
    horizon: int = 1,
    n_mc: int = 200,
    top1_expected_contrib: float = 1e-5,
    R1: float = 0.0,
    R2: float = 0.05,
) -> dict:
    return {
        "amfc_idx": amfc_idx,
        "hv_dm_idx": hv_dm_idx,
        "amfc_agrees_with_hv_dm": amfc_idx == hv_dm_idx,
        "amfc_pick_roi": amfc_pick_roi,
        "amfc_pick_risk": amfc_pick_risk,
        "hv_dm_pick_roi": hv_dm_pick_roi,
        "hv_dm_pick_risk": hv_dm_pick_risk,
        "n_candidates": n_candidates,
        "horizon": horizon,
        "n_mc": n_mc,
        "top1_expected_contrib": top1_expected_contrib,
        "mean_forecast_variance": mean_forecast_variance,
        "amfc_pick_forecast_variance": amfc_pick_forecast_variance,
        "tie_break_fired": tie_break_fired,
        "R1": R1,
        "R2": R2,
    }


class TestSummarizeTelemetry:
    """summarize_telemetry contract."""

    def test_summarize_empty_telemetry(self):
        """Empty input → all-zero defaults with the canonical keys."""
        summary = summarize_telemetry([])
        for k in SUMMARY_KEYS:
            assert k in summary, f"missing canonical key {k!r}"
        assert summary["n_periods"] == 0
        assert summary["agreement_rate"] == 0.0
        assert summary["roi_delta_std"] == 0.0
        assert summary["amfc_picks_more_certain_than_average"] is False

    def test_summarize_single_record(self):
        """Single record → stats reduce to that record's values."""
        rec = _make_record(
            amfc_idx=1,
            hv_dm_idx=2,
            amfc_pick_roi=0.003,
            hv_dm_pick_roi=0.005,
            amfc_pick_risk=0.008,
            hv_dm_pick_risk=0.010,
            mean_forecast_variance=2e-4,
            amfc_pick_forecast_variance=5e-5,
        )
        summary = summarize_telemetry([rec])
        assert summary["n_periods"] == 1
        assert summary["agreement_rate"] == 0.0
        assert summary["mean_amfc_pick_roi"] == 0.003
        assert summary["mean_hv_dm_pick_roi"] == 0.005
        assert abs(summary["roi_delta_mean"] - (0.003 - 0.005)) < 1e-12
        assert summary["roi_delta_std"] == 0.0  # n<2 ⇒ 0
        assert summary["tie_break_fire_rate"] == 0.0
        assert summary["mean_forecast_variance"] == 2e-4
        assert summary["mean_amfc_pick_forecast_variance"] == 5e-5
        assert summary["amfc_picks_more_certain_than_average"] is True

    def test_summarize_multiple_records_agreement_rate(self):
        """Agreement rate is the fraction of records where amfc_idx == hv_dm_idx."""
        # 2/5 agree, 3/5 disagree → 0.4
        records = [
            _make_record(amfc_idx=0, hv_dm_idx=0),  # agree
            _make_record(amfc_idx=0, hv_dm_idx=1),  # disagree
            _make_record(amfc_idx=2, hv_dm_idx=2),  # agree
            _make_record(amfc_idx=3, hv_dm_idx=1),  # disagree
            _make_record(amfc_idx=4, hv_dm_idx=2),  # disagree
        ]
        summary = summarize_telemetry(records)
        assert summary["n_periods"] == 5
        assert abs(summary["agreement_rate"] - 0.4) < 1e-12

    def test_summarize_tie_break_rate(self):
        """tie_break_fire_rate is the fraction with tie_break_fired=True."""
        records = [
            _make_record(tie_break_fired=True),
            _make_record(tie_break_fired=False),
            _make_record(tie_break_fired=True),
            _make_record(tie_break_fired=False),
        ]
        summary = summarize_telemetry(records)
        assert abs(summary["tie_break_fire_rate"] - 0.5) < 1e-12

    def test_amfc_picks_more_certain_when_lower_variance(self):
        """When AMFC pick variance < mean variance on average → flag True."""
        records = [
            _make_record(
                mean_forecast_variance=1e-3, amfc_pick_forecast_variance=1e-5
            ),
            _make_record(
                mean_forecast_variance=2e-3, amfc_pick_forecast_variance=5e-5
            ),
            _make_record(
                mean_forecast_variance=5e-4, amfc_pick_forecast_variance=2e-5
            ),
        ]
        summary = summarize_telemetry(records)
        assert summary["amfc_picks_more_certain_than_average"] is True
        assert (
            summary["mean_amfc_pick_forecast_variance"]
            < summary["mean_forecast_variance"]
        )

    def test_amfc_picks_more_uncertain_returns_false(self):
        """When AMFC pick variance > mean variance on average → flag False."""
        records = [
            _make_record(
                mean_forecast_variance=1e-5, amfc_pick_forecast_variance=1e-3
            ),
            _make_record(
                mean_forecast_variance=2e-5, amfc_pick_forecast_variance=2e-3
            ),
        ]
        summary = summarize_telemetry(records)
        assert summary["amfc_picks_more_certain_than_average"] is False


class TestFormatSummaryMarkdown:
    """format_summary_markdown contract."""

    def test_format_summary_markdown_returns_string_with_keys(self):
        """Returned markdown contains a header and rows for each canonical key."""
        records = [
            _make_record(amfc_idx=0, hv_dm_idx=1),
            _make_record(amfc_idx=1, hv_dm_idx=1),
        ]
        summary = summarize_telemetry(records)
        md = format_summary_markdown(summary)
        assert isinstance(md, str)
        assert "## AMFC-vs-Hv-DM Telemetry Summary" in md
        # Every numeric canonical key appears in the table (column 1).
        for k in ("n_periods", "agreement_rate", "roi_delta_mean", "tie_break_fire_rate"):
            assert k in md, f"expected key {k!r} in markdown"
        # YES/NO rendering for the boolean flag.
        assert ("YES" in md) or ("NO" in md)

    def test_format_summary_markdown_empty_returns_placeholder(self):
        """Empty summary renders an explicit no-data placeholder."""
        summary = summarize_telemetry([])
        md = format_summary_markdown(summary)
        assert "No telemetry records" in md


class TestTelemetryIO:
    """save_telemetry_to_json + load_telemetry_from_json roundtrip."""

    def test_save_and_load_telemetry_roundtrip(self, tmp_path):
        """Records persisted then loaded compare equal."""
        records = [
            _make_record(amfc_idx=0, hv_dm_idx=1),
            _make_record(amfc_idx=2, hv_dm_idx=2, tie_break_fired=True),
        ]
        out = tmp_path / "nested" / "telemetry.json"
        save_telemetry_to_json(records, out)
        assert out.exists(), "save did not create the file"
        loaded = load_telemetry_from_json(out)
        assert loaded == records


class TestCompareSummaries:
    """compare_telemetry_summaries contract."""

    def test_compare_summaries_returns_diff_markdown(self):
        """Comparison markdown contains both labels and a Δ column."""
        summary_a = summarize_telemetry(
            [_make_record(amfc_idx=0, hv_dm_idx=0)]
        )
        summary_b = summarize_telemetry(
            [
                _make_record(amfc_idx=0, hv_dm_idx=1),
                _make_record(amfc_idx=2, hv_dm_idx=3),
            ]
        )
        md = compare_telemetry_summaries(summary_a, summary_b, "A_run", "B_run")
        assert isinstance(md, str)
        assert "A_run" in md
        assert "B_run" in md
        assert "Δ (B − A)" in md
        # n_periods delta is +1 (2 - 1)
        assert "+1" in md
