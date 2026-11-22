"""Regression checks for the read-only multi-video dashboard aggregator."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_multi_video_dashboard", ROOT / "analysis" / "build_multi_video_dashboard.py"
)
dashboard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(dashboard)


def test_dashboard_data_has_all_selectable_stages_and_fixed_video_population():
    data = dashboard.prepare_dashboard_data(ROOT / "data/v2/debate_only/videos")
    assert data["thresholds"] == ["baseline", "0.5", "1.0", "2.0", "5.0"]
    assert data["quality"]["discovered"] == 12
    assert data["quality"]["loaded"] == 12
    for video in data["videos"]:
        assert set(video["stages"]) == set(data["thresholds"])
        assert all(stage["available"] for stage in video["stages"].values())


def test_speaker_counts_are_original_stage_counts_and_clean_shares_normalize():
    data = dashboard.prepare_dashboard_data(ROOT / "data/v2/debate_only/videos")
    video = next(video for video in data["videos"] if video["video_id"] == "dFlv0bkxiLI")
    # This verifies the source artifact rather than retaining the old report's count of six.
    assert video["overview"]["speaker_count"] == 8
    for stage in video["stages"].values():
        shares = stage["speaker_shares"].values()
        assert abs(sum(row["original_percent"] for row in shares) - 100.0) < 0.01
        assert abs(sum(row["clean_percent"] for row in shares) - 100.0) < 0.01


def test_heuristic_hosts_are_not_mislabeled_as_manually_verified():
    data = dashboard.prepare_dashboard_data(ROOT / "data/v2/debate_only/videos")
    for video in data["videos"]:
        for stage in video["stages"].values():
            assert stage["host"]["speaker_id"] is None
            assert stage["host"]["original_share_percent"] is None
