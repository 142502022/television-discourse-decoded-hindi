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


def test_each_stage_resolves_its_own_host_json_and_calculates_host_metrics():
    data = dashboard.prepare_dashboard_data(ROOT / "data/v2/debate_only/videos")
    for video in data["videos"]:
        for stage in video["stages"].values():
            host = stage["host"]
            assert host["speaker_id"]
            assert host["host_id_matches_video"] is True
            assert host["host_found_in_analysis"] is True
            assert host["original_share_percent"] is not None
            assert host["clean_share_percent"] is not None
            assert host["longest_seconds"] is not None
            assert host["mean_seconds"] is not None


def test_host_turn_metrics_match_source_turns_for_two_videos():
    data = dashboard.prepare_dashboard_data(ROOT / "data/v2/debate_only/videos")
    for video_id in ("Cp9kTDGfJk4", "dFlv0bkxiLI"):
        video = next(row for row in data["videos"] if row["video_id"] == video_id)
        host = video["stages"]["baseline"]["host"]
        final = dashboard.load(ROOT / "data/v2/debate_only/videos" / video_id / "baseline" / "final.json", [], "test")
        expected = dashboard.host_turn_stats(final, host["speaker_id"])
        assert host["longest_seconds"] == expected["longest_seconds"]
        assert host["mean_seconds"] == expected["mean_seconds"]


def test_pairwise_metrics_validate_all_videos_and_stages():
    data = dashboard.prepare_dashboard_data(ROOT / "data/v2/debate_only/videos")
    for video in data["videos"]:
        for stage in video["stages"].values():
            item = stage["pairwise_transitions"]
            assert item["data_quality"]["host_mapping_available"]
            assert item["data_quality"]["category_total_matches"]
            assert item["data_quality"]["matrix_total_matches"]
            assert item["total_valid_transitions"] == item["host_to_guest_count"] + item["guest_to_host_count"] + item["guest_to_guest_count"]
            for speaker, row in item["pairwise_transition_counts"].items():
                assert row[speaker] == 0
                normalized = item["pairwise_transition_percentages"][speaker]
                if sum(row.values()):
                    assert abs(sum(value for value in normalized.values() if value is not None) - 100) < 0.01
                else:
                    assert all(value is None for value in normalized.values())


def test_pairwise_synthetic_records_cover_categories_and_invalid_data():
    final = {"episode": {"duration_s": 120}, "speaker_turns": [
        {"speaker": "HOST", "start": 0, "end": 1}, {"speaker": "A", "start": 2, "end": 3},
        {"speaker": "B", "start": 4, "end": 5}, {"speaker": "QUIET", "start": 6, "end": 7},
    ]}
    transitions = {"transition_analysis": {"confirmed_transitions": [
        {"from_speaker": "HOST", "to_speaker": "A", "timestamp": 2, "section_index": 1},
        {"from_speaker": "A", "to_speaker": "HOST", "timestamp": 3, "section_index": 2},
        {"from_speaker": "A", "to_speaker": "B", "timestamp": 4, "section_index": 3},
        {"from_speaker": "A", "to_speaker": "B", "timestamp": 4, "section_index": 3},
        {"from_speaker": "B", "to_speaker": "B", "timestamp": 5, "section_index": 4},
    ]}}
    item = dashboard.pairwise_transition_analysis(transitions, final, "HOST", "0.5")
    assert item["total_valid_transitions"] == 3
    assert (item["host_to_guest_count"], item["guest_to_host_count"], item["guest_to_guest_count"]) == (1, 1, 1)
    assert item["overall_transition_rate_per_minute"] == 1.5
    assert item["pairwise_transition_percentages"]["QUIET"]["HOST"] is None
    assert item["data_quality"]["excluded_record_count"] == 2
    assert dashboard.pairwise_transition_analysis(transitions, final, None, "0.5")["host_to_guest_count"] is None
