import copy
import json

import discourse_summary
from discourse_review import (
    build_review,
    metric_rows,
    rederive,
    review_episode,
    validate_overlaps,
    DEFAULT_THRESHOLDS,
)


def make_final():
    return {
        "schema_version": "x",
        "episode": {"video_id": "v", "title": "t", "duration_s": 100.0},
        "participants": [{"name": "a"}],
        "segments": [{"start": 4.0, "end": 8.0, "text": "yes", "speaker": "S1",
                      "words": [{"word": "yes", "start": 5.0, "end": 6.0, "speaker": "S1"}]}],
        "speaker_turns": [{"speaker": "S1", "start": 0, "end": 10},
                          {"speaker": "S2", "start": 10, "end": 20},
                          {"speaker": "S1", "start": 20, "end": 30}],
        "overlap_candidates": [{"start": 5, "end": 8, "speakers": ["S1", "S2"], "case": "AGREE", "source": "A"},
                               {"start": 6, "end": 7, "speakers": ["S1", "S2"], "case": "OSD_ONLY", "source": "B"}],
        "interruptions": [{"start": 10, "end": 12, "interrupter": "S2", "speakers": ["S1", "S2"]}],
    }


def write_video(tmp_path, name, data):
    directory = tmp_path / "videos" / name
    directory.mkdir(parents=True)
    (directory / "final.json").write_text(json.dumps(data), encoding="utf-8")
    return directory


def test_rederive_hand_computable_values():
    metrics, validation = rederive(make_final(), DEFAULT_THRESHOLDS)
    assert metrics["episode_duration_s"] == 100.0
    assert metrics["number_of_speakers"] == 2
    assert metrics["total_valid_turns"] == 3
    assert metrics["total_speaker_transitions"] == 2
    assert metrics["overlap_candidate_count"] == 2
    assert metrics["unique_elapsed_overlap_s"] == 3.0
    assert metrics["cumulative_candidate_overlap_s"] == 4.0
    assert metrics["interruption_attempt_count"] == 1
    assert metrics["successful_interruption_count"] == 1
    assert metrics["failed_interruption_count"] == 0
    assert metrics["brief_backchannel_count"] == 1
    assert metrics["prolonged_overlap_count"] == 1
    assert validation["overlap_candidates"]["nested"] == 1


def test_metric_rows_match_a_consistent_summary(tmp_path):
    write_video(tmp_path, "v", make_final())
    summary = discourse_summary.build_report(tmp_path / "videos")
    review = build_review(tmp_path / "videos", summary)
    episode = review["episodes"][0]
    assert episode["cross_check"]["mismatched_metrics"] == []
    assert episode["cross_check"]["metrics_compared"] == episode["cross_check"]["matched"]
    assert episode["cross_check"]["metrics_compared"] >= 18


def test_tampered_summary_is_detected_as_mismatch(tmp_path):
    write_video(tmp_path, "v", make_final())
    summary = discourse_summary.build_report(tmp_path / "videos")
    tampered = copy.deepcopy(summary)
    tampered["episodes"][0]["switching"]["total_speaker_transitions"] = 999
    review = build_review(tmp_path / "videos", tampered)
    episode = review["episodes"][0]
    assert "total_speaker_transitions" in episode["cross_check"]["mismatched_metrics"]
    assert any(item["code"] == "metric_mismatch" for item in episode["warnings"])


def test_structural_issues_are_reported():
    data = make_final()
    data["overlap_candidates"] += [
        {"start": 50, "end": 50, "speakers": ["S1"], "case": "AGREE", "source": "A"},
        {"start": 5, "end": 8, "speakers": ["S1", "S2"], "case": "AGREE", "source": "A"},
    ]
    data["interruptions"] += [{"start": 10, "end": 12, "interrupter": "S2", "speakers": ["S1", "S2"]}]
    data["speaker_turns"] = [{"speaker": "S2", "start": 40, "end": 50}, {"speaker": "S1", "start": 0, "end": 10}]
    _, validation = rederive(data, DEFAULT_THRESHOLDS)
    assert validation["overlap_candidates"]["zero_duration"] == 1
    assert validation["overlap_candidates"]["duplicates"] == 1
    assert validation["overlap_candidates"]["out_of_order"] is True
    assert validation["interruptions"]["duplicates"] == 1


def test_validate_overlaps_counts_nested_and_examples():
    raw = [{"start": 0, "end": 10, "speakers": ["A", "B"], "case": "AGREE", "source": "s"},
           {"start": 2, "end": 3, "speakers": ["A", "B"], "case": "OSD_ONLY", "source": "s"},
           {"start": 20, "end": 30, "speakers": ["A", "B"], "case": "AGREE", "source": "s"}]
    _, diagnostics = validate_overlaps(raw)
    assert diagnostics["nested"] == 1
    assert len(diagnostics["nested_examples"]) == 1
    assert diagnostics["nested_examples"][0]["inner"]["start"] == 2


def test_semantic_metrics_marked_unavailable():
    rows = metric_rows({}, {})
    semantic = {row["metric"]: row for row in rows if row["metric"].endswith("overlap_rate_pct")}
    assert semantic and all(row["availability"] == "unavailable_by_design" for row in semantic.values())


def test_episode_without_summary_entry_is_not_reviewed(tmp_path):
    write_video(tmp_path, "v", make_final())
    review = build_review(tmp_path / "videos", {"episodes": [], "report_metadata": {}})
    episode = review["episodes"][0]
    assert episode["status"] == "not_in_summary"
    assert all(row["availability"] == "not_reported" for row in episode["metrics"] if row["rederived_value"] is not None)


def test_missing_final_json_is_listed_not_counted(tmp_path):
    write_video(tmp_path, "v", make_final())
    (tmp_path / "videos" / "empty").mkdir(parents=True)
    review = build_review(tmp_path / "videos", discourse_summary.build_report(tmp_path / "videos"))
    assert review["aggregate"]["processing"] == {"reviewed": 1, "not_reviewed": 1}
    assert review["not_reviewed"][0]["reason"] == "missing_final_json"


def test_review_episode_is_deterministic(tmp_path):
    write_video(tmp_path, "v", make_final())
    summary = discourse_summary.build_report(tmp_path / "videos")
    reported = summary["episodes"][0]
    a = review_episode("v", make_final(), reported, DEFAULT_THRESHOLDS)
    b = review_episode("v", make_final(), reported, DEFAULT_THRESHOLDS)
    assert a == b
