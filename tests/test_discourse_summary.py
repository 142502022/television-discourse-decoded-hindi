import json

from discourse_summary import (
    Thresholds,
    build_report,
    classify_events,
    floor_outcome,
    interruption_metrics,
    merge_turns,
    overlap_metrics,
    participation,
    switching,
    union_duration,
    validate_overlaps,
)


def turn(speaker, start, end):
    return {"speaker": speaker, "start": start, "end": end}


def test_participation_merging_switching_entropy_and_short_turns():
    raw = [turn("A", 0, 1), turn("A", 1.2, 2), turn("B", 2, 2.5), turn("A", 2.5, 4)]
    merged = merge_turns(raw, 0.5)
    result = participation(raw, merged)
    switch = switching(merged, 60, Thresholds())

    assert result["by_speaker"]["A"]["raw_turn_count"] == 3
    assert result["by_speaker"]["A"]["merged_turn_count"] == 2
    assert result["number_of_speakers"] == 2
    assert switch["total_speaker_transitions"] == 2
    assert switch["directional_matrix"] == {"A": {"B": 1}, "B": {"A": 1}}
    assert switch["transition_entropy_bits"] == 1.0
    assert switch["short_turn_count"] == 1


def test_overlap_union_preserves_event_count_but_not_duplicate_elapsed_time():
    events, warnings = validate_overlaps([
        {"start": 1, "end": 4, "speakers": ["A", "B"], "case": "AGREE"},
        {"start": 1, "end": 4, "speakers": ["A", "B"], "case": "AGREE"},
        {"start": 2, "end": 5, "speakers": ["A", "C"], "case": "OSD_ONLY"},
    ])
    metrics = overlap_metrics(events, 60)

    assert warnings[0]["code"] == "duplicate_overlap_candidates"
    assert len(events) == 3
    assert union_duration(events) == 4
    assert metrics["candidate_event_count"] == 3
    assert metrics["unique_elapsed_overlap_s"] == 4.0
    assert metrics["cumulative_candidate_overlap_s"] == 9.0


def test_interruption_floor_outcomes_success_failed_and_ambiguous():
    config = Thresholds(floor_hold_s=1.0, floor_outcome_window_s=2.0)
    successful = {"start": 1, "end": 2, "interrupter": "B", "victims": ["A"]}
    failed = {"start": 1, "end": 2, "interrupter": "B", "victims": ["A"]}

    assert floor_outcome(successful, [turn("A", 0, 2.2), turn("B", 1, 4)], config) == "successful"
    assert floor_outcome(failed, [turn("A", 0, 4), turn("B", 1, 2.2)], config) == "failed"
    assert floor_outcome(failed, [turn("A", 0, 4), turn("B", 1, 4)], config) == "ambiguous"

    metrics, classified = interruption_metrics([successful, failed], [turn("A", 0, 4), turn("B", 1, 2.2)], 60, config)
    assert metrics["total_attempts"] == 2
    assert classified[0]["floor_outcome"] == "failed"


def test_semantic_metrics_are_explicitly_unavailable():
    result = classify_events(
        {"segments": []},
        [turn("A", 0, 1), turn("B", 1, 3)],
        [{"start": 1, "end": 2, "duration": 1, "speakers": ["A", "B"], "case": "AGREE"}],
        [], switching([turn("A", 0, 1), turn("B", 1, 3)], 10, Thresholds()), 10, Thresholds(),
    )

    assert result["agreement_overlap"]["count"] is None
    assert result["disagreement_overlap"]["count"] is None
    assert result["agreement_overlap"]["provenance"] == "unknown"


def test_report_tracks_missing_and_malformed_files(tmp_path):
    videos = tmp_path / "videos"
    (videos / "missing").mkdir(parents=True)
    malformed = videos / "bad"
    malformed.mkdir()
    (malformed / "final.json").write_text("{broken", encoding="utf-8")
    good = videos / "good"
    good.mkdir()
    (good / "final.json").write_text(json.dumps({
        "episode": {"video_id": "good", "duration_s": 10},
        "speaker_turns": [turn("A", 0, 2), turn("B", 2, 3)],
        "overlap_candidates": [],
        "interruptions": [],
        "participants": [],
        "segments": [],
    }), encoding="utf-8")

    report = build_report(videos)

    assert report["aggregate"]["processing"] == {"successfully_analyzed": 1, "skipped": 1, "failed": 1}
    assert report["episodes"][0]["metadata"]["video_id"] == "good"
    assert report["episodes"][0]["event_classification"]["agreement_overlap"]["count"] is None

