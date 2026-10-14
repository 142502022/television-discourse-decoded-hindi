from topic_drift import (
    DriftConfig,
    active_index,
    analyze_episode,
    baseline_comparison,
    boundary_after,
    build_report,
    load_interruptions,
    timeline_runs,
    union_intervals,
    window_key,
)


def run(value, start, end, prob=0.9):
    return {"value": value, "start": start, "end": end, "probability_mean": prob,
            "unit_count": 1, "speakers": ["S1"]}


def report_for(video_id, runs):
    return {
        "subtopics": [{"subtopic_id": value, "candidate_label": f"label {value}"} for value in {r["value"] for r in runs}],
        "broad_topics": [{"broad_topic_id": "broad_000", "candidate_label": "broad label"}],
        "episodes": [{"metadata": {"video_id": video_id, "title": "t", "duration_s": 1000.0},
                      "subtopic_timeline": runs,
                      "broad_timeline": [run("broad_000", runs[0]["start"], runs[-1]["end"])]}],
    }


def final_for(events):
    return {"interruptions": events, "segments": [{"start": 0, "end": 5, "text": "hello world"}]}


def test_union_intervals_merges_overlaps_and_adjacent():
    assert union_intervals([(0, 5), (4, 9), (20, 30)]) == [(0, 9), (20, 30)]
    assert union_intervals([(0, 5), (6, 9)], join_gap=1.0) == [(0, 9)]


def test_timeline_runs_filters_invalid_and_sorts():
    episode = {"subtopic_timeline": [run("b", 10, 20), run("a", 0, 10), "junk",
                                     {"value": "x", "start": 5, "end": 5}]}
    runs = timeline_runs(episode, "subtopic_timeline")
    assert [item["value"] for item in runs] == ["a", "b"]


def test_active_index_inside_run_and_gap_guard():
    runs = timeline_runs({"subtopic_timeline": [run("a", 0, 10), run("b", 20, 30)]}, "subtopic_timeline")
    assert active_index(runs, 5, 60.0) == 0
    assert active_index(runs, 15, 60.0) == 0
    assert active_index(runs, 15, 2.0) is None
    assert active_index(runs, -5, 60.0) is None


def test_boundary_after_requires_different_value_within_horizon():
    runs = timeline_runs({"subtopic_timeline": [run("a", 0, 40), run("b", 40, 80), run("c", 80, 200)]}, "subtopic_timeline")
    assert boundary_after(runs, 30, "a", 45.0)["value"] == "b"
    assert boundary_after(runs, 30, "a", 5.0) is None
    assert boundary_after(runs, 79, "b", 45.0)["value"] == "c"


def test_load_interruptions_dedups_and_drops_invalid():
    events, duplicates = load_interruptions({"interruptions": [
        {"start": 5, "end": 7, "interrupter": "A"},
        {"start": 5.0, "end": 7.0, "interrupter": "A"},
        {"start": 7, "end": 5},
        {"start": 20, "end": 25, "interrupter": "B"},
    ]})
    assert [event["interrupter"] for event in events] == ["A", "B"]
    assert duplicates == 1


def test_baseline_comparison_excludes_interruption_windows():
    runs = timeline_runs({"subtopic_timeline": [run("a", 0, 45), run("b", 45, 90), run("c", 90, 1000)]}, "subtopic_timeline")
    cfg = DriftConfig(outcome_windows=(45.0,), match_seed=7)
    comparison = baseline_comparison(runs, [{"start": 0, "end": 45, "speakers": []}], 200.0, 45.0, cfg)
    assert comparison["interruption"]["events_with_baseline"] == 1
    assert comparison["interruption"]["rate"] == 1.0
    assert comparison["non_interruption"]["windows"] >= 1
    assert isinstance(comparison["lift"], float)


def test_analyze_episode_assesses_and_flags_context():
    runs = timeline_runs({"subtopic_timeline": [run("a", 0, 45), run("b", 45, 200)]}, "subtopic_timeline")
    report = report_for("v", [r for r in runs])
    report["episodes"][0]["subtopic_timeline"] = runs
    episode = analyze_episode("v", report, final_for([
        {"start": 10, "end": 12, "interrupter": "A"},
        {"start": 5, "end": 6, "interrupter": "B"},
    ]), DriftConfig(outcome_windows=(45.0,)))
    assert episode["status"] == "analyzed"
    events = {item["interrupter"]: item for item in episode["events"]}
    assert events["A"]["status"] == "assessed"
    assert events["A"]["baseline"]["subtopic_id"] == "a"
    assert events["A"]["outcomes"][window_key(45.0)]["subtopic"]["change_detected"] is True
    assert events["A"]["outcomes"][window_key(45.0)]["subtopic"]["changed_to"]["topic_id"] == "b"


def test_analyze_episode_insufficient_timeline():
    report = {"subtopics": [], "broad_topics": [], "episodes": [{"metadata": {"video_id": "v", "duration_s": 100.0},
                                                                  "subtopic_timeline": [], "broad_timeline": []}]}
    episode = analyze_episode("v", report, final_for([{"start": 1, "end": 2}]), DriftConfig())
    assert episode["status"] == "insufficient_input"
    assert episode["events"] == []


def test_build_report_end_to_end(tmp_path):
    video = tmp_path / "videos" / "v"
    video.mkdir(parents=True)
    runs = [run("a", 0, 45), run("b", 45, 200)]
    modeling = tmp_path / "topic_modeling.json"
    modeling.write_text(__import__("json").dumps(report_for("v", runs)), encoding="utf-8")
    (video / "final.json").write_text(__import__("json").dumps(final_for([
        {"start": 10, "end": 12, "interrupter": "A"}])), encoding="utf-8")

    result = build_report(tmp_path / "videos", modeling, DriftConfig(outcome_windows=(45.0, 90.0)))
    assert result["schema_version"] == "interruption-topic-drift-1.0"
    assert result["aggregate"]["processing"] == {"analyzed": 1, "insufficient_input": 0, "failed": 0}
    assert result["aggregate"]["interruption_events"] == 1
    assert set(result["aggregate"]["comparisons"]) == {"45", "90"}
    assert result["episodes"][0]["comparisons"]["45"]["subtopic"]["lift"] is not None
