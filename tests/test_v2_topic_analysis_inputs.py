import json

from topic_analysis import TopicConfig, analyze_episode, build_report, extract_units


def seg(start, end, text, speaker="S1"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def codes(warnings):
    return {item["code"] for item in warnings}


def test_extract_units_missing_key():
    units, warnings, diagnostics = extract_units({})
    assert units == []
    assert codes(warnings) == {"no_segments_key"}
    assert diagnostics["segments_present"] is False
    assert diagnostics["valid_unit_count"] == 0


def test_extract_units_empty_list():
    units, warnings, diagnostics = extract_units({"segments": []})
    assert units == []
    assert codes(warnings) == {"empty_segments"}
    assert diagnostics["segments_present"] is True
    assert diagnostics["raw_segment_count"] == 0


def test_extract_units_non_list_container():
    units, warnings, diagnostics = extract_units({"segments": {"0": {}}})
    assert units == []
    assert codes(warnings) == {"invalid_segments_container"}
    assert diagnostics["segments_is_list"] is False


def test_extract_units_counts_rejections_by_reason():
    data = {"segments": [
        seg(0, 5, "good one"),
        "not a dict",
        {"start": 5, "end": 10, "text": "   "},
        {"start": 10, "end": 10, "text": "zero length"},
        seg(12, 20, "good two"),
    ]}
    units, warnings, diagnostics = extract_units(data)
    assert [unit["text"] for unit in units] == ["good one", "good two"]
    assert diagnostics["raw_segment_count"] == 5
    assert diagnostics["valid_unit_count"] == 2
    assert diagnostics["rejected"] == {"not_a_dict": 1, "missing_text": 1, "invalid_time": 1}
    assert codes(warnings) == {"invalid_segment_record", "missing_transcript_text", "invalid_segment_time_excluded"}


def test_extract_units_all_invalid_reports_specific_code():
    data = {"segments": [{"start": 0, "end": 0, "text": "x"}, {"start": 2, "end": 1, "text": "y"}]}
    units, warnings, diagnostics = extract_units(data)
    assert units == []
    assert "all_segments_invalid" in codes(warnings)
    assert diagnostics["valid_unit_count"] == 0


def test_extract_units_sorts_out_of_order_and_warns():
    data = {"segments": [seg(20, 30, "later"), seg(0, 10, "earlier")]}
    units, warnings, _ = extract_units(data)
    assert [unit["text"] for unit in units] == ["earlier", "later"]
    assert "segments_out_of_order" in codes(warnings)


def test_analyze_episode_insufficient_input_is_not_analyzed():
    episode = analyze_episode("v", {"segments": [], "episode": {"video_id": "v", "duration_s": 100}},
                              TopicConfig())
    assert episode["status"] == "insufficient_input"
    assert episode["units"] == [] and episode["broad_topics"] == []
    assert episode["input_diagnostics"]["raw_segment_count"] == 0
    assert "empty_segments" in codes(episode["warnings"])


def test_analyze_episode_single_unit_uses_generic_warning():
    data = {"segments": [seg(0, 5, "only one")], "episode": {"video_id": "v", "duration_s": 100}}
    episode = analyze_episode("v", data, TopicConfig())
    assert episode["status"] == "insufficient_input"
    assert "insufficient_transcript_context" in codes(episode["warnings"])


def test_build_report_separates_insufficient_input_from_analyzed(tmp_path):
    videos = tmp_path / "videos"
    empty = videos / "empty"
    empty.mkdir(parents=True)
    (empty / "final.json").write_text(json.dumps({
        "episode": {"video_id": "empty", "title": "t", "duration_s": 7110.0},
        "pipeline": {"asr": {"language": "en"}},
        "segments": [],
    }), encoding="utf-8")

    report = build_report(videos, TopicConfig(), all_episodes=True)

    processing = report["aggregate"]["processing"]
    assert processing["analyzed"] == 0
    assert processing["insufficient_input"] == 1
    assert report["aggregate"]["topic_segments"] == {"broad_total": 0, "subtopic_total": 0}
    assert report["aggregate"]["shift_events"] == {"total": 0, "with_observed_interruption": 0}
    assert report["episodes"][0]["status"] == "insufficient_input"
