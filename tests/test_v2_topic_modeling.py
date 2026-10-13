import json

import numpy as np

from topic_modeling import (
    TopicConfig,
    build_report,
    build_units,
    contiguous_runs,
    group_broad_topics,
    load_episodes,
    top_terms,
)


def segment(start, end, text, speaker="S1"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def test_build_units_merges_segments_into_windows():
    segments = [
        segment(0, 10, "a", "S1"),
        segment(10, 20, "b", "S2"),
        segment(40, 50, "c", "S1"),
    ]
    windows, warnings = build_units(segments, window_s=15, min_window_s=1, max_chars=1000)

    assert len(windows) == 2
    assert windows[0]["start"] == 0 and windows[0]["end"] == 20
    assert windows[0]["speakers"] == ["S1", "S2"]
    assert windows[0]["source_segment_indices"] == [0, 1]
    assert warnings == []


def test_build_units_drops_invalid_times_and_reports_missing_text():
    segments = [
        segment(0, 5, "ok"),
        {"start": 6, "end": 6, "text": "zero", "speaker": "S1"},
        {"start": 7, "end": 9, "text": "   ", "speaker": "S1"},
    ]
    windows, warnings = build_units(segments, window_s=60, min_window_s=1, max_chars=1000)

    assert len(windows) == 1
    codes = {warning["code"] for warning in warnings}
    assert "invalid_segment_time_excluded" in codes
    assert "missing_transcript_text" in codes


def test_contiguous_runs_merges_adjacent_same_topic():
    assignments = [
        {"start": 0, "end": 5, "speakers": ["A"], "subtopic_id": "t1"},
        {"start": 5, "end": 10, "speakers": ["B"], "subtopic_id": "t1"},
        {"start": 10, "end": 15, "speakers": ["A"], "subtopic_id": "t2"},
    ]
    runs = contiguous_runs(assignments, "subtopic_id")

    assert [run["value"] for run in runs] == ["t1", "t2"]
    assert runs[0]["unit_count"] == 2 and runs[0]["duration_s"] == 10
    assert runs[0]["speakers"] == ["A", "B"]


def test_load_episodes_filters_language_and_surfaces_bad_files(tmp_path):
    videos = tmp_path / "videos"
    for name in ("english", "hindi", "broken", "missing"):
        (videos / name).mkdir(parents=True)
    (videos / "english" / "final.json").write_text(json.dumps({
        "episode": {"video_id": "english", "duration_s": 10},
        "pipeline": {"asr": {"language": "en"}},
        "segments": [],
    }), encoding="utf-8")
    (videos / "hindi" / "final.json").write_text(json.dumps({
        "episode": {"video_id": "hindi", "duration_s": 10},
        "pipeline": {"asr": {"language": "hi"}},
        "segments": [],
    }), encoding="utf-8")
    (videos / "broken" / "final.json").write_text("{broken", encoding="utf-8")

    loaded, skipped, failed = load_episodes(videos, "en")

    assert [video_id for video_id, _ in loaded] == ["english"]
    skipped_reasons = {entry["reason"] for entry in skipped}
    assert "language_filtered" in skipped_reasons and "missing_final_json" in skipped_reasons
    assert [entry["reason"] for entry in failed] == ["invalid_json"]


def test_group_broad_topics_merges_similar_base_topics():
    embeddings = np.array([
        [1.0, 0.0, 0.0], [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.9, 0.1],
        [0.0, 0.0, 1.0], [0.0, 0.1, 0.9],
        [1.0, 0.1, 0.0], [0.8, 0.0, 0.2],
    ], dtype="float32")
    topics = [0, 0, 1, 1, 2, 2, 3, 3]

    mapping = group_broad_topics(embeddings, topics, [0, 1, 2, 3], 2)

    assert set(mapping) == {0, 1, 2, 3}
    assert len(set(mapping.values())) == 2
    assert mapping[0] == mapping[3]
    assert group_broad_topics(embeddings, topics, [0, 1], 8) == {0: 0, 1: 1}


def test_top_terms_prefers_distinctive_vocabulary():
    terms = top_terms(["court election supreme court ruling", "court election ruling"])

    assert "court" in terms


def test_build_report_stops_when_corpus_too_small(tmp_path):
    videos = tmp_path / "videos"
    good = videos / "solo"
    good.mkdir(parents=True)
    (good / "final.json").write_text(json.dumps({
        "episode": {"video_id": "solo", "duration_s": 60},
        "pipeline": {"asr": {"language": "en"}},
        "segments": [segment(0, 10, "hello there"), segment(10, 20, "general kenobi")],
    }), encoding="utf-8")

    report = build_report(videos, TopicConfig(min_topic_size=8))

    assert report["aggregate"]["processing"]["units"] == 0
    assert report["warnings"][0]["code"] == "insufficient_corpus"
    assert "candidate_labels_require_review" not in {w["code"] for w in report["warnings"]}
