import numpy as np

from topic_modeling import (
    TopicConfig,
    build_review_document,
    build_window_records,
    build_windows_document,
    contiguous_runs,
    intersection_seconds,
    simple_tokens,
    snippet_around,
)


def record(index, start, end, text, **overrides):
    base = {
        "index": index, "video_id": "v", "start": start, "end": end,
        "duration_s": end - start, "speakers": ["A", "B"], "text": text,
    }
    base.update(overrides)
    return base


def test_intersection_seconds_basic_and_invalid():
    assert intersection_seconds(0, 10, 5, 15) == 5
    assert intersection_seconds(0, 10, 10, 20) == 0
    assert intersection_seconds(None, 10, 0, 5) == 0.0
    assert intersection_seconds(10, 5, 0, 1) == 0.0


def test_simple_tokens_and_snippet():
    assert simple_tokens("Did they speak, up?") == ["did", "they", "speak", "up"]
    long_text = ("the committee debated the budget for many hours before the "
                 "court finally decided the matter")
    snippet = snippet_around(long_text, "court", before=10)
    assert "court" in snippet and snippet.startswith("…")
    assert snippet_around("short court text", "court") == "short court text"


def test_build_window_records_attaches_topics_quality_and_hits():
    index = [{
        "video_id": "v", "start": 120, "end": 150, "speakers": ["A", "B"],
        "text": "a debate sentence", "source_segment_indices": [0, 1],
    }]
    records = build_window_records(
        index, topics=[3], probabilities=[0.8],
        topic_to_subtopic={3: "subtopic_001"}, broad_of_subtopic={"subtopic_001": "broad_000"},
        episode_segments={"v": [{"start": 120, "end": 130}, {"start": 130, "end": 150}]},
        episode_overlaps={"v": [{"start": 125, "end": 140}, {"start": 400, "end": 401}]},
        episode_interruptions={"v": [{"start": 130, "end": 135}]},
        min_window_seconds=12.0,
    )

    assert len(records) == 1
    row = records[0]
    assert row["subtopic_id"] == "subtopic_001" and row["broad_topic_id"] == "broad_000"
    assert row["probability"] == 0.8
    assert row["overlap"]["count"] == 1 and row["overlap"]["indices"] == [0]
    assert row["overlap"]["total_seconds"] == 15.0
    assert row["interruption"]["count"] == 1
    assert "no_confidence_signals" in row["quality"]["flags"]
    assert row["dropped_reason"] is None


def test_build_window_records_flags_short_single_speaker_window():
    index = [{
        "video_id": "v", "start": 0, "end": 5, "speakers": ["A"],
        "text": "hi", "source_segment_indices": [0],
    }]
    records = build_window_records(
        index, topics=[-1], probabilities=[0.0],
        topic_to_subtopic={}, broad_of_subtopic={},
        episode_segments={"v": [{"start": 0, "end": 5, "avg_logprob": -2.5}]},
        episode_overlaps={"v": []}, episode_interruptions={"v": []},
        min_window_seconds=12.0,
    )
    row = records[0]
    assert row["subtopic_id"] == "subtopic_outlier" and row["broad_topic_id"] == "broad_outlier"
    assert {"short_window", "single_speaker", "low_mean_logprob"} <= set(row["quality"]["flags"])


def test_build_windows_document_shape():
    doc = build_windows_document(
        [record(0, 0, 10, "text")], [{"reason": "ad_trim", "start": 0, "end": 720, "video_id": "v"}],
        config=TopicConfig(), model_meta={"model": "m"}, video_ids=["v"],
    )
    assert doc["schema_version"] == "topic-windows-1.0"
    assert doc["window_count"] == 1 and doc["video_ids"] == ["v"]
    assert doc["dropped_regions"][0]["reason"] == "ad_trim"


def _subtopic(subtopic_id, broad_id, base, label, terms, example):
    return {"subtopic_id": subtopic_id, "broad_topic_id": broad_id, "base_topic_id": base,
            "candidate_label": label, "evidence_terms": terms, "unit_count": 3,
            "representative_example": example}


def test_build_review_document_picks_central_example_not_first_member():
    records = [
        record(0, 0, 10, "unrelated opening chatter"),
        record(1, 10, 20, "the court will decide the election dispute"),
        record(2, 20, 30, "election court ruling expected soon"),
    ]
    embeddings = np.array([[0.0, 1.0], [1.0, 0.05], [0.9, 0.1]], dtype="float32")
    topics = [0, 0, 0]
    subtopics = [_subtopic("subtopic_001", "broad_000", 0, "court · election", ["court", "election"], "first member text")]
    broad_topics = [{"broad_topic_id": "broad_000", "candidate_label": "court", "evidence_terms": ["court"],
                     "subtopic_ids": ["subtopic_001"], "unit_count": 3}]

    review = build_review_document(subtopics, broad_topics, [], records, embeddings, topics)

    central = review["subtopics"][0]["central_example"]
    assert central["window_index"] == 1  # nearest centroid, not members[0] (=0)
    assert central["text"].startswith("the court")
    grounding_terms = {g["term"] for g in review["subtopics"][0]["grounding_examples"]}
    assert grounding_terms == {"court", "election"}
    assert review["hierarchy_diagnostics"]["flat_hierarchy"] is True
    assert any(w["code"] == "flat_hierarchy_no_gain" for w in review["warnings"])


def test_build_review_document_flags_generic_terms():
    records = [record(0, 0, 10, "you know the economy"), record(1, 10, 20, "the economy grows")]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1]], dtype="float32")
    topics = [0, 1]
    subtopics = [
        _subtopic("subtopic_001", "broad_000", 0, "know · economy", ["know", "economy"], "a"),
        _subtopic("subtopic_002", "broad_001", 1, "know · growth", ["know", "growth"], "b"),
    ]
    broad_topics = [
        {"broad_topic_id": "broad_000", "candidate_label": "a", "evidence_terms": ["know"], "subtopic_ids": ["subtopic_001"], "unit_count": 1},
        {"broad_topic_id": "broad_001", "candidate_label": "b", "evidence_terms": ["know"], "subtopic_ids": ["subtopic_002"], "unit_count": 1},
    ]

    review = build_review_document(subtopics, broad_topics, [], records, embeddings, topics)

    know = next(entry for entry in review["stopword_analysis"] if entry["term"] == "know")
    assert know["generic"] is True and know["subtopic_count"] == 2
    assert any(w["code"] == "generic_label_terms_present" for w in review["warnings"])


def test_contiguous_runs_aggregates_probability():
    assignments = [
        {"start": 0, "end": 5, "speakers": ["A"], "subtopic_id": "t1", "probability": 0.2},
        {"start": 5, "end": 10, "speakers": ["A"], "subtopic_id": "t1", "probability": 0.6},
        {"start": 10, "end": 15, "speakers": ["B"], "subtopic_id": "t2", "probability": None},
    ]
    runs = contiguous_runs(assignments, "subtopic_id")

    assert runs[0]["probability_mean"] == 0.4
    assert runs[0]["probability_min"] == 0.2 and runs[0]["probability_max"] == 0.6
    assert "probability_mean" not in runs[1]
