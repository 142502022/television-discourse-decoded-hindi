"""Tests for host identification (pure logic, no models, no episode hardcoding)."""

from v2.identity.host import (
    build_result,
    extract_features,
    has_address,
    infer_host,
    is_question,
    percentile_ranks,
    rank_candidates,
    repeated_phrases,
    tokenize,
)


def _turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def _segment(start, end, text, speaker):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


# ── text helpers ────────────────────────────────────────────────────────────
def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Yes or no?") == ["yes", "or", "no"]
    assert tokenize("Dr. Pandit's view") == ["dr", "pandit's", "view"]


def test_is_question_detects_marker_and_opener():
    assert is_question("Do you support this?")
    assert is_question("What is happening")  # opener without '?'
    assert not is_question("I agree with that")


def test_has_address_detects_second_person():
    assert has_address("Do you support this?")
    assert has_address("Sir, please answer")
    assert not has_address("The economy is growing")


def test_repeated_phrases_filters_stopwords_and_threshold():
    texts = ["do you support this", "do you support that", "do you support it"]
    phrases = repeated_phrases(texts, min_count=3)
    joined = {p["phrase"] for p in phrases}
    assert "do you support" in joined
    assert all(p["count"] >= 3 for p in phrases)


def test_percentile_ranks_handles_ties_and_singleton():
    ranks = percentile_ranks({"a": 1.0, "b": 1.0, "c": 3.0})
    assert ranks["a"] == ranks["b"]
    assert ranks["c"] == 1.0
    assert percentile_ranks({"only": 5.0}) == {"only": 0.5}


# ── feature extraction ──────────────────────────────────────────────────────
def test_extract_features_counts_turns_questions_addresses_interlocutors():
    turns = [
        _turn(0, 4, "HOST"), _turn(4, 8, "G1"), _turn(8, 12, "HOST"),
        _turn(12, 16, "G2"), _turn(16, 20, "HOST"),
    ]
    segments = [
        _segment(0, 4, "Do you support this?", "HOST"),
        _segment(4, 8, "I do not.", "G1"),
        _segment(8, 12, "And you, sir?", "HOST"),
        _segment(12, 16, "Neither do I.", "G2"),
    ]
    features = extract_features(segments, turns, [])
    host = features["HOST"]
    assert host["turn_count"] == 3
    assert host["question_count"] == 2
    assert host["address_count"] == 2
    assert host["distinct_interlocutor_count"] == 2  # yields to G1 and G2
    assert host["speech_seconds"] == 12.0


def test_extract_features_floor_regain_after_overlap():
    turns = [_turn(0, 3, "A"), _turn(3, 6, "B"), _turn(6, 9, "A")]
    overlaps = [{"start": 3, "end": 4, "speakers": ["A", "B"]}]
    features = extract_features([], turns, overlaps)
    assert features["A"]["floor_regains"] == 1
    assert features["B"]["floor_regains"] == 1


def test_extract_features_ignores_invalid_records():
    turns = [_turn(0, 1, "A"), {"start": 5, "end": 5, "speaker": "A"},
             {"start": 6, "end": 7, "speaker": ""}, "garbage"]
    features = extract_features([], turns, [])
    assert features["A"]["turn_count"] == 1


# ── ranking and prediction ──────────────────────────────────────────────────
def _moderator_episode():
    """A synthetic debate with a clear moderator and two panellists."""
    turns, segments = [], []
    t = 0.0
    for _ in range(6):
        turns.append(_turn(t, t + 3, "MODERATOR"))
        segments.append(_segment(t, t + 3, "Why do you say that, and how will you respond to this?", "MODERATOR"))
        t += 3
        turns.append(_turn(t, t + 3, "PANEL_A"))
        segments.append(_segment(t, t + 3, "I believe the policy is sound for our country.", "PANEL_A"))
        t += 3
        turns.append(_turn(t, t + 3, "MODERATOR"))
        segments.append(_segment(t, t + 3, "Do you support this, sir?", "MODERATOR"))
        t += 3
        turns.append(_turn(t, t + 3, "PANEL_B"))
        segments.append(_segment(t, t + 3, "No, I disagree with the whole approach.", "PANEL_B"))
        t += 3
    overlaps = [{"start": 2, "end": 4, "speakers": ["MODERATOR", "PANEL_A"]}]
    return segments, turns, overlaps


def test_infer_host_picks_moderator_not_just_longest_talker():
    segments, turns, overlaps = _moderator_episode()
    result = infer_host(segments, turns, overlaps)
    assert result["prediction"]["speaker_label"] == "MODERATOR"
    assert result["prediction"]["inferred_role"] == "host_moderator"
    assert result["prediction"]["uncertain"] in (True, False)  # band is coarse


def test_infer_host_reference_match_is_evaluation_only():
    segments, turns, overlaps = _moderator_episode()
    result = infer_host(segments, turns, overlaps, reference_label="MODERATOR")
    comparison = result["reference_comparison"]
    assert comparison["match"] is True
    assert comparison["reference_rank"] == 1

    result_wrong = infer_host(segments, turns, overlaps, reference_label="PANEL_A")
    assert result_wrong["reference_comparison"]["match"] is False
    # reference does not change the prediction
    assert result_wrong["prediction"]["speaker_label"] == "MODERATOR"


def test_infer_host_returns_uncertain_when_no_speakers():
    result = infer_host([], [], [])
    assert result["prediction"]["uncertain"] is True
    assert result["prediction"]["speaker_label"] is None


def test_infer_host_single_speaker_is_uncertain_band():
    segments = [_segment(0, 5, "Hello and welcome to the debate", "ONLY")]
    turns = [_turn(0, 5, "ONLY")]
    result = infer_host(segments, turns, [])
    assert result["prediction"]["speaker_label"] == "ONLY"
    assert result["prediction"]["confidence"]["band"] == "uncertain"


def test_rank_candidates_respects_signal_agreement():
    segments, turns, overlaps = _moderator_episode()
    features = extract_features(segments, turns, overlaps)
    ranked = rank_candidates(features)
    assert ranked["ranking"][0][0] == "MODERATOR"
    # the moderator should win at least the question-oriented signals
    assert ranked["signal_winner"]["question_rate"] == "MODERATOR"
    assert ranked["signal_winner"]["distinct_interlocutors"] == "MODERATOR"


def test_build_result_records_participants_provenance():
    segments, turns, overlaps = _moderator_episode()
    final_json = {
        "episode": {"video_id": "TEST", "title": "t", "duration_s": 100},
        "participants": [],
        "segments": segments,
        "speaker_turns": turns,
        "overlap_candidates": overlaps,
    }
    result = build_result(final_json, video_id="TEST", reference_label="MODERATOR")
    assert result["video"]["video_id"] == "TEST"
    assert result["participants_provenance"]["record_count"] == 0
    assert result["prediction"]["speaker_label"] == "MODERATOR"
    assert result["reference_comparison"]["match"] is True
    # schema is JSON-serialisable
    import json
    json.dumps(result)
