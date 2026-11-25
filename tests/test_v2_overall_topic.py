import json

from analysis.overall_topic import build_windows, detect_one, extract_segments, validate_overall_result, validate_window_result


def seg(start, end, text, speaker="SPEAKER_00"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def test_segments_use_transcript_text_and_preserve_speaker_timestamps():
    records = extract_segments({"speaker_turns": [{"start": 0, "end": 9, "speaker": "SPEAKER_99"}], "segments": [seg(1, 2, "actual transcript", "SPEAKER_01")]})
    assert records == [{"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_01", "text": "actual transcript"}]


def test_windows_do_not_split_or_double_count_and_keep_final_partial_window():
    records = extract_segments({"segments": [seg(590, 610, "crosses boundary"), seg(601, 605, "final")]} )
    windows = build_windows(records)
    assert [(w["start"], w["end"], len(w["segments"])) for w in windows] == [(0.0, 600.0, 1), (600.0, 610.0, 1)]


def test_empty_transcript_produces_no_windows():
    assert build_windows(extract_segments({"segments": []})) == []


def test_validation_discards_fabricated_evidence_and_normalizes_alignment():
    window = {"window_id": "W001", "start": 0.0, "end": 10.0, "segments": extract_segments({"segments": [seg(1, 2, "real evidence")]})}
    analysis = validate_window_result({"main_issue": "Issue", "summary": "Summary", "discussion_points": ["Point"], "evidence": [{"text": "invented"}]}, window)
    assert analysis["evidence"] == []
    overall = validate_overall_result({"topic": "Topic", "supporting_evidence": [{"text": "invented"}], "topic_alignment": {"status": "wrong"}}, [analysis])
    assert overall["topic_alignment"]["status"] == "uncertain"
    assert overall["supporting_evidence"] == []


def test_detect_one_is_variant_local_and_preserves_inputs(tmp_path):
    source = tmp_path / "videos" / "vid" / "baseline" / "merge_final.json"
    metadata = tmp_path / "videos" / "vid" / "metadata" / "metadata.json"
    source.parent.mkdir(parents=True); metadata.parent.mkdir(parents=True)
    source.write_text(json.dumps({"segments": [seg(1, 2, "debate transcript")]}, ensure_ascii=False))
    metadata.write_text(json.dumps({"title": "Metadata title"}))
    original = source.read_bytes()
    calls = []
    def fake(_key, _model, _instruction, payload):
        calls.append(payload)
        if "segments" in payload:
            return {"main_issue": "Issue", "summary": "Summary", "discussion_points": ["Point"], "entities": [], "evidence": [{"text": "debate transcript"}]}
        return {"topic": "Overall issue", "description": "Description", "key_discussion_points": ["Point"], "supporting_evidence": [{"text": "debate transcript"}], "topic_alignment": {"status": "aligned", "reason": "Matches"}}
    output = tmp_path / "out" / "baseline" / "vid" / "overall_topic.json"
    windows, result = detect_one("vid", "baseline", source, metadata, output, api_key="test", model="test", llm=fake)
    assert windows == 1 and result["variant"] == "baseline" and output.is_file()
    assert source.read_bytes() == original
    assert calls[0]["segments"][0]["text"] == "debate transcript"
