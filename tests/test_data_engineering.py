import pytest

from src.data_engineering.transforms import (
    build_episode_record,
    build_gender_age_records,
    build_segment_records,
    build_toxicity_records,
    build_transcript_records,
)
from src.data_engineering.validation import (
    EpisodeRecord,
    GenderAgeLabelRecord,
    SegmentRecord,
    ToxicityScoreRecord,
    TranscriptRecord,
    validate_no_orphaned_segments,
)


def test_build_episode_record_from_video_metadata():
    record = build_episode_record(
        {
            "yt_vid_id": "abc123",
            "yt_vid_url": "https://www.youtube.com/watch?v=abc123",
            "publish_time": "2020-10-02T18:52:22Z",
            "total_duration": 3234,
        }
    )

    assert record.episode_id == "abc123"
    assert record.source == "https://www.youtube.com/watch?v=abc123"
    assert record.duration == 3234


def test_segment_validation_rejects_invalid_time_bounds():
    with pytest.raises(ValueError):
        SegmentRecord(
            segment_id="seg-1",
            episode_id="ep-1",
            speaker_id="speaker-1",
            start_time=10,
            end_time=5,
        )


def test_transcript_validation_rejects_blank_text():
    with pytest.raises(ValueError):
        TranscriptRecord(
            segment_id="seg-1",
            transcript_text="   ",
            language="hi",
            whisper_confidence=0.8,
        )


def test_toxicity_validation_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        ToxicityScoreRecord(
            segment_id="seg-1",
            toxicity_score=1.5,
            detoxify_labels={"TOXICITY": 1.5},
        )


def test_gender_age_validation_rejects_unexpected_gender():
    with pytest.raises(ValueError):
        GenderAgeLabelRecord(
            segment_id="seg-1",
            speaker_gender="not-a-label",
            age_estimate=35,
            confidence=0.9,
        )


def test_build_segment_records_from_diarization_output():
    records = build_segment_records(
        "ep-1",
        [
            [{"start": 1.5, "end": 3.0}, {"track": "SPEAKER_00"}],
            [{"start": 4.0, "end": 8.0}, {"track": "SPEAKER_01"}],
        ],
    )

    assert [record.segment_id for record in records] == ["ep-1:0", "ep-1:1"]
    assert records[0].speaker_id == "SPEAKER_00"


def test_build_transcript_records_maps_no_speech_probability_to_confidence():
    records = build_transcript_records(
        "ep-1",
        [{"text": "नमस्ते", "language": "hi", "no_speech_prob": 0.25}],
    )

    assert records[0].segment_id == "ep-1:0"
    assert records[0].whisper_confidence == 0.75


def test_build_toxicity_records_extracts_perspective_style_scores():
    records = build_toxicity_records(
        "ep-1",
        [
            {
                "perspective": {
                    "attributeScores": {
                        "TOXICITY": {"summaryScore": {"value": 0.42}},
                        "INSULT": {"summaryScore": {"value": 0.2}},
                    }
                }
            }
        ],
    )

    assert records[0].toxicity_score == 0.42
    assert records[0].detoxify_labels["INSULT"] == 0.2


def test_build_gender_age_records_normalizes_gender_label():
    records = build_gender_age_records(
        "ep-1",
        [{"speaker_gender": "Female", "age_estimate": 34, "confidence": 0.88}],
    )

    assert records[0].segment_id == "ep-1:0"
    assert records[0].speaker_gender == "female"
    assert records[0].age_estimate == 34


def test_no_orphaned_segments_sanity_check():
    episodes = [EpisodeRecord(episode_id="ep-1", source="youtube")]
    segments = [
        SegmentRecord(
            segment_id="seg-1",
            episode_id="missing",
            speaker_id="speaker-1",
            start_time=0,
            end_time=1,
        )
    ]

    with pytest.raises(ValueError, match="Orphaned segments"):
        validate_no_orphaned_segments(episodes, segments)
