from __future__ import annotations

from typing import Any

from .validation import (
    EpisodeRecord,
    GenderAgeLabelRecord,
    SegmentRecord,
    ToxicityScoreRecord,
    TranscriptRecord,
)


def build_episode_record(video_details: dict[str, Any]) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=video_details["yt_vid_id"],
        source=video_details.get("yt_vid_url", video_details.get("source", "youtube")),
        air_date=video_details.get("publish_time", "")[:10] or None,
        duration=video_details.get("total_duration"),
    )


def build_segment_records(episode_id: str, diarization_data: list[Any]) -> list[SegmentRecord]:
    records = []
    for index, utterance in enumerate(diarization_data):
        bounds, speaker_map = utterance
        speaker_id = str(next(iter(speaker_map.values())))
        records.append(
            SegmentRecord(
                segment_id=f"{episode_id}:{index}",
                episode_id=episode_id,
                speaker_id=speaker_id,
                start_time=float(bounds["start"]),
                end_time=float(bounds["end"]),
            )
        )
    return records


def build_transcript_records(
    episode_id: str, transcript_data: list[dict[str, Any]]
) -> list[TranscriptRecord]:
    records = []
    for index, utterance in enumerate(transcript_data):
        no_speech_prob = utterance.get("no_speech_prob")
        confidence = None if no_speech_prob is None else 1 - float(no_speech_prob)
        records.append(
            TranscriptRecord(
                segment_id=f"{episode_id}:{index}",
                transcript_text=utterance["text"],
                language=utterance.get("language", "hi"),
                whisper_confidence=confidence,
            )
        )
    return records


def build_toxicity_records(
    episode_id: str, perspective_data: list[dict[str, Any]]
) -> list[ToxicityScoreRecord]:
    records = []
    for index, utterance in enumerate(perspective_data):
        labels = _extract_detoxify_labels(utterance.get("perspective", {}))
        records.append(
            ToxicityScoreRecord(
                segment_id=f"{episode_id}:{index}",
                toxicity_score=float(labels.get("TOXICITY", 0.0)),
                detoxify_labels=labels,
            )
        )
    return records


def build_gender_age_records(
    episode_id: str, gender_data: list[dict[str, Any]]
) -> list[GenderAgeLabelRecord]:
    records = []
    for index, label in enumerate(gender_data):
        records.append(
            GenderAgeLabelRecord(
                segment_id=f"{episode_id}:{index}",
                speaker_gender=str(label.get("speaker_gender", "unknown")).lower(),
                age_estimate=label.get("age_estimate"),
                confidence=label.get("confidence"),
            )
        )
    return records


def _extract_detoxify_labels(perspective_response: dict[str, Any]) -> dict[str, float]:
    attribute_scores = perspective_response.get("attributeScores", {})
    labels = {}
    for label, score_data in attribute_scores.items():
        summary = score_data.get("summaryScore", {})
        labels[label] = float(summary.get("value", 0.0))
    return labels

