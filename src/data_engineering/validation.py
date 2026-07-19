from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

GenderLabel = Literal["male", "female", "unknown", "other"]


class EpisodeRecord(BaseModel):
    episode_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    air_date: Optional[date] = None
    duration: Optional[float] = Field(default=None, ge=0)
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class SegmentRecord(BaseModel):
    segment_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        return self


class TranscriptRecord(BaseModel):
    segment_id: str = Field(min_length=1)
    transcript_text: str = Field(min_length=1)
    language: str = Field(min_length=1)
    whisper_confidence: Optional[float] = Field(default=None, ge=0, le=1)

    @field_validator("transcript_text")
    @classmethod
    def transcript_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transcript_text must not be blank")
        return value


class ToxicityScoreRecord(BaseModel):
    segment_id: str = Field(min_length=1)
    toxicity_score: float = Field(ge=0, le=1)
    detoxify_labels: dict[str, Any]


class GenderAgeLabelRecord(BaseModel):
    segment_id: str = Field(min_length=1)
    speaker_gender: GenderLabel
    age_estimate: Optional[int] = Field(default=None, ge=0, le=120)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


def validate_no_orphaned_segments(
    episodes: list[EpisodeRecord], segments: list[SegmentRecord]
) -> None:
    episode_ids = {episode.episode_id for episode in episodes}
    orphaned = [
        segment.segment_id
        for segment in segments
        if segment.episode_id not in episode_ids
    ]
    if orphaned:
        raise ValueError(f"Orphaned segments found: {orphaned}")
