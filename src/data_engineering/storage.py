from sqlalchemy.orm import Session

from .database import Episode, GenderAgeLabel, Segment, ToxicityScore, Transcript
from .validation import (
    EpisodeRecord,
    GenderAgeLabelRecord,
    SegmentRecord,
    ToxicityScoreRecord,
    TranscriptRecord,
    validate_no_orphaned_segments,
)


def store_episode_bundle(
    session: Session,
    episode: EpisodeRecord,
    segments: list[SegmentRecord],
    transcripts: list[TranscriptRecord],
    toxicity_scores: list[ToxicityScoreRecord],
    gender_age_labels: list[GenderAgeLabelRecord],
) -> None:
    validate_no_orphaned_segments([episode], segments)

    session.merge(Episode(**episode.model_dump()))
    for segment in segments:
        session.merge(Segment(**segment.model_dump()))
    for transcript in transcripts:
        session.merge(Transcript(**transcript.model_dump()))
    for toxicity_score in toxicity_scores:
        session.merge(ToxicityScore(**toxicity_score.model_dump()))
    for gender_age_label in gender_age_labels:
        session.merge(GenderAgeLabel(**gender_age_label.model_dump()))

    session.commit()

