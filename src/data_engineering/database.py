import os
from datetime import date, datetime
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

load_dotenv()


class Base(DeclarativeBase):
    pass


class Episode(Base):
    __tablename__ = "episodes"

    episode_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    air_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    segments: Mapped[list["Segment"]] = relationship(
        back_populates="episode", cascade="all, delete-orphan"
    )


class Segment(Base):
    __tablename__ = "segments"

    segment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    episode_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("episodes.episode_id"), nullable=False, index=True
    )
    speaker_id: Mapped[str] = mapped_column(String(128), nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)

    episode: Mapped[Episode] = relationship(back_populates="segments")
    transcript: Mapped["Transcript"] = relationship(
        back_populates="segment", cascade="all, delete-orphan"
    )
    toxicity_score: Mapped["ToxicityScore"] = relationship(
        back_populates="segment", cascade="all, delete-orphan"
    )
    gender_age_label: Mapped["GenderAgeLabel"] = relationship(
        back_populates="segment", cascade="all, delete-orphan"
    )


class Transcript(Base):
    __tablename__ = "transcripts"

    segment_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("segments.segment_id"), primary_key=True
    )
    transcript_text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    whisper_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    segment: Mapped[Segment] = relationship(back_populates="transcript")


class ToxicityScore(Base):
    __tablename__ = "toxicity_scores"

    segment_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("segments.segment_id"), primary_key=True
    )
    toxicity_score: Mapped[float] = mapped_column(Float, nullable=False)
    detoxify_labels: Mapped[dict] = mapped_column(JSON, nullable=False)

    segment: Mapped[Segment] = relationship(back_populates="toxicity_score")


class GenderAgeLabel(Base):
    __tablename__ = "gender_age_labels"

    segment_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("segments.segment_id"), primary_key=True
    )
    speaker_gender: Mapped[str] = mapped_column(String(32), nullable=False)
    age_estimate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    segment: Mapped[Segment] = relationship(back_populates="gender_age_label")


def get_database_url() -> str:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set for Postgres access.")
    return database_url


def get_engine(database_url: Optional[str] = None):
    return create_engine(database_url or get_database_url(), future=True)


def create_tables(database_url: Optional[str] = None) -> None:
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)


def get_session_factory(database_url: Optional[str] = None):
    engine = get_engine(database_url)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
