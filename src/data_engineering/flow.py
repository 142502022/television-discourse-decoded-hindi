import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from prefect import flow, task

from .database import get_session_factory
from .storage import store_episode_bundle
from .transforms import (
    build_episode_record,
    build_gender_age_records,
    build_segment_records,
    build_toxicity_records,
    build_transcript_records,
)
from .validation import validate_no_orphaned_segments

logger = logging.getLogger(__name__)


@task
def ingest_episode(video_details_path: str, episode_id: str) -> dict[str, Any]:
    with open(video_details_path, encoding="utf-8") as file_obj:
        videos = json.load(file_obj)

    for video in videos:
        if video.get("yt_vid_id") == episode_id:
            return video
    raise ValueError(f"Episode not found in metadata: {episode_id}")


@task
def run_vad_diarization(episode_id: str) -> str:
    return _run_pipeline_module("src.diarization_vad_osd_related.run_pipeline_osd_vad", episode_id)


@task
def run_transcription(episode_id: str) -> str:
    return _run_pipeline_module("src.transcription_related.run_pipeline_transcription", episode_id)


@task
def run_toxicity_scoring() -> str:
    return _run_pipeline_module("src.perspective_related.run_pipeline_perspective")


@task
def run_gender_age_detection(episode_id: str) -> str:
    logger.warning(
        "Gender/age detection is known to be blocked by ad interference for some videos."
    )
    return _run_pipeline_module("src.gender_data.gender_data", episode_id)


@task
def validate_outputs(
    episode_details: dict[str, Any],
    diarization_path: str,
    transcript_path: str,
    toxicity_path: str,
    gender_age_path: str | None = None,
):
    episode = build_episode_record(episode_details)
    segments = build_segment_records(episode.episode_id, _load_json(diarization_path))
    transcripts = build_transcript_records(episode.episode_id, _load_json(transcript_path))
    toxicity_scores = build_toxicity_records(episode.episode_id, _load_json(toxicity_path))
    gender_age_labels = (
        build_gender_age_records(episode.episode_id, _load_json(gender_age_path))
        if gender_age_path and os.path.exists(gender_age_path)
        else []
    )

    validate_no_orphaned_segments([episode], segments)
    return episode, segments, transcripts, toxicity_scores, gender_age_labels


@task
def store_to_postgres(validated_bundle, database_url: str | None = None) -> None:
    session_factory = get_session_factory(database_url)
    with session_factory() as session:
        store_episode_bundle(session, *validated_bundle)


@flow(name="mtp-data-engineering-pipeline")
def mtp_data_engineering_flow(
    episode_id: str,
    video_details_path: str = "data/video_details.json",
    results_dir: str = "data/results",
    database_url: str | None = None,
    run_gender_stage: bool = False,
) -> None:
    episode_details = ingest_episode(video_details_path, episode_id)
    run_vad_diarization(episode_id)
    run_transcription(episode_id)
    run_toxicity_scoring()

    gender_age_path = str(Path(results_dir) / "gender_data" / f"{episode_id}.json")
    if run_gender_stage:
        run_gender_age_detection(episode_id)

    validated_bundle = validate_outputs(
        episode_details,
        str(Path(results_dir) / "diarization_data" / f"{episode_id}.json"),
        str(Path(results_dir) / "transcription_data" / f"{episode_id}.json"),
        str(Path(results_dir) / "perspective_data" / f"{episode_id}.json"),
        gender_age_path,
    )
    store_to_postgres(validated_bundle, database_url)


def _run_pipeline_module(module_name: str, episode_id: str | None = None) -> str:
    command = [sys.executable, "-m", module_name]
    if episode_id:
        command.append(episode_id)

    logger.info("Running pipeline command: %s", " ".join(command))
    subprocess.run(command, check=True)
    return "completed"


def _load_json(path: str):
    with open(path, encoding="utf-8") as file_obj:
        return json.load(file_obj)
