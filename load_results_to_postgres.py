#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data_engineering.database import create_tables, get_session_factory
from src.data_engineering.storage import store_episode_bundle
from src.data_engineering.transforms import (
    build_episode_record,
    build_gender_age_records,
    build_segment_records,
    build_toxicity_records,
    build_transcript_records,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Load already-generated pipeline JSON outputs into PostgreSQL."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.create_tables:
        create_tables(args.database_url)

    video_details = load_video_details(Path(args.video_details_path))
    video_ids = resolve_video_ids(args, Path(args.results_dir))
    session_factory = get_session_factory(args.database_url)

    added = 0
    skipped = 0
    failed = 0

    with session_factory() as session:
        for video_id in video_ids:
            LOGGER.info("Processing %s...", video_id)

            episode_details = video_details.get(video_id)
            if not episode_details:
                LOGGER.warning("Skipped: missing video_details.json entry.")
                skipped += 1
                continue

            missing_paths = missing_required_paths(video_id, Path(args.results_dir))
            if missing_paths:
                LOGGER.warning("Skipped: missing %s", ", ".join(missing_paths))
                skipped += 1
                continue

            try:
                bundle = build_episode_bundle(
                    video_id,
                    episode_details,
                    Path(args.results_dir),
                )
                store_episode_bundle(session, *bundle)
            except Exception as exc:
                LOGGER.exception("Failed: %s", exc)
                failed += 1
                continue

            LOGGER.info("Stored metadata and stage outputs.")
            added += 1

    LOGGER.info("Total IDs: %s", len(video_ids))
    LOGGER.info("Stored: %s", added)
    LOGGER.info("Skipped: %s", skipped)
    LOGGER.info("Failed: %s", failed)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-file", default="ids.json")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--video-details-path", default="data/video_details.json")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--create-tables", action="store_true")
    return parser.parse_args()


def load_video_details(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load video metadata keyed by YouTube video ID."""
    with path.open("r", encoding="utf-8") as file_obj:
        rows = json.load(file_obj)
    return {
        row["yt_vid_id"]: row
        for row in rows
        if isinstance(row, dict) and row.get("yt_vid_id")
    }


def resolve_video_ids(args: argparse.Namespace, results_dir: Path) -> List[str]:
    """Resolve video IDs from CLI, ids.json, or result files."""
    if args.video_id:
        return args.video_id

    ids_path = Path(args.ids_file)
    if ids_path.exists():
        with ids_path.open("r", encoding="utf-8") as file_obj:
            ids = json.load(file_obj)
        return [str(video_id).strip() for video_id in ids if str(video_id).strip()]

    diarization_dir = results_dir / "diarization_data"
    return sorted(path.stem for path in diarization_dir.glob("*.json"))


def missing_required_paths(video_id: str, results_dir: Path) -> List[str]:
    """Return required result paths that do not exist for a video."""
    required_paths = [
        results_dir / "diarization_data" / f"{video_id}.json",
        results_dir / "transcription_data" / f"{video_id}.json",
        results_dir / "perspective_data" / f"{video_id}.json",
    ]
    return [str(path) for path in required_paths if not path.exists()]


def build_episode_bundle(
    video_id: str,
    episode_details: Dict[str, Any],
    results_dir: Path,
):
    """Build validated records for one episode from existing result JSON files."""
    episode = build_episode_record(episode_details)
    segments = build_segment_records(
        video_id,
        load_json(results_dir / "diarization_data" / f"{video_id}.json"),
    )
    transcripts = build_transcript_records(
        video_id,
        load_json(results_dir / "transcription_data" / f"{video_id}.json"),
    )
    toxicity_scores = build_toxicity_records(
        video_id,
        load_json(results_dir / "perspective_data" / f"{video_id}.json"),
    )

    gender_path = results_dir / "gender_data" / f"{video_id}.json"
    gender_age_labels = (
        build_gender_age_records(video_id, load_json(gender_path))
        if gender_path.exists()
        else []
    )

    return episode, segments, transcripts, toxicity_scores, gender_age_labels


def load_json(path: Path):
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


if __name__ == "__main__":
    main()
