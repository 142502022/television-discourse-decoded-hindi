import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from src.speech.io import save_json

LOGGER = logging.getLogger(__name__)


def chunk_proxy_video(
    proxy_path: Path,
    output_dir: Path,
    chunk_seconds: int = 900,
) -> List[Path]:
    """Cut a proxy video into fixed-size chunks using ffmpeg segment muxing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / f"{proxy_path.stem}_chunk_%03d.mp4"

    existing_chunks = sorted(output_dir.glob(f"{proxy_path.stem}_chunk_*.mp4"))
    if existing_chunks:
        LOGGER.info("Video chunks already exist: %s", output_dir)
        return existing_chunks

    command = [
        "ffmpeg",
        "-i",
        str(proxy_path),
        "-c",
        "copy",
        "-map",
        "0",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        "-y",
        str(output_pattern),
    ]

    LOGGER.info("Chunking proxy video %s", proxy_path.name)
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return sorted(output_dir.glob(f"{proxy_path.stem}_chunk_*.mp4"))


def slice_episode_json_for_chunks(
    episode_json_path: Path,
    chunk_paths: List[Path],
    output_dir: Path,
    chunk_seconds: int = 900,
) -> List[Path]:
    """Slice episode JSON into one JSON file per video chunk."""
    with episode_json_path.open("r", encoding="utf-8") as file_obj:
        episode = json.load(file_obj)

    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_json_paths = []

    for chunk_index, chunk_path in enumerate(chunk_paths):
        start = chunk_index * chunk_seconds
        end = start + chunk_seconds
        chunk_payload = {
            "episode_id": episode["episode_id"],
            "chunk_index": chunk_index,
            "chunk_start": start,
            "chunk_end": end,
            "proxy_video": str(chunk_path),
            "segments": _slice_timed_items(episode.get("segments", []), start, end),
            "overlap_candidates": _slice_timed_items(
                episode.get("overlap_candidates", []),
                start,
                end,
            ),
            "participants": episode.get("participants", []),
            "speaker_links": episode.get("speaker_links", []),
        }
        output_path = output_dir / f"{episode['episode_id']}_chunk_{chunk_index:03d}.json"
        save_json(chunk_payload, output_path)
        chunk_json_paths.append(output_path)

    return chunk_json_paths


def _slice_timed_items(
    items: List[Dict[str, Any]],
    chunk_start: float,
    chunk_end: float,
) -> List[Dict[str, Any]]:
    sliced = []
    for item in items:
        item_start = float(item.get("start", 0))
        item_end = float(item.get("end", item_start))
        if item_start < chunk_end and item_end > chunk_start:
            new_item = dict(item)
            new_item["chunk_relative_start"] = max(0.0, item_start - chunk_start)
            new_item["chunk_relative_end"] = min(chunk_end, item_end) - chunk_start
            sliced.append(new_item)
    return sliced
