#!/usr/bin/env python3
"""
Add YouTube metadata for Hindi dataset video IDs.

Requirements:
    pip install yt-dlp
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from yt_dlp import YoutubeDL

BASE_DIR = Path(__file__).resolve().parent
IDS_PATH = BASE_DIR / "data" / "ids.json"
VIDEO_DETAILS_PATH = BASE_DIR / "data" / "video_details.json"

LOGGER = logging.getLogger(__name__)


def load_ids(path: Path = IDS_PATH) -> List[str]:
    """Load YouTube video IDs from a JSON file."""
    with path.open("r", encoding="utf-8") as file_obj:
        video_ids = json.load(file_obj)

    if not isinstance(video_ids, list):
        raise ValueError(f"{path} must contain a JSON array of YouTube IDs.")

    cleaned_ids = []
    for video_id in video_ids:
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"Invalid YouTube ID in {path}: {video_id!r}")
        cleaned_ids.append(video_id.strip())

    return cleaned_ids


def load_video_details(path: Path = VIDEO_DETAILS_PATH) -> List[Dict[str, Any]]:
    """Load existing video metadata entries from video_details.json."""
    with path.open("r", encoding="utf-8") as file_obj:
        video_details = json.load(file_obj)

    if not isinstance(video_details, list):
        raise ValueError(f"{path} must contain a JSON array of video metadata.")

    return video_details


def fetch_metadata(video_id: str) -> Dict[str, Any]:
    """Fetch YouTube metadata for a video ID using yt-dlp without downloading."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        metadata = ydl.extract_info(video_url, download=False)

    if not isinstance(metadata, dict):
        raise ValueError(f"yt-dlp returned invalid metadata for {video_id}.")

    return metadata


def create_video_entry(
    video_id: str,
    metadata: Dict[str, Any],
    video_idx: int,
) -> Dict[str, Any]:
    """Create a video_details.json entry matching the existing schema."""
    duration = _safe_int(metadata.get("duration"))

    return {
        "video_idx": video_idx,
        "yt_vid_id": video_id,
        "yt_vid_url": metadata.get(
            "webpage_url", f"https://www.youtube.com/watch?v={video_id}"
        ),
        "major_label": "Hindi",
        "minor_labels": [],
        "yt_stats": {
            "viewCount": _stat_to_string(metadata.get("view_count")),
            "likeCount": _stat_to_string(metadata.get("like_count")),
            "favoriteCount": _stat_to_string(metadata.get("favorite_count")),
            "commentCount": _stat_to_string(metadata.get("comment_count")),
        },
        "publish_time": _extract_publish_time(metadata),
        "vid_title": str(metadata.get("title") or ""),
        "total_duration": duration,
        "total_duration_str": _seconds_to_iso8601_duration(duration),
        "hashtags_detected": _extract_hashtags(metadata),
    }


def save_video_details(
    video_details: List[Dict[str, Any]],
    path: Path = VIDEO_DETAILS_PATH,
) -> None:
    """Persist video metadata with stable UTF-8 pretty JSON formatting."""
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(video_details, file_obj, ensure_ascii=False, indent=4)
        file_obj.write("\n")


def main() -> None:
    """Append metadata for new IDs from data/ids.json to video_details.json."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        video_ids = load_ids()
        video_details = load_video_details()
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        LOGGER.error("Failed: %s", exc)
        raise SystemExit(1) from exc

    existing_ids = _existing_video_ids(video_details)
    next_video_idx = _next_video_idx(video_details)

    added = 0
    skipped = 0
    failed = 0

    for video_id in video_ids:
        LOGGER.info("Processing %s...", video_id)

        if video_id in existing_ids:
            LOGGER.info("Skipped (already exists).")
            skipped += 1
            continue

        try:
            metadata = fetch_metadata(video_id)
            entry = create_video_entry(video_id, metadata, next_video_idx)
        except Exception as exc:
            LOGGER.error("Failed: %s", exc)
            failed += 1
            continue

        video_details.append(entry)
        existing_ids.add(video_id)
        next_video_idx += 1
        added += 1
        LOGGER.info("Added metadata.")

    if added:
        save_video_details(video_details)

    LOGGER.info("Total IDs: %s", len(video_ids))
    LOGGER.info("Added: %s", added)
    LOGGER.info("Skipped: %s", skipped)
    LOGGER.info("Failed: %s", failed)


def _existing_video_ids(video_details: Iterable[Dict[str, Any]]) -> Set[str]:
    """Return the set of YouTube IDs already present in video details."""
    return {
        str(entry["yt_vid_id"])
        for entry in video_details
        if isinstance(entry, dict) and entry.get("yt_vid_id")
    }


def _next_video_idx(video_details: Iterable[Dict[str, Any]]) -> int:
    """Return the next video_idx after the highest existing index."""
    indexes = [
        int(entry["video_idx"])
        for entry in video_details
        if isinstance(entry, dict) and _is_int_like(entry.get("video_idx"))
    ]
    return max(indexes, default=-1) + 1


def _extract_publish_time(metadata: Dict[str, Any]) -> str:
    """Extract publish time in ISO-8601 UTC format when available."""
    timestamp = metadata.get("timestamp") or metadata.get("release_timestamp")
    if timestamp is not None:
        try:
            return datetime.fromtimestamp(
                int(timestamp),
                tz=timezone.utc,
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError, OSError):
            pass

    upload_date = metadata.get("upload_date")
    if isinstance(upload_date, str) and re.fullmatch(r"\d{8}", upload_date):
        return (
            datetime.strptime(upload_date, "%Y%m%d")
            .replace(tzinfo=timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )

    return ""


def _extract_hashtags(metadata: Dict[str, Any]) -> List[str]:
    """Extract hashtags from yt-dlp metadata, title, description, and tags."""
    hashtags = []

    raw_hashtags = metadata.get("hashtags") or []
    if isinstance(raw_hashtags, list):
        hashtags.extend(str(tag) for tag in raw_hashtags if str(tag).strip())

    for text_key in ("title", "description"):
        text = metadata.get(text_key)
        if isinstance(text, str):
            hashtags.extend(re.findall(r"#[\w]+", text, flags=re.UNICODE))

    tags = metadata.get("tags") or []
    if isinstance(tags, list):
        for tag in tags:
            tag_text = str(tag).strip()
            if tag_text.startswith("#"):
                hashtags.append(tag_text)

    return _deduplicate_hashtags(hashtags)


def _deduplicate_hashtags(hashtags: Iterable[str]) -> List[str]:
    """Normalize hashtag values and preserve first-seen order."""
    seen = set()
    normalized = []

    for hashtag in hashtags:
        clean_hashtag = hashtag.strip()
        if not clean_hashtag:
            continue
        if not clean_hashtag.startswith("#"):
            clean_hashtag = f"#{clean_hashtag}"
        if clean_hashtag not in seen:
            seen.add(clean_hashtag)
            normalized.append(clean_hashtag)

    return normalized


def _seconds_to_iso8601_duration(seconds: int) -> str:
    """Convert a duration in seconds to an ISO-8601 duration string."""
    if seconds <= 0:
        return "PT0S"

    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    duration_parts = ["PT"]
    if hours:
        duration_parts.append(f"{hours}H")
    if minutes:
        duration_parts.append(f"{minutes}M")
    if secs or not (hours or minutes):
        duration_parts.append(f"{secs}S")

    return "".join(duration_parts)


def _stat_to_string(value: Any) -> str:
    """Convert a YouTube statistic value to a string, defaulting to zero."""
    if value is None:
        return "0"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "0"


def _safe_int(value: Any) -> int:
    """Convert a value to int, defaulting to zero when unavailable."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_int_like(value: Any) -> bool:
    """Return whether a value can be safely converted to int."""
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    main()
