"""Curated metadata for a downloaded video."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from yt_dlp import YoutubeDL

from v2.io import save_json

LOGGER = logging.getLogger(__name__)

CURATED_FIELDS = (
    "id",
    "url",
    "webpage_url",
    "title",
    "channel",
    "channel_id",
    "upload_date",
    "duration",
    "view_count",
    "thumbnail",
    "extractor",
)


class MetadataError(Exception):
    """Raised when metadata cannot be fetched or saved."""


def fetch_metadata(
    video_id: str,
    *,
    cookies_from_browser: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch and curate metadata for ``video_id`` without downloading."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": False,
    }
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise MetadataError(f"Failed to fetch metadata for {video_id}") from exc

    return curate_metadata(info or {})


def curate_metadata(info: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a yt-dlp info dict to the fields we want to persist."""
    curated: Dict[str, Any] = {
        field: info.get(field)
        for field in CURATED_FIELDS
        if info.get(field) is not None
    }
    curated["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return curated


def save_metadata(metadata: Dict[str, Any], output_path: Path) -> Path:
    """Save curated metadata as JSON.

    An existing file is reused only if it is valid JSON; a corrupt or empty
    file is replaced rather than trusted.
    """
    return save_json(metadata, output_path)