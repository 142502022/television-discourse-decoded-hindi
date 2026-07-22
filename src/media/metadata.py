from pathlib import Path
import json
import logging
from typing import Any, Dict

from yt_dlp import YoutubeDL

LOGGER = logging.getLogger(__name__)


class MetadataError(Exception):
    """Raised when metadata extraction fails."""


def fetch_metadata(
    video_id: str,
    use_browser_cookies: bool = True,
) -> Dict[str, Any]:
    """
    Fetch metadata from YouTube without downloading the video.
    """

    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": False,
    }

    if use_browser_cookies:
        ydl_opts["cookiesfrombrowser"] = ("chrome",)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            metadata = ydl.extract_info(url, download=False)

    except Exception as exc:
        raise MetadataError(
            f"Failed to fetch metadata for {video_id}"
        ) from exc

    return metadata


def save_metadata(
    metadata: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    Save metadata as JSON.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = metadata["id"]

    output_path = output_dir / f"{video_id}.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )

    LOGGER.info("Metadata saved.")

    return output_path
