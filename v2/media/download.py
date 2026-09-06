"""Video download using yt-dlp."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from yt_dlp import YoutubeDL

from .metadata import curate_metadata

LOGGER = logging.getLogger(__name__)

_FORMAT = (
    "bestvideo[ext=mp4][protocol^=http]+bestaudio[ext=m4a][protocol^=http]/"
    "best[ext=mp4][protocol^=http]/best[protocol^=http]/"
    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
)

_ORIGINAL_GLOB = "original.*"


@dataclass(frozen=True)
class DownloadResult:
    video_path: Path
    metadata: Dict[str, Any]


class VideoDownloadError(Exception):
    """Raised when a YouTube video cannot be downloaded."""


def find_original_video(source_dir: Path) -> Optional[Path]:
    """Return the first non-empty downloaded video file, if one exists."""
    for match in source_dir.glob(_ORIGINAL_GLOB):
        if match.is_file() and match.stat().st_size > 0:
            LOGGER.info("Found existing download: %s", match)
            return match
    return None


def download_video(
    video_id: str,
    source_dir: Path,
    *,
    cookies_from_browser: Optional[str] = None,
) -> DownloadResult:
    """Download ``video_id`` to ``source_dir`` as ``original.<ext>``.

    Idempotent: if a non-empty video already exists in ``source_dir`` the
    download is skipped and the existing path is returned (without metadata).

    Returns:
        The downloaded video path and its yt-dlp info dictionary.
    """
    source_dir.mkdir(parents=True, exist_ok=True)

    existing = find_original_video(source_dir)
    if existing is not None:
        return DownloadResult(video_path=existing, metadata={})

    ydl_opts = {
        "format": _FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(source_dir / _ORIGINAL_GLOB.replace("*", "%(ext)s")),
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
    }
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
    except Exception as exc:
        raise VideoDownloadError(f"Failed to download {video_id}") from exc

    info = info or {}
    video_path = _resolve_downloaded_path(source_dir, info)
    if video_path is None:
        raise VideoDownloadError(
            f"Download finished but no non-empty video was produced for {video_id}"
        )

    LOGGER.info("Download complete: %s", video_path.name)
    return DownloadResult(video_path=video_path, metadata=curate_metadata(info))


def _resolve_downloaded_path(source_dir: Path, info: Dict[str, Any]) -> Optional[Path]:
    requested = info.get("requested_downloads") or []
    if requested:
        filepath = (requested[0] or {}).get("filepath")
        if filepath and Path(filepath).is_file() and Path(filepath).stat().st_size > 0:
            return Path(filepath)
    return find_original_video(source_dir)