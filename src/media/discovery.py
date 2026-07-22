import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from yt_dlp import YoutubeDL

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeCandidate:
    video_id: str
    title: str
    url: str
    upload_date: Optional[str]


def last_calendar_month(today: Optional[date] = None) -> tuple[date, date]:
    """Return start and end dates for the previous calendar month."""
    today = today or date.today()
    first_of_this_month = today.replace(day=1)
    last_of_previous_month = first_of_this_month - timedelta(days=1)
    first_of_previous_month = last_of_previous_month.replace(day=1)
    return first_of_previous_month, last_of_previous_month


def discover_channel_episodes(
    channel_url: str,
    start_date: date,
    end_date: date,
    limit: int = 5,
    use_browser_cookies: bool = True,
) -> List[EpisodeCandidate]:
    """Discover recent channel videos whose upload dates fall in a date range."""
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "playlistend": 100,
    }

    if use_browser_cookies:
        ydl_opts["cookiesfrombrowser"] = ("chrome",)

    with YoutubeDL(ydl_opts) as ydl:
        playlist = ydl.extract_info(channel_url, download=False)

        entries = playlist.get("entries", []) if isinstance(playlist, dict) else []
        candidates = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            video_id = entry.get("id")
            if not video_id:
                continue

            metadata = entry
            upload_date = _parse_upload_date(metadata)
            if upload_date is None:
                metadata = _fetch_video_metadata(ydl, str(video_id))
                upload_date = _parse_upload_date(metadata)
            if upload_date is None:
                LOGGER.info("Skipping %s because upload date is unavailable.", video_id)
                continue
            if not (start_date <= upload_date <= end_date):
                continue

            candidates.append(_candidate_from_entry(metadata))
            if len(candidates) >= limit:
                break

    LOGGER.info("Discovered %s candidate episodes.", len(candidates))
    return candidates


def _fetch_video_metadata(ydl: YoutubeDL, video_id: str) -> Dict[str, Any]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    metadata = ydl.extract_info(url, download=False)
    if not isinstance(metadata, dict):
        return {"id": video_id}
    return metadata


def _candidate_from_entry(entry: Dict[str, Any]) -> EpisodeCandidate:
    video_id = str(entry["id"])
    return EpisodeCandidate(
        video_id=video_id,
        title=str(entry.get("title") or ""),
        url=str(entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"),
        upload_date=str(entry.get("upload_date") or "") or None,
    )


def _parse_upload_date(entry: Dict[str, Any]) -> Optional[date]:
    upload_date = entry.get("upload_date")
    if isinstance(upload_date, str) and len(upload_date) == 8:
        return datetime.strptime(upload_date, "%Y%m%d").date()
    timestamp = entry.get("timestamp")
    if timestamp is not None:
        return datetime.utcfromtimestamp(int(timestamp)).date()
    return None
