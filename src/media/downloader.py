from pathlib import Path
import logging

from yt_dlp import YoutubeDL

LOGGER = logging.getLogger(__name__)


class VideoDownloadError(Exception):
    """Raised when a YouTube video cannot be downloaded."""


def download_video(
    video_id: str,
    output_dir: Path,
    use_browser_cookies: bool = True,
) -> Path:
    """
    Download a YouTube video as an MP4.

    Args:
        video_id: YouTube video ID.
        output_dir: Directory where the video will be saved.
        use_browser_cookies: Whether to use Chrome cookies.

    Returns:
        Path to the downloaded MP4.

    Raises:
        VideoDownloadError: If the download fails.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{video_id}.mp4"

    if output_path.exists():
        LOGGER.info("Video already exists: %s", output_path)
        return output_path

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "format": (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
            "best[ext=mp4]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(output_path.with_suffix("")),
        "quiet": False,
        "noplaylist": True,
        "retries": 5,
    }

    if use_browser_cookies:
        ydl_opts["cookiesfrombrowser"] = ("chrome",)

    try:
        LOGGER.info("Downloading %s", video_id)

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

    except Exception as exc:
        raise VideoDownloadError(
            f"Failed to download {video_id}"
        ) from exc

    if not output_path.exists():
        raise VideoDownloadError(
            f"Download finished but {output_path.name} was not created."
        )

    LOGGER.info("Download complete.")

    return output_path
