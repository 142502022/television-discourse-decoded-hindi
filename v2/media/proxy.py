"""Low-resolution proxy video generation with FFmpeg."""

import logging
from pathlib import Path
import subprocess

LOGGER = logging.getLogger(__name__)


class ProxyGenerationError(Exception):
    """Raised when proxy generation fails."""


def create_proxy(video_path: Path, output_path: Path) -> Path:
    """Create a 480p H.264 MP4 proxy of ``video_path``.

    Output is suited to later manual/visual speaker verification:
    - H.264 in MP4 for universal playback.
    - Height capped at 480 (even-width so the encoder stays valid).
    - Audio re-encoded as AAC so timestamps stay in sync.
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        LOGGER.info("Proxy already exists: %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "scale=-2:480",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    try:
        LOGGER.info("Generating proxy video from %s", video_path.name)
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise ProxyGenerationError(
            f"Failed to create proxy for {video_path}"
        ) from exc

    LOGGER.info("Proxy generation complete: %s", output_path.name)
    return output_path