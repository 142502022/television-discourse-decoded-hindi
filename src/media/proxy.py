

from pathlib import Path
import logging
import subprocess

LOGGER = logging.getLogger(__name__)


class ProxyGenerationError(Exception):
    """Raised when proxy generation fails."""


def create_proxy(
    video_path: Path,
    output_dir: Path,
) -> Path:
    """
    Create a 480p H.264 proxy video.

    Args:
        video_path: Original video.
        output_dir: Directory for proxy videos.

    Returns:
        Path to proxy video.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    proxy_path = output_dir / f"{video_path.stem}_proxy.mp4"

    if proxy_path.exists():
        LOGGER.info("Proxy already exists: %s", proxy_path)
        return proxy_path

    command = [
        "ffmpeg",
        "-i", str(video_path),

        # Scale while keeping aspect ratio
        "-vf", "scale=-2:480",

        # Video codec
        "-c:v", "libx264",

        # Compression quality
        "-crf", "23",

        # Encoding speed
        "-preset", "medium",

        # Audio codec
        "-c:a", "aac",
        "-b:a", "128k",

        "-movflags", "+faststart",

        "-y",
        str(proxy_path),
    ]

    try:
        LOGGER.info("Generating proxy video...")

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

    LOGGER.info("Proxy generation complete.")

    return proxy_path
