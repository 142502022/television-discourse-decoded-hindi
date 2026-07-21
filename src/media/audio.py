from pathlib import Path
import logging
import subprocess

LOGGER = logging.getLogger(__name__)


class AudioExtractionError(Exception):
    """Raised when audio extraction fails."""


def extract_audio(
    video_path: Path,
    output_dir: Path,
) -> Path:
    """
    Extract mono 16 kHz WAV audio from a video.

    Args:
        video_path: Path to the input video.
        output_dir: Directory where the WAV will be stored.

    Returns:
        Path to the extracted WAV.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / f"{video_path.stem}.wav"

    if audio_path.exists():
        LOGGER.info("Audio already exists: %s", audio_path)
        return audio_path

    command = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        str(audio_path),
    ]

    try:
        LOGGER.info("Extracting audio from %s", video_path.name)

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    except subprocess.CalledProcessError as exc:
        raise AudioExtractionError(
            f"Failed to extract audio from {video_path}"
        ) from exc

    LOGGER.info("Audio extraction complete.")

    return audio_path
