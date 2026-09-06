"""Audio extraction with FFmpeg."""

import logging
from pathlib import Path
import subprocess

LOGGER = logging.getLogger(__name__)


class AudioExtractionError(Exception):
    """Raised when audio extraction fails."""


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio from a video as 16 kHz mono PCM WAV.

    WhisperX and Pyannote both expect 16 kHz audio; mono removes channel
    ambiguity. WAV avoids further transcoding and is safe to re-read many
    times, which suits an inspectable-intermediates pipeline.
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        LOGGER.info("Audio already exists: %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
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

    LOGGER.info("Audio extraction complete: %s", output_path.name)
    return output_path