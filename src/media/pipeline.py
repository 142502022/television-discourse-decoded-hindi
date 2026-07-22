
from pathlib import Path

from .downloader import download_video
from .audio import extract_audio
from .proxy import create_proxy
from .metadata import fetch_metadata, save_metadata


def prepare_media(video_id: str):

    video = download_video(
        video_id,
        Path("data/raw/videos"),
    )

    audio = extract_audio(
        video,
        Path("data/raw/audio"),
    )

    proxy = create_proxy(
        video,
        Path("data/raw/proxy"),
    )

    metadata = fetch_metadata(video_id)

    metadata_file = save_metadata(
        metadata,
        Path("data/raw/metadata"),
    )

    return {
        "video": video,
        "audio": audio,
        "proxy": proxy,
        "metadata": metadata_file,
    }
