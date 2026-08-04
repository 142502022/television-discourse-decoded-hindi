
from pathlib import Path

from .downloader import download_video
from .audio import extract_audio
from .proxy import create_proxy
from .metadata import fetch_metadata, save_metadata


project_root = Path(__file__).resolve().parent.parent.parent
data_raw_dir = project_root / "data" / "raw"

def prepare_media(video_id: str):

    video = download_video(
        video_id,
        data_raw_dir / "video",
    )
    audio = extract_audio(
        video,
        data_raw_dir / "audio",
    )

    proxy = create_proxy(
        video,
        data_raw_dir / "proxy",
    )

    metadata = fetch_metadata(video_id)

    metadata_file = save_metadata(
        metadata,
        data_raw_dir / "metdadata",
    )

    return {
        "video": video,
        "audio": audio,
        "proxy": proxy,
        "metadata": metadata_file,
    }
