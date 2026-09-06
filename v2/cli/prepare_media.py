"""Milestone 1: media acquisition and preprocessing for one video."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

from v2.media.audio import extract_audio
from v2.media.download import download_video
from v2.media.layout import LAYOUT_ENV_VAR, build_layout
from v2.media.metadata import fetch_metadata, save_metadata
from v2.media.proxy import create_proxy

LOGGER = logging.getLogger("v2.cli.prepare_media")


def prepare_media(
    video_id: str,
    data_dir: Optional[Path] = None,
    cookies_from_browser: Optional[str] = None,
) -> Dict[str, Path]:
    """Run download -> audio -> proxy -> metadata and return their paths."""
    layout = build_layout(video_id, data_dir)

    result = download_video(
        video_id,
        layout.source_dir,
        cookies_from_browser=cookies_from_browser,
    )

    metadata = result.metadata or fetch_metadata(
        video_id,
        cookies_from_browser=cookies_from_browser,
    )
    if not metadata:
        raise RuntimeError(f"No metadata could be obtained for {video_id}")

    metadata_path = save_metadata(metadata, layout.metadata_path)
    audio_path = extract_audio(result.video_path, layout.audio_path)
    proxy_path = create_proxy(result.video_path, layout.proxy_path)

    return {
        "video_id": video_id,
        "video": str(result.video_path),
        "audio": str(audio_path),
        "proxy": str(proxy_path),
        "metadata": str(metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one YouTube debate video and prepare audio + proxy.",
    )
    parser.add_argument("--video-id", required=True, help="YouTube video ID.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            f"Data root (default: ${LAYOUT_ENV_VAR} or data/v2/videos)."
        ),
    )
    parser.add_argument(
        "--cookies-from-browser",
        help="Browser to read YouTube cookies from, e.g. chrome (for bot-checked videos).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    paths = prepare_media(
        args.video_id,
        data_dir=args.data_dir,
        cookies_from_browser=args.cookies_from_browser,
    )
    for kind in ("video", "audio", "proxy", "metadata"):
        LOGGER.info("Wrote %s: %s", kind, paths[kind])
    LOGGER.info("Done for %s", args.video_id)


if __name__ == "__main__":
    main()