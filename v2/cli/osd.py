"""Milestone 4: overlapping speech detection on one video, with analysis."""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from v2.io import load_json, save_json
from v2.media.layout import LAYOUT_ENV_VAR, build_layout
from v2.osd.detect import detect_overlap
from v2.osd.overlap import build_analysis

LOGGER = logging.getLogger("v2.cli.osd")


def _env_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect overlapping speech and analyse crosstalk / interruptions.",
    )
    parser.add_argument("--video-id", required=True, help="YouTube video ID.")
    parser.add_argument(
        "--model",
        default="pyannote/segmentation-3.0",
        help="Pyannote segmentation model used for OSD (gated on Hugging Face).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        help="Override auto device selection (cuda/mps/cpu).",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token. Defaults to $HF_TOKEN.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=f"Data root (default: ${LAYOUT_ENV_VAR} or data/v2/videos).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    token = args.token or _env_token()
    if not token:
        LOGGER.error(
            "No Hugging Face token found. Pass --token or set HF_TOKEN; the "
            "overlap model is gated and requires accepting its license."
        )
        raise SystemExit(1)

    layout = build_layout(args.video_id, args.data_dir)
    overlap_path = detect_overlap(
        layout.audio_path,
        layout.overlap_path,
        model_name=args.model,
        device=args.device,
        token=token,
    )

    from v2.io import load_json

    diarization = load_json(layout.diarization_path)
    overlap_payload = load_json(overlap_path)
    analysis = build_analysis(
        overlap_payload["regions"],
        diarization["turns"],
        {"model": overlap_payload["model"], "device": overlap_payload["device"]},
    )
    save_json(analysis, layout.analysis_path)
    LOGGER.info("OSD analysis saved: %s", layout.analysis_path)


if __name__ == "__main__":
    main()