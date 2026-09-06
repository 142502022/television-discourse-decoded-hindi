"""Milestone 3: speaker diarization on one video, merged into the transcript."""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from v2.media.layout import LAYOUT_ENV_VAR, build_layout
from v2.speech.diarize import build_diarized_transcript_file, diarize_audio

LOGGER = logging.getLogger("v2.cli.diarize")


def _env_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diarize one video and tag the transcript with speakers.",
    )
    parser.add_argument("--video-id", required=True, help="YouTube video ID.")
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="Pyannote pipeline to use (gated on Hugging Face).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        help="Override auto device selection (cuda/mps/cpu).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Force exact speaker count (else auto-detect at runtime).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Upper bound on speaker count for auto-detection.",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token (gated model). Defaults to $HF_TOKEN.",
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
            "diarization model is gated and requires accepting its license."
        )
        raise SystemExit(1)

    layout = build_layout(args.video_id, args.data_dir)
    diarization_path = diarize_audio(
        layout.audio_path,
        layout.diarization_path,
        model_name=args.model,
        device=args.device,
        token=token,
        num_speakers=args.num_speakers,
        max_speakers=args.max_speakers,
    )
    merged_path = build_diarized_transcript_file(
        layout.asr_path,
        diarization_path,
        layout.diarized_transcript_path,
    )
    LOGGER.info("Diarized transcript saved: %s", merged_path)


if __name__ == "__main__":
    main()