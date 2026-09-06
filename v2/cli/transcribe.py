"""Milestone 2: direct ASR on one video's audio (no VAD / diarization)."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from v2.media.layout import LAYOUT_ENV_VAR, build_layout
from v2.speech.transcribe import transcribe_audio

LOGGER = logging.getLogger("v2.cli.transcribe")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe one video's extracted audio with WhisperX.",
    )
    parser.add_argument("--video-id", required=True, help="YouTube video ID.")
    parser.add_argument("--model", default="medium", help="Whisper model size.")
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        help="Override auto device selection (cuda/mps/cpu).",
    )
    parser.add_argument(
        "--compute-type",
        choices=("float16", "int8", "float32"),
        help="Override compute type (default: float16 on GPU, int8 on CPU).",
    )
    parser.add_argument("--language", help="Force source language (e.g. en, hi).")
    parser.add_argument("--batch-size", type=int, help="Transcription batch size.")
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

    layout = build_layout(args.video_id, args.data_dir)
    transcript_path = transcribe_audio(
        layout.audio_path,
        layout.asr_path,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        batch_size=args.batch_size,
    )
    LOGGER.info("Transcript saved: %s", transcript_path)


if __name__ == "__main__":
    main()