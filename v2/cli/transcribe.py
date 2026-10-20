"""Milestone 2: direct ASR on one video's audio (no VAD / diarization)."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from v2.media.layout import LAYOUT_ENV_VAR, build_layout
from v2.speech.transcribe import transcribe_audio

LOGGER = logging.getLogger("v2.cli.transcribe")

# Corpus-level overrides: some videos open with non-source-language audio
# (e.g. Hindi ads before an English debate), which fools WhisperX's short
# auto-detect window. This file forces a language per video_id.
DEFAULT_OVERRIDES_PATH = Path("data/v2/language_overrides.json")


def load_language_overrides(path: Optional[Path] = DEFAULT_OVERRIDES_PATH) -> Dict[str, str]:
    """Read ``{video_id: language}``; missing/invalid file yields an empty map."""
    if path is None or not Path(path).exists():
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Ignoring unreadable language overrides: %s", path)
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if isinstance(value, str) and value}


def resolve_language(
    video_id: str,
    cli_language: Optional[str],
    overrides: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    return cli_language or "en"
    """Explicit ``--language`` wins; else the per-video override; else None (auto)."""
    # if cli_language:
    #     return cli_language
    # overrides = load_language_overrides() if overrides is None else overrides
    # return overrides.get(video_id)


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
    parser.add_argument(
        "--language-overrides",
        type=Path,
        default=DEFAULT_OVERRIDES_PATH,
        help="Per-video language override map (default: data/v2/language_overrides.json).",
    )
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
    # language = resolve_language(
    #     args.video_id,
    #     args.language,
    #     load_language_overrides(args.language_overrides),
    # )
    # if language and not args.language:
    #     LOGGER.info("Using per-video language override '%s' for %s", language, args.video_id)
    transcript_path = transcribe_audio(
        layout.audio_path,
        layout.asr_path,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language="en",
        batch_size=args.batch_size,
    )
    LOGGER.info("Transcript saved: %s", transcript_path)


if __name__ == "__main__":
    main()