"""Milestone 5: identity linking — speaker embeddings + role inference."""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from v2.io import load_json, save_json
from v2.identity.embeddings import speaker_embeddings
from v2.identity.roles import infer_roles
from v2.media.layout import LAYOUT_ENV_VAR, build_layout

LOGGER = logging.getLogger("v2.cli.identity")


def _env_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identity linking: embed speakers and infer their debate role.",
    )
    parser.add_argument("--video-id", required=True, help="YouTube video ID.")
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
            "embedding model is gated and requires accepting its license."
        )
        raise SystemExit(1)

    layout = build_layout(args.video_id, args.data_dir)

    embeddings_path = speaker_embeddings(
        layout.audio_path,
        layout.diarization_path,
        layout.embeddings_path,
        device=args.device,
        token=token,
    )

    diarized = load_json(layout.diarized_transcript_path)
    analysis = load_json(layout.analysis_path)
    roles = infer_roles(diarized["segments"], analysis["stats"]["interruptions_by_pair"])
    save_json(roles, layout.roles_path)

    LOGGER.info("Embeddings saved: %s", embeddings_path)
    LOGGER.info("Roles saved: %s", layout.roles_path)


if __name__ == "__main__":
    main()