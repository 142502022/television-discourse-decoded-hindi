"""Step 7: LLM (Gemini) roster extraction on one video."""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from v2.io import load_json, save_json
from v2.media.layout import LAYOUT_ENV_VAR, build_layout

LOGGER = logging.getLogger("v2.cli.roster")


def _env_token() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY")


def _call_gemini(
    system: str,
    user: str,
    api_key: str,
    model: str,
) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=genai.types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a named participant roster from a diarized "
        "transcript using Gemini."
    )
    parser.add_argument("--video-id", required=True, help="YouTube video ID.")
    parser.add_argument(
        "--model",
        default="gemini-3.6-flash",
        help="Gemini model (default: gemini-3.6-flash).",
    )
    parser.add_argument(
        "--token",
        help="Gemini API key. Defaults to $GEMINI_API_KEY.",
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
            "No Gemini API key found. Pass --token or set GEMINI_API_KEY."
        )
        raise SystemExit(1)

    from v2.roster.extract import (
        build_prompt,
        build_roster_artifact,
        parse_roster,
        system_prompt,
    )

    layout = build_layout(args.video_id, args.data_dir)
    diarized = load_json(layout.diarized_transcript_path)

    LOGGER.info("Calling Gemini '%s' for roster...", args.model)
    raw = _call_gemini(system_prompt(), build_prompt(diarized["segments"]), token, args.model)
    payload = parse_roster(raw)
    roster = build_roster_artifact(payload["participants"], args.model)

    for p in roster["participants"]:
        LOGGER.info(
            "- %-24s %-13s %s (label=%s)",
            p["name"],
            p["role"],
            p["affiliation"] or "-",
            p["speaker_label"],
        )
    save_json(roster, layout.roster_path)
    LOGGER.info("Roster saved: %s", layout.roster_path)


if __name__ == "__main__":
    main()