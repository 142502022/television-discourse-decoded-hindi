"""Step 9: cluster-to-name linking on one video."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from v2.identity.link import link_roster
from v2.io import load_json, save_json
from v2.media.layout import LAYOUT_ENV_VAR, build_layout

LOGGER = logging.getLogger("v2.cli.link")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge roster (step 7) and role evidence (step 5) into a "
        "linked participant roster with real names."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--data-dir", type=Path,
        help=f"Data root (default: ${LAYOUT_ENV_VAR} or data/v2/videos).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    layout = build_layout(args.video_id, args.data_dir)

    roster_data = None
    if layout.roster_path.exists():
        roster_data = load_json(layout.roster_path)
    else:
        LOGGER.warning("roster.json not found — linking from roles only.")

    roles_data = None
    if layout.roles_path.exists():
        roles_data = load_json(layout.roles_path)
    else:
        LOGGER.warning("roles.json not found — linking from roster only.")

    participants = roster_data.get("participants", []) if roster_data else []
    linked = link_roster(participants, roles_data)

    for p in linked["participants"]:
        LOGGER.info(
            "  %-24s [%s] role=%s  src=%s",
            p["name"],
            p["speaker_label"],
            p["role"],
            p["source"],
        )
    save_json(linked, layout.linked_path)
    LOGGER.info("Linked roster saved: %s", layout.linked_path)


if __name__ == "__main__":
    main()
