"""Shared JSON artifact persistence for pipeline stages."""

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def is_valid_json_file(path: Path) -> bool:
    """True if ``path`` exists, is non-empty, and parses as JSON."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            json.load(file_obj)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return True


def load_json(path: Path) -> Any:
    """Load a JSON artifact. Raises FileNotFoundError if missing or invalid."""
    if not is_valid_json_file(path):
        raise FileNotFoundError(f"Missing or corrupt JSON artifact: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(data: Any, output_path: Path) -> Path:
    """Write ``data`` as JSON; an existing valid artifact is reused."""
    if is_valid_json_file(output_path):
        LOGGER.info("Artifact already exists: %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, ensure_ascii=False)
    LOGGER.info("Artifact saved: %s", output_path.name)
    return output_path