import json
from pathlib import Path


def load_json(input_path: Path):
    """Load JSON data from disk."""
    with input_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(result, output_path: Path):
    """Save JSON data to disk with stable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, indent=4, ensure_ascii=False)
        file_obj.write("\n")


def save_whisperx_json(
    result,
    output_path: Path,
):
    save_json(result, output_path)
