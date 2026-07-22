import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import OverlappedSpeechDetection

from .io import save_json

LOGGER = logging.getLogger(__name__)


def run_pyannote_osd(
    audio_path: Path,
    output_path: Path,
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run Pyannote overlapped speech detection and save intervals."""
    if output_path.exists():
        LOGGER.info("OSD output already exists: %s", output_path)
        return _load_existing(output_path)

    hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("HUGGINGFACE_TOKEN is required for Pyannote OSD.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentation_model = Model.from_pretrained(
        "pyannote/segmentation",
        use_auth_token=hf_token,
    ).to(device)
    osd = OverlappedSpeechDetection(segmentation=segmentation_model)
    osd.instantiate(
        {
            "onset": 0.5,
            "offset": 0.5,
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
        }
    )

    LOGGER.info("Running Pyannote OSD on %s", audio_path)
    output = osd(str(audio_path))
    intervals = [
        {"start": float(segment.start), "end": float(segment.end)}
        for segment in output.get_timeline().support()
    ]
    save_json(intervals, output_path)
    return intervals


def _load_existing(output_path: Path) -> List[Dict[str, Any]]:
    import json

    with output_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)
