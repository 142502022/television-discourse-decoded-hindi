"""Milestone 4: overlapping speech detection (OSD).

Runs a neural overlap detector and saves ``osd/overlap.json`` with the raw
regions. Uses pyannote's ``OverlappedSpeechDetection`` built on
``pyannote/segmentation-3.0`` (the segmentation model the diarization pipeline
already relies on, so the same accepted gated model + token work here).
The richer event/statistics layer lives in ``v2.osd.overlap`` and is applied by
the CLI after detection.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from v2.io import is_valid_json_file, save_json
from v2.speech.transcribe import resolve_device

LOGGER = logging.getLogger(__name__)

DEFAULT_OVERLAP_MODEL = "pyannote/segmentation-3.0"

# Faithful to the pyannote segmentation-3.0 card: powerset model (overlap
# thresholds fixed at 0.5), no smoothing.
DEFAULT_HYPERPARAMETERS: Dict[str, float] = {
    "min_duration_on": 0.0,
    "min_duration_off": 0.0,
}


def _round3(value: float) -> float:
    return round(float(value), 3)


def extract_regions(detection: object) -> list:
    """Flatten a pyannote annotation into ``[start, end]`` dicts."""
    regions = []
    for segment, _, _ in detection.itertracks(yield_label=True):
        regions.append({"start": _round3(segment.start), "end": _round3(segment.end)})
    return regions


def detect_overlap(
    audio_path: Path,
    output_path: Path,
    *,
    model_name: str = DEFAULT_OVERLAP_MODEL,
    device: Optional[str] = None,
    token: Optional[str] = None,
    hyperparameters: Optional[Dict[str, float]] = None,
) -> Path:
    """Detect overlapping-speech regions in ``audio_path`` and save them."""
    if is_valid_json_file(output_path):
        LOGGER.info("Overlap detection already exists: %s", output_path)
        return output_path

    from pyannote.audio import Model
    from pyannote.audio.pipelines import OverlappedSpeechDetection

    import torch

    device = device or resolve_device()
    LOGGER.info("Loading overlap model '%s' on %s", model_name, device)
    model = Model.from_pretrained(model_name, use_auth_token=token)
    if model is None:
        raise RuntimeError(
            f"Could not load '{model_name}'. Accept its license on the Hugging "
            "Face model page, then retry with --token / $HF_TOKEN."
        )

    try:
        pipeline = OverlappedSpeechDetection(
            segmentation=model,
            use_auth_token=token,
            device=torch.device(device),
        )
    except Exception as exc:
        LOGGER.warning("Overlap inference on %s failed (%s); falling back to cpu.", device, exc)
        pipeline = OverlappedSpeechDetection(
            segmentation=model,
            use_auth_token=token,
            device=torch.device("cpu"),
        )
        device = "cpu"

    pipeline.instantiate(hyperparameters or DEFAULT_HYPERPARAMETERS)

    LOGGER.info("Running overlap detection on %s", audio_path)
    result = pipeline(str(audio_path))
    regions = extract_regions(result)
    payload: Dict = {
        "model": model_name,
        "device": device,
        "regions": regions,
    }
    LOGGER.info("Overlap detection complete: %d regions.", len(regions))
    save_json(payload, output_path)
    return output_path