import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import whisperx

from .io import save_json

LOGGER = logging.getLogger(__name__)


def run_whisperx(
    audio_path: Path,
    output_path: Path,
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
    language: str = "en",
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Run WhisperX transcription, word alignment, and diarization."""
    if output_path.exists():
        LOGGER.info("WhisperX output already exists: %s", output_path)
        return _load_existing(output_path)

    hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("HUGGINGFACE_TOKEN is required for WhisperX diarization.")

    LOGGER.info("Running WhisperX on %s", audio_path)
    model = whisperx.load_model(
        "large-v3",
        device=device,
        compute_type=compute_type,
        language=language,
    )
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token,
        device=device,
    )
    diarize_segments = diarize_model(audio)
    assigned = whisperx.assign_word_speakers(diarize_segments, aligned)
    assigned["diarization"] = _diarization_to_records(diarize_segments)

    save_json(assigned, output_path)
    return assigned


def _diarization_to_records(diarize_segments) -> list[dict[str, Any]]:
    records = []
    for _, row in diarize_segments.iterrows():
        records.append(
            {
                "start": float(row["start"]),
                "end": float(row["end"]),
                "speaker": str(row["speaker"]),
            }
        )
    return records


def _load_existing(output_path: Path) -> Dict[str, Any]:
    import json

    with output_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)
