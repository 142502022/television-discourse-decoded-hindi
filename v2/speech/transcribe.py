"""WhisperX direct ASR: transcription and word alignment, no diarization."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from v2.io import is_valid_json_file, save_json

LOGGER = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when transcription fails."""


def resolve_device() -> str:
    """Pick cuda > mps > cpu based on what is available locally."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_compute_type(device: str) -> str:
    """float16 for accelerators, int8 for CPU."""
    return "int8" if device == "cpu" else "float16"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _segment_to_record(segment: Dict[str, Any], aligned: bool) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "start": float(segment["start"]),
        "end": float(segment["end"]),
        "text": segment["text"].strip(),
    }
    if aligned:
        words = segment.get("words") or []
        record["words"] = [
            {"word": w.get("word"), "start": w.get("start"), "end": w.get("end")}
            for w in words
            if w.get("start") is not None and w.get("end") is not None
        ]
    return record


def build_transcript_payload(
    whisper_result: Dict[str, Any],
    *,
    model: str,
    device: str,
    compute_type: str,
    aligned: bool,
) -> Dict[str, Any]:
    """Map a raw WhisperX transcribe/align result into the saved artifact shape."""
    return {
        "model": model,
        "language": whisper_result.get("language"),
        "device": device,
        "compute_type": compute_type,
        "aligned": aligned,
        "segments": [
            _segment_to_record(seg, aligned)
            for seg in whisper_result.get("segments") or []
        ],
    }


def transcribe_audio(
    audio_path: Path,
    output_path: Path,
    *,
    model: str = "medium",
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Path:
    """Transcribe ``audio_path`` and save segments with word timestamps.

    Uses WhisperX transcription + wav2vec2 word alignment. Diarization is
    intentionally not run here; it is a later stage.
    """
    import whisperx

    if is_valid_json_file(output_path):
        LOGGER.info("Transcript already exists: %s", output_path)
        return output_path

    device = device or resolve_device()
    # WhisperX transcription runs on faster-whisper/CTranslate2, which only
    # supports CUDA and CPU. MPS falls back to CPU for this step; the torch
    # based word alignment can still try MPS (with CPU fallback) below.
    transcribe_device = device if device in ("cuda", "cpu") else "cpu"
    compute_type = compute_type or default_compute_type(transcribe_device)
    batch_size = batch_size or _env_int("TVD_BATCH_SIZE", 8)

    LOGGER.info(
        "Loading WhisperX model '%s' on %s (%s)",
        model,
        transcribe_device,
        compute_type,
    )
    model_obj = whisperx.load_model(
        model,
        device=transcribe_device,
        compute_type=compute_type,
        language=language,
    )

    audio = whisperx.load_audio(str(audio_path))
    transcribe_result = model_obj.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
    )

    aligned = False
    align_language = transcribe_result.get("language") or language
    for align_device in (device, "cpu"):
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=align_language,
                device=align_device,
            )
            aligned_result = whisperx.align(
                transcribe_result["segments"],
                align_model,
                align_metadata,
                audio,
                align_device,
                return_char_alignments=False,
            )
            transcribe_result["segments"] = aligned_result["segments"]
            aligned = True
            break
        except Exception as exc:
            LOGGER.warning(
                "Word alignment failed on %s (%s); falling back or skipping.",
                align_device,
                exc,
            )

    payload = build_transcript_payload(
        transcribe_result,
        model=model,
        device=transcribe_device,
        compute_type=compute_type,
        aligned=aligned,
    )
    if not aligned:
        LOGGER.warning("Saving transcript WITHOUT word-level timestamps.")

    segment_count = len(payload["segments"])
    LOGGER.info("Transcription complete: %d segments (%saligned).", segment_count, "" if aligned else "not ")
    save_json(payload, output_path)
    return output_path