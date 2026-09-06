"""Milestone 3: speaker diarization with pyannote, merged into the transcript.

Produces two artifacts:
  - ``asr/diarization.json``          raw speaker turns from the pipeline
  - ``asr/diarized_transcript.json``  transcription segments (and words) tagged
                                      with a speaker label, self-contained

The pure merging logic lives above the heavy pyannote import so it can be unit
tested without loading the model stack.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from v2.io import is_valid_json_file, save_json
from v2.speech.transcribe import resolve_device

LOGGER = logging.getLogger(__name__)

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

DIARIZATION_KEY = "diarization"
SPEAKER_TURNS_KEY = "speaker_turns"


def _round3(value: float) -> float:
    return round(float(value), 3)


def dominant_speaker(
    start: float,
    end: float,
    turns: List[Dict[str, Any]],
) -> Optional[str]:
    """Speaker whose turns cover the most of ``[start, end)``, or None.

    Ties fall to whichever speaker appears first; segments with no overlap get
    no speaker (matches the "don't invent speakers" behavior of WhisperX).
    """
    best_speaker: Optional[str] = None
    best_seconds = 0.0
    for turn in turns:
        overlap = min(end, turn["end"]) - max(start, turn["start"])
        if overlap <= 0:
            continue
        if overlap > best_seconds:
            best_seconds = overlap
            best_speaker = turn["speaker"]
    return best_speaker


def assign_speakers_to_segments(
    segments: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return copies of ``segments`` with a ``speaker`` on each segment and word."""
    result: List[Dict[str, Any]] = []
    for segment in segments:
        tagged = dict(segment)
        tagged["speaker"] = dominant_speaker(segment["start"], segment["end"], turns)
        words = segment.get("words")
        if words:
            tagged["words"] = [
                dict(word) | {"speaker": dominant_speaker(word["start"], word["end"], turns)}
                for word in words
            ]
        result.append(tagged)
    return result


def build_diarized_transcript(
    transcript_payload: Dict[str, Any],
    turns: List[Dict[str, Any]],
    diarization_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge speaker turns into a copy of the transcript artifact."""
    payload = dict(transcript_payload)
    payload["segments"] = assign_speakers_to_segments(transcript_payload["segments"], turns)
    payload[DIARIZATION_KEY] = dict(diarization_meta)
    payload[SPEAKER_TURNS_KEY] = turns
    return payload


def extract_turns(diarization: Any) -> List[Dict[str, Any]]:
    """Flatten a pyannote diarization result into [[start, end, speaker]] dicts."""
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            {
                "start": _round3(turn.start),
                "end": _round3(turn.end),
                "speaker": str(speaker),
            }
        )
    return turns


def load_pyannote_pipeline(model_name: str, token: Optional[str]) -> Any:
    """Load a pyannote pipeline, raising a clear error if it is gated/None.

    pyannote's ``from_pretrained`` returns ``None`` (and prints a hint) when it
    cannot load the model, so a plain call is not enough.
    """
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(
            f"Could not load '{model_name}'. It may be gated: accept its license "
            "on the Hugging Face model page, then retry with --token / $HF_TOKEN."
        )
    return pipeline


def diarize_audio(
    audio_path: Path,
    output_path: Path,
    *,
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    device: Optional[str] = None,
    token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Path:
    """Run speaker diarization on ``audio_path`` and save the turns.

    Skips (returns the path unchanged) if ``output_path`` already holds a valid
    artifact. Pyannote models are gated on Hugging Face; ``token`` is the HF
    access token (or set ``HF_TOKEN`` before calling).
    """
    if is_valid_json_file(output_path):
        LOGGER.info("Diarization already exists: %s", output_path)
        return output_path

    import torch

    device = device or resolve_device()
    LOGGER.info("Loading diarization pipeline '%s' on %s", model_name, device)
    pipeline = load_pyannote_pipeline(model_name, token)
    if device != "cpu":
        try:
            pipeline = pipeline.to(torch.device(device))
        except Exception as exc:
            LOGGER.warning("Diarization on %s failed (%s); falling back to cpu.", device, exc)
            pipeline = pipeline.to(torch.device("cpu"))
            device = "cpu"

    LOGGER.info("Running diarization on %s", audio_path)
    diarization = pipeline(
        str(audio_path),
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    turns = extract_turns(diarization)
    payload = {
        "model": model_name,
        "device": device,
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "turns": turns,
    }
    LOGGER.info("Diarization complete: %d turns.", len(turns))
    save_json(payload, output_path)
    return output_path


def build_diarized_transcript_file(
    transcript_path: Path,
    diarization_path: Path,
    output_path: Path,
) -> Path:
    """Merge an existing transcription artifact with a diarization artifact."""
    from v2.io import load_json

    transcript = load_json(transcript_path)
    diarization = load_json(diarization_path)
    payload = build_diarized_transcript(
        transcript,
        diarization["turns"],
        {key: diarization[key] for key in ("model", "device", "num_speakers")},
    )
    save_json(payload, output_path)
    return output_path