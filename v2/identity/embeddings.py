"""Milestone 5, part 1: per-speaker voice embeddings.

Builds one averaged speaker embedding per diarized speaker using pyannote's
``pyannote/embedding`` model. These vectors are the bridge to identity: they
let us match speaker labels across videos (STABLE identities) and, with
reference audio, to named individuals in the cast.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from v2.io import is_valid_json_file, load_json, save_json
from v2.speech.transcribe import resolve_device

LOGGER = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "pyannote/embedding"

# Skip turns shorter than this when computing embeddings (too little signal).
MIN_TURN_SECONDS = 0.5


def _embed_turn(inference: object, audio_path: Path, start: float, end: float):
    """Return the embedding vector for one audio window."""
    import numpy as np

    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(audio_path), frame_offset=int(start * 16000), num_frames=int((end - start) * 16000))
    if waveform.shape[1] == 0:
        return None
    embedding = inference({"waveform": waveform, "sample_rate": sample_rate})
    if torch.is_tensor(embedding):
        embedding = embedding.detach().cpu().numpy()
    return np.asarray(embedding).squeeze()


def speaker_embeddings(
    audio_path: Path,
    diarization_path: Path,
    output_path: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """Embed one vector per diarized speaker and save ``speakers -> vector``."""
    if is_valid_json_file(output_path):
        LOGGER.info("Embeddings already exist: %s", output_path)
        return output_path

    from pyannote.audio import Inference, Model

    import numpy as np

    import torch

    turns = load_json(diarization_path)["turns"]
    device = device or resolve_device()
    LOGGER.info("Loading embedding model '%s' on %s", model_name, device)
    model = Model.from_pretrained(model_name, use_auth_token=token)
    if model is None:
        raise RuntimeError(
            f"Could not load '{model_name}'. Accept its license on the Hugging "
            "Face model page, then retry with --token / $HF_TOKEN."
        )
    inference = Inference(model, window="whole", device=torch.device(device))

    accumulators: Dict[str, List] = {}
    for turn in turns:
        start, end, speaker = turn["start"], turn["end"], turn["speaker"]
        if end - start < MIN_TURN_SECONDS:
            continue
        embedding = _embed_turn(inference, audio_path, start, end)
        if embedding is None:
            continue
        accumulators.setdefault(speaker, []).append(embedding)

    speakers = {}
    for speaker, vectors in sorted(accumulators.items()):
        mean = np.mean(np.stack(vectors), axis=0)
        speakers[speaker] = {
            "mean_embedding": [float(x) for x in mean],
            "embedding_dim": int(mean.shape[0]),
            "turns_embedded": len(vectors),
        }

    payload: Dict = {
        "model": model_name,
        "device": device,
        "speakers": speakers,
    }
    LOGGER.info("Embeddings complete: %d speakers.", len(speakers))
    save_json(payload, output_path)
    return output_path