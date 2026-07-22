import whisperx


def load_models(device: str = "cuda", compute_type: str = "float16"):
    whisper_model = whisperx.load_model(
        "large-v3",
        device=device,
        compute_type=compute_type,
    )
    return whisper_model


def load_alignment_model(language: str, device: str):
    return whisperx.load_align_model(
        language_code=language,
        device=device,
    )
