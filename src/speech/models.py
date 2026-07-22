from typing import Tuple

import whisperx

def load_models(device: str="cuda", compute_type: str = "float16",):
    whisper_model = whisperx.load_model(
        "large-v3",
        device=device,
        computer_type=computer_type,
    )
    return whisper_model
def load_alignment_model(laguage: str, device:str,):
    return whisperx.load_align_model(
        langauge_code=laguage,
        device=device,
    )