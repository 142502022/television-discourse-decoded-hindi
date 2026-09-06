"""Per-video directory layout for the V2 pipeline."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LAYOUT_ENV_VAR = "TVD_DATA_DIR"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "v2" / "videos"


def default_data_dir() -> Path:
    """Return the configured data root, from env or the repo default."""
    configured = os.environ.get(LAYOUT_ENV_VAR)
    if configured:
        return Path(configured)
    return DEFAULT_DATA_DIR


@dataclass(frozen=True)
class VideoLayout:
    """Fixed, deterministic paths for one video's artifacts."""

    video_id: str
    root: Path

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def proxy_dir(self) -> Path:
        return self.root / "proxy"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def asr_dir(self) -> Path:
        return self.root / "asr"

    @property
    def osd_dir(self) -> Path:
        return self.root / "osd"

    @property
    def identity_dir(self) -> Path:
        return self.root / "identity"

    @property
    def original_video(self) -> Path:
        return self.source_dir / "original.mp4"

    @property
    def audio_path(self) -> Path:
        return self.audio_dir / "audio.wav"

    @property
    def proxy_path(self) -> Path:
        return self.proxy_dir / "proxy_480p.mp4"

    @property
    def metadata_path(self) -> Path:
        return self.metadata_dir / "metadata.json"

    @property
    def asr_path(self) -> Path:
        return self.asr_dir / "transcription.json"

    @property
    def diarization_path(self) -> Path:
        return self.asr_dir / "diarization.json"

    @property
    def diarized_transcript_path(self) -> Path:
        return self.asr_dir / "diarized_transcript.json"

    @property
    def overlap_path(self) -> Path:
        return self.osd_dir / "overlap.json"

    @property
    def analysis_path(self) -> Path:
        return self.osd_dir / "analysis.json"

    @property
    def embeddings_path(self) -> Path:
        return self.identity_dir / "embeddings.json"

    @property
    def roles_path(self) -> Path:
        return self.identity_dir / "roles.json"


def build_layout(video_id: str, data_dir: Optional[Path] = None) -> VideoLayout:
    """Create and return the layout for one video under the data root.

    Future processing stages (asr, diarization, osd, overlap, roster, final)
    will add their own sibling directories under ``root``.
    """
    root = data_dir if data_dir is not None else default_data_dir()
    layout = VideoLayout(video_id=video_id, root=root / video_id)
    for directory in (
        layout.root,
        layout.source_dir,
        layout.audio_dir,
        layout.proxy_dir,
        layout.metadata_dir,
        layout.asr_dir,
        layout.osd_dir,
        layout.identity_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout