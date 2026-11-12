import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))


def _load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_variant_runner_writes_isolated_merged_analysis(tmp_path):
    runner = _load_module("variant_runner", "analysis/run_variant_pipeline.py")
    source = {
        "schema_version": "1.0",
        "episode": {"video_id": "sample", "title": "Sample", "duration_s": 12.0},
        "participants": [],
        "speaker_turns": [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01"},
        ],
        "segments": [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello?"},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01", "text": "Response."},
        ],
        "overlap_candidates": [],
        "interruptions": [],
        "osd_stats": {},
    }
    original = copy.deepcopy(source)

    runner.run_variant(tmp_path / "baseline", source, "baseline", overwrite=True)

    assert source == original
    merged = json.loads((tmp_path / "baseline" / "merge_final.json").read_text())
    assert len(merged["speaker_turns"]) == 2
    for filename in (
        "turn_gap_analysis.json",
        "speaker_transition_analysis.json",
        "final_result.json",
        "final_result_numeric.json",
        "short_result_numeric.json",
    ):
        assert (tmp_path / "baseline" / filename).is_file()
