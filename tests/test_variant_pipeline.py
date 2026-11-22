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


def test_dashboard_data_uses_stage_local_metrics_and_preserves_missing_ads(tmp_path):
    dashboard = _load_module("multi_dashboard", "analysis/build_multi_video_dashboard.py")
    video = tmp_path / "video"; (video / "comparison").mkdir(parents=True)
    stages = {}
    for selector, stage_name in dashboard.STAGES.items():
        directory = video / ("baseline" if selector == "baseline" else f"threshold_{selector}")
        directory.mkdir(parents=True)
        (directory / "final.json").write_text(json.dumps({"speaker_turns": [{"speaker": "S00", "start": 0, "end": 10}]}))
        stages[stage_name] = {"speaker_participation": {"by_speaker": {"S00": {
            "speech_share_percent_of_attributed_total": 100, "clean_speech_share_percent": 100,
            "attributed_speaking_seconds": 10, "clean_speech_seconds": 10,
            "speech_excluded_by_retained_overlap_seconds": 0}}},
            "host_analysis": {"identified_host_speaker": "S00"},
            "overlap_statistics": {"total_overlap_events": 2, "duration": {"total": 3}},
            "gap_analysis": {"clean_silence_duration": 4, "total_inter_turn_gap_duration": {"total": 5},
                             "gaps_containing_overlap_duration": 1, "clean_silence_gap_count": 1,
                             "gaps_containing_overlap_count": 1}}
    stages["original"] = {"timeline_statistics": {"episode_duration_seconds": 120}}
    (video / "comparison" / "five_stage_comparison.json").write_text(json.dumps({"stage_results": stages}))
    data = dashboard.build_video(video, [])
    assert data["overview"]["ads_removed"] is None
    assert data["stages"]["5.0"]["clean_silence_duration_seconds"] == 4
    assert data["stages"]["0.5"]["speaker_shares"]["S00"]["clean_percent"] == 100
