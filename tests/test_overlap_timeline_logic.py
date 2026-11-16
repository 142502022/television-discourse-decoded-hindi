import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))


def module(name):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_merge_preserves_same_speaker_turns_across_genuine_silence():
    merger = module("merge_speaker_turns")
    turns = [
        {"start": 1.0, "end": 5.0, "speaker": "S00"},
        {"start": 7.0, "end": 10.0, "speaker": "S00"},
    ]
    assert len(merger.merge_speaker_turns(turns)) == 2
    assert len(merger.merge_speaker_turns(turns, merge_gap=2.0)) == 1


def test_gap_statistics_do_not_treat_overlap_as_clean_silence():
    gaps = module("build_turn_gap_analysis")
    turns = [
        {"turn_index": 0, "speaker": "S00", "start": 1.0, "end": 5.0, "duration": 4.0},
        {"turn_index": 1, "speaker": "S01", "start": 7.0, "end": 9.0, "duration": 2.0},
    ]
    raw_gaps = gaps.calculate_gaps(turns, [{"start": 5.5, "end": 6.0}])
    result = gaps.build_statistics(turns, raw_gaps, gaps.merge_turns(turns, raw_gaps, 1.0))
    assert result["debate_level"]["total_gap_statistics"]["count"] == 1
    assert result["debate_level"]["clean_gap_count"] == 0


def test_merged_osd_episode_is_thresholded_by_span_and_preserves_sources():
    overlap = module("merge_overlap_regions")
    preprocess = module("preprocess_overlap")
    original = [{"start": 10.0, "end": 10.5, "speakers": ["S00", "S01"]},
                {"start": 11.0, "end": 11.7, "speakers": ["S01", "S02"]}]
    merged, _ = overlap.merge_overlap_regions(original)
    assert merged[0]["end"] - merged[0]["start"] == 1.6999999999999993
    kept, removed, identities = preprocess.filter_merged_osd_regions(merged, 1.0)
    assert len(kept) == 1 and not removed and not identities
    kept, removed, identities = preprocess.filter_merged_osd_regions(merged, 2.0)
    assert not kept and len(removed) == 1 and len(identities) == 2
    assert original == [{"start": 10.0, "end": 10.5, "speakers": ["S00", "S01"]},
                        {"start": 11.0, "end": 11.7, "speakers": ["S01", "S02"]}]


def test_floor_transition_classification_is_exclusive(tmp_path):
    threshold = module("threshold")
    data = {"episode": {"video_id": "sample"}, "overlap_candidates": [{"start": 4.0, "end": 7.0, "speakers": ["S00", "S01"]}],
            "speaker_turns": [{"start": 1.0, "end": 5.0, "speaker": "S00"},
                              {"start": 4.0, "end": 7.0, "speaker": "S01"},
                              {"start": 7.0, "end": 11.0, "speaker": "S01"}]}
    path = tmp_path / "final.json"
    path.write_text(json.dumps(data))
    result = threshold.analyze(path)
    totals = result["transition_analysis"]["totals"]
    assert totals == {"confirmed": 1, "candidate": 0, "uncertain": 0, "overlap_related": 1}

    resumed = copy.deepcopy(data)
    resumed["speaker_turns"][-1]["speaker"] = "S00"
    path.write_text(json.dumps(resumed))
    result = threshold.analyze(path)
    totals = result["transition_analysis"]["totals"]
    assert totals == {"confirmed": 0, "candidate": 0, "uncertain": 1, "overlap_related": 0}
