"""Milestone 6: reconcile overlap alarms from OSD and diarization.

Pure, device-free logic that classifies every overlapping-speech candidate
interval into one of three cases:

- ``AGREE``            : OSD fires AND >= 2 diarized speakers are active there.
                          Highest confidence.
- ``OSD_ONLY``         : OSD fires but diarization shows < 2 active clusters.
- ``DIARIZATION_ONLY`` : >= 2 speaker turns intersect but OSD did not fire.

The merged list of classified intervals is what downstream stages (roster
extraction, annotation) consume.
"""

from typing import Any, Dict, List, Tuple

from v2.osd.overlap import speakers_in

# Confidence weights assigned to each case. AGREE is the most trustworthy
# (two independent detectors agree); OSD_ONLY and DIARIZATION_ONLY rely on a
# single detector and are therefore rated lower. Tunable.
CONFIDENCE_WEIGHTS = {
    "AGREE": 1.0,
    "OSD_ONLY": 0.7,
    "DIARIZATION_ONLY": 0.6,
}

MIN_INTERSECTION_S = 0.1  # ignore sub-100ms turn fights as noise
MIN_CANDIDATE_S = 0.1     # ignore candidate intervals shorter than this
COVERED_RATIO = 0.5       # diarization intersection "covered" by OSD if this
                          # fraction of it lies inside OSD regions


def _round3(value: float) -> float:
    return round(float(value), 3)


def speaker_intersections(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Intervals where two different diarized speakers vocalize together."""
    overlaps: List[Dict[str, Any]] = []
    turns = sorted(turns, key=lambda t: (t["start"], t["end"]))
    for i, left in enumerate(turns):
        for right in turns[i + 1 :]:
            if right["start"] >= left["end"]:
                break
            if left["speaker"] == right["speaker"]:
                continue
            start = max(left["start"], right["start"])
            end = min(left["end"], right["end"])
            if end - start >= MIN_INTERSECTION_S:
                overlaps.append(
                    {
                        "start": _round3(start),
                        "end": _round3(end),
                        "speakers": sorted({left["speaker"], right["speaker"]}),
                    }
                )
    overlaps.sort(key=lambda i: (i["start"], i["end"]))
    return overlaps


def covered_by_osd(
    interval: Dict[str, Any],
    regions: List[Dict[str, Any]],
    ratio: float = COVERED_RATIO,
) -> bool:
    """True if at least ``ratio`` of the interval sits inside OSD regions."""
    if interval["end"] - interval["start"] <= 0:
        return False
    covered = 0.0
    for region in regions:
        start = max(interval["start"], region["start"])
        end = min(interval["end"], region["end"])
        if end > start:
            covered += end - start
    return covered / (interval["end"] - interval["start"]) >= ratio


def classify_candidates(
    regions: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    min_candidate_s: float = MIN_CANDIDATE_S,
    covered_ratio: float = COVERED_RATIO,
) -> List[Dict[str, Any]]:
    """Classify OSD regions and mutually-overlapping speaker turns into the
    three cases, returning one candidate interval per alarm."""
    candidates: List[Dict[str, Any]] = []

    for region in regions:
        if region["end"] - region["start"] < min_candidate_s:
            continue
        active = speakers_in(region, turns)
        case = "AGREE" if len(active) >= 2 else "OSD_ONLY"
        candidates.append(
            {
                "case": case,
                "start": _round3(region["start"]),
                "end": _round3(region["end"]),
                "duration": _round3(region["end"] - region["start"]),
                "speakers": active,
                "source": "osd",
                "confidence": CONFIDENCE_WEIGHTS[case],
            }
        )

    for interval in speaker_intersections(turns):
        if interval["end"] - interval["start"] < min_candidate_s:
            continue
        if covered_by_osd(interval, regions, covered_ratio):
            continue  # already corroborated as an AGREE candidate
        candidates.append(
            {
                "case": "DIARIZATION_ONLY",
                "start": interval["start"],
                "end": interval["end"],
                "duration": _round3(interval["end"] - interval["start"]),
                "speakers": interval["speakers"],
                "source": "diarization",
                "confidence": CONFIDENCE_WEIGHTS["DIARIZATION_ONLY"],
            }
        )

    candidates.sort(key=lambda c: (c["start"], c["end"]))
    return candidates


def summarize(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts per case plus an ordered list of the case labels."""
    counts: Dict[str, int] = {}
    for candidate in candidates:
        counts[candidate["case"]] = counts.get(candidate["case"], 0) + 1
    return {
        "num_candidates": len(candidates),
        "num_agree": counts.get("AGREE", 0),
        "num_osd_only": counts.get("OSD_ONLY", 0),
        "num_diarization_only": counts.get("DIARIZATION_ONLY", 0),
        "confidence_weights": dict(CONFIDENCE_WEIGHTS),
    }


def build_verification(
    regions: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    min_candidate_s: float = MIN_CANDIDATE_S,
    covered_ratio: float = COVERED_RATIO,
) -> Dict[str, Any]:
    """Full step-6 artifact: classified candidate intervals + tally."""
    candidates = classify_candidates(regions, turns, min_candidate_s, covered_ratio)
    return {
        "method": "three-case-reconciliation",
        "cases": sorted(CONFIDENCE_WEIGHTS),
        "summary": summarize(candidates),
        "candidates": candidates,
    }