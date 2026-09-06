"""Milestone 4: overlap analysis.

Pure, device-free logic that turns raw overlap regions (from a neural overlap
detector) and speaker turns (diarization) into crosstalk events, directed
interruption events, and per-speaker-pair statistics.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _round3(value: float) -> float:
    return round(float(value), 3)


def speakers_in(region: Dict[str, Any], turns: List[Dict[str, Any]]) -> List[str]:
    """Distinct speakers with turns intersecting the region, sorted by name."""
    involved = {
        turn["speaker"]
        for turn in turns
        if turn["end"] > region["start"] and turn["start"] < region["end"]
    }
    return sorted(involved)


def ongoing_at(region: Dict[str, Any], turns: List[Dict[str, Any]]) -> List[str]:
    """Speakers already holding the floor when the region begins (strict:
    they must have started speaking before the region's start)."""
    ongoing = {
        turn["speaker"]
        for turn in turns
        if turn["start"] < region["start"] < turn["end"]
    }
    return sorted(ongoing)


def interrupter_of(
    region: Dict[str, Any],
    turns: List[Dict[str, Any]],
) -> Optional[str]:
    """The speaker that starts talking inside the region while another holds
    the floor, or None if this overlap is not an interruption (e.g. spontaneous
    crosstalk where both start together)."""
    ongoing = set(ongoing_at(region, turns))
    starters = [
        turn
        for turn in turns
        if region["start"] <= turn["start"] < region["end"]
        and turn["speaker"] not in ongoing
    ]
    if not ongoing or not starters:
        return None
    starters.sort(key=lambda turn: turn["start"])
    return starters[0]["speaker"]


def overlap_events(
    regions: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach speaker attribution to raw overlap regions."""
    events = []
    for region in regions:
        involved = speakers_in(region, turns)
        events.append(
            {
                "start": _round3(region["start"]),
                "end": _round3(region["end"]),
                "duration": _round3(region["end"] - region["start"]),
                "speakers": involved,
                "interrupter": interrupter_of(region, turns),
            }
        )
    return events


def pair_stats(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate crosstalk seconds/count per unordered speaker pair."""
    seconds: Dict[Tuple[str, ...], float] = defaultdict(float)
    count: Dict[Tuple[str, ...], int] = defaultdict(int)
    for event in events:
        if len(event["speakers"]) < 2:
            continue
        key = tuple(event["speakers"])
        seconds[key] += event["duration"]
        count[key] += 1
    return [
        {"pair": list(pair), "seconds": _round3(seconds[pair]), "count": count[pair]}
        for pair in sorted(seconds)
    ]


def interruption_stats(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate directed interruption events (interrupter -> victim)."""
    seconds: Dict[Tuple[str, str], float] = defaultdict(float)
    count: Dict[Tuple[str, str], int] = defaultdict(int)
    for event in events:
        interrupter = event["interrupter"]
        if interrupter is None:
            continue
        victims_in_event = [s for s in event["speakers"] if s != interrupter]
        for victim in victims_in_event:
            key = (interrupter, victim)
            seconds[key] += event["duration"]
            count[key] += 1
    return [
        {
            "interrupter": key[0],
            "victim": key[1],
            "seconds": _round3(seconds[key]),
            "count": count[key],
        }
        for key in sorted(seconds)
    ]


def build_analysis(
    regions: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    detector_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Full OSD artifact: raw regions, attributed events, and summary stats."""
    events = overlap_events(regions, turns)
    interruptions = [e for e in events if e["interrupter"] is not None]
    total_duration = max(
        [r["end"] for r in regions] + [t["end"] for t in turns],
        default=0.0,
    )
    total_overlap = sum(e["duration"] for e in events)
    stats = {
        "total_duration_s": _round3(total_duration),
        "overlap_seconds": _round3(total_overlap),
        "num_regions": len(regions),
        "mean_overlap_s": _round3(total_overlap / len(events)) if events else 0.0,
        "max_overlap_s": _round3(max((e["duration"] for e in events), default=0.0)),
        "num_interruptions": len(interruptions),
        "by_pair": pair_stats(events),
        "interruptions_by_pair": interruption_stats(events),
    }
    return {
        "detector": dict(detector_meta),
        "regions": [{"start": r["start"], "end": r["end"]} for r in regions],
        "events": events,
        "interruptions": interruptions,
        "stats": stats,
    }