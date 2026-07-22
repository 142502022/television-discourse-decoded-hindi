from typing import Any, Dict, Iterable, List


def diarization_intersections(
    diarization: Iterable[Dict[str, Any]],
    min_overlap_seconds: float = 0.25,
) -> List[Dict[str, Any]]:
    """Find intervals where two diarized speaker turns overlap."""
    turns = sorted(
        [
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "speaker": str(item.get("speaker", "")),
            }
            for item in diarization
        ],
        key=lambda item: (item["start"], item["end"]),
    )
    intersections = []

    for left_index, left in enumerate(turns):
        for right in turns[left_index + 1:]:
            if right["start"] >= left["end"]:
                break
            if left["speaker"] == right["speaker"]:
                continue
            start = max(left["start"], right["start"])
            end = min(left["end"], right["end"])
            if end - start >= min_overlap_seconds:
                intersections.append(
                    {
                        "start": start,
                        "end": end,
                        "source": "diarization_intersection",
                        "speakers": sorted([left["speaker"], right["speaker"]]),
                    }
                )

    return intersections


def merge_overlap_candidates(
    osd_intervals: Iterable[Dict[str, Any]],
    diarization_overlap_intervals: Iterable[Dict[str, Any]],
    tolerance_seconds: float = 0.5,
) -> List[Dict[str, Any]]:
    """Merge OSD and diarization overlap candidates with confidence labels."""
    osd_items = [
        {
            "start": float(item["start"]),
            "end": float(item["end"]),
            "source": "pyannote_osd",
        }
        for item in osd_intervals
    ]
    diar_items = [
        {
            "start": float(item["start"]),
            "end": float(item["end"]),
            "source": "diarization_intersection",
            "speakers": item.get("speakers", []),
        }
        for item in diarization_overlap_intervals
    ]

    candidates = []
    used_diar = set()

    for osd_index, osd_item in enumerate(osd_items):
        matches = [
            (idx, diar_item)
            for idx, diar_item in enumerate(diar_items)
            if idx not in used_diar
            and _intervals_touch(osd_item, diar_item, tolerance_seconds)
        ]
        if matches:
            diar_index, diar_item = matches[0]
            used_diar.add(diar_index)
            candidates.append(
                {
                    "start": min(osd_item["start"], diar_item["start"]),
                    "end": max(osd_item["end"], diar_item["end"]),
                    "confidence": "high",
                    "reason": "pyannote_osd_and_diarization_agree",
                    "sources": ["pyannote_osd", "diarization_intersection"],
                    "speakers": diar_item.get("speakers", []),
                }
            )
        else:
            candidates.append(
                {
                    "start": osd_item["start"],
                    "end": osd_item["end"],
                    "confidence": "medium",
                    "reason": "pyannote_osd_only",
                    "sources": ["pyannote_osd"],
                    "speakers": [],
                }
            )

    for diar_index, diar_item in enumerate(diar_items):
        if diar_index in used_diar:
            continue
        candidates.append(
            {
                "start": diar_item["start"],
                "end": diar_item["end"],
                "confidence": "medium",
                "reason": "diarization_intersection_only",
                "sources": ["diarization_intersection"],
                "speakers": diar_item.get("speakers", []),
            }
        )

    return sorted(candidates, key=lambda item: (item["start"], item["end"]))


def _intervals_touch(
    left: Dict[str, Any],
    right: Dict[str, Any],
    tolerance_seconds: float,
) -> bool:
    return (
        left["start"] <= right["end"] + tolerance_seconds
        and right["start"] <= left["end"] + tolerance_seconds
    )
