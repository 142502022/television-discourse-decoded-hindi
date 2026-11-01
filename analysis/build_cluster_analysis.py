
#!/usr/bin/env python3
"""
Build overlap_cluster_analysis.json for every video_id directory.

Cluster A:
    Clean turn transfer — analyzed across ALL sequential speaker transitions.

Clusters B-G:
    Classified using the existing overlap classification logic and
    overlap_followup_analysis.json.

Inputs per video:
    final.json
    overlap_followup_analysis.json

Output per video:
    overlap_cluster_analysis.json

Original source files are never modified.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path.cwd()
VIDEOS_DIR = PROJECT_ROOT / "data" / "v2" / "videos"

DEFAULT_FOLLOWUP_WINDOW = 3.0
DEFAULT_PROLONGED_OVERLAP = 1.0
DEFAULT_OVERLAP_MERGE_GAP = 1.0
DEFAULT_CLEAN_TRANSITION_MAX_GAP = 1.0

CLUSTER_NAMES = {
    "A": "clean_turn_transfer",
    "B": "short_supportive_switch",
    "C": "short_disagreement",
    "D": "successful_interruption",
    "E": "failed_interruption",
    "F": "unresolved_floor_competition",
    "G": "prolonged_overlap",
}

SUPPORTIVE_FUNCTIONS = {
    "supporting",
    "agreeing",
    "backchanneling",
    "acknowledging",
}

DISAGREEMENT_FUNCTIONS = {
    "challenging",
    "disagreeing",
    "contradicting",
    "rejecting",
}


# =========================================================
# UTILITIES
# =========================================================

def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def valid_interval(item):
    if not isinstance(item, dict):
        return False

    start = to_float(item.get("start"))
    end = to_float(item.get("end"))

    return (
        start is not None
        and end is not None
        and end > start
    )


def normalize_speaker(speaker):
    if speaker is None:
        return None

    return str(speaker)


def interval_intersects(a_start, a_end, b_start, b_end):
    return a_start < b_end and a_end > b_start


def normalize_text(value):
    if value is None:
        return ""

    return " ".join(str(value).strip().lower().split())


# =========================================================
# SOURCE EXTRACTION
# =========================================================

def extract_speaker_turns(final_data):
    """
    Extract original diarization speaker turns from final.json.

    Expected:
        final_data["speaker_turns"]

    Do not use merged turns from turn_gap_analysis.
    """

    turns = final_data.get("speaker_turns", [])

    result = []

    for index, turn in enumerate(turns):

        if not valid_interval(turn):
            continue

        speaker = normalize_speaker(turn.get("speaker"))

        if not speaker:
            continue

        result.append({
            "source_turn_index": index,
            "speaker": speaker,
            "start": float(turn["start"]),
            "end": float(turn["end"]),
        })

    return sorted(
        result,
        key=lambda x: (
            x["start"],
            x["end"],
            x["source_turn_index"],
        ),
    )


def extract_overlap_candidates(final_data):
    """
    Extract reconciled overlap candidates from final.json.

    Expected:
        final_data["overlap_candidates"]
    """

    candidates = final_data.get("overlap_candidates", [])

    result = []

    for index, item in enumerate(candidates):

        if not valid_interval(item):
            continue

        speakers = item.get("speakers", [])

        result.append({
            "source_overlap_index": index,
            "start": float(item["start"]),
            "end": float(item["end"]),
            "duration": float(item["end"]) - float(item["start"]),
            "speakers": [
                normalize_speaker(speaker)
                for speaker in speakers
                if speaker is not None
            ],
            "case": item.get("case"),
            "source": item.get("source"),
            "confidence": item.get("confidence"),
        })

    return sorted(
        result,
        key=lambda x: (x["start"], x["end"]),
    )


def extract_followup_events(followup_data):
    if not isinstance(followup_data, dict):
        return []

    return followup_data.get("overlap_events", [])


# =========================================================
# TRANSITION EXTRACTION
# =========================================================

def build_transitions(speaker_turns):
    """
    Build ALL sequential speaker transitions from original turns.

    Consecutive turns with the same speaker are not transitions.
    Negative gaps represent overlapping turns.

    Cluster A candidates are derived from non-overlapping transitions
    with a small nonnegative gap and no nearby reconciled overlap.
    """

    transitions = []

    for index in range(1, len(speaker_turns)):

        previous_turn = speaker_turns[index - 1]
        current_turn = speaker_turns[index]

        previous_speaker = previous_turn["speaker"]
        current_speaker = current_turn["speaker"]

        if previous_speaker == current_speaker:
            continue

        gap = current_turn["start"] - previous_turn["end"]

        transitions.append({
            "transition_id": f"ST{len(transitions) + 1:05d}",
            "from_speaker": previous_speaker,
            "to_speaker": current_speaker,
            "timestamp": round(current_turn["start"], 3),
            "previous_turn_end": round(previous_turn["end"], 3),
            "current_turn_start": round(current_turn["start"], 3),
            "turn_gap": round(gap, 3),
            "is_overlapping_transition": gap < 0,
        })

    return transitions


def classify_cluster_a_transitions(
    transitions,
    overlaps,
    clean_gap_max=DEFAULT_CLEAN_TRANSITION_MAX_GAP,
):
    """
    Analyze Cluster A across the complete transition list.

    A candidate must:
      1. Have a nonnegative gap.
      2. Have a gap <= clean_gap_max.
      3. Have no reconciled overlap intersecting the transition boundary.

    This is a structural candidate, not definitive semantic proof.
    """

    classified = []

    for transition in transitions:

        gap = transition["turn_gap"]
        transition_time = transition["timestamp"]

        overlap_nearby = False

        for overlap in overlaps:

            overlap_start = overlap["start"]
            overlap_end = overlap["end"]

            # Overlap touching or crossing the transition boundary.
            if (
                overlap_start <= transition_time
                and overlap_end >= transition["previous_turn_end"]
            ):
                overlap_nearby = True
                break

        is_candidate = (
            0 <= gap <= clean_gap_max
            and not overlap_nearby
            and not transition["is_overlapping_transition"]
        )

        record = dict(transition)

        record["overlap_near_transition"] = overlap_nearby

        if is_candidate:
            record["primary_cluster"] = "A"
            record["cluster_name"] = CLUSTER_NAMES["A"]
            record["classification_status"] = "provisional_candidate"
            record["classification_evidence"] = [
                "Sequential speaker transition",
                f"Turn gap {gap:.3f}s is within configured threshold",
                "No reconciled overlap detected near transition",
            ]
            record["classification_note"] = (
                "Structural candidate for clean turn transfer. "
                "Conversational context is needed for confirmation."
            )

        else:
            record["primary_cluster"] = None
            record["cluster_name"] = "not_cluster_A"
            record["classification_status"] = "not_automatically_classified"
            record["classification_evidence"] = []
            record["classification_note"] = (
                "Transition does not satisfy the structural "
                "criteria for an automatic Cluster A candidate."
            )

        classified.append(record)

    return classified


# =========================================================
# OVERLAP MERGING
# =========================================================

def merge_overlap_regions(overlaps, merge_gap=1.0):
    """
    Preserve your previous temporal merging logic.

    Merge overlap regions if the temporal gap between them
    is <= merge_gap. Original source indices are retained.
    """

    if not overlaps:
        return []

    sorted_overlaps = sorted(
        overlaps,
        key=lambda x: (x["start"], x["end"]),
    )

    merged = []

    current = {
        "merged_overlap_id": "MO0001",
        "start": sorted_overlaps[0]["start"],
        "end": sorted_overlaps[0]["end"],
        "source_overlap_indices": [
            sorted_overlaps[0]["source_overlap_index"]
        ],
        "speakers": set(sorted_overlaps[0]["speakers"]),
    }

    for overlap in sorted_overlaps[1:]:

        gap = overlap["start"] - current["end"]

        if gap <= merge_gap:

            current["end"] = max(
                current["end"],
                overlap["end"],
            )

            current["source_overlap_indices"].append(
                overlap["source_overlap_index"]
            )

            current["speakers"].update(
                overlap["speakers"]
            )

        else:

            current["duration"] = (
                current["end"] - current["start"]
            )

            current["speakers"] = sorted(
                current["speakers"]
            )

            merged.append(current)

            current = {
                "merged_overlap_id":
                    f"MO{len(merged) + 1:04d}",
                "start": overlap["start"],
                "end": overlap["end"],
                "source_overlap_indices": [
                    overlap["source_overlap_index"]
                ],
                "speakers": set(overlap["speakers"]),
            }

    current["duration"] = current["end"] - current["start"]
    current["speakers"] = sorted(current["speakers"])
    merged.append(current)

    return merged


# =========================================================
# FOLLOW-UP HELPERS
# =========================================================

def get_followup_speakers(event):
    if not event:
        return None, []

    observations = event.get("observations", {})

    first_speaker = observations.get(
        "first_new_speaker_after_overlap"
    )

    speakers_in_window = observations.get(
        "speakers_in_window", []
    )

    return first_speaker, speakers_in_window


def get_event_functions(event):
    """
    Preserve existing conversational-function labels.
    Do not create new predictions here.
    """

    if not event:
        return []

    functions = []

    for speech_event in event.get("speech_events", []):

        function = speech_event.get(
            "conversational_function"
        )

        if function:
            functions.append(
                normalize_text(function)
            )

    return functions


def match_followup_event(overlap, followup_events):
    """
    Match by overlap timestamps instead of list position.

    Timestamp tolerance accounts for rounding differences.
    """

    tolerance = 0.5

    overlap_start = overlap["start"]
    overlap_end = overlap["end"]

    best_match = None
    best_distance = None

    for event in followup_events:

        observations = event.get("observations", {})

        event_start = to_float(
            event.get("overlap_start", event.get("start"))
        )

        event_end = to_float(
            event.get("overlap_end", event.get("end"))
        )

        # Some follow-up formats store timestamps in observations.
        if event_start is None:
            event_start = to_float(
                observations.get("overlap_start")
            )

        if event_end is None:
            event_end = to_float(
                observations.get("overlap_end")
            )

        if event_start is None or event_end is None:
            continue

        start_distance = abs(event_start - overlap_start)
        end_distance = abs(event_end - overlap_end)

        if (
            start_distance <= tolerance
            and end_distance <= tolerance
        ):
            distance = start_distance + end_distance

            if best_distance is None or distance < best_distance:
                best_match = event
                best_distance = distance

    return best_match


# =========================================================
# B-G CLASSIFICATION
# =========================================================

def classify_overlap_event(
    overlap,
    followup_event,
    prolonged_threshold=DEFAULT_PROLONGED_OVERLAP,
):
    """
    Preserve the original B-G classification logic.

    Priority:
        G: Prolonged overlap
        D/E: Interruption outcome using post-overlap evidence
        B: Supportive conversational function
        C: Disagreement conversational function
        F: Unresolved competition
        Otherwise: Unclassified

    A is deliberately NOT classified here.
    """

    duration = overlap["duration"]
    speakers = overlap["speakers"]

    first_new_speaker, speakers_in_window = (
        get_followup_speakers(followup_event)
        if followup_event
        else (None, [])
    )

    functions = (
        get_event_functions(followup_event)
        if followup_event
        else []
    )

    # G: Prolonged overlap
    if duration >= prolonged_threshold:

        return {
            "cluster": "G",
            "cluster_name": CLUSTER_NAMES["G"],
            "classification_status": "classified",
            "evidence": [
                f"Overlap duration {duration:.3f}s "
                f">= threshold {prolonged_threshold:.3f}s"
            ],
        }

    # D/E: post-overlap floor outcome
    if first_new_speaker and len(speakers) >= 2:

        interrupted_speaker = next(
            (
                speaker
                for speaker in speakers
                if speaker != first_new_speaker
            ),
            None,
        )

        if interrupted_speaker:

            # E: original overlapping speaker resumes
            if interrupted_speaker in speakers_in_window:

                return {
                    "cluster": "E",
                    "cluster_name": CLUSTER_NAMES["E"],
                    "classification_status": "classified",
                    "evidence": [
                        "Original overlapping speaker resumed "
                        "within the post-overlap observation window."
                    ],
                }

            # D: provisional floor retention
            return {
                "cluster": "D",
                "cluster_name": CLUSTER_NAMES["D"],
                "classification_status": "provisional",
                "evidence": [
                    "A new speaker entered the post-overlap window.",
                    "Floor retention requires contextual confirmation.",
                ],
            }

    # B: supportive overlap
    if any(
        function in SUPPORTIVE_FUNCTIONS
        for function in functions
    ):

        return {
            "cluster": "B",
            "cluster_name": CLUSTER_NAMES["B"],
            "classification_status": "classified",
            "evidence": [
                "Existing follow-up conversational-function "
                "label indicates supportive acknowledgment."
            ],
        }

    # C: disagreement overlap
    if any(
        function in DISAGREEMENT_FUNCTIONS
        for function in functions
    ):

        return {
            "cluster": "C",
            "cluster_name": CLUSTER_NAMES["C"],
            "classification_status": "classified",
            "evidence": [
                "Existing follow-up conversational-function "
                "label indicates disagreement/challenge."
            ],
        }

    # F: unresolved competition
    if len(speakers) >= 2 and not first_new_speaker:

        return {
            "cluster": "F",
            "cluster_name": CLUSTER_NAMES["F"],
            "classification_status": "provisional",
            "evidence": [
                "Multiple speakers overlap.",
                "No clear first new speaker identified after overlap.",
            ],
        }

    return {
        "cluster": None,
        "cluster_name": "unclassified",
        "classification_status": "pending",
        "evidence": [
            "Available structural and follow-up evidence "
            "is insufficient for reliable classification."
        ],
    }


# =========================================================
# B-G STATISTICS
# =========================================================

def build_overlap_cluster_statistics(classified_events):
    counts = Counter()
    speaker_counts = defaultdict(Counter)

    classified_count = 0

    for event in classified_events:

        cluster = event.get("primary_cluster")

        if cluster not in ("B", "C", "D", "E", "F", "G"):
            continue

        counts[cluster] += 1
        classified_count += 1

        for speaker in event.get("speakers", []):
            speaker_counts[speaker][cluster] += 1

    all_clusters = ["B", "C", "D", "E", "F", "G"]

    percentages = {
        cluster: (
            round(counts[cluster] / classified_count * 100, 2)
            if classified_count
            else 0.0
        )
        for cluster in all_clusters
    }

    return {
        "debate_level": {
            "total_overlap_events": len(classified_events),
            "classified_event_count": classified_count,
            "unclassified_event_count": (
                len(classified_events) - classified_count
            ),
            "cluster_counts": {
                cluster: counts[cluster]
                for cluster in all_clusters
            },
            "cluster_percentages": percentages,
        },
        "speaker_level": {
            speaker: {
                "cluster_counts": {
                    cluster: speaker_counts[speaker][cluster]
                    for cluster in all_clusters
                }
            }
            for speaker in sorted(speaker_counts)
        },
    }


# =========================================================
# CLUSTER A STATISTICS
# =========================================================

def build_cluster_a_statistics(transitions):
    """
    Statistics are computed across ALL speaker transitions.

    A is a separate transition category and is not included
    in B-G overlap percentages.
    """

    candidate_a = [
        transition
        for transition in transitions
        if transition.get("primary_cluster") == "A"
    ]

    pairwise_counts = Counter(
        (
            transition["from_speaker"],
            transition["to_speaker"],
        )
        for transition in transitions
    )

    outgoing = Counter(
        transition["from_speaker"]
        for transition in transitions
    )

    incoming = Counter(
        transition["to_speaker"]
        for transition in transitions
    )

    return {
        "total_speaker_transitions": len(transitions),
        "cluster_A_candidate_count": len(candidate_a),
        "cluster_A_candidate_percentage": (
            round(
                len(candidate_a) / len(transitions) * 100,
                2,
            )
            if transitions
            else 0.0
        ),
        "pairwise_transition_counts": [
            {
                "from_speaker": from_speaker,
                "to_speaker": to_speaker,
                "count": count,
            }
            for (from_speaker, to_speaker), count
            in sorted(pairwise_counts.items())
        ],
        "speaker_transition_totals": {
            speaker: {
                "outgoing_transitions": outgoing[speaker],
                "incoming_transitions": incoming[speaker],
            }
            for speaker in sorted(
                set(outgoing) | set(incoming)
            )
        },
    }


# =========================================================
# ARTIFACT CONSTRUCTION
# =========================================================

def build_artifact(
    final_data,
    followup_data,
    merge_gap=DEFAULT_OVERLAP_MERGE_GAP,
    prolonged_threshold=DEFAULT_PROLONGED_OVERLAP,
    clean_gap_max=DEFAULT_CLEAN_TRANSITION_MAX_GAP,
):
    episode = final_data.get("episode", {})

    speaker_turns = extract_speaker_turns(final_data)
    overlaps = extract_overlap_candidates(final_data)

    # ALL transitions, including non-overlap transitions
    transitions = build_transitions(speaker_turns)

    # Cluster A classification is separate from overlap clusters
    classified_transitions = classify_cluster_a_transitions(
        transitions,
        overlaps,
        clean_gap_max=clean_gap_max,
    )

    merged_overlaps = merge_overlap_regions(
        overlaps,
        merge_gap=merge_gap,
    )

    followup_events = extract_followup_events(
        followup_data
    )

    classified_events = []

    for index, overlap in enumerate(overlaps):

        followup_event = match_followup_event(
            overlap,
            followup_events,
        )

        classification = classify_overlap_event(
            overlap,
            followup_event,
            prolonged_threshold=prolonged_threshold,
        )

        classified_events.append({
            "event_id": f"OC{index + 1:04d}",
            "source_overlap_index": (
                overlap["source_overlap_index"]
            ),
            "start": round(overlap["start"], 3),
            "end": round(overlap["end"], 3),
            "duration": round(overlap["duration"], 3),
            "speakers": overlap["speakers"],
            "case": overlap["case"],
            "source": overlap["source"],
            "confidence": overlap["confidence"],
            "primary_cluster": classification["cluster"],
            "cluster_name": classification["cluster_name"],
            "classification_status": (
                classification["classification_status"]
            ),
            "classification_evidence": classification["evidence"],
            "followup_matched": followup_event is not None,
        })

    overlap_statistics = build_overlap_cluster_statistics(
        classified_events
    )

    transition_statistics = build_cluster_a_statistics(
        classified_transitions
    )

    return {
        "schema_version": "overlap-cluster-analysis-2.0",

        "episode": {
            "video_id": episode.get("video_id"),
            "title": episode.get("title"),
            "duration_s": episode.get("duration"),
        },

        "configuration": {
            "overlap_merge_gap_seconds": merge_gap,
            "prolonged_overlap_threshold_seconds": (
                prolonged_threshold
            ),
            "post_overlap_followup_window_seconds": (
                followup_data.get(
                    "window_seconds",
                    DEFAULT_FOLLOWUP_WINDOW,
                )
                if isinstance(followup_data, dict)
                else DEFAULT_FOLLOWUP_WINDOW
            ),
            "cluster_A_clean_transition_max_gap_seconds": (
                clean_gap_max
            ),
            "primary_cluster_policy": (
                "A is classified from all speaker transitions. "
                "B-G are classified from overlap events. "
                "A is excluded from overlap-event percentages."
            ),
        },

        "source": {
            "final_json": "final.json",
            "followup_json": "overlap_followup_analysis.json",
            "original_speaker_turn_count": len(speaker_turns),
            "all_speaker_transition_count": len(transitions),
            "raw_overlap_candidate_count": len(overlaps),
            "followup_event_count": len(followup_events),
        },

        "cluster_definitions": CLUSTER_NAMES,

        "statistics": {
            "speaker_transitions": transition_statistics,
            "overlap_interaction_clusters": (
                overlap_statistics
            ),
        },

        # A: all sequential transitions
        "speaker_transitions": classified_transitions,

        # B-G: overlap events
        "merged_overlap_regions": merged_overlaps,
        "classified_events": classified_events,

        "notes": [
            "This is a separate analytical artifact.",
            "final.json and overlap_followup_analysis.json "
            "are not modified.",
            "Cluster A is analyzed across all sequential "
            "speaker transitions, not only overlaps.",
            "Cluster A candidates are structural and require "
            "contextual confirmation.",
            "Clusters B-G are based on overlap interaction events.",
            "D/E floor outcomes should be interpreted cautiously; "
            "timing alone does not prove interruption success.",
            "Unclassified events are preserved rather than forced "
            "into a cluster.",
        ],
    }


# =========================================================
# PROCESS ONE VIDEO
# =========================================================

def process_episode(
    video_dir,
    merge_gap,
    prolonged_threshold,
    clean_gap_max,
):
    final_path = video_dir / "final.json"
    followup_path = (
        video_dir / "overlap_followup_analysis.json"
    )

    if not final_path.exists():
        print(f"[SKIP] Missing {final_path}")
        return

    if not followup_path.exists():
        print(f"[SKIP] Missing {followup_path}")
        return

    final_data = load_json(final_path)
    followup_data = load_json(followup_path)

    artifact = build_artifact(
        final_data,
        followup_data,
        merge_gap=merge_gap,
        prolonged_threshold=prolonged_threshold,
        clean_gap_max=clean_gap_max,
    )

    output_path = (
        video_dir / "overlap_cluster_analysis.json"
    )

    save_json(artifact, output_path)

    transition_stats = (
        artifact["statistics"]["speaker_transitions"]
    )

    overlap_stats = (
        artifact["statistics"]
        ["overlap_interaction_clusters"]
        ["debate_level"]
    )

    print(f"\nEpisode: {video_dir.name}")
    print(
        "All speaker transitions:",
        transition_stats["total_speaker_transitions"],
    )
    print(
        "Cluster A candidates:",
        transition_stats["cluster_A_candidate_count"],
    )
    print(
        "Raw overlap candidates:",
        overlap_stats["total_overlap_events"],
    )
    print(
        "B-G classified:",
        overlap_stats["classified_event_count"],
    )
    print(
        "B-G unclassified:",
        overlap_stats["unclassified_event_count"],
    )
    print(f"Output: {output_path}")


# =========================================================
# MAIN: BATCH PROCESS ALL video_id DIRECTORIES
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build overlap_cluster_analysis.json "
            "for each video_id directory."
        )
    )

    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Process only one video ID.",
    )

    parser.add_argument(
        "--merge-gap",
        type=float,
        default=DEFAULT_OVERLAP_MERGE_GAP,
        help="Merge overlap regions when gap <= this threshold.",
    )

    parser.add_argument(
        "--prolonged-threshold",
        type=float,
        default=DEFAULT_PROLONGED_OVERLAP,
        help="Duration threshold for Cluster G.",
    )

    parser.add_argument(
        "--clean-gap-max",
        type=float,
        default=DEFAULT_CLEAN_TRANSITION_MAX_GAP,
        help="Maximum gap for Cluster A structural candidates.",
    )

    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=VIDEOS_DIR,
        help="Root directory containing video_id folders.",
    )

    args = parser.parse_args()

    if args.merge_gap < 0:
        parser.error("--merge-gap cannot be negative.")

    if args.prolonged_threshold <= 0:
        parser.error(
            "--prolonged-threshold must be greater than zero."
        )

    if args.clean_gap_max < 0:
        parser.error("--clean-gap-max cannot be negative.")

    if not args.videos_dir.exists():
        parser.error(
            f"Videos directory not found: {args.videos_dir}"
        )

    if args.video_id:

        video_dir = args.videos_dir / args.video_id

        if not video_dir.is_dir():
            parser.error(
                f"Video directory not found: {video_dir}"
            )

        process_episode(
            video_dir,
            args.merge_gap,
            args.prolonged_threshold,
            args.clean_gap_max,
        )

    else:

        video_dirs = sorted(
            path
            for path in args.videos_dir.iterdir()
            if path.is_dir()
        )

        print(f"Found {len(video_dirs)} video directories.")

        for video_dir in video_dirs:

            process_episode(
                video_dir,
                args.merge_gap,
                args.prolonged_threshold,
                args.clean_gap_max,
            )


if __name__ == "__main__":
    main()
