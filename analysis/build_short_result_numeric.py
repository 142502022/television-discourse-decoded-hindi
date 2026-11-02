#!/usr/bin/env python3
"""Build compact short_result_numeric.json from the flattened final_result_numeric.json.

The source numeric file stores values in numeric_summary under dotted paths.
This script reads those exact paths and leaves both source files unchanged.
"""
import argparse
import json
import statistics
from pathlib import Path

NUMERIC_FILENAME = "final_result_numeric.json"
FINAL_FILENAME = "final.json"
OUTPUT_FILENAME = "short_result_numeric.json"



def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARNING] Cannot read {path}: {exc}")
        return None


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[SAVED] {path}")


def num(value, digits=3):
    if value is None:
        return None
    try:
        value = float(value)
        if not (value == float("inf") or value == float("-inf")):
            return round(value, digits)
    except (TypeError, ValueError):
        pass
    return None


def get_flat(summary, key, default=None):
    return summary.get(key, default) if isinstance(summary, dict) else default



def extract_metadata(data, final_data, video_dir, summary):
    episode = (final_data or {}).get("episode", {}) if isinstance(final_data, dict) else {}
    video_id = data.get("video_id") or episode.get("video_id") or video_dir.name
    title = episode.get("title") or episode.get("video_title")
    language = episode.get("language") or get_flat(summary, "episode.language") or "Hindi"
    duration = get_flat(summary, "speaker_participation.floor_holding.source.episode.duration_s")
    if duration is None:
        duration = episode.get("duration_s") or episode.get("duration_seconds")
    return {
        "video_id": video_id,
        "video_title": title,
        "language": language,
        "video_duration_seconds": num(duration),
    }


def extract_speaker_data(data, summary):
    speakers = data.get("speaker_metrics", {})
    shares, original_turn_frequency, merged_stats = {}, {}, {}
    source_prefix = "speaker_participation.speakers."
    floor_prefix = "speaker_participation.floor_holding.source.statistics.speaker_level."
    for speaker_id, metrics in speakers.items():
        if not isinstance(metrics, dict):
            continue
        share = metrics.get("speech_share_percent")
        if share is None:
            share = get_flat(summary, f"{source_prefix}{speaker_id}.speech_share_percent")
        turns = metrics.get("turn_count")
        if turns is None:
            turns = get_flat(summary, f"{source_prefix}{speaker_id}.turn_count")
        if share is not None:
            shares[speaker_id] = num(share)
        if turns is not None:
            original_turn_frequency[speaker_id] = turns
        merged_count = get_flat(summary, f"{floor_prefix}{speaker_id}.merged_turn_count")
        mean_duration = get_flat(summary, f"{floor_prefix}{speaker_id}.floor_holding_duration_statistics.mean")
        merged_stats[speaker_id] = {
            "merged_turn_frequency": merged_count,
            "mean_merged_turn_duration_seconds": num(mean_duration),
            "original_turn_frequency": turns,
        }
    return shares, original_turn_frequency, merged_stats


def extract_transition_matrix(data):
    matrix = data.get("speaker_transition_matrix", {})
    result = {}
    if not isinstance(matrix, dict):
        return result
    for source, destinations in matrix.items():
        if not isinstance(destinations, dict):
            continue
        result[source] = {target: count for target, count in destinations.items() if target != source}
    return result


def extract_gap_stats(summary):
    prefix = "speaker_participation.floor_holding.source.statistics.debate_level.total_gap_statistics."
    return {
        "count": get_flat(summary, prefix + "count"),
        "total_seconds": num(get_flat(summary, prefix + "total")),
        "mean_seconds": num(get_flat(summary, prefix + "mean")),
        "median_seconds": num(get_flat(summary, prefix + "median")),
        "minimum_seconds": num(get_flat(summary, prefix + "min")),
        "maximum_seconds": num(get_flat(summary, prefix + "max")),
    }


def extract_merged_turn_stats(summary, final_data):
    """Calculate merged-turn threshold frequencies from individual merged turns.


    The flattened numeric summary has only aggregate statistics; the full
    final_result.json contains source.merged_turns with each duration.
    """
    prefix = "speaker_participation.floor_holding.source.statistics.debate_level.floor_holding_duration_statistics."
    count = get_flat(summary, prefix + "count")

    merged_turns = []
    if isinstance(final_data, dict):
        current = final_data
        for key in ("speaker_participation", "floor_holding", "source", "merged_turns"):
            current = current.get(key, {}) if isinstance(current, dict) else {}
        merged_turns = current if isinstance(current, list) else []

    durations = []
    if isinstance(merged_turns, list):
        for turn in merged_turns:
            if not isinstance(turn, dict):
                continue
            duration = num(turn.get("floor_holding_duration"), digits=10)
            if duration is not None:
                durations.append(float(turn["floor_holding_duration"]))

    short_frequency = sum(d <= SHORT_TURN_THRESHOLD for d in durations) if durations else None
    long_frequency = sum(d >= LONG_TURN_THRESHOLD for d in durations) if durations else None
    denominator = len(durations)

    result = {
        "count": count,
        "mean_seconds": num(get_flat(summary, prefix + "mean")),
        "median_seconds": num(get_flat(summary, prefix + "median")),
        "minimum_seconds": num(get_flat(summary, prefix + "min")),
        "maximum_seconds": num(get_flat(summary, prefix + "max")),
    }

    if not durations:
        result["note"] = (
            "Individual merged-turn durations were not found in final.json; "
            "threshold frequencies could not be calculated."
        )
    elif count is not None and len(durations) != count:
        result["note"] = (
            f"Calculated from {len(durations)} individual durations, but numeric summary count is {count}."
        )

    return result


def extract_clusters(summary):
    prefix = "overlap_analysis.summary.cluster_B_to_G_source_summary.debate_level.cluster_counts."
    result = {letter: get_flat(summary, prefix + letter, 0) for letter in "BCDEFG"}
    # A and H are transition-related clusters, not included in the overlap B-G block.
    transition_prefix = "interaction_clusters.statistics.speaker_transitions."
    result["A"] = get_flat(summary, transition_prefix + "cluster_A_candidate_count", 0)
    result["H"] = get_flat(summary, transition_prefix + "cluster_H_candidate_count", 0)
    return {letter: result.get(letter, 0) for letter in "ABCDEFGH"}



def find_key_recursive(obj, keys):
    """Return the first value found for any matching key in nested JSON."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys:
                return value
        for value in obj.values():
            found = find_key_recursive(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key_recursive(value, keys)
            if found is not None:
                return found
    return None


def extract_host_identification(video_dir):
    """Read host identification artifact and keep only compact host identity fields."""
    candidates = [
        video_dir / "host_identification" / "host_identification.json",
        video_dir / "host_identification.json",
        video_dir / "host_identification" / "result.json",
    ]
    host_data = None
    for path in candidates:
        host_data = load_json(path)
        if host_data is not None:
            break

    if not isinstance(host_data, dict):
        return {"host_speaker_id": None, "host_identification_confidence": None}

    # Prefer the explicit predicted-host field, avoiding generic recursive
    # speaker_id matches that may accidentally pick a candidate.
    speaker_id = (host_data.get("predicted_host")
                  or host_data.get("host_speaker_id")
                  or host_data.get("predicted_host_speaker")
                  or host_data.get("host_id"))
    if isinstance(speaker_id, dict):
        speaker_id = (speaker_id.get("speaker") or speaker_id.get("speaker_id")
                     or speaker_id.get("id"))

    # The artifact's confidence is represented by the predicted candidate's
    # ranking_score. Do not use a generic recursively-found "score" field.
    candidates_list = (host_data.get("top_3_candidates")
                      or host_data.get("top_candidates")
                      or host_data.get("candidates") or [])
    confidence = host_data.get("host_identification_confidence")
    if confidence is None and isinstance(candidates_list, list):
        for candidate in candidates_list:
            if not isinstance(candidate, dict):
                continue
            candidate_speaker = candidate.get("speaker") or candidate.get("speaker_id")
            if candidate_speaker == speaker_id or candidate.get("is_predicted_host") is True:
                confidence = candidate.get("ranking_score")
                break

    return {
        "host_speaker_id": speaker_id,
        "host_identification_confidence": num(confidence),
    }


def extract_host_cluster_counts(summary, video_dir, host_id):
    """Extract A-H counts specifically associated with the identified host."""
    candidate_paths = [
        video_dir / "host_character_behavior.json",
        video_dir / "host_cluster_analysis.json",
        video_dir / "host_interaction_analysis.json",
        video_dir / "host_identification" / "host_cluster_counts.json",
        video_dir / "host_identification" / "host_character_behavior.json",
        video_dir / "host_identification" / "host_identification.json",
    ]
    sources = []
    for path in candidate_paths:
        obj = load_json(path)
        if isinstance(obj, dict):
            sources.append(obj)
    if isinstance(summary, dict):
        sources.append(summary)

    letters = "ABCDEFGH"
    counts = {letter: 0 for letter in letters}
    found_any = False

    def read_counts(obj):
        """Accept a direct A-H dictionary or a nested cluster_counts object."""
        if not isinstance(obj, dict):
            return None
        for key in ("host_cluster_counts", "host_clusters", "cluster_counts",
                    "cluster_frequency", "cluster_counts_by_speaker"):
            value = obj.get(key)
            if isinstance(value, dict):
                # A speaker-indexed dictionary: select the identified host only.
                if host_id and isinstance(value.get(host_id), dict):
                    value = value[host_id]
                if any(letter in value for letter in letters):
                    return value
        # Also accept a direct cluster dictionary (A-H keys).
        if any(letter in obj for letter in letters):
            return obj
        return None

    # First seek explicit host-specific counts.
    for source in sources:
        host_specific = find_key_recursive(source, {
            "host_cluster_counts", "host_clusters", "host_cluster_frequency",
            "host_cluster_count", "host_cluster_counts_by_speaker"
        })
        cluster_dict = read_counts(host_specific)
        if cluster_dict is None and host_id:
            # Some artifacts store cluster counts by speaker ID, e.g.
            # speaker_cluster_counts.SPEAKER_02 = {A:..., B:...}
            speaker_block = find_key_recursive(source, {
                "speaker_cluster_counts", "cluster_counts_by_speaker",
                "speaker_clusters", "clusters_per_speaker"
            })
            if isinstance(speaker_block, dict):
                cluster_dict = read_counts(speaker_block.get(host_id))
        if cluster_dict is not None:
            for letter in letters:
                value = cluster_dict.get(letter, 0)
                if isinstance(value, dict):
                    value = value.get("count", value.get("frequency", value.get("total", 0)))
                counts[letter] = value if value is not None else 0
            found_any = True
            break

    # Flattened numeric-summary paths may include speaker-specific entries.
    if not found_any and isinstance(summary, dict) and host_id:
        prefixes = [
            f"host_cluster_counts.{host_id}.",
            f"host_clusters.{host_id}.",
            f"host_character_behavior.host_cluster_counts.{host_id}.",
            f"speaker_cluster_counts.{host_id}.",
            f"cluster_counts_by_speaker.{host_id}.",
            f"speaker_clusters.{host_id}.",
        ]
        for letter in letters:
            for prefix in prefixes:
                key = prefix + letter
                if key in summary:
                    counts[letter] = summary[key]
                    found_any = True
                    break

    total = None
    if found_any:
        total = sum(v for v in counts.values() if isinstance(v, (int, float)))
    return {
        **counts,
        "total_host_cluster_events": total if total is not None else 0,
    }

def build_short_result(video_dir):
    numeric_path = video_dir / NUMERIC_FILENAME
    final_path = video_dir / FINAL_FILENAME
    data = load_json(numeric_path)
    if data is None:
        print(f"[SKIP] Missing or invalid {numeric_path}")
        return False
    final_data = load_json(final_path)
    summary = data.get("numeric_summary", {})

    metadata = extract_metadata(data, final_data, video_dir, summary)
    duration = metadata["video_duration_seconds"]
    shares, speaker_turns, speaker_merged = extract_speaker_data(data, summary)
    transitions = extract_transition_matrix(data)
    host_identification = extract_host_identification(video_dir)
    host_clusters = extract_host_cluster_counts(summary, video_dir, host_identification.get("host_speaker_id"))

    transition_count = get_flat(summary, "speaker_transitions.total_transitions", 0)
    overlap_count = get_flat(summary, "overlap_analysis.summary.overlap_event_count",
                              get_flat(summary, "episode_overview.overlap_candidate_count", 0))
    overlap_duration = get_flat(summary, "overlap_analysis.summary.total_overlap_duration_seconds", 0)
    overlap_ratio = (num(overlap_duration / duration * 100) if duration and duration > 0 else None)

    original_turns = get_flat(summary, "episode_overview.speaker_turn_count", 0)
    merged_turns = get_flat(summary, "speaker_participation.floor_holding.source.statistics.debate_level.merged_turn_count")
    merged_operations = None if original_turns is None or merged_turns is None else original_turns - merged_turns

    short_result = {
        "metadata": metadata,
        "interaction_summary": {
            "total_speakers": get_flat(summary, "episode_overview.speaker_count", len(shares)),
            "total_original_speaker_turns": original_turns,
            "total_speaker_transitions": transition_count,
            "transition_rate_per_minute": num(transition_count / (duration / 60)) if duration and duration > 0 else None,
            "total_overlap_events": overlap_count,
            "total_overlap_duration_seconds": num(overlap_duration),
            "overlap_ratio_percent": overlap_ratio,
            "total_merged_turns": merged_turns,
            "total_merged_operations": merged_operations,
        },
        "speaker_shares_percent": shares,
        "speaker_original_turn_frequency": speaker_turns,
        "pairwise_transition_frequency": transitions,
        "merged_turn_duration_statistics": extract_merged_turn_stats(summary, final_data),
        "speaker_wise_merged_turn_statistics": speaker_merged,
        "gap_duration_statistics": extract_gap_stats(summary),
        "interaction_cluster_counts": extract_clusters(summary),
        "host_character_behavior": {
            "host_identification": host_identification,
            "host_cluster_counts": host_clusters,
        },
    }
    save_json(short_result, video_dir / OUTPUT_FILENAME)
    return True


def main():
    parser = argparse.ArgumentParser(description="Build compact short_result_numeric.json from flattened numeric summaries.")
    parser.add_argument("--root", default="data/v2/videos", help="Root directory containing video-ID folders")
    parser.add_argument("--video-id", default=None, help="Process only one video ID")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] Root directory does not exist: {root}")
        return
    if args.video_id:
        target = root / args.video_id
        if not target.is_dir():
            print(f"[ERROR] Video directory not found: {target}")
            return
        print(f"Processed: {int(build_short_result(target))} video")
        return
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    success = skipped = 0
    print(f"Found {len(folders)} video directories.")
    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] {folder.name}")
        if build_short_result(folder):
            success += 1
        else:
            skipped += 1
    print(f"\nFinished. Successful: {success}; skipped: {skipped}; total: {len(folders)}")


if __name__ == "__main__":
    main()