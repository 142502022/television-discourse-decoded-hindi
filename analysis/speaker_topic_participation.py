#!/usr/bin/env python3
"""Speaker-wise participation by subtopic using speaker_turns as the primary source.

For each subtopic, each speaker turn is clipped to the subtopic interval:
contribution = max(0, min(turn_end, topic_end) - max(turn_start, topic_start)).

This preserves consistency with final_result_numeric.json, whose speaking-time
metric is also calculated from speaker-turn intervals.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def intersection_duration(start_a, end_a, start_b, end_b):
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def calculate(transcript, subtopic_data):
    turns = transcript.get("speaker_turns")
    if not isinstance(turns, list):
        raise ValueError("merge_final.json does not contain a valid speaker_turns list")

    valid_turns = []
    for index, turn in enumerate(turns):
        speaker = turn.get("speaker") or turn.get("speaker_id")
        start, end = turn.get("start"), turn.get("end")
        if not speaker or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if float(end) <= float(start):
            continue
        valid_turns.append({
            "speaker_id": str(speaker),
            "start": float(start),
            "end": float(end),
            "source_turn_index": index,
        })

    subtopics = subtopic_data.get("subtopics", [])
    if not subtopics:
        raise ValueError("subtopic_analysis.json contains no subtopics")

    results = []
    episode_speaker_time = defaultdict(float)
    episode_speaker_turns = defaultdict(int)
    speaker_topic_coverage = defaultdict(set)

    for topic in subtopics:
        topic_id = topic.get("subtopic_id")
        topic_start, topic_end = float(topic["start"]), float(topic["end"])
        if topic_end <= topic_start:
            raise ValueError(f"Invalid time range for {topic_id}: {topic_start}–{topic_end}")

        speaker_time = defaultdict(float)
        speaker_turn_count = defaultdict(int)

        for turn in valid_turns:
            contribution = intersection_duration(
                turn["start"], turn["end"], topic_start, topic_end
            )
            if contribution > 0:
                speaker = turn["speaker_id"]
                speaker_time[speaker] += contribution
                # Count an intersecting source turn once in this subtopic.
                speaker_turn_count[speaker] += 1

        total_speaking_time = sum(speaker_time.values())
        speaker_ids = sorted(set(speaker_time) | set(speaker_turn_count))
        speaker_rows = []

        for speaker in speaker_ids:
            seconds = speaker_time[speaker]
            share = seconds / total_speaking_time * 100 if total_speaking_time else 0.0
            speaker_rows.append({
                "speaker_id": speaker,
                "speaking_time_seconds": round(seconds, 3),
                "speech_share_percent": round(share, 3),
                "turn_count": int(speaker_turn_count[speaker]),
            })
            episode_speaker_time[speaker] += seconds
            episode_speaker_turns[speaker] += speaker_turn_count[speaker]
            if seconds > 0:
                speaker_topic_coverage[speaker].add(topic_id)

        speaker_rows.sort(key=lambda row: (-row["speaking_time_seconds"], row["speaker_id"]))
        positive_rows = [row for row in speaker_rows if row["speaking_time_seconds"] > 0]
        dominant_speaker = positive_rows[0]["speaker_id"] if positive_rows else None

        results.append({
            "subtopic_id": topic_id,
            "title": topic.get("title", ""),
            "start": round(topic_start, 3),
            "end": round(topic_end, 3),
            "duration_seconds": round(topic_end - topic_start, 3),
            "total_speaking_time_seconds": round(total_speaking_time, 3),
            "dominant_speaker": dominant_speaker,
            "speakers": speaker_rows,
        })

    episode_total = sum(episode_speaker_time.values())
    episode_rows = []
    for speaker in sorted(episode_speaker_time, key=lambda s: (-episode_speaker_time[s], s)):
        seconds = episode_speaker_time[speaker]
        topics = sorted(speaker_topic_coverage[speaker])
        episode_rows.append({
            "speaker_id": speaker,
            "total_speaking_time_seconds": round(seconds, 3),
            "speech_share_percent": round(seconds / episode_total * 100, 3) if episode_total else 0.0,
            "turn_count_across_subtopics": int(episode_speaker_turns[speaker]),
            "subtopics_participated": topics,
            "subtopic_count": len(topics),
        })

    return {
        "schema_version": "speaker-topic-participation-1.1",
        "video_id": subtopic_data.get("video_id"),
        "variant": subtopic_data.get("variant"),
        "method": {
            "speaking_time": "speaker_turn intervals clipped to each subtopic boundary; each turn contributes only its intersecting duration",
            "turn_count": "number of source speaker_turn intervals intersecting each subtopic; a turn crossing a boundary is counted once in each intersected subtopic",
            "speech_share": "speaker clipped turn-duration sum divided by total clipped speaker-turn duration within the same subtopic, multiplied by 100",
            "dominant_speaker": "speaker with the greatest clipped speaker-turn duration in the subtopic",
            "topic_coverage": "subtopics with positive clipped speaking time",
            "consistency": "uses the same speaker_turn duration basis as final_result_numeric.json",
            "overlap_note": "speaker turns may overlap; if they do, their durations are counted for each speaker, matching speaker-wise duration accounting in the existing pipeline"
        },
        "summary": {
            "total_subtopics": len(results),
            "total_speakers": len(episode_rows),
            "total_speaking_time_seconds_across_subtopics": round(episode_total, 3),
        },
        "subtopic_participation": results,
        "speaker_episode_summary": episode_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Speaker-wise topic participation analysis")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--root", default="data/v2/debate_only")
    parser.add_argument("--transcript", help="Optional explicit merge_final.json path")
    parser.add_argument("--subtopics", help="Optional explicit subtopic_analysis.json path")
    parser.add_argument("--output", help="Optional explicit output JSON path")
    args = parser.parse_args()

    root = Path(args.root)
    transcript_path = Path(args.transcript) if args.transcript else (
        root / "videos" / args.video_id / args.variant / "merge_final.json"
    )
    subtopics_path = Path(args.subtopics) if args.subtopics else (
        root / "subtopic_analysis" / args.variant / args.video_id / "subtopic_analysis.json"
    )
    output_path = Path(args.output) if args.output else (
        root / "speaker_topic_participation" / args.variant / args.video_id /
        "speaker_topic_participation.json"
    )

    for input_path in (transcript_path, subtopics_path):
        if not input_path.is_file():
            raise FileNotFoundError(f"Required input file not found: {input_path}")

    transcript = load_json(transcript_path)
    subtopic_data = load_json(subtopics_path)

    transcript_video_id = transcript.get("episode", {}).get("video_id")
    if transcript_video_id not in (None, args.video_id):
        raise ValueError(f"Transcript video ID mismatch: {transcript_video_id}")
    if subtopic_data.get("video_id") not in (None, args.video_id):
        raise ValueError(f"Subtopic video ID mismatch: {subtopic_data.get('video_id')}")
    if subtopic_data.get("variant") not in (None, args.variant):
        raise ValueError(f"Subtopic variant mismatch: {subtopic_data.get('variant')}")

    output = calculate(transcript, subtopic_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {output_path}")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()