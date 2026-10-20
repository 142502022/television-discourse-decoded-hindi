
import json
from pathlib import Path

# Change this Threshold afret sometime.
def classify_duration(duration):
    if duration < 0.2:
        return "Very short"
    elif duration < 0.5:
        return "Short"
    elif duration <= 1.0:
        return "Moderate"
    else:
        return "Prolonged"


from collections import Counter

duration_counts = Counter()



videos_dir = Path("data/v2/videos")
json_files = sorted(videos_dir.glob("*/final.json"))

print("Number of episode files:", len(json_files))


for json_path in json_files:
    video_id = json_path.parent.name

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    candidates = data.get("overlap_candidates", [])
    duration_counts = Counter()
    total_duration = 0.0
    overlap_events = []

    for candidate in candidates:
        duration = candidate["duration"]
        category = classify_duration(duration)

        duration_counts[category] += 1
        total_duration += duration

        overlap_events.append({
            "start": candidate["start"],
            "end": candidate["end"],
            "duration": duration,
            "category": category,
            "speakers": candidate.get("speakers", []),
            "case": candidate.get("case"),
            "confidence": candidate.get("confidence"),
        })

    total_candidates = len(candidates)

    percentages = {}
    for category in ["Very short", "Short", "Moderate", "Prolonged"]:
        if total_candidates > 0:
            percentages[category] = round(
                duration_counts[category] / total_candidates * 100, 2
            )
        else:
            percentages[category] = 0.0

    summary = {
        "video_id": video_id,
        "total_overlap_candidates": total_candidates,
        "total_candidate_duration_seconds": round(total_duration, 3),
        "duration_category_counts": {
            category: duration_counts[category]
            for category in ["Very short", "Short", "Moderate", "Prolonged"]
        },
        "duration_category_percentages": percentages,
    }

    output_dir = json_path.parent / "overlapped"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "duration_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    events_path = output_dir / "overlap_events.json"
    with events_path.open("w", encoding="utf-8") as file:
        json.dump(overlap_events, file, indent=4, ensure_ascii=False)

    print(f"{video_id}: saved {total_candidates} overlap candidates")
    print(f"Summary: {summary_path}")
    print(f"Events:  {events_path}")
