import json
from pathlib import Path
import matplotlib.pyplot as plt

report_path = Path("data/v2/analysis/discourse_summary.json")

with report_path.open(encoding="utf-8") as f:
    report = json.load(f)

episodes = report["episodes"]
ids = [e["metadata"]["video_id"] for e in episodes]
intr_per_min = [e["interruptions"]["attempts_per_minute"] or 0 for e in episodes]
overlap_pct = [e["overlap"]["overlap_ratio_pct"] or 0 for e in episodes]

plt.figure(figsize=(12, 5))
plt.bar(ids, intr_per_min)
plt.xticks(rotation=60, ha="right")
plt.ylabel("Interruptions per minute")
plt.title("Interruption rate by episode")
plt.tight_layout()
plt.savefig("data/v2/analysis/interruptions_by_episode.png", dpi=200)
plt.show()

plt.figure(figsize=(12, 5))
plt.bar(ids, overlap_pct)
plt.xticks(rotation=60, ha="right")
plt.ylabel("Unique overlap ratio (%)")
plt.title("Overlap by episode")
plt.tight_layout()
plt.savefig("data/v2/analysis/overlap_by_episode.png", dpi=200)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(overlap_pct, intr_per_min)
plt.xlabel("Unique overlap ratio (%)")
plt.ylabel("Interruptions per minute")
plt.title("Overlap and interruption rate")
plt.tight_layout()
plt.savefig("data/v2/analysis/overlap_vs_interruptions.png", dpi=200)
plt.show()