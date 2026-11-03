#!/usr/bin/env python3
"""Merge speaker_turns into solo/overlap sections and count transitions.

Does not modify final.json. Writes analysis/speaker_transition_analysis.json
for one input, or speaker_transition_comparison.json for multiple inputs.

Examples:
  python speaker_transition_analysis.py --video-id eiY9KJ2SHnA
  python speaker_transition_analysis.py --input normal.json --label normal \
      --input t05.json --label threshold_0.5 \
      --input t10.json --label threshold_1.0 \
      --input t20.json --label threshold_2.0

Definitions implemented:
* Overlap sections are temporally connected overlapping intervals; their
  original composition changes are preserved in composition_timeline.
* Candidate A -> B: count once per overlap section for each other speaker B
  appearing in that section's composition, relative to the preceding solo A.
* Confirmed A -> B: candidate B becomes the solo speaker immediately after
  that overlap section.
* Overlap-related A -> B: the above confirmed transition initiated in overlap.
* Clean solo-to-solo change is confirmed, but not candidate/overlap-related.
* Consecutive same-speaker solo intervals are merged; gaps are not bridged.
"""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path


LOG = logging.getLogger("speaker_transition_analysis")


def normalize_speakers(value):
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = value.replace("&", ",").replace("+", ",").split(",")
    elif value is None:
        items = []
    else:
        items = [str(value)]
    return sorted({str(x).strip() for x in items if str(x).strip()})


def load_final(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("speaker_turns"), list):
        raise ValueError(f"{path}: missing/list-invalid speaker_turns")

    turns = []
    for i, item in enumerate(data["speaker_turns"]):
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path}: invalid start/end in turn #{i}") from e
        if end <= start:
            LOG.warning("Skipping turn #%d with end <= start", i)
            continue
        speakers = normalize_speakers(item.get("speaker", item.get("speakers")))
        if not speakers:
            LOG.warning("Skipping turn #%d with no speaker", i)
            continue
        turns.append({"start": start, "end": end, "speakers": speakers,
                      "source_index": i})
    turns.sort(key=lambda x: (x["start"], x["end"], x["source_index"]))
    return data, turns


def make_sections(turns, overlap_candidates=None):
    """Create solo sections and overlap components from diarization turns AND
    the top-level overlap_candidates artifact. Candidate intervals are included
    regardless of case (AGREE, DISAGREE, etc.).
    """
    all_turns = list(turns)

    # overlap_candidates is the canonical OSD/verified overlap source in final.json.
    # Add each event as a multi-speaker interval so it participates in section
    # construction even when speaker_turns contains only solo turns.
    for i, event in enumerate(overlap_candidates or []):
        try:
            start, end = float(event["start"]), float(event["end"])
        except (KeyError, TypeError, ValueError):
            LOG.warning("Skipping overlap candidate #%d with invalid start/end", i)
            continue
        if end <= start:
            continue
        speakers = normalize_speakers(event.get("speakers"))
        if len(speakers) < 2:
            LOG.warning("Skipping overlap candidate #%d with fewer than 2 speakers", i)
            continue
        all_turns.append({
            "start": start,
            "end": end,
            "speakers": speakers,
            "source_index": f"overlap_candidate_{i}",
            "source": event.get("source", "overlap_candidates"),
            "case": event.get("case"),
        })

    all_turns.sort(key=lambda x: (x["start"], x["end"], str(x["source_index"])))
    sections, active = [], []
    active_end = float("-inf")

    def flush():
        nonlocal active, active_end
        if not active:
            return
        sections.append({
            "section_type": "overlap",
            "start": min(t["start"] for t in active),
            "end": max(t["end"] for t in active),
            "speakers": sorted({s for t in active for s in t["speakers"]}),
            "composition_timeline": [
                {"start": t["start"], "end": t["end"],
                 "speakers": t["speakers"], "source_index": t["source_index"]}
                for t in active
            ],
        })
        active, active_end = [], float("-inf")

    for turn in all_turns:
        # A multi-speaker turn begins/continues an overlap component.
        if len(turn["speakers"]) > 1:
            if active and turn["start"] < active_end:
                active.append(turn)
                active_end = max(active_end, turn["end"])
            else:
                flush()
                active, active_end = [turn], turn["end"]
            continue


        # A solo interval crossing an active overlap belongs inside it as an
        # internal composition change (e.g. A,B -> A), not a separate section.
        if active and turn["start"] < active_end:
            active.append(turn)
            active_end = max(active_end, turn["end"])
            continue

        flush()
        sections.append({"section_type": "solo", "start": turn["start"],
                        "end": turn["end"], "speaker": turn["speakers"][0],
                        "speakers": turn["speakers"],
                        "source_indices": [turn["source_index"]]})

    flush()
    sections.sort(key=lambda s: (s["start"], s["end"]))
    return sections


def merge_adjacent_solos(sections):
    """Merge consecutive same-speaker solo intervals; never merge across gaps."""
    out = []
    for sec in sections:
        sec = dict(sec)
        if (sec["section_type"] == "solo" and out
                and out[-1]["section_type"] == "solo"
                and out[-1]["speaker"] == sec["speaker"]
                and sec["start"] >= out[-1]["end"]):
            out[-1]["end"] = sec["end"]
            out[-1].setdefault("source_indices", []).extend(sec.get("source_indices", []))
        else:
            out.append(sec)
    return out


def count_transitions(sections):
    confirmed, candidate, overlap_related = defaultdict(int), defaultdict(int), defaultdict(int)
    confirmed_events, candidate_events = [], []
    last_solo = None
    pending_candidates = []

    for section_idx, sec in enumerate(sections):
        if sec["section_type"] == "solo":
            solo = sec["speaker"]

            # Resolve candidates from the preceding overlap section.
            for event in pending_candidates:
                if solo == event["to_speaker"]:
                    pair = f'{event["from_speaker"]} -> {event["to_speaker"]}'
                    confirmed[pair] += 1
                    overlap_related[pair] += 1
                    confirmed_events.append({**event, "confirmed_at": sec["start"],
                                            "confirmed_solo_section_index": section_idx})
            pending_candidates = []

            # A clean change from one solo speaker to another is confirmed.
            if last_solo is not None and last_solo != solo:
                pair = f"{last_solo} -> {solo}"
                confirmed[pair] += 1
                confirmed_events.append({"from_speaker": last_solo,
                    "to_speaker": solo, "transition_type": "clean_solo",
                    "timestamp": sec["start"], "section_index": section_idx})
            last_solo = solo
            continue


        # One candidate per distinct entering speaker per overlap section.
        if last_solo is not None:
            targets = [s for s in sec["speakers"] if s != last_solo]
            for target in sorted(set(targets)):
                pair = f"{last_solo} -> {target}"
                candidate[pair] += 1
                event = {"from_speaker": last_solo, "to_speaker": target,
                    "transition_type": "overlap_candidate",
                    "overlap_section_index": section_idx,
                    "overlap_start": sec["start"], "overlap_end": sec["end"]}
                candidate_events.append(event)
                pending_candidates.append(event)

    return {
        "confirmed_transition_counts": dict(sorted(confirmed.items())),
        "candidate_transition_counts": dict(sorted(candidate.items())),
        "overlap_related_transition_counts": dict(sorted(overlap_related.items())),
        "confirmed_transitions": confirmed_events,
        "candidate_transitions": candidate_events,
        "totals": {"confirmed": sum(confirmed.values()),
                   "candidate": sum(candidate.values()),
                   "overlap_related": sum(overlap_related.values())},
    }


def analyze(path, label=None, output=None, merged_output=None):
    data, turns = load_final(path)
    overlap_candidates = data.get("overlap_candidates", [])
    if not isinstance(overlap_candidates, list):
        LOG.warning("%s: overlap_candidates is not a list; treating as empty", path)
        overlap_candidates = []
    sections = merge_adjacent_solos(make_sections(turns, overlap_candidates))
    transitions = count_transitions(sections)
    merged_result = {
        "schema_version": "1.0",
        "source_file": str(path),
        "label": label or path.stem,
        "video_id": data.get("episode", {}).get("video_id"),
        "input_turn_count": len(turns),
        "section_count": len(sections),
        "solo_section_count": sum(s["section_type"] == "solo" for s in sections),
        "overlap_section_count": sum(s["section_type"] == "overlap" for s in sections),
        "merged_sections": sections,
    }
    if merged_output:
        merged_output.parent.mkdir(parents=True, exist_ok=True)
        with merged_output.open("w", encoding="utf-8") as f:
            json.dump(merged_result, f, indent=2, ensure_ascii=False)
            f.write("\n")
        LOG.info("Saved merged speaker turns: %s", merged_output)

    result = {
        "schema_version": "1.0",
        "source_file": str(path),
        "label": label or path.stem,
        "video_id": data.get("episode", {}).get("video_id"),
        "input_turn_count": len(turns),
        "section_count": len(sections),
        "solo_section_count": sum(s["section_type"] == "solo" for s in sections),
        "overlap_section_count": sum(s["section_type"] == "overlap" for s in sections),
        "merged_sections": sections,
        "transition_analysis": transitions,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")
        LOG.info("Saved: %s", output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/v2/videos"),
                  help="Root containing <video_id>/final.json")
    p.add_argument("--video-id", help="Analyze one video under --root")
    p.add_argument("--input", action="append", default=[],
                  help="Explicit final.json path; repeat for comparison")
    p.add_argument("--label", action="append", default=[],
                  help="Label for each --input, in matching order")
    p.add_argument("--output", type=Path,
                  help="Exact output JSON path for a single --input")
    p.add_argument("--comparison-output", type=Path,
                  default=Path("speaker_transition_comparison.json"))
    p.add_argument("--no-write", action="store_true",
                  help="Print results without writing output files")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.input:
        paths = [Path(x) for x in args.input]
        if args.label and len(args.label) != len(paths):
            p.error("Provide exactly one --label for every --input")
        labels = args.label or [x.stem for x in paths]

        if len(paths) == 1:
            out = None if args.no_write else (args.output or
                paths[0].parent / "analysis" / "speaker_transition_analysis.json")
            merged_out = None if args.no_write else paths[0].parent / "merged_speaker_turns.json"
            result = analyze(paths[0], labels[0], out, merged_out)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

        full = {
            label: analyze(
                path, label,
                output=None if args.no_write else path.parent / "transition_analysis.json",
                merged_output=None if args.no_write else path.parent / "merged_speaker_turns.json",
            )
            for path, label in zip(paths, labels)
        }
        metrics = ("confirmed_transition_counts", "candidate_transition_counts",
                   "overlap_related_transition_counts")
        all_pairs = sorted({pair for res in full.values() for metric in metrics
                           for pair in res["transition_analysis"][metric]})
        comparison = {
            "schema_version": "1.0",
            "files": {label: str(path) for path, label in zip(paths, labels)},
            "totals_by_version": {label: res["transition_analysis"]["totals"]
                                  for label, res in full.items()},
            "directional_counts": {
                pair: {label: {"confirmed": res["transition_analysis"][metrics[0]].get(pair, 0),
                             "candidate": res["transition_analysis"][metrics[1]].get(pair, 0),
                             "overlap_related": res["transition_analysis"][metrics[2]].get(pair, 0)}
                       for label, res in full.items()}
                for pair in all_pairs},
            "full_results": full,
        }
        if not args.no_write:
            args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
            with args.comparison_output.open("w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
                f.write("\n")
            LOG.info("Saved comparison: %s", args.comparison_output)
        print(json.dumps({"totals_by_version": comparison["totals_by_version"],
                         "directional_counts": comparison["directional_counts"]},
                        indent=2, ensure_ascii=False))
        return

    if not args.video_id:
        p.error("Specify --video-id or --input")
    source = args.root / args.video_id / "final.json"
    out = None if args.no_write else ((args.output or
        args.root / args.video_id / "analysis" / "speaker_transition_analysis.json"))
    merged_out = None if args.no_write else (
        args.output.parent / "merged_speaker_turns.json" if args.output
        else args.root / args.video_id / "merged_speaker_turns.json"
    )
    result = analyze(source, "normal", out, merged_out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()