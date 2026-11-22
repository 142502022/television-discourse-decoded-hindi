"""Rules-only Hindi advertisement detection and timestamp-preserving masking."""
from __future__ import annotations

import copy
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

PROMO = ("ऑफर", "छूट", "डिस्काउंट", "कीमत", "खरीदें", "खरीदिये", "ऑर्डर", "कॉल करें", "आज ही", "फ्री डिलीवरी", "गारंटी", "कैशबैक", "सेल", "बुक करें", "डाउनलोड करें", "buy now", "order now", "limited offer", "discount", "call now", "download now", "free delivery", "cashback", "offer")
PRODUCT = ("रुपये", "₹", "ऐप", "इंश्योरेंस", "बीमा", "रिचार्ज", "स्कीम", "उत्पाद", "सर्विस", "product", "service", "price", "loan", "insurance")
CTA = ("करें", "कीजिए", "मिलता है", "पाएं", "जाइए", "visit", "subscribe", "dial", "apply")
MIN_ADVERTISEMENT_DURATION_SECONDS = 15.0


def _number(value: Any) -> float | None:
    try: return float(value)
    except (TypeError, ValueError): return None


def score(text: str) -> tuple[int, list[str]]:
    lower = text.lower()
    hits = []
    for family, words in (("promotional", PROMO), ("product_or_price", PRODUCT), ("call_to_action", CTA)):
        if any(word in lower for word in words): hits.append(family)
    # At least one unambiguous marketing term and a second independent cue are
    # required; this prevents ordinary debate brand mentions becoming ads.
    return len(hits), hits


def candidates(transcript: dict) -> list[dict]:
    rows=[]
    for index, segment in enumerate(transcript.get("segments", [])):
        if not isinstance(segment, dict): continue
        start,end=_number(segment.get("start")),_number(segment.get("end"))
        text=str(segment.get("text") or "").strip()
        if start is None or end is None or end<=start or not text: continue
        value,evidence=score(text)
        if value:
            rows.append({"source_segment_indices":[index],"start_timestamp":start,"end_timestamp":end,"duration_seconds":round(end-start,3),"transcript_evidence":text,"rule_score":value,"evidence_categories":evidence,"classification":"advertisement_candidate" if value>=2 else "uncertain"})
    return rows


def continuous_blocks(rows: list[dict], max_gap: float = 2.0,
                      min_duration_seconds: float = MIN_ADVERTISEMENT_DURATION_SECONDS) -> list[dict]:
    """Merge only adjacent high-confidence rows; never bridge debate speech."""
    blocks=[]
    for row in rows:
        if row["classification"] != "advertisement_candidate": continue
        if blocks and row["start_timestamp"]-blocks[-1]["end_timestamp"] <= max_gap:
            current=blocks[-1]; current["end_timestamp"]=row["end_timestamp"]; current["source_segment_indices"] += row["source_segment_indices"]; current["evidence"].append(row["transcript_evidence"]); current["evidence_categories"].update(row["evidence_categories"])
        else:
            blocks.append({"source_segment_indices":list(row["source_segment_indices"]),"start_timestamp":row["start_timestamp"],"end_timestamp":row["end_timestamp"],"evidence":[row["transcript_evidence"]],"evidence_categories":set(row["evidence_categories"])})
    for i, block in enumerate(blocks,1):
        block["advertisement_id"]=f"AD_{i:03d}"; block["duration_seconds"]=round(block["end_timestamp"]-block["start_timestamp"],3); block["evidence_categories"]=sorted(block["evidence_categories"]); block["transcript_evidence"]=" ".join(block.pop("evidence")); block["language"]="Hindi_or_Hinglish"; block["classification"]="advertisement"; block["confidence"]="rule_based_high"; block["exclusion_status"]="excluded" if block["duration_seconds"] >= min_duration_seconds else "retained"; block["exclusion_reason"]=f"Hindi advertisement duration is at least {min_duration_seconds:g} seconds" if block["duration_seconds"] >= min_duration_seconds else f"duration is below {min_duration_seconds:g} seconds"
    return blocks


def exclusion_record(video_id: str, transcript: dict,
                     min_duration_seconds: float = MIN_ADVERTISEMENT_DURATION_SECONDS) -> dict:
    if min_duration_seconds < 0:
        raise ValueError("min_duration_seconds must be non-negative")
    raw=candidates(transcript); blocks=continuous_blocks(raw, min_duration_seconds=min_duration_seconds); excluded=[x for x in blocks if x["exclusion_status"]=="excluded"]; retained=[x for x in blocks if x["exclusion_status"]=="retained"]
    return {"schema_version":"advertisement-exclusion-1.0","video_id":video_id,"advertisement_detection":{"method":"conservative_rules_only","language":"Hindi_or_Hinglish","duration_threshold_seconds":min_duration_seconds,"exclusion_rule":f"duration >= {min_duration_seconds:g} seconds","continuous_gap_seconds":2.0},"summary":{"total_advertisement_candidates":len(raw),"total_continuous_advertisements":len(blocks),"total_excluded_advertisements":len(excluded),"total_retained_advertisements":len(retained),"total_excluded_duration_seconds":round(sum(x["duration_seconds"] for x in excluded),3),"total_retained_duration_seconds":round(sum(x["duration_seconds"] for x in retained),3),"uncertain_candidate_count":sum(x["classification"]=="uncertain" for x in raw)},"advertisement_candidates":raw,"excluded_advertisements":excluded,"retained_advertisements":retained,"validation":{"human_verified":False,"review_required":bool(raw),"warning":"Rules do not establish commercial intent; uncertain candidates are retained."}}


def filter_transcript(transcript: dict, excluded: list[dict]) -> dict:
    intervals=[(x["start_timestamp"],x["end_timestamp"]) for x in excluded]
    out=copy.deepcopy(transcript); kept=[]; removed=[]
    for segment in out.get("segments",[]):
        start,end=_number(segment.get("start")),_number(segment.get("end"))
        if start is not None and end is not None and any(start < b and end > a for a,b in intervals): removed.append({"start":start,"end":end,"reason":"intersects excluded advertisement"})
        else: kept.append(segment)
    out["segments"]=kept; out["advertisement_filtering"]={"timestamp_domain":"original_media","removed_transcript_segments":len(removed),"removed_segments":removed,"excluded_intervals": [{"start":a,"end":b} for a,b in intervals]}; return out


def mask_audio(input_path: Path, output_path: Path, excluded: list[dict]) -> None:
    output_path.parent.mkdir(parents=True,exist_ok=True)
    if output_path.exists() and output_path.stat().st_size: return
    filters=[f"volume=enable='between(t,{x['start_timestamp']},{x['end_timestamp']})':volume=0" for x in excluded]
    command=["ffmpeg","-y","-i",str(input_path)]
    if filters: command += ["-af",",".join(filters)]
    command += ["-acodec","pcm_s16le","-ar","16000","-ac","1",str(output_path)]
    subprocess.run(command,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def save_record(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
