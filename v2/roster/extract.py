"""Step 7: extract a named participant roster from a diarized transcript.

Pure, model-free helpers to build the prompt for an LLM (Gemini) and to parse
its JSON response into a structured roster. The actual API call lives in the
CLI so this module stays unit-testable without network access.
"""

import json
import re
from typing import Any, Dict, List, Optional

OUTPUT_SCHEMA = {
    "participants": [
        {
            "name": "full display name (e.g. 'Geeta Mohan')",
            "speaker_label": "anonymous cluster, e.g. SPEAKER_00 (as it appears in the transcript)",
            "role": "anchor | panelist | correspondent | reporter | caller | unknown",
            "affiliation": "party/organisation, or null if not mentioned",
            "gender_as_addressed": "male | female | unknown (based on how the show addresses/references them, not appearance)",
            "evidence_timestamps_s": [
                "list of a few timestamps (seconds) where this person is speaking or referenced"
            ],
            "evidence_quote": "short verbatim quote that names them or establishes their role",
        }
    ]
}

SYSTEM_INSTRUCTION = """You are an expert analyst of Indian TV news debate shows.

From the diarized transcript of one episode below, extract the roster of every
named participant who speaks or is named on air. A diarized transcript uses
anonymous labels (SPEAKER_00, SPEAKER_01, ...) because the speaker-identifying
system is not yet bound to real names.

For each participant return:
- name: the person's real name if it can be determined from the transcript
  (e.g. the anchor names a guest, or a guest introduces themselves). If a name
  appears only once, still use it. If truly not determinable, use "Unknown".
- speaker_label: the anonymous SPEAKER_xx cluster this person belongs to.
- role: one of anchor / panelist / correspondent / reporter / caller / unknown.
- affiliation: party or organisation if explicitly mentioned, else null.
- gender_as_addressed: based solely on how the show ADDRESSES or REFERS to the
  person in words (sir / madam / he said / she said / pronouns, added titles).
  Do not guess from voice or appearance. "male"/"female"/"unknown".
- evidence_timestamps_s: a few seconds timestamps (e.g. 4.2, 17.9) where this
  person is speaking or is named. Prefer their own speaking turns.
- evidence_quote: one short verbatim quote that names them or establishes role.

Rules:
- One participant per human being, even if the same label hosts multiple turns.
- If multiple names map to the SAME speaker_label, still list them separately
  only if they are clearly different people; typically one label = one person.
- If a speaker label has no gliding name at all, use name "Unknown" for it.
- Output ONLY valid JSON, no markdown fences, matching the schema below.
"""


def build_prompt(
    segments: List[Dict[str, Any]],
    schema: Optional[str] = None,
) -> str:
    """Turn diarized segments into a numbered speaker-tagged transcript."""
    if schema is None:
        schema = json.dumps(OUTPUT_SCHEMA, indent=2)
    lines = []
    for segment in segments:
        start = segment.get("start", 0)
        end = segment.get("end", start)
        speaker = segment.get("speaker", "?")
        text = (segment.get("text") or "").strip()
        lines.append(f"[{start:.1f}-{end:.1f}] {speaker}: {text}")
    transcript = "\n".join(lines)
    return (
        "You will receive a diarized transcript of one Indian TV news debate episode.\n"
        f"Return a JSON object exactly matching this schema:\n{schema}\n\n"
        "EPISODE TRANSCRIPT (seconds of each spoken turn):\n"
        f"{transcript}"
    )


def system_prompt(schema: Optional[str] = None) -> str:
    if schema is None:
        schema = json.dumps(OUTPUT_SCHEMA, indent=2)
    return f"{SYSTEM_INSTRUCTION}\n\nReturn ONLY valid JSON:\n{schema}"


def parse_roster(raw: str) -> Dict[str, Any]:
    """Parse the model's answer, tolerating markdown fences and stray prose."""
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text)
        text = text.replace("```", "").strip()
    # Crop to the outermost JSON block if prose surrounds it.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    payload = json.loads(text[start : end + 1])
    participants = payload.get("participants")
    if not isinstance(participants, list):
        raise ValueError("Response has no 'participants' list.")
    return {"participants": participants}


def normalize_roster(participants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fill defaults and coerce field types so the artifact is uniform."""
    roles = {"anchor", "panelist", "correspondent", "reporter", "caller", "unknown"}
    cleaned = []
    for p in participants:
        name = str(p.get("name") or "Unknown").strip()
        label = str(p.get("speaker_label") or "").strip()
        role = str(p.get("role") or "unknown").strip().lower()
        if role not in roles:
            role = "unknown"
        affiliation = p.get("affiliation")
        if affiliation is not None:
            affiliation = str(affiliation).strip() or None
        gender = str(p.get("gender_as_addressed") or "unknown").strip().lower()
        if gender not in {"male", "female", "unknown"}:
            gender = "unknown"
        timestamps = p.get("evidence_timestamps_s") or []
        if isinstance(timestamps, list):
            timestamps = [float(t) for t in timestamps]
        else:
            timestamps = []
        cleaned.append(
            {
                "name": name,
                "speaker_label": label,
                "role": role,
                "affiliation": affiliation,
                "gender_as_addressed": gender,
                "evidence_timestamps_s": timestamps,
                "evidence_quote": str(p.get("evidence_quote") or "").strip(),
            }
        )
    cleaned.sort(key=lambda p: p["name"].lower())
    return cleaned


def build_roster_artifact(participants: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    normalized = normalize_roster(participants)
    return {
        "method": "llm-extraction",
        "model": model,
        "num_participants": len(normalized),
        "participants": normalized,
    }