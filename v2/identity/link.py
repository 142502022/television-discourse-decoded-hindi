"""Step 9: cluster-to-name linking.

Merges the roster (from step 7, Gemini) with the role evidence (from step 5)
to produce a single authoritative participant roster per episode, tagged with
real names and affiliations where available.

If roster.json is missing, the linked roster uses speaker labels with role
labels from identity/roles.json.
"""

from typing import Any, Dict, List, Optional

from v2.osd.overlap import speakers_in


def _role_for(label: str, roles_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract role evidence for a speaker label from the roles artifact."""
    if not roles_data or "roles" not in roles_data:
        return {}
    return roles_data["roles"].get(label, {})


def link_roster(
    participants: List[Dict[str, Any]],
    roles_data: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge roster and roles evidence into a linked participant list.

    Each entry represents one real (or anonymous) person with all known
    information: name, speaker_label, role, affiliation, gender, evidence.
    """
    linked: List[Dict[str, Any]] = []
    seen_labels: set = set()

    for p in participants:
        label = p.get("speaker_label", "")
        role_info = _role_for(label, roles_data)
        heuristic = role_info.get("evidence", {})
        linked.append(
            {
                "name": p.get("name") or "Unknown",
                "speaker_label": label,
                "role": role_info.get("role", p.get("role", "unknown")),
                "affiliation": p.get("affiliation"),
                "gender_as_addressed": p.get("gender_as_addressed", "unknown"),
                "role_score": role_info.get("score"),
                "speech_seconds": heuristic.get("speech_seconds"),
                "speech_share": heuristic.get("speech_share"),
                "segment_count": heuristic.get("segment_count"),
                "questions_asked": heuristic.get("questions", 0),
                "interruptions_initiated": heuristic.get("interruptions_initiated", 0),
                "interruptions_received": heuristic.get("interruptions_received", 0),
                "first_start_s": heuristic.get("first_start_s"),
                "evidence_quote": p.get("evidence_quote", ""),
                "evidence_timestamps_s": p.get("evidence_timestamps_s", []),
                "source": "roster+heuristic" if p.get("name") else "heuristic",
            }
        )
        seen_labels.add(label)

    anchor = roles_data.get("anchor") if roles_data else None
    anchor_speaker = roles_data.get("anchor_speaker") if roles_data else None

    return {
        "method": "cluster-to-name",
        "anchor_label": anchor_speaker or anchor,
        "anchor_name": next(
            (p["name"] for p in linked if p["speaker_label"] == (anchor_speaker or anchor)),
            None,
        ),
        "num_participants": len(linked),
        "participants": linked,
    }
