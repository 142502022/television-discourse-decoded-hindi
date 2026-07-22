from typing import Any, Dict, List


def empty_roster_payload() -> Dict[str, Any]:
    """Return an explicit placeholder for roster extraction not yet reviewed."""
    return {
        "participants": [],
        "evidence_audit": {
            "sample_size": 0,
            "checked": 0,
            "errors": 0,
            "notes": "Roster extraction requires an LLM call and manual evidence review.",
        },
    }


def speaker_name_links() -> List[Dict[str, Any]]:
    """Return an empty speaker-link list until roster evidence is verified."""
    return []
