from __future__ import annotations

from typing import Any, Dict, Optional


def synthesize_output(
    task: Dict[str, Any],
    draft_output: str,
    reasoning_metadata: Dict[str, Any],
    reflection_result: Dict[str, Any],
    user_profile: Optional[Dict[str, Any]],
) -> str:
    """
    Decide whether to show draft as-is or patch it lightly according
    to reflection + user preferences (e.g., more concise).
    """
    text = draft_output
    quality = reflection_result.get("quality_score", 0.7)
    needs_revision = reflection_result.get("needs_revision", False)
    suggested_fix = reflection_result.get("suggested_fix")

    prefs = (user_profile or {}).get("preferences", {}) if user_profile else {}
    want_concise = prefs.get("detail_level") == "concise"

    # Very minimal:
    if want_concise and len(text) > 1500:
        text = text[:1400] + "\n\n(Shortened for brevity.)"

    if needs_revision and isinstance(suggested_fix, str) and suggested_fix.strip():
        # For now, just append a note; later you can run a second-pass LLM edit.
        text += "\n\n---\n(Internal critique suggests improvement)\n"
        text += suggested_fix

    # You might also inject disclaimers when hallucination_risk is high.
    if reflection_result.get("hallucination_risk") == "high":
        text += "\n\n[Note: This answer may contain speculative or uncertain information. Please double-check critical facts.]"

    return text
