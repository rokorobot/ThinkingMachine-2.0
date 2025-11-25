from __future__ import annotations

from typing import Any, Dict, List, Optional

from libs import user_memory as um


def build_user_context(
    user_id: Optional[str],
    user_profile: Optional[Dict[str, Any]],
    input_text: str,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Combines user profile + semantic & recent memories into a textual block
    for system prompt, plus returns the raw memories for logging/meta.
    """
    memories: List[Dict[str, Any]] = []

    if user_id:
        sem = um.search_user_memories(user_id, query=input_text, top_k=3, min_importance=2)
        rec = um.get_top_recent_memories(user_id, limit=3, min_importance=3)
        seen = set()
        for m in sem + rec:
            if m["id"] not in seen:
                seen.add(m["id"])
                memories.append(m)

    lines: List[str] = []
    if user_profile:
        prefs = user_profile.get("preferences")
        if prefs:
            lines.append("User preferences:")
            for k, v in prefs.items():
                lines.append(f"- {k}: {v}")

    if memories:
        lines.append("Key user memories:")
        for m in memories[:5]:
            lines.append(f"- ({m['kind']}) {m['text']}")

    ctx = "\n".join(lines) if lines else ""
    return ctx, memories
