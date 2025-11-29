from __future__ import annotations

from typing import Any, Dict, List


def should_retrieve(task: Dict[str, Any], world_model_ctx: Dict[str, Any]) -> bool:
    """
    Decide whether to do retrieval.
    Very naive: always retrieve for factual / medical / coding queries.
    """
    t = world_model_ctx.get("task_nature")
    if t in ("medical_like", "coding"):
        return True
    # You might also check task["force_retrieval"] etc.
    return False


def retrieval_pass(
    task: Dict[str, Any],
    world_model_ctx: Dict[str, Any],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Hook to external search/index systems.
    For now, just placeholder: you can wire in:
      - Web search microservice
      - Vector DB (docs, API manuals, etc.)
    """
    if not should_retrieve(task, world_model_ctx):
        return {"used": False, "snippets": []}

    # TODO: call your search service / vector DB
    # Stub: pretend we found one "source"
    snippets: List[Dict[str, Any]] = [
        {
            "source": "stub://knowledge",
            "title": "Placeholder snippet",
            "content": "This is where retrieved knowledge would go.",
        }
    ]

    return {
        "used": True,
        "snippets": snippets,
    }
