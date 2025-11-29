from __future__ import annotations

from typing import Any, Dict


def world_model_pass(
    task: Dict[str, Any],
    policy: Dict[str, Any],
    user_context: str,
) -> Dict[str, Any]:
    """
    Skeleton World Model layer.
    Later: use a smaller LLM or classifier to:
      - infer task nature (advice, planning, creative, coding)
      - predict user reaction
      - suggest constraints and focus points
    For now, we just tag a few things heuristically.
    """
    input_text = task["input_text"].lower()

    # Super simple heuristics for now
    if "plan" in input_text or "roadmap" in input_text:
        task_nature = "planning"
    elif "code" in input_text or "bug" in input_text:
        task_nature = "coding"
    elif "diagnose" in input_text or "symptom" in input_text:
        task_nature = "medical_like"
    else:
        task_nature = "general"

    constraints = []
    if task_nature == "medical_like":
        constraints.append("Do not give definitive medical diagnosis.")
        constraints.append("Encourage consulting a real doctor.")
    if len(input_text) > 800:
        constraints.append("User sent long input, prioritize summarization and structure.")

    world_state = {
        "task_nature": task_nature,
        "constraints": constraints,
        "signals": {
            "needs_high_safety": task_nature in ("medical_like",),
            "needs_structured_output": task_nature in ("planning", "coding"),
        },
    }
    return world_state
