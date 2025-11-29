from __future__ import annotations

import uuid
from typing import Any, Dict, Tuple, List, Optional

from libs import db
from libs.llm.client import LLMClient
from libs import user_memory as um
from thinking_core.pipeline import run_thinking_pipeline


llm_client = LLMClient.from_env()


def deep_merge(a: dict, b: dict) -> dict:
    """
    Recursively merge dict b into dict a (a wins type conflicts, b overrides values).
    Returns a new dict.
    """
    import copy
    result = copy.deepcopy(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def apply_user_policy_overlay(
    base_policy: Dict[str, Any],
    user_id: Optional[str],
) -> Dict[str, Any]:
    """
    Load user overlay (if any) and merge into base policy.
    """
    if not user_id:
        return base_policy

    overlay = db.get_active_user_policy_overlay(user_id)
    if not overlay:
        return base_policy

    merged = dict(base_policy)
    # base_policy is a DB row; our Policy schema uses JSON columns for routing/tool_use
    routing = merged.get("routing", {})
    tool_use = merged.get("tool_use", {})

    routing_ov = overlay.get("routing_override") or {}
    tool_use_ov = overlay.get("tool_use_override") or {}

    routing = deep_merge(routing, routing_ov)
    tool_use = deep_merge(tool_use, tool_use_ov)

    merged["routing"] = routing
    merged["tool_use"] = tool_use
    return merged


def build_user_context_block(memories: List[Dict[str, Any]], user_profile: Optional[Dict[str, Any]]) -> str:
    """
    Format user memories + profile into a concise system-usable text block.
    Keep it compact: top 3–5 facts/preferences/projects.
    """
    lines = []
    if user_profile:
        prefs = user_profile.get("preferences")
        if prefs:
            lines.append("User preferences:")
            for k, v in prefs.items():
                lines.append(f"- {k}: {v}")

    if memories:
        lines.append("Key user memories:")
        for mem in memories[:5]:
            lines.append(f"- ({mem['kind']}) {mem['text']}")

    return "\n".join(lines) if lines else ""


def build_messages(
    input_text: str,
    policy: Dict[str, Any],
    self_prompt: Dict[str, Any],
    user_context_block: str,
) -> list[Dict[str, str]]:
    """
    Build chat-style messages from:
      - global self-prompt
      - global policies (if you want)
      - user-specific long-term context
    """

    system_instructions = self_prompt.get("merged") or self_prompt.get("editable") or {}
    system_text = (
        "You are a self-modifying Thinking Machine.\n"
        "Follow safety rules, use tools when needed, and avoid hallucinations.\n"
        "Adapt behavior to the specific user based on the provided user context.\n\n"
        "=== Core Meta-Instructions ===\n"
        f"{system_instructions}\n\n"
    )

    if user_context_block:
        system_text += "=== User Context (long-term memory) ===\n"
        system_text += user_context_block + "\n\n"

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": input_text},
    ]
    return messages


def reasoning_engine(
    task: Dict[str, Any],
    policy: Dict[str, Any],
    self_prompt: Dict[str, Any],
    user_context_block: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Real reasoning call using LLMClient.
    """
    input_text = task["input_text"]
    messages = build_messages(input_text, policy, self_prompt, user_context_block)
    output_text = llm_client.chat(messages)

    metadata = {
        "latency_ms": 0,              # TODO: measure realistically
        "hallucination_flag": False,  # TODO: connect to judge/safety pipeline
        "low_confidence_flag": False,
        "reward_score": 0.8,
    }
from __future__ import annotations

import uuid
from typing import Any, Dict, Tuple, List, Optional

from libs import db
from libs.llm.client import LLMClient
from libs import user_memory as um


llm_client = LLMClient.from_env()


def deep_merge(a: dict, b: dict) -> dict:
    """
    Recursively merge dict b into dict a (a wins type conflicts, b overrides values).
    Returns a new dict.
    """
    import copy
    result = copy.deepcopy(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def apply_user_policy_overlay(
    base_policy: Dict[str, Any],
    user_id: Optional[str],
) -> Dict[str, Any]:
    """
    Load user overlay (if any) and merge into base policy.
    """
    if not user_id:
        return base_policy

    overlay = db.get_active_user_policy_overlay(user_id)
    if not overlay:
        return base_policy

    merged = dict(base_policy)
    # base_policy is a DB row; our Policy schema uses JSON columns for routing/tool_use
    routing = merged.get("routing", {})
    tool_use = merged.get("tool_use", {})

    routing_ov = overlay.get("routing_override") or {}
    tool_use_ov = overlay.get("tool_use_override") or {}

    routing = deep_merge(routing, routing_ov)
    tool_use = deep_merge(tool_use, tool_use_ov)

    merged["routing"] = routing
    merged["tool_use"] = tool_use
    return merged


def build_user_context_block(memories: List[Dict[str, Any]], user_profile: Optional[Dict[str, Any]]) -> str:
    """
    Format user memories + profile into a concise system-usable text block.
    Keep it compact: top 3–5 facts/preferences/projects.
    """
    lines = []
    if user_profile:
        prefs = user_profile.get("preferences")
        if prefs:
            lines.append("User preferences:")
            for k, v in prefs.items():
                lines.append(f"- {k}: {v}")

    if memories:
        lines.append("Key user memories:")
        for mem in memories[:5]:
            lines.append(f"- ({mem['kind']}) {mem['text']}")

    return "\n".join(lines) if lines else ""


def build_messages(
    input_text: str,
    policy: Dict[str, Any],
    self_prompt: Dict[str, Any],
    user_context_block: str,
) -> list[Dict[str, str]]:
    """
    Build chat-style messages from:
      - global self-prompt
      - global policies (if you want)
      - user-specific long-term context
    """

    system_instructions = self_prompt.get("merged") or self_prompt.get("editable") or {}
    system_text = (
        "You are a self-modifying Thinking Machine.\n"
        "Follow safety rules, use tools when needed, and avoid hallucinations.\n"
        "Adapt behavior to the specific user based on the provided user context.\n\n"
        "=== Core Meta-Instructions ===\n"
        f"{system_instructions}\n\n"
    )

    if user_context_block:
        system_text += "=== User Context (long-term memory) ===\n"
        system_text += user_context_block + "\n\n"

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": input_text},
    ]
    return messages


def reasoning_engine(
    task: Dict[str, Any],
    policy: Dict[str, Any],
    self_prompt: Dict[str, Any],
    user_context_block: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Real reasoning call using LLMClient.
    """
    input_text = task["input_text"]
    messages = build_messages(input_text, policy, self_prompt, user_context_block)
    output_text = llm_client.chat(messages)

    metadata = {
        "latency_ms": 0,              # TODO: measure realistically
        "hallucination_flag": False,  # TODO: connect to judge/safety pipeline
        "low_confidence_flag": False,
        "reward_score": 0.8,
    }
    return output_text, metadata


def handle_task(task: Dict[str, Any]) -> str:
    """
    End-to-end for one user task:
      - calls Thinking Machine 2.0 pipeline
      - logs trace
      - returns final output
    """
    session_id = task.get("session_id") or str(uuid.uuid4())
    task_id = task.get("task_id") or str(uuid.uuid4())
    task_type = task.get("task_type", "chat")
    domain = task.get("domain", "general")
    input_text = task["input_text"]

    # Run the full pipeline
    result = run_thinking_pipeline(task)

    final_output = result["final_output"]
    metadata = result["metadata"]
    user_id = result["user_id"]
    policy_id = result["policy_id"]
    self_prompt_id = result["self_prompt_id"]

    # ----- trace logging -----
    if policy_id and self_prompt_id:
        db.insert_trace(
            session_id=session_id,
            task_id=task_id,
            task_type=task_type,
            domain=domain,
            input_text=input_text,
            output_text=final_output,
            metadata=metadata,
            policy_version_id=policy_id,
            self_prompt_id=self_prompt_id,
            experiment_run_id=None,
            user_feedback=None,
            user_id=user_id,
        )

    # ----- optional memory writeback -----
    # (The pipeline doesn't currently return memory notes to write back, 
    # but we can keep the explicit note logic if it comes from the task input)
    if user_id and task.get("remember", True):
        if task.get("memory_note"):
            um.add_user_memory(
                user_id=user_id,
                text=task["memory_note"],
                kind="preference",
                importance=4,
            )

    return final_output
