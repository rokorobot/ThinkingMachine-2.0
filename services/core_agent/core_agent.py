from __future__ import annotations

import uuid
from typing import Any, Dict, Tuple, List, Optional

from libs import db
from libs.llm.client import LLMClient
from libs import user_memory as um


llm_client = LLMClient.from_env()


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
      - resolve user_id (optional)
      - load active policy & self-prompt
      - pull user memories
      - run reasoning
      - insert trace (with user_id)
      - optionally write new memory
    """
    policy = db.get_active_policy_version()
    self_prompt = db.get_active_self_prompt()
    
    if not policy:
        policy = {"id": None, "routing": {}, "tool_use": {}, "safety_overrides": {}}
    if not self_prompt:
        self_prompt = {"id": None, "merged": {}, "editable": {}}

    session_id = task.get("session_id") or str(uuid.uuid4())
    task_id = task.get("task_id") or str(uuid.uuid4())
    task_type = task.get("task_type", "chat")
    domain = task.get("domain", "general")
    input_text = task["input_text"]

    # ----- user resolution -----
    user_id = None
    user_profile = None
    user_external_id = task.get("user_external_id")
    if user_external_id:
        user_row = um.get_or_create_user(user_external_id, default_profile={"preferences": {}})
        user_id = user_row["id"]
        user_profile = user_row.get("profile") or {}

    # ----- user memory retrieval -----
    user_memories: List[Dict[str, Any]] = []
    if user_id:
        # semantic + recency combo
        sem = um.search_user_memories(user_id, query=input_text, top_k=3, min_importance=2)
        rec = um.get_top_recent_memories(user_id, limit=3, min_importance=3)
        # de-duplicate by id
        seen = set()
        combo = []
        for m in sem + rec:
            if m["id"] not in seen:
                seen.add(m["id"])
                combo.append(m)
        user_memories = combo

    user_context_block = build_user_context_block(user_memories, user_profile)

    # ----- reasoning -----
    output_text, metadata = reasoning_engine(task, policy, self_prompt, user_context_block)

    # ----- trace logging -----
    if policy.get("id") and self_prompt.get("id"):
        db.insert_trace(
            session_id=session_id,
            task_id=task_id,
            task_type=task_type,
            domain=domain,
            input_text=input_text,
            output_text=output_text,
            metadata=metadata,
            policy_version_id=policy["id"],
            self_prompt_id=self_prompt["id"],
            experiment_run_id=None,
            user_feedback=None,
            user_id=user_id,
        )

    # ----- optional memory writeback -----
    if user_id and task.get("remember", True):
        # 1) explicit memory note from the user
        if task.get("memory_note"):
            um.add_user_memory(
                user_id=user_id,
                text=task["memory_note"],
                kind="preference",
                importance=4,
            )
        # 2) or some heuristic from the task/output (stub for now)
        # Example heuristic: if user says "I prefer X", you could extract that phrase
        # with a simple regex or a classifier and store as memory.

    return output_text
