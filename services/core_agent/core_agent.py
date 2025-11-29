# services/core_agent/core_agent.py
"""
Core Agent - Thinking Machine 2.0

Main entry point for the cognitive engine.
Delegates to the full thinking pipeline which includes:
- User Context (memory)
- World Model (task classification)
- RAG Retrieval (knowledge base + web)
- Multi-Agent Reasoning
- Reflection & Critique
- Output Synthesis
"""
from __future__ import annotations

from typing import Any, Dict
import uuid
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from libs import db
from libs import user_memory as um
from thinking_core.pipeline import run_thinking_pipeline


def handle_task(task: Dict[str, Any]) -> str:
    """
    Core entry point for the Thinking Machine 2.0 cognitive engine.

    Responsibilities:
      - Ensure session_id / task_id
      - Delegate reasoning to thinking_core.pipeline.run_thinking_pipeline()
        (which includes User Context, World Model, RAG, Multi-Agent, Reflection, Output Synthesis)
      - Log a trace row (including RAG-related metadata)
      - Optionally write user memories
      - Return final output text

    Args:
        task: Task dictionary with input_text, domain, user_external_id, etc.

    Returns:
        str: The final output text
    """

    # --------- normalize IDs ---------
    session_id = task.get("session_id") or str(uuid.uuid4())
    task_id = task.get("task_id") or str(uuid.uuid4())
    task_type = task.get("task_type", "chat")
    domain = task.get("domain", "general")
    input_text = task["input_text"]

    # --------- run full TM 2.0 pipeline (includes RAG) ---------
    pipeline_result = run_thinking_pipeline(task)
    final_output = pipeline_result["final_output"]
    metadata = pipeline_result.get("metadata", {})
    user_id = pipeline_result.get("user_id")
    policy_id = pipeline_result.get("policy_id")
    self_prompt_id = pipeline_result.get("self_prompt_id")

    # --------- log trace ---------
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

    # --------- optional memory writeback ---------
    if user_id and task.get("remember", True):
        # 1) Explicit memory note from the user
        if task.get("memory_note"):
            um.add_user_memory(
                user_id=user_id,
                text=task["memory_note"],
                kind="preference",
                importance=4,
            )
        # 2) Could add automatic memory extraction here
        # e.g., detect important facts from the conversation

    return final_output
