from __future__ import annotations

from typing import Any, Dict, Optional

from libs import db
from libs import user_memory as um
from libs.llm.client import LLMClient

from .user_context.assembler import build_user_context
from .world_model.simulator import world_model_pass
from .retrieval.retriever import retrieval_pass
from .reasoning.multi_agent import multi_agent_reasoning_pass
from .reflection.critic import reflection_pass
from .output.synthesizer import synthesize_output


llm_client = LLMClient.from_env()


def run_thinking_pipeline(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full Thinking Machine 2.0 pipeline:
      1) Resolve user + context
      2) World Model Simulator
      3) Retrieval & Knowledge Core
      4) Multi-Agent Reasoning Core
      5) Reflection / Critique Loop
      6) Output Synthesis

    Returns a dict with:
      {
        "final_output": str,
        "draft": str,
        "retrieval_ctx": ...,
        "world_model_ctx": ...,
        "reflection_notes": ...,
        "metadata": {...},
        "user_id": str | None,
        "policy_id": str,
        "self_prompt_id": str
      }
    """
    # ----- Load global state -----
    policy = db.get_active_policy_version()
    self_prompt = db.get_active_self_prompt()
    if not policy or not self_prompt:
        # Fallback if DB is empty or not ready, to avoid crashing
        policy = policy or {"id": "fallback", "routing": {}, "tool_use": {}, "safety_overrides": {}}
        self_prompt = self_prompt or {"id": "fallback", "merged": {}, "editable": {}}

    # ----- Resolve user / user_id / profile / memories -----
    user_external_id = task.get("user_external_id")
    user_row = None
    if user_external_id:
        user_row = um.get_or_create_user(user_external_id, default_profile={"preferences": {}})
    user_id = user_row["id"] if user_row else None
    user_profile = (user_row or {}).get("profile") if user_row else None

    user_context_block, user_memories = build_user_context(
        user_id=user_id,
        user_profile=user_profile,
        input_text=task["input_text"],
    )

    # ----- World Model pass -----
    world_model_ctx = world_model_pass(
        task=task,
        policy=policy,
        user_context=user_context_block,
    )

    # ----- Retrieval pass -----
    retrieval_ctx = retrieval_pass(
        task=task,
        world_model_ctx=world_model_ctx,
        policy=policy,
    )

    # ----- Multi-Agent Reasoning pass -----
    reasoning_result = multi_agent_reasoning_pass(
        task=task,
        policy=policy,
        self_prompt=self_prompt,
        user_context=user_context_block,
        world_model_ctx=world_model_ctx,
        retrieval_ctx=retrieval_ctx,
        llm_client=llm_client,
    )
    draft_output = reasoning_result["draft_output"]
    reasoning_metadata = reasoning_result.get("metadata", {})

    # ----- Reflection / Critique pass -----
    reflection_result = reflection_pass(
        task=task,
        draft_output=draft_output,
        policy=policy,
        self_prompt=self_prompt,
        user_context=user_context_block,
        world_model_ctx=world_model_ctx,
        retrieval_ctx=retrieval_ctx,
    )

    final_output = synthesize_output(
        task=task,
        draft_output=draft_output,
        reasoning_metadata=reasoning_metadata,
        reflection_result=reflection_result,
        user_profile=user_profile,
    )

    # ----- Attach IDs and metadata for logging -----
    result = {
        "final_output": final_output,
        "draft": draft_output,
        "retrieval_ctx": retrieval_ctx,
        "world_model_ctx": world_model_ctx,
        "reflection_notes": reflection_result,
        "metadata": {
            "reasoning": reasoning_metadata,
            "world_model": world_model_ctx.get("signals") if isinstance(world_model_ctx, dict) else None,
            "reflection_quality": reflection_result.get("quality_score") if isinstance(reflection_result, dict) else None,
        },
        "user_id": user_id,
        "policy_id": policy["id"],
        "self_prompt_id": self_prompt["id"],
    }
    return result
