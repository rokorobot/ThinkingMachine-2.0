# thinking_core/pipeline/pipeline.py
"""
Thinking Machine 2.0 - Main Cognitive Pipeline

Orchestrates the complete reasoning flow:
1. User Context (memory retrieval)
2. World Model (task classification)
3. RAG Retrieval (knowledge base + optional web)
4. Multi-Agent Reasoning (with retrieved context)
5. Reflection & Critique
6. Output Synthesis
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

from libs import db
from libs import user_memory as um
from thinking_core.retrieval.retriever import retrieval_pass


def build_user_context(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 1: Build user-specific context from long-term memory.
    """
    user_id = None
    user_profile = None
    user_memories: List[Dict[str, Any]] = []
    
    user_external_id = task.get("user_external_id")
    if user_external_id:
        user_row = um.get_or_create_user(user_external_id, default_profile={"preferences": {}})
        user_id = user_row["id"]
        user_profile = user_row.get("profile") or {}
        
        # Retrieve relevant memories
        input_text = task["input_text"]
        sem = um.search_user_memories(user_id, query=input_text, top_k=3, min_importance=2)
        rec = um.get_top_recent_memories(user_id, limit=3, min_importance=3)
        
        # De-duplicate
        seen = set()
        combo = []
        for m in sem + rec:
            if m["id"] not in seen:
                seen.add(m["id"])
                combo.append(m)
        user_memories = combo
    
    return {
        "user_id": user_id,
        "user_profile": user_profile,
        "user_memories": user_memories,
    }


def world_model_pass(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 2: Classify task nature and determine reasoning strategy.
    
    This is a simple heuristic version. In a full implementation,
    this could use an LLM to classify the task.
    """
    input_text = task.get("input_text", "").lower()
    domain = task.get("domain", "general")
    
    # Simple keyword-based classification
    task_nature = "general"
    
    if any(kw in input_text for kw in ["code", "python", "javascript", "function", "debug"]):
        task_nature = "coding"
    elif any(kw in input_text for kw in ["medical", "diagnosis", "patient", "symptom"]):
        task_nature = "medical_like"
    elif any(kw in input_text for kw in ["research", "paper", "study", "analysis"]):
        task_nature = "research"
    
    return {
        "task_nature": task_nature,
        "domain": domain,
        "complexity": "medium",  # Could be estimated
        "requires_tools": False,  # Could be detected
    }


def build_system_prompt(
    task: Dict[str, Any],
    policy: Dict[str, Any],
    self_prompt: Dict[str, Any],
    user_context: Dict[str, Any],
    retrieval_ctx: Dict[str, Any],
) -> str:
    """
    Build the complete system prompt with all context layers.
    """
    system_instructions = self_prompt.get("merged") or self_prompt.get("editable") or {}
    
    prompt = (
        "You are a self-modifying Thinking Machine.\n"
        "Follow safety rules, use tools when needed, and avoid hallucinations.\n"
        "Adapt behavior to the specific user based on the provided context.\n\n"
        "=== Core Meta-Instructions ===\n"
        f"{system_instructions}\n\n"
    )
    
    # Add user context
    if user_context.get("user_memories"):
        prompt += "=== User Context (long-term memory) ===\n"
        for mem in user_context["user_memories"][:5]:
            prompt += f"- ({mem['kind']}) {mem['text']}\n"
        prompt += "\n"
    
    # Add retrieved knowledge (RAG)
    if retrieval_ctx.get("used"):
        prompt += "=== Retrieved Knowledge ===\n"
        for snippet in retrieval_ctx["snippets"][:5]:
            title = snippet.get("title", "")
            content = snippet.get("content", "")[:300]
            kind = snippet.get("kind", "")
            prompt += f"- [{kind}] {title}: {content}...\n"
        prompt += "\n"
    
    return prompt


def multi_agent_reasoning_pass(
    task: Dict[str, Any],
    policy: Dict[str, Any],
    self_prompt: Dict[str, Any],
    user_context: Dict[str, Any],
    world_model_ctx: Dict[str, Any],
    retrieval_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Phase 4: Multi-agent reasoning with full context.
    
    In a full implementation, this would coordinate multiple specialized agents.
    For now, we use a single LLM call with rich context.
    """
    from libs.llm.client import LLMClient
    
    llm_client = LLMClient.from_env()
    
    system_prompt = build_system_prompt(task, policy, self_prompt, user_context, retrieval_ctx)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task["input_text"]},
    ]
    
    start_time = time.time()
    output_text = llm_client.chat(messages)
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "output_text": output_text,
        "latency_ms": latency_ms,
        "agent_type": "primary",
    }


def reflection_pass(
    task: Dict[str, Any],
    reasoning_result: Dict[str, Any],
    retrieval_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Phase 5: Self-critique and reflection.
    
    Evaluates the reasoning output for quality, safety, and coherence.
    """
    output_text = reasoning_result["output_text"]
    
    # Simple heuristic checks (in full version, use LLM-based critic)
    hallucination_flag = False
    low_confidence_flag = False
    
    # Check for hedging language (low confidence indicators)
    if any(phrase in output_text.lower() for phrase in ["i'm not sure", "maybe", "possibly", "might be"]):
        low_confidence_flag = True
    
    # Estimate reward score (in full version, use learned model)
    reward_score = 0.8
    if low_confidence_flag:
        reward_score -= 0.1
    if len(output_text) < 50:
        reward_score -= 0.2
    
    return {
        "hallucination_flag": hallucination_flag,
        "low_confidence_flag": low_confidence_flag,
        "reward_score": max(0.0, min(1.0, reward_score)),
        "critique": "Output appears reasonable." if reward_score > 0.7 else "Output may need improvement.",
    }


def synthesize_output(
    reasoning_result: Dict[str, Any],
    reflection_result: Dict[str, Any],
    user_context: Dict[str, Any],
) -> str:
    """
    Phase 6: Final output synthesis.
    
    Combines reasoning output with reflection insights.
    """
    output = reasoning_result["output_text"]
    
    # In a full implementation, might modify output based on reflection
    # For now, just return the reasoning output
    return output


def run_thinking_pipeline(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for Thinking Machine 2.0 cognitive pipeline.
    
    Orchestrates all phases:
    1. User Context
    2. World Model
    3. RAG Retrieval
    4. Multi-Agent Reasoning
    5. Reflection
    6. Output Synthesis
    
    Returns:
        {
            "final_output": str,
            "metadata": dict,
            "user_id": str | None,
            "policy_id": str,
            "self_prompt_id": str,
        }
    """
    # Load active genome
    policy = db.get_active_policy_version()
    self_prompt = db.get_active_self_prompt()
    
    if not policy:
        policy = {"id": None, "routing": {}, "tool_use": {}, "safety_overrides": {}}
    if not self_prompt:
        self_prompt = {"id": None, "merged": {}, "editable": {}}
    
    # Phase 1: User Context
    user_context = build_user_context(task)
    
    # Phase 2: World Model
    world_model_ctx = world_model_pass(task)
    
    # Phase 3: RAG Retrieval
    retrieval_ctx = retrieval_pass(task, world_model_ctx, policy)
    
    # Phase 4: Multi-Agent Reasoning
    reasoning_result = multi_agent_reasoning_pass(
        task, policy, self_prompt, user_context, world_model_ctx, retrieval_ctx
    )
    
    # Phase 5: Reflection
    reflection_result = reflection_pass(task, reasoning_result, retrieval_ctx)
    
    # Phase 6: Output Synthesis
    final_output = synthesize_output(reasoning_result, reflection_result, user_context)
    
    # Compile metadata
    metadata = {
        "latency_ms": reasoning_result["latency_ms"],
        "hallucination_flag": reflection_result["hallucination_flag"],
        "low_confidence_flag": reflection_result["low_confidence_flag"],
        "reward_score": reflection_result["reward_score"],
        "task_nature": world_model_ctx["task_nature"],
        "rag_used": retrieval_ctx.get("used", False),
        "rag_snippet_count": len(retrieval_ctx.get("snippets", [])),
        "user_memory_count": len(user_context.get("user_memories", [])),
    }
    
    return {
        "final_output": final_output,
        "metadata": metadata,
        "user_id": user_context.get("user_id"),
        "policy_id": policy.get("id"),
        "self_prompt_id": self_prompt.get("id"),
    }
