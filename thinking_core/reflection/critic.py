from __future__ import annotations

from typing import Any, Dict

from libs.llm.client import LLMClient

# You can configure a separate judge backend via env vars if desired.
judge_client = LLMClient.from_env()


def reflection_pass(
    task: Dict[str, Any],
    draft_output: str,
    policy: Dict[str, Any],
    self_prompt: Dict[str, Any],
    user_context: str,
    world_model_ctx: Dict[str, Any],
    retrieval_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Reflection & critique step:
      - Assess safety, correctness, helpfulness.
      - Optionally propose brief corrections.
    """
    input_text = task["input_text"]

    system = (
        "You are a critical evaluator of another AI's answer.\n"
        "Your job is to check for: factual errors, unsafe advice, unclear reasoning.\n"
        "Be strict but concise. Mark a quality score 0-1.\n\n"
    )
    if world_model_ctx:
        system += f"World Model Signals: {world_model_ctx}\n\n"

    prompt = f"""
[USER QUESTION]
{input_text}

[DRAFT ANSWER]
{draft_output}

Evaluate:

1. Safety issues (especially medical/financial/legal).
2. Hallucinations or overconfident claims.
3. Clarity and structure.
4. Alignment with likely user preferences.

Respond in JSON:

{{
  "quality_score": float between 0 and 1,
  "needs_revision": true/false,
  "safety_concerns": [],
  "hallucination_risk": "low" | "medium" | "high",
  "suggested_fix": "short textual guidance to improve the answer"
}}
"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    critique_raw = judge_client.chat(messages)

    # For now, trust it's JSON-ish; in production, wrap in robust parser.
    import json
    try:
        critique = json.loads(critique_raw)
    except Exception:
        critique = {
            "quality_score": 0.7,
            "needs_revision": False,
            "safety_concerns": [],
            "hallucination_risk": "medium",
            "suggested_fix": "Could not parse critique; use answer as-is with caution.",
        }

    return critique
