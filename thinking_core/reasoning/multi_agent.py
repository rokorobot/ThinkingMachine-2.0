from __future__ import annotations

from typing import Any, Dict, List

from libs.llm.client import LLMClient


def _build_base_system_prompt(policy: Dict[str, Any], self_prompt: Dict[str, Any], user_context: str,
                              world_model_ctx: Dict[str, Any], retrieval_ctx: Dict[str, Any]) -> str:
    base_instr = self_prompt.get("merged") or self_prompt.get("editable") or {}
    txt = (
        "You are a self-modifying Thinking Machine.\n"
        "Follow safety rules, avoid hallucinations, and adapt to the specific user.\n\n"
        "=== Core Meta-Instructions ===\n"
        f"{base_instr}\n\n"
    )
    if user_context:
        txt += "=== User Context ===\n" + user_context + "\n\n"
    if world_model_ctx:
        txt += "=== World Model Signals ===\n" + str(world_model_ctx) + "\n\n"
    if retrieval_ctx.get("used"):
        txt += "=== Retrieved Knowledge (snippets) ===\n"
        for sn in retrieval_ctx["snippets"]:
            txt += f"- {sn.get('title','')} :: {sn.get('content','')[:300]}\n"
        txt += "\n"
    return txt


def _call_agent(
    role: str,
    input_text: str,
    system_prompt: str,
    llm_client: LLMClient,
) -> str:
    """
    Single-agent call with a role-specific instruction.
    """
    agent_header = f"You are acting as the {role} within a multi-agent thinking system."
    system = system_prompt + "\n" + agent_header

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": input_text},
    ]
    return llm_client.chat(messages)


def multi_agent_reasoning_pass(
    task: Dict[str, Any],
    policy: Dict[str, Any],
    self_prompt: Dict[str, Any],
    user_context: str,
    world_model_ctx: Dict[str, Any],
    retrieval_ctx: Dict[str, Any],
    llm_client: LLMClient,
) -> Dict[str, Any]:
    """
    Run multiple reasoning agents and combine their outputs into a draft.
    """
    input_text = task["input_text"]

    system_prompt = _build_base_system_prompt(
        policy=policy,
        self_prompt=self_prompt,
        user_context=user_context,
        world_model_ctx=world_model_ctx,
        retrieval_ctx=retrieval_ctx,
    )

    # Agents
    direct_ans = _call_agent("Direct Responder (fast, intuitive)", input_text, system_prompt, llm_client)
    planner = _call_agent("Planner (step-by-step reasoning)", input_text, system_prompt, llm_client)
    cautious = _call_agent("Cautious Checker (looks for risks & mistakes)", input_text, system_prompt, llm_client)

    # Combine via a short fusion call (could be same or smaller model)
    fusion_prompt = system_prompt + """
You are the Synthesizer Agent. You have three candidate responses:
[Direct]
{direct}
[Planner]
{planner}
[Cautious]
{cautious}

Task:
- Merge the strengths of these.
- Prefer safety & correctness over style.
- If they disagree, explain the most likely correct resolution.
- Return a single, coherent answer for the user.
""".format(direct=direct_ans, planner=planner, cautious=cautious)

    synth_messages = [
        {"role": "system", "content": fusion_prompt},
        {"role": "user", "content": input_text},
    ]
    draft_output = llm_client.chat(synth_messages)

    metadata = {
        "agents": {
            "direct": direct_ans,
            "planner": planner,
            "cautious": cautious,
        }
    }

    return {
        "draft_output": draft_output,
        "metadata": metadata,
    }
