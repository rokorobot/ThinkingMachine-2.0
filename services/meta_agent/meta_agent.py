from __future__ import annotations

from typing import Any, Dict, List

from libs import db


def analyze_traces_and_build_payloads(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Very simple heuristic:
      - if many 'medical' hallucinations → tighten medical routing policy
      - if many low_confidence → update self-prompt (not fully implemented here)
    Returns a list of proposal dicts (type + payload + reason).
    """
    if not traces:
        return []

    medical_errors = [t for t in traces if t.get("domain") == "medical" and t["metadata"].get("hallucination_flag")]
    low_conf = [t for t in traces if t["metadata"].get("low_confidence_flag")]

    proposals: List[Dict[str, Any]] = []

    if len(medical_errors) >= 5:
        proposals.append(
            {
                "proposal_type": "policy_update",
                "payload": {
                    "domain": "medical",
                    "change": {
                        "require_multi_source_check": True,
                        "min_sources": 2,
                        "ask_clarifying_if_ambiguous": True,
                    },
                },
                "reason": f"{len(medical_errors)} hallucinations in medical domain in last window",
            }
        )

    if len(low_conf) >= 10:
        proposals.append(
            {
                "proposal_type": "self_prompt_update",
                "payload": {
                    "edit_type": "append_instruction",
                    "instruction": "When unsure, explicitly state uncertainty and suggest external verification.",
                },
                "reason": f"{len(low_conf)} low-confidence traces detected",
            }
        )

    return proposals


def run_reflection_cycle(hours: int = 24, limit: int = 100) -> None:
    traces = db.get_problematic_traces(hours=hours, limit=limit)
    if not traces:
        print("meta_agent: no problematic traces found")
        return

    active_policy = db.get_active_policy_version()
    active_prompt = db.get_active_self_prompt()
    current_policy_id = active_policy["id"] if active_policy else None
    current_prompt_id = active_prompt["id"] if active_prompt else None

    proposal_specs = analyze_traces_and_build_payloads(traces)
    if not proposal_specs:
        print("meta_agent: no proposals constructed from traces")
        return

    for spec in proposal_specs:
        proposal_id = db.insert_proposal(
            created_by="meta_agent",
            proposal_type=spec["proposal_type"],
            payload=spec["payload"],
            current_policy_version=current_policy_id,
            current_self_prompt_id=current_prompt_id,
            reason=spec["reason"],
        )
        print(f"meta_agent: created proposal {proposal_id} ({spec['proposal_type']})")
