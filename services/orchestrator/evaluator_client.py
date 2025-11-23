from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
from libs.policy_store import Proposal, Policy

@dataclass
class EvalResult:
    proposal_id: str
    score: float
    safety_ok: bool
    details: Dict[str, Any]

class EvaluatorClient:
    """
    Talks to eval_judge service to score a candidate state.
    """

    def __init__(self):
        pass

    def evaluate_candidate(self, proposal: Proposal, candidate_policy: Policy) -> EvalResult:
        # TODO: Implement RPC/HTTP call to eval_judge
        # For skeleton, we fake a score.
        fake_score = 0.85
        safety_ok = True

        return EvalResult(
            proposal_id=proposal.id,
            score=fake_score,
            safety_ok=safety_ok,
            details={"note": "stub evaluation"}
        )
