from __future__ import annotations

import copy
from typing import Any, Dict

from libs.policy_store import PolicyStore, Proposal, ProposalType, Policy

class CandidateStateBuilder:
    """
    Creates a candidate policy/self-prompt state by applying a proposal
    on top of the current state.
    """

    def __init__(self, policy_store: PolicyStore):
        self.policy_store = policy_store

    def build_candidate_policy(self, proposal: Proposal) -> Policy:
        base_policy = self.policy_store.load_current_policy()
        candidate = copy.deepcopy(base_policy)

        if proposal.proposal_type == ProposalType.POLICY_UPDATE:
            self._apply_policy_update(candidate, proposal.payload)
        
        return candidate

    def _apply_policy_update(self, policy: Policy, payload: Dict[str, Any]) -> None:
        domain = payload.get("domain")
        change = payload.get("change", {})

        # Example: update routing config for a domain
        # In a real system, we'd have a more robust schema for where 'change' applies.
        # Here we assume 'routing' has keys like 'medical', 'coding', etc.
        if domain:
            current_cfg = policy.routing.get(domain, {})
            if isinstance(current_cfg, dict):
                current_cfg.update(change)
                policy.routing[domain] = current_cfg
            else:
                # If it didn't exist or wasn't a dict, just set it
                policy.routing[domain] = change
