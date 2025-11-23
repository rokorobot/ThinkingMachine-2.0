from __future__ import annotations

import logging
import time
import asyncio
from libs.policy_store import PolicyStore
from services.orchestrator.candidate_runner import CandidateStateBuilder
from services.orchestrator.evaluator_client import EvaluatorClient
from libs.logging.logger import get_logger

logger = get_logger("orchestrator")

class Orchestrator:
    """
    Periodically:
      - loads pending proposals
      - builds candidate states
      - sends them for evaluation
      - promotes or rejects based on scores & safety
    """

    def __init__(self, policy_store: PolicyStore):
        self.policy_store = policy_store
        self.candidate_builder = CandidateStateBuilder(policy_store)
        self.evaluator = EvaluatorClient()
        self.min_score_for_acceptance = 0.80

    async def run_loop(self, interval_seconds: int = 30):
        logger.info("Orchestrator loop started.")
        while True:
            try:
                self._process_pending_proposals()
            except Exception as e:
                logger.exception(f"Error in orchestrator loop: {e}")
            await asyncio.sleep(interval_seconds)

    def _process_pending_proposals(self) -> None:
        proposals = self.policy_store.load_pending_proposals()
        if not proposals:
            return

        logger.info(f"Found {len(proposals)} pending proposals")

        for proposal in proposals:
            logger.info(f"Evaluating proposal {proposal.id} ({proposal.proposal_type.value})")
            
            # 1. Build Candidate
            candidate_policy = self.candidate_builder.build_candidate_policy(proposal)
            
            # 2. Evaluate
            eval_result = self.evaluator.evaluate_candidate(proposal, candidate_policy)

            # 3. Decide
            if not eval_result.safety_ok:
                logger.warning(f"Proposal {proposal.id} failed safety check")
                self.policy_store.update_proposal_status(
                    proposal_id=proposal.id,
                    status="rejected",
                    reason="safety_check_failed"
                )
                continue

            if eval_result.score >= self.min_score_for_acceptance:
                logger.info(f"Proposal {proposal.id} accepted with score {eval_result.score}")
                self.policy_store.save_policy(candidate_policy, label=f"auto_promote_{proposal.id}", author="orchestrator")
                self.policy_store.update_proposal_status(
                    proposal_id=proposal.id,
                    status="accepted",
                    reason=f"score={eval_result.score}"
                )
            else:
                logger.info(f"Proposal {proposal.id} rejected with score {eval_result.score}")
                self.policy_store.update_proposal_status(
                    proposal_id=proposal.id,
                    status="rejected",
                    reason=f"score={eval_result.score}"
                )

if __name__ == "__main__":
    store = PolicyStore()
    orchestrator = Orchestrator(store)
    asyncio.run(orchestrator.run_loop(interval_seconds=10))
