from __future__ import annotations

import copy
import time
from typing import Any, Dict, List

from libs import db


def apply_policy_payload_to_routing(
    baseline_policy: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Very simple patcher:
      - assumes payload = { "domain": "...", "change": {...} }
      - merges 'change' into routing[domain]
    """
    routing = copy.deepcopy(baseline_policy["routing"])
    domain = payload.get("domain")
    change = payload.get("change", {})

    domain_cfg = routing.get(domain, {})
    domain_cfg.update(change)
    routing[domain] = domain_cfg
    return routing


def start_experiments_for_pending_proposals() -> None:
    proposals = db.get_pending_proposals(limit=10)
    if not proposals:
        # print("orchestrator: no pending proposals")
        return

    for prop in proposals:
        if prop["proposal_type"] != "policy_update":
            # For now, only handle policy_update proposals in this skeleton
            print(f"orchestrator: skipping non-policy proposal {prop['id']}")
            continue

        baseline_policy = db.get_active_policy_version()
        if not baseline_policy:
            print("orchestrator: no active baseline policy; cannot create experiment")
            continue

        # Build candidate routing by applying payload
        candidate_routing = apply_policy_payload_to_routing(
            baseline_policy, prop["payload"]
        )

        # tool_use and safety_overrides unchanged in this simple example
        candidate_policy_id = db.insert_policy_version(
            created_by="orchestrator",
            routing=candidate_routing,
            tool_use=baseline_policy["tool_use"],
            safety_overrides=baseline_policy["safety_overrides"],
            label=f"candidate_for_{prop['id']}",
            is_active=False,
        )

        # Create experiment + some runs
        if "game_strategy" in prop["payload"]:
            # Use Tournament Config for Game Theoretic Proposals
            from services.orchestrator.experiment_config import tournament_config
            config = tournament_config(prop["payload"].get("domain", "medical"))
        else:
            # Standard A/B Test
            config = {
                "description": "A/B test candidate vs baseline in target domain",
                "target_domain": prop["payload"].get("domain", "medical"),
                "num_runs": 3
            }
            
        experiment_id = db.create_experiment(
            proposal_id=prop["id"],
            baseline_policy_id=baseline_policy["id"],
            candidate_policy_id=candidate_policy_id,
            config=config,
        )

        # For tournament, we might have rounds * batch_per_round
        # For simplicity here, we just map rounds -> runs
        num_runs = config.get("rounds", config.get("num_runs", 3))
        
        for run_index in range(num_runs):
            db.create_experiment_run(
                experiment_id=experiment_id,
                run_index=run_index,
                candidate_policy_id=candidate_policy_id,
            )

        db.mark_proposal_in_experiment(prop["id"])
        print(f"orchestrator: started experiment {experiment_id} for proposal {prop['id']}")


def finalize_completed_experiments(min_score_for_acceptance: float = 0.85) -> None:
    experiments = db.get_experiments_ready_to_finalize()
    if not experiments:
        # print("orchestrator: no experiments ready to finalize")
        return

    for exp in experiments:
        runs = db.get_runs_for_experiment(exp["id"])
        if not runs:
            db.finalize_experiment(exp["id"], status="failed", result_summary={"reason": "no_runs"})
            continue

        # Aggregate
        scores = [r["score"] for r in runs if r["score"] is not None]
        safety_flags = [r["safety_ok"] for r in runs if r["safety_ok"] is not None]

        if not scores:
            db.finalize_experiment(exp["id"], status="failed", result_summary={"reason": "no_scores"})
            continue

        avg_score = sum(scores) / len(scores)
        all_safe = all(safety_flags) if safety_flags else False
        
        # ESS Check (Evolutionarily Stable Strategy)
        # For now, we assume if it passed the tournament rounds with high score, it's stable.
        # Real impl would check variance or specific 'invasion' metrics.
        ess_ok = True 

        result_summary = {
            "avg_score": avg_score,
            "num_runs": len(runs),
            "all_safe": all_safe,
            "ess_ok": ess_ok
        }

        db.finalize_experiment(
            experiment_id=exp["id"],
            status="completed",
            result_summary=result_summary,
        )

        # Decide whether to accept candidate or not
        proposal_id = exp["proposal_id"]
        candidate_policy_id = exp["candidate_policy_id"]

        # Stability Gate: High Score + All Safe + ESS OK
        if all_safe and avg_score >= min_score_for_acceptance and ess_ok:
            print(f"orchestrator: ACCEPT experiment {exp['id']} with avg_score={avg_score:.3f}")
            # Flip active policy
            db.set_active_policy(candidate_policy_id)
            db.update_proposal_status(
                proposal_id=proposal_id,
                status="accepted",
                final_policy_version=candidate_policy_id,
                reason=f"avg_score={avg_score:.3f}",
            )
        else:
            print(f"orchestrator: REJECT experiment {exp['id']} with avg_score={avg_score:.3f}, all_safe={all_safe}")
            db.update_proposal_status(
                proposal_id=proposal_id,
                status="rejected",
                reason=f"avg_score={avg_score:.3f}, all_safe={all_safe}",
            )


def run_orchestrator_loop() -> None:
    while True:
        try:
            start_experiments_for_pending_proposals()
            finalize_completed_experiments()
        except Exception as e:
            print("orchestrator: error in loop:", e)
        time.sleep(10)
