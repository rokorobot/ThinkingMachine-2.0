from __future__ import annotations

from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from libs import db
from genome_store.skills.code.game_strategy import recommend_policy_patch_from_metrics


router = APIRouter(prefix="/admin", tags=["admin"])


class GameTheoryOptimizeRequest(BaseModel):
    domain: str = "medical"
    hours: int = 24
    commit: bool = False
    weights: Optional[Dict[str, float]] = None


class MixView(BaseModel):
    player: str
    strategies: List[str]
    mix: List[float]


class GameTheoryOptimizeResponse(BaseModel):
    domain: str
    metrics: Dict[str, float]
    chosen_strategy: str
    patch: Dict[str, Any]
    mixes: List[MixView]
    proposal_id: Optional[str] = None
    message: str


def compute_window_metrics(domain: str, hours: int) -> Dict[str, float]:
    """
    Aggregate some basic metrics from traces for the given domain & time window.
    This is where you define what the game sees as 'state of the world'.
    """
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            # Accuracy proxy: avg reward_score
            cur.execute(
                """
                SELECT
                  COALESCE(AVG((metadata->>'reward_score')::float), 0.7) AS accuracy,
                  COALESCE(1.0 - AVG(CASE WHEN (metadata->>'hallucination_flag')::bool THEN 1 ELSE 0 END), 0.9) AS safety,
                  COALESCE(AVG((metadata->>'latency_ms')::float), 200.0) AS latency,
                  COALESCE(AVG(CASE WHEN (user_feedback->>'thumbs_up')::bool THEN 1 ELSE 0 END), 0.6) AS user_sat
                FROM traces
                WHERE created_at > NOW() - (%s || ' hours')::interval
                  AND domain = %s
                """,
                (str(hours), domain),
            )
            row = cur.fetchone()
            if row is None:
                return {"accuracy": 0.7, "safety": 0.9, "latency": 200.0, "user_sat": 0.6}
            accuracy, safety, latency, user_sat = row
            return {
                "accuracy": float(accuracy),
                "safety": float(safety),
                "latency": float(latency),
                "user_sat": float(user_sat),
            }


@router.post("/game-theory/optimize", response_model=GameTheoryOptimizeResponse)
def optimize_adaptability(req: GameTheoryOptimizeRequest):
    """
    Full Phase 1–6 for adaptability:
      PH1: Problem/players are implicit (Thinking Machine vs Safety vs User)
      PH2: Incentive mapping via metrics
      PH3: Strategy set is defined in game_strategy
      PH4: Equilibrium via fictitious play
      PH5: Translate to concrete patch
      PH6: Optionally commit as a proposal for orchestrator
    """
    # Phase 2: metrics from traces
    metrics = compute_window_metrics(domain=req.domain, hours=req.hours)

    # Phase 3–4: equilibrium & patch recommendation
    rec = recommend_policy_patch_from_metrics(metrics, req.weights)
    chosen_strategy = rec["chosen_strategy"]
    patch = rec["patch"]
    mixes = rec["mixes"]

    proposal_id: Optional[str] = None
    msg = "Preview only; proposal not created."

    if req.commit:
        active_policy = db.get_active_policy_version()
        active_prompt = db.get_active_self_prompt()
        if not active_policy or not active_prompt:
            raise HTTPException(status_code=400, detail="No active policy or self-prompt configured")

        payload = {
            "domain": req.domain,
            "change": patch,
            "game_strategy": chosen_strategy,
            "mixes": mixes,
            "metrics": metrics,
        }

        proposal_id = db.insert_proposal(
            created_by="meta_agent",  # or 'admin:manual'
            proposal_type="policy_update",
            payload=payload,
            current_policy_version=active_policy["id"],
            current_self_prompt_id=active_prompt["id"],
            reason=f"Game-theory equilibrium favored {chosen_strategy} for domain={req.domain}",
        )
        msg = f"Proposal {proposal_id} created and ready for orchestrated experiments."

    return GameTheoryOptimizeResponse(
        domain=req.domain,
        metrics=metrics,
        chosen_strategy=chosen_strategy,
        patch=patch,
        mixes=[MixView(**m) for m in mixes],
        proposal_id=proposal_id,
        message=msg,
    )


@router.get("/game-theory/preview", response_model=GameTheoryOptimizeResponse)
def preview_equilibrium(
    domain: str = Query("medical"),
    hours: int = Query(24, ge=1, le=168),
):
    """
    Shortcut GET endpoint: just compute metrics + equilibrium & patch, no proposal.
    """
    metrics = compute_window_metrics(domain=domain, hours=hours)
    rec = recommend_policy_patch_from_metrics(metrics, weights=None)

    return GameTheoryOptimizeResponse(
        domain=domain,
        metrics=metrics,
        chosen_strategy=rec["chosen_strategy"],
        patch=rec["patch"],
        mixes=[MixView(**m) for m in rec["mixes"]],
        proposal_id=None,
        message="Preview equilibrium only; no proposal created.",
    )
