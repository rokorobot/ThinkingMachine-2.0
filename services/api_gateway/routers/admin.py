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
# NEW imports for distillation + training
from services.distillation.dataset_builder import export_distillation_dataset
from services.distillation.trainer_launcher import create_training_run, launch_local_training


# ========== Distill & Train ==========

class DistillAndTrainRequest(BaseModel):
    base_model: str                  # e.g. "qwen-32b-instruct" or "llama-3-70b"
    target_name: str                 # e.g. "tm-v2"
    min_reward: float = 0.85
    require_safety_ok: bool = True
    domains: Optional[List[str]] = None
    dataset_path: str = "data/distill/tm_v2_train.jsonl"
    training_config: Dict[str, Any] = {}
    auto_launch: bool = True         # if False: only create training_runs row


class DistillAndTrainResponse(BaseModel):
    training_run_id: str
    dataset_path: str
    message: str


@router.post("/distill-and-train", response_model=DistillAndTrainResponse)
def distill_and_train(req: DistillAndTrainRequest):
    """
    High-level admin endpoint:

    1) Build a distillation dataset from high-quality traces.
    2) Create a training_runs row.
    3) Optionally launch the training script as a subprocess.
    """
    # 1) Export dataset
    dataset_path = export_distillation_dataset(
        output_path=req.dataset_path,
        min_reward=req.min_reward,
        require_safety_ok=req.require_safety_ok,
        domains=req.domains,
    )

    # 2) Create training run
    run_id = create_training_run(
        base_model=req.base_model,
        target_name=req.target_name,
        dataset_path=dataset_path,
        config=req.training_config,
    )

    # 3) Optionally launch training job
    if req.auto_launch:
        try:
            launch_local_training(run_id)
            msg = f"Training run {run_id} created and training launched."
        except Exception as e:
            msg = f"Training run {run_id} created, but failed to launch training: {e}"
            # You might want to set status='failed_to_launch' here
    else:
        msg = f"Training run {run_id} created. Launch manually when ready."

    return DistillAndTrainResponse(
        training_run_id=run_id,
        dataset_path=dataset_path,
        message=msg,
    )


# ========== Training run status ==========

class TrainingRunStatusResponse(BaseModel):
    id: str
    base_model: str
    target_name: str
    status: str
    dataset_path: Optional[str] = None
    logs_path: Optional[str] = None
    metrics: Dict[str, Any]
    model_version_id: Optional[str] = None


@router.get("/training-runs/{run_id}", response_model=TrainingRunStatusResponse)
def get_training_run_status(run_id: str):
    """
    Poll the status of a training run.
    Useful for dashboards and monitoring.
    """
    with db.get_conn() as conn:
        with conn.cursor(cursor_factory=db.psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, base_model, target_name, status,
                       dataset_path, logs_path, metrics, model_version_id
                FROM training_runs
                WHERE id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="training_run not found")

    return TrainingRunStatusResponse(
        id=str(row["id"]),
        base_model=row["base_model"],
        target_name=row["target_name"],
        status=row["status"],
        dataset_path=row.get("dataset_path"),
        logs_path=row.get("logs_path"),
        metrics=row.get("metrics") or {},
        model_version_id=str(row["model_version_id"]) if row.get("model_version_id") else None,
    )
