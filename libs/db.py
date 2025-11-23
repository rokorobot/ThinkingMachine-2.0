from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
import json

import psycopg2
import psycopg2.extras


DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@postgres:5432/thinking_machine")


@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


# ---------- Policy + self-prompt ----------

def get_active_policy_version() -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM policy_versions
                WHERE is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return dict(row) if row else None


def get_active_self_prompt() -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM self_prompts
                WHERE is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return dict(row) if row else None


def insert_policy_version(
    created_by: str,
    routing: Dict[str, Any],
    tool_use: Dict[str, Any],
    safety_overrides: Dict[str, Any],
    label: Optional[str] = None,
    is_active: bool = False,
) -> str:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO policy_versions (created_by, label, routing, tool_use, safety_overrides, is_active)
                VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                RETURNING id
                """,
                (created_by, label, json.dumps(routing),
                 json.dumps(tool_use),
                 json.dumps(safety_overrides),
                 is_active),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        return str(new_id)


def set_active_policy(policy_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            # set all inactive
            cur.execute("UPDATE policy_versions SET is_active = FALSE WHERE is_active = TRUE")
            # set new active
            cur.execute("UPDATE policy_versions SET is_active = TRUE WHERE id = %s", (policy_id,))
        conn.commit()


# ---------- Traces ----------

def insert_trace(
    session_id: str,
    task_id: str,
    task_type: str,
    domain: str,
    input_text: str,
    output_text: str,
    metadata: Dict[str, Any],
    policy_version_id: str,
    self_prompt_id: str,
    experiment_run_id: Optional[str] = None,
    user_feedback: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO traces (
                    session_id, task_id, task_type, domain,
                    input_text, output_text, metadata,
                    policy_version_id, self_prompt_id, experiment_run_id,
                    user_feedback, user_id
                )
                VALUES (%s, %s, %s, %s, %s, %s,
                        %s::jsonb, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    session_id,
                    task_id,
                    task_type,
                    domain,
                    input_text,
                    output_text,
                    json.dumps(metadata),
                    policy_version_id,
                    self_prompt_id,
                    experiment_run_id,
                    json.dumps(user_feedback or {}),
                    user_id,
                ),
            )
        conn.commit()


def get_problematic_traces(hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Pick traces with hallucination flag or thumbs_down feedback.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM traces
                WHERE created_at > NOW() - (%s || ' hours')::interval
                  AND (
                    (metadata->>'hallucination_flag')::bool = TRUE
                    OR (user_feedback->>'thumbs_down')::bool = TRUE
                  )
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (str(hours), limit),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]


# ---------- Proposals ----------

def insert_proposal(
    created_by: str,
    proposal_type: str,
    payload: Dict[str, Any],
    current_policy_version: Optional[str],
    current_self_prompt_id: Optional[str],
    reason: str,
) -> str:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO proposals (
                    created_by, proposal_type, payload,
                    current_policy_version, current_self_prompt_id, reason
                )
                VALUES (%s, %s::proposal_type, %s::jsonb, %s, %s, %s)
                RETURNING id
                """,
                (
                    created_by,
                    proposal_type,
                    json.dumps(payload),
                    current_policy_version,
                    current_self_prompt_id,
                    reason,
                ),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        return str(new_id)


def get_pending_proposals(limit: int = 20) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM proposals
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def update_proposal_status(
    proposal_id: str,
    status: str,
    final_policy_version: Optional[str] = None,
    final_self_prompt_id: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE proposals
                SET status = %s::proposal_status,
                    final_policy_version = COALESCE(%s, final_policy_version),
                    final_self_prompt_id = COALESCE(%s, final_self_prompt_id),
                    reason = COALESCE(%s, reason)
                WHERE id = %s
                """,
                (status, final_policy_version, final_self_prompt_id, reason, proposal_id),
            )
        conn.commit()


def mark_proposal_in_experiment(proposal_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE proposals SET status = 'in_experiment' WHERE id = %s",
                (proposal_id,),
            )
        conn.commit()


# ---------- Experiments ----------

def create_experiment(
    proposal_id: str,
    baseline_policy_id: str,
    candidate_policy_id: str,
    config: Dict[str, Any],
) -> str:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO experiments (
                    proposal_id, baseline_policy_id, candidate_policy_id, config
                )
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING id
                """,
                (proposal_id, baseline_policy_id, candidate_policy_id, json.dumps(config)),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        return str(new_id)


def create_experiment_run(
    experiment_id: str,
    run_index: int,
    candidate_policy_id: str,
) -> str:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO experiment_runs (
                    experiment_id, run_index, candidate_policy_id
                )
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (experiment_id, run_index, candidate_policy_id),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        return str(new_id)


def get_pending_experiment_runs(limit: int = 10) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM experiment_runs
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def update_experiment_run_result(
    run_id: str,
    score: float,
    safety_ok: bool,
    metrics: Dict[str, Any],
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE experiment_runs
                SET status = 'completed',
                    score = %s,
                    safety_ok = %s,
                    metrics = %s::jsonb
                WHERE id = %s
                """,
                (score, safety_ok, json.dumps(metrics), run_id),
            )
        conn.commit()


def get_experiments_ready_to_finalize() -> List[Dict[str, Any]]:
    """
    Experiments where all runs are completed (or some failed) → ready to aggregate.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT e.*
                FROM experiments e
                WHERE e.status IN ('pending', 'running')
                  AND NOT EXISTS (
                    SELECT 1
                    FROM experiment_runs r
                    WHERE r.experiment_id = e.id
                      AND r.status IN ('pending', 'running')
                  )
                """
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def get_runs_for_experiment(experiment_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM experiment_runs
                WHERE experiment_id = %s
                """,
                (experiment_id,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def finalize_experiment(experiment_id: str, status: str, result_summary: Dict[str, Any]) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE experiments
                SET status = %s::experiment_status,
                    result_summary = %s::jsonb
                WHERE id = %s
                """,
                (status, json.dumps(result_summary), experiment_id),
            )
        conn.commit()
