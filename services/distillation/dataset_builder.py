from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from libs import db


def select_good_traces(
    min_reward: float = 0.85,
    require_safety_ok: bool = True,
    max_rows: int = 50000,
    domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Pulls high-quality traces from DB.
    These become candidates for distillation.
    """
    where_clauses = ["metadata->>'reward_score' IS NOT NULL"]
    params: List[Any] = []

    where_clauses.append("(metadata->>'reward_score')::float >= %s")
    params.append(min_reward)

    if require_safety_ok:
        where_clauses.append(
            "COALESCE((metadata->>'hallucination_flag')::bool,FALSE) = FALSE"
        )

    if domains:
        where_clauses.append("domain = ANY(%s)")
        params.append(domains)

    where_sql = " AND ".join(where_clauses)

    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, input_text, output_text, metadata, domain, policy_version_id
                FROM traces
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (*params, max_rows),
            )
            rows = cur.fetchall()

    results = []
    for trace_id, inp, out, meta, domain, policy_id in rows:
        results.append(
            {
                "trace_id": trace_id,
                "prompt": inp,
                "ideal_response": out,
                "metadata": meta or {},
                "domain": domain,
                "policy_version_id": policy_id,
            }
        )
    return results


def insert_distillation_samples(samples: List[Dict[str, Any]], source_model: str) -> None:
    """
    Store selected examples into distillation_samples.
    """
    if not samples:
        return

    with db.get_conn() as conn:
        with conn.cursor() as cur:
            for s in samples:
                reward = s["metadata"].get("reward_score")
                safety_ok = not bool(s["metadata"].get("hallucination_flag", False))
                cur.execute(
                    """
                    INSERT INTO distillation_samples (
                        trace_id, policy_version_id, source_model,
                        prompt, ideal_response, reward_score, safety_ok,
                        domain, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        s["trace_id"],
                        s["policy_version_id"],
                        source_model,
                        s["prompt"],
                        s["ideal_response"],
                        reward,
                        safety_ok,
                        s["domain"],
                        json.dumps(s["metadata"]),
                    ),
                )
        conn.commit()


def export_distillation_dataset(
    output_path: str,
    min_reward: float = 0.85,
    require_safety_ok: bool = True,
    domains: Optional[List[str]] = None,
) -> str:
    """
    High-level:
      - select good traces
      - insert into distillation_samples (audit trail)
      - write JSONL dataset to disk (or mount → then sync to S3)
    Returns output_path.
    """
    samples = select_good_traces(
        min_reward=min_reward,
        require_safety_ok=require_safety_ok,
        max_rows=50000,
        domains=domains,
    )
    insert_distillation_samples(samples, source_model="tm-v1")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            rec = {
                "prompt": s["prompt"],
                "ideal_response": s["ideal_response"],
                "domain": s["domain"],
                "reward_score": s["metadata"].get("reward_score"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return str(out)
