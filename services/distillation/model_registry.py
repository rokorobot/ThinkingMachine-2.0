from __future__ import annotations

from typing import Any, Dict, Optional
import json

from libs import db


def register_model_version(
    name: str,
    base_model: str,
    location: str,
    status: str = "candidate",
    performance_score: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Insert a new model_versions row and return its id.
    """
    metadata = metadata or {}
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_versions (name, base_model, location, status,
                                            performance_score, metadata)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                RETURNING id
                """,
                (name, base_model, location, status, performance_score, json.dumps(metadata)),
            )
            model_id = cur.fetchone()[0]
        conn.commit()
    return str(model_id)


def set_model_status(model_id: str, status: str) -> None:
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE model_versions SET status = %s WHERE id = %s",
                (status, model_id),
            )
        conn.commit()


def get_active_model() -> Optional[Dict[str, Any]]:
    """
    Your primary serving model.
    """
    with db.get_conn() as conn:
        with conn.cursor(cursor_factory=db.psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM model_versions
                WHERE status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return dict(row) if row else None
