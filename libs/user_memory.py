from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from . import db  # reuse get_conn and DATABASE_URL from libs/db.py

# You can plug in OpenAI embeddings, local embeddings, etc.
# For now we stub with a fake dense vector for skeleton purposes.
import numpy as np


EMBED_DIM = int(os.getenv("USER_MEMORY_EMBED_DIM", "1536"))


def embed_text(text: str) -> List[float]:
    """
    Stub embedding generator; replace with real embedding call.
    Examples:
      - OpenAI: client.embeddings.create(model="text-embedding-3-large", input=text)
      - Local: sentence-transformers in a separate service
    """
    # DO NOT use this in production; this is just to keep schema coherent.
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vec = rng.normal(size=EMBED_DIM)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


# ---------- users ----------

def get_or_create_user(external_id: str, default_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Look up user by external_id; if not exists, create.
    Returns full users row as dict.
    """
    default_profile = default_profile or {}

    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, external_id, created_at, profile FROM users WHERE external_id = %s",
                (external_id,),
            )
            row = cur.fetchone()
            if row:
                user_id, external_id, created_at, profile = row
                return {
                    "id": str(user_id),
                    "external_id": external_id,
                    "created_at": created_at,
                    "profile": profile,
                }

            # Create new user
            cur.execute(
                """
                INSERT INTO users (external_id, profile)
                VALUES (%s, %s::jsonb)
                RETURNING id, external_id, created_at, profile
                """,
                (external_id, db.psycopg2.extras.Json(default_profile)),  # type: ignore
            )
            new_row = cur.fetchone()
        conn.commit()

    user_id, external_id, created_at, profile = new_row
    return {
        "id": str(user_id),
        "external_id": external_id,
        "created_at": created_at,
        "profile": profile,
    }


def update_user_profile(user_id: str, profile_patch: Dict[str, Any]) -> None:
    """
    Shallow-merge patch into profile JSON.
    """
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT profile FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                return
            current_profile = row[0] or {}
            current_profile.update(profile_patch)
            cur.execute(
                "UPDATE users SET profile = %s::jsonb WHERE id = %s",
                (db.psycopg2.extras.Json(current_profile), user_id),  # type: ignore
            )
        conn.commit()


# ---------- memories ----------

def add_user_memory(
    user_id: str,
    text: str,
    kind: str = "fact",
    importance: int = 1,
    embed: bool = True,
) -> int:
    """
    Insert a new memory for the user, optionally with embedding.
    Returns memory id.
    """
    emb = embed_text(text) if embed else None

    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_memories (user_id, text, kind, importance, embedding)
                VALUES (%s, %s, %s, %s,
                        %s::vector)
                RETURNING id
                """,
                (
                    user_id,
                    text,
                    kind,
                    importance,
                    emb,
                ),
            )
            mem_id = cur.fetchone()[0]
        conn.commit()
    return mem_id


def touch_user_memory(mem_id: int) -> None:
    """
    Update last_seen_at for a memory.
    """
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE user_memories SET last_seen_at = NOW() WHERE id = %s",
                (mem_id,),
            )
        conn.commit()


def search_user_memories(
    user_id: str,
    query: str,
    top_k: int = 5,
    min_importance: int = 1,
) -> List[Dict[str, Any]]:
    """
    Vector search over user_memories via pgvector.
    Returns list of memories with distance.
    """
    q_emb = embed_text(query)

    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, text, kind, importance,
                       1 - (embedding <#> %s::vector) AS similarity
                FROM user_memories
                WHERE user_id = %s
                  AND importance >= %s
                ORDER BY embedding <#> %s::vector ASC
                LIMIT %s
                """,
                (q_emb, user_id, min_importance, q_emb, top_k),
            )
            rows = cur.fetchall()

    memories: List[Dict[str, Any]] = []
    for mem_id, text, kind, importance, similarity in rows:
        memories.append(
            {
                "id": mem_id,
                "text": text,
                "kind": kind,
                "importance": importance,
                "similarity": float(similarity),
            }
        )
        # Update last_seen asynchronously if you want; for now we keep it simple
    return memories


def get_top_recent_memories(
    user_id: str,
    limit: int = 5,
    min_importance: int = 2,
) -> List[Dict[str, Any]]:
    """
    Non-semantic, recency-based fetch. Useful as a fallback or in addition
    to vector search.
    """
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, text, kind, importance
                FROM user_memories
                WHERE user_id = %s
                  AND importance >= %s
                ORDER BY last_seen_at DESC
                LIMIT %s
                """,
                (user_id, min_importance, limit),
            )
            rows = cur.fetchall()

    return [
        {
            "id": mem_id,
            "text": text,
            "kind": kind,
            "importance": importance,
        }
        for mem_id, text, kind, importance in rows
    ]
