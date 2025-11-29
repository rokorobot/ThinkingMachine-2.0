# thinking_core/retrieval/retriever.py
"""
Full RAG retrieval system with:
- pgvector-backed knowledge base search
- Optional web search integration
- Policy-driven retrieval decisions
"""
from __future__ import annotations

from typing import Any, Dict, List
import os

import psycopg2
import psycopg2.extras
import requests

from libs.db import get_conn
from libs.user_memory import embed_text as embed_text_user  # reuse embedding logic


SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL")  # e.g. http://search-service:8081


def embed_text(text: str) -> List[float]:
    """
    Use the same embedding strategy as user_memories.
    """
    return embed_text_user(text)


# ---------- retrieval policy ----------

def should_retrieve(task: Dict[str, Any], world_model_ctx: Dict[str, Any], policy: Dict[str, Any]) -> bool:
    """
    Decide whether to use RAG retrieval for this task.
    
    Basic heuristic:
      - retrieve for coding, medical-like, or QA tasks
      - can be extended with policy-based toggles:
          policy['routing']['enable_rag'] etc.
    """
    t = world_model_ctx.get("task_nature", "")
    if t in ("coding", "medical_like"):
        return True
    if task.get("task_type") == "qa":
        return True

    # You can also inspect domain:
    domain = task.get("domain", "")
    if domain in ("research", "science", "finance", "medical"):
        return True

    return False


def should_use_web(task: Dict[str, Any], world_model_ctx: Dict[str, Any], policy: Dict[str, Any]) -> bool:
    """
    Decide whether to call web search.
    For safety, you might limit this to certain domains or flags in policy.
    """
    if not SEARCH_SERVICE_URL:
        return False

    # Example: only allow in 'general', 'news', 'research' domains
    domain = task.get("domain", "")
    if domain in ("news", "research", "general"):
        return True

    # Alternatively, only allow when explicitly requested:
    input_text = task.get("input_text", "").lower()
    if "latest" in input_text or "current" in input_text or "recent" in input_text:
        return True

    return False


# ---------- vector KB search ----------

def vector_search_knowledge(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search over knowledge_chunks.
    Returns top-k most relevant chunks.
    """
    emb = embed_text(query)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                  kc.id AS chunk_id,
                  kc.content,
                  kd.title,
                  kd.uri,
                  kd.source,
                  1 - (kc.embedding <#> %s::vector) AS similarity
                FROM knowledge_chunks kc
                JOIN knowledge_documents kd
                  ON kc.document_id = kd.id
                ORDER BY kc.embedding <#> %s::vector ASC
                LIMIT %s
                """,
                (emb, emb, top_k),
            )
            rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "chunk_id": r["chunk_id"],
                "content": r["content"],
                "title": r["title"],
                "uri": r["uri"],
                "source": r["source"],
                "similarity": float(r["similarity"]),
            }
        )
    return results


# ---------- web search client ----------

def web_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Calls an external search microservice.
    That service is responsible for calling Bing/Brave/Exa/etc.
    
    Returns list of search results with title, url, snippet.
    """
    if not SEARCH_SERVICE_URL:
        return []

    try:
        resp = requests.get(
            f"{SEARCH_SERVICE_URL}/search",
            params={"q": query, "k": top_k},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[retrieval] web_search error: {e}")
        return []

    results_raw = data.get("results", [])
    results: List[Dict[str, Any]] = []
    for r in results_raw:
        results.append(
            {
                "source": r.get("url") or "web",
                "title": r.get("title") or "",
                "content": r.get("snippet") or "",
                "similarity": None,
            }
        )
    return results


# ---------- main retrieval pass ----------

def retrieval_pass(
    task: Dict[str, Any],
    world_model_ctx: Dict[str, Any],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Main RAG hook:
      - optional vector KB search (pgvector-backed)
      - optional web search (via external microservice)
    Combines snippets into a single context object.
    
    Returns:
        {
            "used": bool,
            "snippets": [
                {
                    "source": str,
                    "title": str,
                    "content": str,
                    "similarity": float | None,
                    "kind": "kb" | "web"
                },
                ...
            ]
        }
    """
    if not should_retrieve(task, world_model_ctx, policy):
        return {"used": False, "snippets": []}

    query = task["input_text"]

    # 1) Vector KB search
    kb_chunks = vector_search_knowledge(query, top_k=5)

    snippets: List[Dict[str, Any]] = []
    for c in kb_chunks:
        snippets.append(
            {
                "source": f"{c['source']}::{c['uri']}",
                "title": c["title"],
                "content": c["content"],
                "similarity": c["similarity"],
                "kind": "kb",
            }
        )

    # 2) Optional web search
    if should_use_web(task, world_model_ctx, policy):
        web_results = web_search(query, top_k=3)
        for w in web_results:
            snippets.append(
                {
                    "source": w["source"],
                    "title": w["title"],
                    "content": w["content"],
                    "similarity": w.get("similarity"),
                    "kind": "web",
                }
            )

    if not snippets:
        return {"used": False, "snippets": []}

    return {
        "used": True,
        "snippets": snippets,
    }
