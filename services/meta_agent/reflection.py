from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import json

from libs.policy_store import PolicyStore, ProposalType, Proposal

@dataclass
class Trace:
    id: str
    task_type: str
    input_text: str
    output_text: str
    metadata: Dict[str, Any]

class TraceRepository:
    """Skeleton for pulling traces from DB."""
    def __init__(self, db_url: str):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def get_problematic_traces(self, limit: int = 50) -> List[Trace]:
        from sqlalchemy import text
        with self.Session() as session:
            # Example query: traces with negative feedback or specific flags
            result = session.execute(
                text("""
                    SELECT id, task_type, input_text, output_text, metadata
                    FROM traces
                    WHERE (metadata->>'hallucination_flag')::boolean = TRUE
                       OR (user_feedback->>'thumbs_down')::boolean = TRUE
                    ORDER BY created_at DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            )
            traces = []
            for row in result:
                traces.append(Trace(
                    id=str(row.id),
                    task_type=row.task_type or "unknown",
                    input_text=row.input_text,
                    output_text=row.output_text,
                    metadata=row.metadata
                ))
            return traces

class MetaAgentReflector:
    """
    Uses traces to propose changes to policies / self-prompt etc.
    """

    def __init__(self, policy_store: PolicyStore, trace_repo: TraceRepository):
        self.policy_store = policy_store
        self.trace_repo = trace_repo

    def run_reflection_cycle(self) -> List[Proposal]:
        traces = self.trace_repo.get_problematic_traces(limit=50)
        if not traces:
            return []

        proposals: List[Proposal] = []

        # --- Simple Heuristics (Placeholder for LLM Logic) ---
        
        # 1. Medical Errors
        medical_errors = [t for t in traces if t.metadata.get("domain") == "medical"]
        if len(medical_errors) >= 3: # Lower threshold for demo
            payload = {
                "domain": "medical",
                "change": {
                    "require_multi_source_check": True,
                    "min_sources": 2
                }
            }
            proposal = self.policy_store.create_proposal(
                proposal_type=ProposalType.POLICY_UPDATE,
                payload=payload,
                author="meta_agent",
                reason=f"Detected {len(medical_errors)} problematic medical traces"
            )
            proposals.append(proposal)

        return proposals
