from __future__ import annotations

import os
import uuid
import shutil
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List

import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

GENOME_ROOT = os.environ.get("GENOME_ROOT", "/app/genome_store")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@postgres:5432/thinking_machine")

class ProposalType(str, Enum):
    POLICY_UPDATE = "policy_update"
    SELF_PROMPT_UPDATE = "self_prompt_update"
    SKILL_UPDATE = "skill_update"
    ADAPTER_REGISTRATION = "adapter_registration"

@dataclass
class Policy:
    """In-memory representation of effective policies."""
    routing: Dict[str, Any] = field(default_factory=dict)
    tool_use: Dict[str, Any] = field(default_factory=dict)
    safety_overrides: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Proposal:
    id: str
    proposal_type: ProposalType
    created_at: datetime
    author: str
    payload: Dict[str, Any]
    status: str = "pending"
    reason: Optional[str] = None

class PolicyStore:
    """
    Reads/writes policy files under genome_store/ AND syncs with Postgres.
    """

    def __init__(self, root: str = GENOME_ROOT, db_url: str = DATABASE_URL):
        self.root = root
        self.policies_dir = os.path.join(root, "policies")
        self.self_prompt_dir = os.path.join(root, "self_prompt")
        os.makedirs(self.policies_dir, exist_ok=True)
        
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    # ---------- core load/save ----------

    def load_current_policy(self) -> Policy:
        # Try loading from DB first for active policy? 
        # For now, stick to FS as source of truth for the running container, 
        # but DB is the source of truth for metadata.
        routing_path = os.path.join(self.policies_dir, "routing.yaml")
        tool_use_path = os.path.join(self.policies_dir, "tool_use.yaml")
        safety_path = os.path.join(self.policies_dir, "safety_overrides.yaml")

        def _load_yaml(p: str) -> Dict[str, Any]:
            if not os.path.exists(p):
                return {}
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}

        return Policy(
            routing=_load_yaml(routing_path),
            tool_use=_load_yaml(tool_use_path),
            safety_overrides=_load_yaml(safety_path),
        )

    def save_policy(self, policy: Policy, label: Optional[str] = None, author: str = "system") -> str:
        """
        Save current policy as main + create a versioned snapshot in DB and FS.
        Returns version_id (UUID).
        """
        # 1. Save to Filesystem
        routing_path = os.path.join(self.policies_dir, "routing.yaml")
        tool_use_path = os.path.join(self.policies_dir, "tool_use.yaml")
        safety_path = os.path.join(self.policies_dir, "safety_overrides.yaml")

        with open(routing_path, "w") as f:
            yaml.safe_dump(policy.routing, f)
        with open(tool_use_path, "w") as f:
            yaml.safe_dump(policy.tool_use, f)
        with open(safety_path, "w") as f:
            yaml.safe_dump(policy.safety_overrides, f)

        # 2. Save to DB
        with self.Session() as session:
            # Deactivate old policies
            session.execute(text("UPDATE policy_versions SET is_active = FALSE WHERE is_active = TRUE"))
            
            # Insert new policy
            result = session.execute(
                text("""
                    INSERT INTO policy_versions (created_by, label, routing, tool_use, safety_overrides, is_active)
                    VALUES (:created_by, :label, :routing, :tool_use, :safety_overrides, TRUE)
                    RETURNING id
                """),
                {
                    "created_by": author,
                    "label": label,
                    "routing": json.dumps(policy.routing),
                    "tool_use": json.dumps(policy.tool_use),
                    "safety_overrides": json.dumps(policy.safety_overrides)
                }
            )
            version_id = str(result.scalar())
            session.commit()

        return version_id

    # ---------- self-prompt ----------

    def load_self_prompt(self) -> Dict[str, Any]:
        base_path = os.path.join(self.self_prompt_dir, "base.yaml")
        editable_path = os.path.join(self.self_prompt_dir, "editable.yaml")

        def _load_yaml(p: str) -> Dict[str, Any]:
            if not os.path.exists(p):
                return {}
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}

        base = _load_yaml(base_path)
        editable = _load_yaml(editable_path)
        return {**base, **editable}

    def save_editable_self_prompt(self, editable: Dict[str, Any]) -> None:
        editable_path = os.path.join(self.self_prompt_dir, "editable.yaml")
        with open(editable_path, "w") as f:
            yaml.safe_dump(editable, f)
        
        # TODO: Sync with DB (self_prompts table)

    # ---------- proposals ----------

    def create_proposal(
        self,
        proposal_type: ProposalType,
        payload: Dict[str, Any],
        author: str = "meta_agent",
        reason: str = None
    ) -> Proposal:
        
        with self.Session() as session:
            # Get current active policy ID
            current_policy_id = session.execute(text("SELECT id FROM policy_versions WHERE is_active = TRUE")).scalar()
            
            result = session.execute(
                text("""
                    INSERT INTO proposals (created_by, proposal_type, payload, current_policy_version, reason, status)
                    VALUES (:created_by, :proposal_type, :payload, :current_policy_version, :reason, 'pending')
                    RETURNING id, created_at
                """),
                {
                    "created_by": author,
                    "proposal_type": proposal_type.value,
                    "payload": json.dumps(payload),
                    "current_policy_version": current_policy_id,
                    "reason": reason
                }
            )
            row = result.fetchone()
            session.commit()
            
            return Proposal(
                id=str(row.id),
                proposal_type=proposal_type,
                created_at=row.created_at,
                author=author,
                payload=payload,
                status="pending",
                reason=reason
            )

    def load_pending_proposals(self) -> List[Proposal]:
        with self.Session() as session:
            result = session.execute(
                text("""
                    SELECT id, proposal_type, created_at, created_by, payload, status, reason
                    FROM proposals
                    WHERE status = 'pending'
                """)
            )
            proposals = []
            for row in result:
                proposals.append(Proposal(
                    id=str(row.id),
                    proposal_type=ProposalType(row.proposal_type),
                    created_at=row.created_at,
                    author=row.created_by,
                    payload=row.payload,
                    status=row.status,
                    reason=row.reason
                ))
            return proposals

    def update_proposal_status(self, proposal_id: str, status: str, reason: str | None = None) -> None:
        with self.Session() as session:
            session.execute(
                text("UPDATE proposals SET status = :status, reason = :reason WHERE id = :id"),
                {"status": status, "reason": reason, "id": proposal_id}
            )
            session.commit()
