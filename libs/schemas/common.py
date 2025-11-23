from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID

class Policy(BaseModel):
    id: UUID
    name: str
    rules: Dict[str, Any]
    is_active: bool

class Trace(BaseModel):
    id: UUID
    task_input: str
    result_output: str
    policy_id: Optional[UUID]
    metadata: Dict[str, Any]
    user_feedback: Optional[Dict[str, Any]]
    created_at: datetime

class Proposal(BaseModel):
    id: UUID
    type: str # 'policy_patch', 'skill_patch'
    payload: Dict[str, Any]
    reasoning: str
    status: str # 'pending', 'approved', 'rejected'
    created_at: datetime
