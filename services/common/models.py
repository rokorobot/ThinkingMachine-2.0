from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Text, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class SelfPrompt(Base):
    __tablename__ = 'self_prompts'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    is_active = Column(Boolean, default=False)

class PolicyVersion(Base):
    __tablename__ = 'policy_versions'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    rules = Column(JSONB, nullable=False)
    parent_id = Column(UUID(as_uuid=True), ForeignKey('policy_versions.id'))
    created_at = Column(DateTime, server_default=func.now())
    is_active = Column(Boolean, default=False)
    
    parent = relationship('PolicyVersion', remote_side=[id])

class Trace(Base):
    __tablename__ = 'traces'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_input = Column(Text)
    result_output = Column(Text)
    policy_version_id = Column(UUID(as_uuid=True), ForeignKey('policy_versions.id'))
    self_prompt_id = Column(UUID(as_uuid=True), ForeignKey('self_prompts.id'))
    metadata_ = Column('metadata', JSONB)
    user_feedback = Column(JSONB)
    created_at = Column(DateTime, server_default=func.now())

class Proposal(Base):
    __tablename__ = 'proposals'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(String, nullable=False)
    payload = Column(JSONB, nullable=False)
    reasoning = Column(Text)
    status = Column(String, default='pending')
    created_at = Column(DateTime, server_default=func.now())

class Experiment(Base):
    __tablename__ = 'experiments'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    baseline_policy_id = Column(UUID(as_uuid=True), ForeignKey('policy_versions.id'))
    candidate_policy_id = Column(UUID(as_uuid=True), ForeignKey('policy_versions.id'))
    status = Column(String, default='running')
    result_summary = Column(JSONB)

class ExperimentRun(Base):
    __tablename__ = 'experiment_runs'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey('experiments.id'))
    trace_id = Column(UUID(as_uuid=True), ForeignKey('traces.id'))
    score = Column(Float)
    safety_ok = Column(Boolean)
