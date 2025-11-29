-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

CREATE TABLE IF NOT EXISTS users (
  id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  external_id  TEXT UNIQUE,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  profile      JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS user_memories (
  id           BIGSERIAL PRIMARY KEY,
  user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  importance   INT NOT NULL DEFAULT 1,
  kind         TEXT NOT NULL,
  text         TEXT NOT NULL,
  embedding    VECTOR(1536)
);

CREATE INDEX IF NOT EXISTS idx_user_memories_user_id
  ON user_memories(user_id);

CREATE INDEX IF NOT EXISTS idx_user_memories_embedding
  ON user_memories
  USING ivfflat (embedding vector_cosine_ops);

-- Update traces to include user_id
ALTER TABLE traces
  ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id);

-- High-level document metadata for knowledge base
CREATE TABLE IF NOT EXISTS knowledge_documents (
  id           BIGSERIAL PRIMARY KEY,
  source       TEXT NOT NULL,           -- e.g. 'local_corpus', 'manual_upload', 'web_capture'
  uri          TEXT NOT NULL,           -- absolute file path, URL, or logical ID
  title        TEXT,
  doc_type     TEXT,                    -- 'pdf','txt','md',...
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  metadata     JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Chunk-level embeddings (pgvector)
CREATE TABLE IF NOT EXISTS knowledge_chunks (
  id             BIGSERIAL PRIMARY KEY,
  document_id    BIGINT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
  chunk_index    INT NOT NULL,
  content        TEXT NOT NULL,
  embedding      VECTOR(1536),
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_doc_id
  ON knowledge_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding
  ON knowledge_chunks
  USING ivfflat (embedding vector_cosine_ops);

-- 1. Policy Versions
CREATE TABLE policy_versions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by          TEXT NOT NULL,              -- 'system', 'meta_agent', 'human:robert'
    label               TEXT,                      -- e.g. 'v20251107_1200_medical_stricter'
    routing             JSONB NOT NULL,            -- mirrors policies/routing.yaml
    tool_use            JSONB NOT NULL,            -- mirrors policies/tool_use.yaml
    safety_overrides    JSONB NOT NULL,            -- mirrors policies/safety_overrides.yaml
    is_active           BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE UNIQUE INDEX idx_policy_versions_active_true
ON policy_versions (is_active)
WHERE is_active = TRUE;

CREATE INDEX idx_policy_versions_created_at
ON policy_versions (created_at);

-- 2. Self-Prompt Versions
CREATE TABLE self_prompts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by      TEXT NOT NULL,           -- 'meta_agent' or human
    base            JSONB NOT NULL,          -- immutable core (snapshot)
    editable        JSONB NOT NULL,          -- current overlay
    merged          JSONB NOT NULL,          -- base + editable (cached for speed)
    is_active       BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE UNIQUE INDEX idx_self_prompts_active_true
ON self_prompts (is_active)
WHERE is_active = TRUE;

CREATE INDEX idx_self_prompts_created_at
ON self_prompts (created_at);

-- 3. Proposals
CREATE TYPE proposal_type AS ENUM (
    'policy_update',
    'self_prompt_update',
    'skill_update',
    'adapter_registration'
);

CREATE TYPE proposal_status AS ENUM (
    'pending',
    'in_experiment',
    'accepted',
    'rejected'
);

CREATE TABLE proposals (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT NOT NULL,           -- 'meta_agent', 'human:robert'
    proposal_type           proposal_type NOT NULL,
    status                  proposal_status NOT NULL DEFAULT 'pending',
    payload                 JSONB NOT NULL,          -- exact requested change
    current_policy_version  UUID REFERENCES policy_versions(id),
    current_self_prompt_id  UUID REFERENCES self_prompts(id),
    reason                  TEXT,                    -- human / meta-agent explanation
    final_policy_version    UUID REFERENCES policy_versions(id),
    final_self_prompt_id    UUID REFERENCES self_prompts(id)
);

CREATE INDEX idx_proposals_status
ON proposals (status);

CREATE INDEX idx_proposals_created_at
ON proposals (created_at);

CREATE INDEX idx_proposals_type
ON proposals (proposal_type);

-- 4. Experiments
CREATE TYPE experiment_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed'
);

CREATE TABLE experiments (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    proposal_id         UUID NOT NULL REFERENCES proposals(id) ON DELETE CASCADE,
    baseline_policy_id  UUID NOT NULL REFERENCES policy_versions(id),
    candidate_policy_id UUID REFERENCES policy_versions(id),  -- optional if policy change
    baseline_prompt_id  UUID REFERENCES self_prompts(id),
    candidate_prompt_id UUID REFERENCES self_prompts(id),
    config              JSONB NOT NULL,           -- eval suite config (tasks, domains, thresholds)
    status              experiment_status NOT NULL DEFAULT 'pending',
    result_summary      JSONB                     -- aggregated metrics
);

CREATE INDEX idx_experiments_proposal_id
ON experiments (proposal_id);

CREATE INDEX idx_experiments_status
ON experiments (status);

-- 5. Experiment Runs
CREATE TYPE run_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed'
);

CREATE TABLE experiment_runs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id       UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_index           INT NOT NULL,          -- 0,1,2,... for each experiment
    status              run_status NOT NULL DEFAULT 'pending',
    candidate_policy_id UUID REFERENCES policy_versions(id),
    candidate_prompt_id UUID REFERENCES self_prompts(id),
    score               DOUBLE PRECISION,      -- aggregate score for this run
    safety_ok           BOOLEAN,              -- high-level safety result
    metrics             JSONB                 -- detailed metrics: per-task scores, latencies, etc.
);

CREATE INDEX idx_experiment_runs_experiment_id
ON experiment_runs (experiment_id);

-- 6. Traces
CREATE TABLE traces (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id          TEXT,                  -- user session / conversation id
    task_id             TEXT,                  -- internal task/job id
    task_type           TEXT,                  -- 'chat', 'planning', 'coding', 'medical_advice', etc.
    domain              TEXT,                  -- 'medical', 'marketing', 'code', etc.
    input_text          TEXT NOT NULL,
    output_text         TEXT NOT NULL,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,  -- scores, flags, tool traces
    policy_version_id   UUID REFERENCES policy_versions(id),
    self_prompt_id      UUID REFERENCES self_prompts(id),
    experiment_run_id   UUID REFERENCES experiment_runs(id), -- null for normal live traffic
    user_feedback       JSONB                               -- thumbs up/down, corrections, etc.
);

CREATE INDEX idx_traces_created_at
ON traces (created_at);

CREATE INDEX idx_traces_domain_created_at
ON traces (domain, created_at);

CREATE INDEX idx_traces_policy_version_id
ON traces (policy_version_id);

CREATE INDEX idx_traces_metadata_gin
ON traces
USING GIN (metadata);
