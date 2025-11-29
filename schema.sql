-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users Table (for long-term memory & preferences)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id TEXT UNIQUE NOT NULL, -- The ID from the upstream system (e.g. Slack ID)
    profile JSONB DEFAULT '{}'::jsonb, -- { "preferences": { "tone": "direct" } }
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 1. The Genome (Policies & Prompts)
CREATE TABLE self_prompts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    content TEXT NOT NULL, -- The actual system prompt text
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE
);

CREATE TABLE policy_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    rules JSONB NOT NULL, -- Structured policy rules/heuristics
    parent_id UUID REFERENCES policy_versions(id),
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE
);

-- User-Specific Policy Overlays
CREATE TABLE IF NOT EXISTS user_policies (
  id                BIGSERIAL PRIMARY KEY,
  user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  base_policy_id    UUID NOT NULL REFERENCES policy_versions(id),
  routing_override  JSONB NOT NULL DEFAULT '{}'::jsonb,
  tool_use_override JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  is_active         BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_user_policies_user_id
  ON user_policies(user_id);

CREATE INDEX IF NOT EXISTS idx_user_policies_user_active
  ON user_policies(user_id)
  WHERE is_active = TRUE;

-- 2. Experience (Traces & Feedback)
CREATE TABLE traces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_input TEXT,
    result_output TEXT,
    policy_version_id UUID REFERENCES policy_versions(id),
    self_prompt_id UUID REFERENCES self_prompts(id),
    metadata JSONB, -- { "hallucination_flag": true, "latency_ms": 400 }
    user_feedback JSONB, -- { "thumbs_down": true, "comment": "Too verbose" }
    user_id UUID REFERENCES users(id), -- Link trace to user
    created_at TIMESTAMP DEFAULT NOW()
);

-- 3. Evolution (Proposals & Experiments)
CREATE TABLE proposals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type TEXT NOT NULL, -- 'prompt_patch', 'new_policy'
    payload JSONB NOT NULL, -- The proposed change
    reasoning TEXT, -- Why this change? (Game Theory analysis)
    status TEXT DEFAULT 'pending', -- pending, accepted, rejected
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    baseline_policy_id UUID REFERENCES policy_versions(id),
    candidate_policy_id UUID REFERENCES policy_versions(id),
    status TEXT DEFAULT 'running',
    result_summary JSONB -- { "win_rate": 0.6, "safety_score": 0.99 }
);

CREATE TABLE experiment_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(id),
    trace_id UUID REFERENCES traces(id), -- Link to the actual run
    score FLOAT,
    safety_ok BOOLEAN
);

-- 4. Distillation & Model Training
CREATE TABLE IF NOT EXISTS model_versions (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name              TEXT UNIQUE NOT NULL,   -- 'tm-v1', 'tm-v2'
  base_model        TEXT NOT NULL,          -- underlying foundation
  location          TEXT NOT NULL,          -- e.g. 'hf://...', 's3://...', 'local:/models/tm-v2'
  status            TEXT NOT NULL,          -- 'candidate','active','retired'
  trained_from_run  UUID,                   -- FK added later to avoid circular dependency issues if needed, or just add now
  performance_score FLOAT,
  metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training_runs (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  base_model        TEXT NOT NULL,    -- e.g. 'qwen-32b-instruct', 'llama-3-70b'
  target_name       TEXT NOT NULL,    -- e.g. 'tm-v2'
  status            TEXT NOT NULL,    -- 'pending','running','failed','completed'
  config            JSONB NOT NULL,
  dataset_path      TEXT,             -- e.g. 's3://.../tm_v2_distill.parquet'
  logs_path         TEXT,
  metrics           JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  model_version_id  UUID REFERENCES model_versions(id)
);

-- Add the FK from model_versions to training_runs now that training_runs exists
ALTER TABLE model_versions 
ADD CONSTRAINT fk_model_versions_training_run 
FOREIGN KEY (trained_from_run) REFERENCES training_runs(id);

CREATE TABLE IF NOT EXISTS distillation_samples (
  id                BIGSERIAL PRIMARY KEY,
  trace_id          UUID REFERENCES traces(id) ON DELETE SET NULL,
  policy_version_id UUID REFERENCES policy_versions(id),
  source_model      TEXT,         -- e.g. 'gpt-5.1', 'tm-v1'
  prompt            TEXT NOT NULL,
  ideal_response    TEXT NOT NULL,
  reward_score      FLOAT,
  safety_ok         BOOLEAN,
  domain            TEXT,
  metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_distill_domain ON distillation_samples(domain);
