-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

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

-- 2. Experience (Traces & Feedback)
CREATE TABLE traces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_input TEXT,
    result_output TEXT,
    policy_version_id UUID REFERENCES policy_versions(id),
    self_prompt_id UUID REFERENCES self_prompts(id),
    metadata JSONB, -- { "hallucination_flag": true, "latency_ms": 400 }
    user_feedback JSONB, -- { "thumbs_down": true, "comment": "Too verbose" }
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
