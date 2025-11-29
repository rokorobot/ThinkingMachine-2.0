# Thinking Machine 2.0 - DGX Spark Edition

## System Overview
This is a **Level-3 self-modifying AI system** optimized for a single DGX Spark node. It features:
- **Genome Store** (Git-based policy & prompt versioning)
- **Self-Training** (LoRA fine-tuning)
- **RAG** (Retrieval-Augmented Generation with pgvector)
- **Game Theory** optimization
- **Long-Term Memory** (per-user context)
- **Safety Guard** (immutable core rules)

### Architecture
The system implements a complete cognitive pipeline:
1. **User Context** - Retrieves user-specific memories
2. **World Model** - Classifies task nature
3. **RAG Retrieval** - Searches knowledge base + optional web
4. **Multi-Agent Reasoning** - LLM with full context
5. **Reflection & Critique** - Self-evaluation
6. **Output Synthesis** - Final response generation

## Development Workflow

### 1. Local Development (CPU-only)
Use VS Code with Dev Containers for a consistent environment.
1. Open the repository in VS Code.
2. Click "Reopen in Container" when prompted (or use command palette).
3. This uses `.devcontainer/devcontainer.json` and `infra/docker-compose.dev.yml`.
4. Run services:
    ```bash
    cd infra
    docker-compose -f docker-compose.dev.yml up
    ```
5. Access Monitor at `http://localhost:8501`.

### 2. Production Deployment (DGX Spark)
On the DGX node with NVIDIA Container Toolkit:
1. Clone the repo.
2. Set environment variables in `infra/env/*.env`.
3. Run with GPU support:
    ```bash
    cd infra
    docker-compose up -d
    ```
    *Note: `docker-compose.yml` is configured with `deploy.resources.reservations.devices` for GPU access.*

## Quick Start

### 1. Set Environment Variables
Create a `.env` file in the root directory (or rely on `infra/env/*.env`):
```bash
OPENAI_API_KEY=sk-...
LLM_BACKEND=openai # or tgi, vllm
LLM_MODEL=gpt-4o
DATABASE_URL=postgresql://tm_user:tm_pass@postgres:5432/thinking_machine
```

### 2. Initialize Database
```bash
docker-compose exec postgres psql -U tm_user -d thinking_machine -f /docker-entrypoint-initdb.d/init_db.sql
```

### 3. Interact with the Agent
```bash
curl -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "Explain quantum computing",
    "domain": "science",
    "user_external_id": "user_001"
  }'
```

## The Self-Reprogramming Loop

1. **Trace**: User interactions are logged to the database.
2. **Reflection**: The **Meta Agent** analyzes failing traces using Game Theory.
3. **Proposal**: It proposes a patch to `genome_store/` (e.g., a new policy rule).
4. **Validation**: The **Safety Guard** checks the proposal against `immutable_core.yaml`.
5. **Experiment**: The **Orchestrator** spawns a candidate agent with the patch.
6. **Evaluation**: The **Eval Judge** scores the candidate (supports "Tournament" style multi-round games).
7. **Evolution**: If successful (High Score + Stable Strategy), the patch is applied to the main Genome.

## RAG (Retrieval-Augmented Generation)

The system includes a full RAG pipeline for knowledge-grounded responses.

### Knowledge Base Ingestion

**Folder-based ingestion** (bulk):
```bash
# Place PDFs/TXT/MD files in data/docs/
python scripts/ingest_knowledge.py --path data/docs/ --source "my_knowledge"
```

**API-based upload** (single file):
```bash
curl -X POST "http://localhost:8080/admin/knowledge/upload?source=manual" \
  -F "file=@document.pdf"
```

### How RAG Works

When enabled (automatically for research/medical/coding domains):
1. **Vector Search**: Queries the knowledge base using pgvector similarity
2. **Web Search**: Optionally fetches current information (if `SEARCH_SERVICE_URL` is set)
3. **Context Injection**: Retrieved snippets are injected into the LLM system prompt
4. **Grounded Responses**: The agent answers based on your knowledge base

### View Knowledge Stats
```bash
curl "http://localhost:8080/admin/knowledge/stats"
```

## Game Theory Integration

The system uses Game Theory to optimize its adaptability.

### Admin API
You can manually trigger the Game Theory optimization loop via the Admin API:

1. **Preview Equilibrium**: See what strategy the system recommends based on recent metrics.
    ```bash
    curl "http://localhost:8080/admin/game-theory/preview?domain=medical&hours=24"
    ```

2. **Optimize & Propose**: Commit the recommendation as a proposal for the Orchestrator.
    ```bash
    curl -X POST http://localhost:8080/admin/game-theory/optimize \
      -H "Content-Type: application/json" \
      -d '{"domain": "medical", "hours": 24, "commit": true}'
    ```

## Long-Term Memory

The system supports persistent user memory using `pgvector`.

### User-Aware Interaction
To use memory, provide a `user_external_id` in your request. The agent will recall previous context and store new notes.

```bash
curl -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "What projects am I working on?",
    "user_external_id": "robert_123",
    "memory_note": "I am working on the Thinking Machine project."
  }'
```

## Mission Control Dashboard

The **Monitor** service (`http://localhost:8501`) provides a full "Mission Control" interface with 6 tabs:

1. **🚀 Ops & KPIs**: System health, success rates, latency, and active user counts.
2. **🧠 Cognitive Engine**:
   - **Memory**: Stats on total users and memories
   - **Knowledge**: Status of the World Model and Vector DB
   - **User Inspector**: Look up user profiles by `external_id`
3. **📚 Knowledge Base (RAG)**:
   - **Document Stats**: Total documents and chunks ingested
   - **RAG Metrics**: Query count, average snippets per query
   - **Document Sources**: Pie chart of document sources
   - **Recent Ingestions**: Latest documents added to the knowledge base
   - **Recent Retrievals**: Queries that used RAG with snippet counts
4. **🧬 Self-Reprogramming**:
   - **Active Genome**: View the currently active Policy and Self-Prompt
   - **Game Theory**: Visualizes the live **Strategy Equilibrium**
   - **Evolution**: Tracks Proposals and Experiments
5. **🛡️ Safety & Governance**:
   - **Immutable Core**: Read-only view of `immutable_core.yaml`
   - **Audit Log**: History of all accepted/rejected proposals
   - **Human-in-the-Loop**: Interface to manually **Approve** or **Reject** pending proposals
6. **💬 Interaction & Traces**:
   - **Trace Explorer**: Filter traces by domain or error status
   - **Meta-Cognition**: Inspect Reward Scores, Latency, and Hallucination flags

### Admin Actions
The sidebar includes an **Operator Actions** section to manually trigger specific system functions, such as running a **Game Theory Optimization** cycle.

## Directory Structure
```text
thinking-machine-2.0/
├── .devcontainer/          # VS Code Dev Container Config
├── infra/                  # Docker Compose & Env
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   └── scripts/
│       └── init_db.sql     # Database initialization
├── libs/                   # Shared Code
│   ├── llm/                # LLM Client (OpenAI, vLLM, TGI)
│   ├── db.py               # Database Access
│   ├── user_memory.py      # Long-Term Memory (pgvector)
│   └── knowledge_ingest.py # RAG document processing
├── genome_store/           # The "Mind"
│   ├── policies/           # Routing & tool-use policies
│   ├── prompts/            # Self-prompts (editable)
│   ├── skills/
│   │   └── code/
│   │       └── game_strategy.py # Game Theory Logic
│   └── safety/
│       └── immutable_core.yaml  # Safety rules
├── thinking_core/          # Cognitive Pipeline
│   ├── pipeline.py         # Main orchestration
│   ├── user_context/       # Memory assembly
│   ├── world_model/        # Task classification
│   ├── retrieval/          # RAG (vector search + web)
│   ├── reasoning/          # Multi-agent reasoning
│   ├── reflection/         # Self-critique
│   └── output/             # Output synthesis
├── scripts/
│   └── ingest_knowledge.py # CLI for document ingestion
├── data/                   # Logs, Traces, Checkpoints
│   ├── docs/               # Documents for RAG ingestion
│   ├── logs/
│   ├── traces/
│   └── checkpoints/
└── services/               # Microservices
    ├── api_gateway/        # Public API + Admin endpoints
    ├── core_agent/         # Main cognitive engine
    ├── meta_agent/         # Self-modification proposals
    ├── orchestrator/       # Experiment management
    ├── eval_judge/         # Candidate scoring
    ├── training_worker/    # LoRA fine-tuning
    ├── safety_guard/       # Safety validation
    ├── distillation/       # Model distillation
    └── monitor/            # Streamlit dashboard
```

## Key Features
## Key Features

### ✅ Implemented

#### 1. Full Cognitive Pipeline (6 Phases)
The system implements a complete reasoning loop inspired by human cognition:
- **Phase 1: User Context** - Retrieves user-specific memories and preferences from long-term storage
- **Phase 2: World Model** - Classifies task nature (coding, medical, research, etc.) to adapt reasoning strategy
- **Phase 3: RAG Retrieval** - Searches knowledge base using vector similarity and optionally fetches live web data
- **Phase 4: Multi-Agent Reasoning** - LLM processes the task with full context (user memories + retrieved knowledge + policies)
- **Phase 5: Reflection & Critique** - Self-evaluates output for quality, safety, and coherence
- **Phase 6: Output Synthesis** - Generates final response incorporating reflection insights

#### 2. RAG with pgvector
Retrieval-Augmented Generation for knowledge-grounded responses:
- **Document Ingestion**: Supports PDF, TXT, and Markdown files via CLI (`scripts/ingest_knowledge.py`) or API (`/admin/knowledge/upload`)
- **Vector Search**: Uses PostgreSQL with pgvector extension for semantic similarity search over embedded text chunks
- **Web Integration Hook**: Ready for external web search microservice (Bing/Brave/Exa) for live information
- **Automatic Activation**: Triggers for research, medical, and coding domains based on task classification
- **Metrics Tracking**: Logs RAG usage, snippet counts, and retrieval quality in traces

#### 3. Long-Term User Memory
Per-user context that evolves over time:
- **Persistent Storage**: Each user (identified by `user_external_id`) has a dedicated memory store
- **Semantic Retrieval**: Vector search finds relevant past interactions based on query similarity
- **Recency Weighting**: Recent high-importance memories are prioritized
- **Memory Types**: Supports facts, preferences, projects, and custom categories
- **Privacy**: User memories are isolated and only accessible to that specific user

#### 4. Game Theory Optimization
Strategic adaptation using multi-player game theory:
- **N-Player Equilibrium**: Models interactions between Agent, Regulator (safety), and User as a strategic game
- **ESS Checks**: Validates Evolutionarily Stable Strategies to ensure robust adaptations
- **Tournament Experiments**: Tests proposed changes in controlled multi-round scenarios
- **Admin API**: Manual triggers for optimization cycles (`/admin/game-theory/preview`, `/optimize`)
- **Strategy Visualization**: Mission Control displays current equilibrium and recommended strategies

#### 5. Mission Control Dashboard
Real-time monitoring and control interface:
- **6 Specialized Tabs**: Ops & KPIs, Cognitive Engine, Knowledge Base (RAG), Self-Reprogramming, Safety & Governance, Interaction & Traces
- **Live Metrics**: System health, success rates, latency, RAG usage, user activity
- **Genome Viewer**: Inspect active policies and self-prompts
- **Human-in-the-Loop**: Review and approve/reject self-modification proposals
- **Visualizations**: Plotly charts for game theory equilibrium, trace analysis, and knowledge sources

#### 6. Self-Modification Loop
Continuous improvement through self-evolution:
1. **Trace**: All interactions logged with metadata (reward scores, latency, errors)
2. **Reflect**: Meta-agent analyzes failing traces to identify improvement opportunities
3. **Propose**: Generates patches to policies, prompts, or skills (stored in `genome_store/`)
4. **Validate**: Safety guard checks proposals against immutable core rules
5. **Experiment**: Orchestrator spawns candidate agents with proposed changes
6. **Evaluate**: Eval judge scores candidates using reward metrics and tournament games
7. **Evolve**: Successful proposals (high score + stable strategy) are merged into main genome

#### 7. Safety Layer
Multi-level safety guarantees:
- **Immutable Core**: Read-only YAML file (`genome_store/safety/immutable_core.yaml`) defines non-negotiable rules
- **Audit Log**: Complete history of all proposals (accepted/rejected) with timestamps and reasons
- **Proposal Review**: Human operators can manually approve/reject changes via Mission Control
- **Safety Guard Service**: Validates all modifications before deployment (currently placeholder)
- **Trace Monitoring**: Flags hallucinations, low confidence, and policy violations in real-time

#### 8. Genome Store
Git-based versioning for the AI's "mind":
- **Policy Versioning**: Routing rules, tool-use policies, and safety overrides tracked in database
- **Prompt Evolution**: Self-prompts (system instructions) can be modified and rolled back
- **Skill Library**: Code-based capabilities stored in `genome_store/skills/code/`
- **Immutable Safety**: Core safety rules separated from mutable components
- **Rollback Support**: Can revert to previous genome versions if issues arise

#### 9. Training Worker (LoRA Fine-Tuning)
Self-improvement through model distillation and fine-tuning:
- **LoRA Adapters**: Low-Rank Adaptation for efficient fine-tuning with minimal parameters
- **Distillation Pipeline**: Mines high-quality traces (reward > threshold) to create training datasets
- **Automated Training**: `train_tm_model.py` handles complete training workflow from dataset to model registration
- **Model Versioning**: Trained models registered as candidates in `model_versions` table
- **Tournament Evaluation**: New models tested against baseline before promotion
- **GPU Optimization**: Supports bf16, gradient checkpointing, and gradient accumulation for DGX deployment
- **Multi-GPU Support**: Fully configured for DGX Spark (8x H200) using Accelerate and DeepSpeed Zero3

### 🧪 Multi-GPU LoRA Training Workflow

The system includes a "Research Lab in a Box" setup for high-performance training on DGX nodes.

#### 1. Configuration
We provide a reproducible, version-pinned stack for DGX Spark (CUDA 12.1, PyTorch 2.2.2, Transformers 4.40.1):
- **Docker**: `docker/Dockerfile.training` (Pre-built image with all dependencies)
- **Conda**: `environment.yml` (Alternative environment setup)
- **Accelerate**: `configs/accelerate/accelerate_deepspeed_zero3.yaml` (8-GPU orchestration)
- **DeepSpeed**: `configs/deepspeed/ds_zero3_tm.json` (Zero Stage 3 optimization)

#### 2. Triggering Training
You trigger the distillation and training process via the Admin API. This creates a training run entry in the database.

```bash
curl -X POST http://localhost:8080/admin/distill-and-train \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "qwen-32b-instruct",
    "target_name": "tm-v2",
    "min_reward": 0.88,
    "require_safety_ok": true,
    "domains": ["general","coding"],
    "dataset_path": "data/distill/tm_v2_train.jsonl",
    "training_config": {
      "epochs": 1,
      "batch_size": 2,
      "grad_accum_steps": 8,
      "learning_rate": 1e-4,
      "max_length": 1024,
      "lora_r": 8,
      "lora_alpha": 16,
      "lora_dropout": 0.05
    },
    "auto_launch": false
  }'
```

#### 3. Execution on DGX
Once the run is created (and data distilled), launch the training worker on the DGX node using Accelerate:

**Option A: Docker**
```bash
docker build -t tm-train -f docker/Dockerfile.training .
docker run --gpus all -it tm-train
# Inside container:
accelerate launch --config_file configs/accelerate/accelerate_deepspeed_zero3.yaml train_tm_model.py --run-id <RUN_ID>
```

**Option B: Conda**
```bash
conda env create -f environment.yml
conda activate tm-train
accelerate launch --config_file configs/accelerate/accelerate_deepspeed_zero3.yaml train_tm_model.py --run-id <RUN_ID>
```

The worker will:
1. Load the run config and dataset
2. Apply LoRA adapters to the base model
3. Fine-tune across 8 GPUs using DeepSpeed Zero3
4. Save the adapter to `models/`
5. Register the new model version as a `candidate` for evaluation

### 🚧 In Progress
- **Real Safety Guard** (currently placeholder) - Advanced safety validation logic
- **Skill Evolution** (code-level mutations) - Automatic code generation and testing
- **Web Search Service** (external microservice for live data) - Integration with search APIs

## License

MIT License - See LICENSE file for details.

## Contributing

This is an experimental research project. Contributions welcome via pull requests.
