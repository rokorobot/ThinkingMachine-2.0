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

The **Monitor** service (`http://localhost:8501`) provides a full "Mission Control" interface with 5 tabs:

1. **🚀 Ops & KPIs**: System health, success rates, latency, and active user counts.
2. **🧠 Cognitive Engine**:
   - **Memory**: Stats on total users and memories
   - **Knowledge**: Status of the World Model and Vector DB
   - **User Inspector**: Look up user profiles by `external_id`
3. **🧬 Self-Reprogramming**:
   - **Active Genome**: View the currently active Policy and Self-Prompt
   - **Game Theory**: Visualizes the live **Strategy Equilibrium**
   - **Evolution**: Tracks Proposals and Experiments
4. **🛡️ Safety & Governance**:
   - **Immutable Core**: Read-only view of `immutable_core.yaml`
   - **Audit Log**: History of all accepted/rejected proposals
   - **Human-in-the-Loop**: Interface to manually **Approve** or **Reject** pending proposals
5. **💬 Interaction & Traces**:
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

### ✅ Implemented
- **Full Cognitive Pipeline** (6 phases: Context → World Model → RAG → Reasoning → Reflection → Synthesis)
- **RAG with pgvector** (PDF/TXT/MD ingestion, vector search, web integration hook)
- **Long-Term User Memory** (per-user context, semantic + recency retrieval)
- **Game Theory Optimization** (n-player equilibrium, ESS checks, tournament experiments)
- **Mission Control Dashboard** (real-time metrics, genome viewer, human-in-the-loop)
- **Self-Modification Loop** (trace → reflect → propose → validate → experiment → evolve)
- **Safety Layer** (immutable core, audit log, proposal review)

### 🚧 In Progress
- **Real Safety Guard** (currently placeholder)
- **Training Worker** (LoRA fine-tuning logic)
- **Skill Evolution** (code-level mutations)
- **Web Search Service** (external microservice for live data)

## License

MIT License - See LICENSE file for details.

## Contributing

This is an experimental research project. Contributions welcome via pull requests.
