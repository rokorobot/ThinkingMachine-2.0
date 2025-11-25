# Thinking Machine - Walkthrough (DGX Spark Edition)

## System Overview
This is a "Level-3" self-modifying AI system optimized for a single DGX Spark node. It features a **Genome Store** (Git-based), **Self-Training** (LoRA), and a **Safety Guard**.

### Services
## Development Workflow ("Vibecoding")

### 1. Local Development (CPU-only)
Use VS Code with Dev Containers for a consistent environment.
1.  Open the repository in VS Code.
2.  Click "Reopen in Container" when prompted (or use command palette).
3.  This uses `.devcontainer/devcontainer.json` and `infra/docker-compose.dev.yml`.
4.  Run services:
    ```bash
    cd infra
    docker-compose -f docker-compose.dev.yml up
    ```
5.  Access Monitor at `http://localhost:8501`.

### 2. Production Deployment (DGX Spark)
On the DGX node with NVIDIA Container Toolkit:
1.  Clone the repo.
2.  Set environment variables in `infra/env/*.env`.
3.  Run with GPU support:
    ```bash
    cd infra
    docker-compose up -d
    ```
    *Note: `docker-compose.yml` is configured with `deploy.resources.reservations.devices` for GPU access.*

## How to Run

1.  **Set Environment Variables**
    Create a `.env` file in the root directory (or rely on `infra/env/*.env`):
    ```bash
    OPENAI_API_KEY=sk-...
    LLM_BACKEND=openai # or tgi, vllm
    LLM_MODEL=gpt-4o
    ```

2.  **Interact with the Agent**
    ```bash
    curl -X POST http://localhost:8080/task \
      -H "Content-Type: application/json" \
      -d '{"input_text": "How do I build a rocket?", "domain": "science"}'
    ```

## The Self-Reprogramming Loop

1.  **Trace**: User interactions are logged to `data/traces`.
2.  **Reflection**: The **Meta Agent** analyzes failing traces using Game Theory.
3.  **Proposal**: It proposes a patch to `genome_store/` (e.g., a new policy rule).
4.  **Validation**: The **Safety Guard** checks the proposal against `immutable_core.yaml`.
5.  **Experiment**: The **Orchestrator** spawns a candidate agent with the patch.
6.  **Evaluation**: The **Eval Judge** scores the candidate (supports "Tournament" style multi-round games).
7.  **Evolution**: If successful (High Score + Stable Strategy), the patch is applied to the main Genome.

## Game Theory Integration

The system uses Game Theory to optimize its adaptability.

### Admin API
You can manually trigger the Game Theory optimization loop via the Admin API:

1.  **Preview Equilibrium**: See what strategy the system recommends based on recent metrics.
    ```bash
    curl "http://localhost:8080/admin/game-theory/preview?domain=medical&hours=24"
    ```

2.  **Optimize & Propose**: Commit the recommendation as a proposal for the Orchestrator.
    ```bash
    curl -X POST http://localhost:8080/admin/game-theory/optimize \
      -H "Content-Type: application/json" \
      -d '{"domain": "medical", "hours": 24, "commit": true}'
    ```

## Long-Term Memory

The system now supports persistent user memory using `pgvector`.

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

## User-Specific Policy Overlays

The system now adapts its behavior to individual users by inferring their preferences from interaction traces.

### How it works
1.  **Trace Analysis**: The **Meta Agent** periodically analyzes recent traces for each user.
2.  **Preference Inference**: It infers preferences like `tone` (direct vs gentle), `detail_level` (concise vs detailed), and `safety_bias`.
3.  **Policy Overlay**: These preferences are converted into a **Policy Overlay** (stored in `user_policies`) that overrides specific routing and tool-use rules in the global policy.
4.  **Runtime Application**: When the Core Agent handles a task, it merges the active global policy with the user's specific overlay.

### Example Preferences
*   **Tone**: `direct` (high directness) vs `gentle` (low directness)
*   **Detail**: `concise` (short replies) vs `detailed` (long replies)
*   **Safety**: `strict` (extra checks) vs `relaxed`
## Mission Control Dashboard

The **Monitor** service (`http://localhost:8501`) has been upgraded to a full "Mission Control" interface with 5 tabs reflecting the system's core capabilities:

1.  **🚀 Ops & KPIs**: System health, success rates (reward > 0.5), latency, and active user counts.
2.  **🧠 Cognitive Engine**:
    *   **Memory**: Stats on total users and memories, plus a view of recent user memories.
    *   **Knowledge**: Status of the World Model and Vector DB.
    *   **User Inspector**: Look up user profiles by `external_id`.
3.  **🧬 Self-Reprogramming**:
    *   **Active Genome**: View the currently active Policy and Self-Prompt.
    *   **Game Theory**: Visualizes the live **Strategy Equilibrium** (Agent vs Regulator vs User).
    *   **Evolution**: Tracks Proposals (evolution) and Experiments (validation).
4.  **🛡️ Safety & Governance**:
    *   **Immutable Core**: Read-only view of the safety constitution (`immutable_core.yaml`).
    *   **Audit Log**: History of all accepted/rejected proposals.
    *   **Human-in-the-Loop**: Interface to manually **Approve** or **Reject** pending proposals.
5.  **💬 Interaction & Traces**:
    *   **Trace Explorer**: Filter traces by domain or error status.
    *   **Meta-Cognition**: Inspect Reward Scores, Latency, and Hallucination flags for each interaction.

### Admin Actions
The sidebar includes an **Operator Actions** section to manually trigger specific system functions, such as running a **Game Theory Optimization** cycle.

## Thinking Machine 2.0 Architecture

The system has been upgraded to **Thinking Machine 2.0**, introducing a sophisticated reasoning pipeline orchestrated by the `thinking_core` package.

### Core Pipeline
The `core_agent` now delegates task execution to a 6-stage pipeline:

1.  **User Context**: Assembles user profile and relevant memories.
2.  **World Model**: Simulates task nature, constraints, and safety signals.
3.  **Retrieval**: Fetches external knowledge (Web/Docs) if needed.
4.  **Multi-Agent Reasoning**:
    *   **Direct Responder**: Fast, intuitive answers.
    *   **Planner**: Step-by-step reasoning.
    *   **Cautious Checker**: Safety and error analysis.
    *   **Synthesizer**: Merges agent outputs.
5.  **Reflection**: A "Critic" loop (Judge) evaluates the draft for safety and quality.
6.  **Output Synthesis**: Final shaping based on critique and user preferences.

## Directory Structure
```text
thinking-machine/
├── .devcontainer/          # VS Code Dev Container Config
├── infra/                  # Docker Compose & Env
├── libs/                   # Shared Code
│   ├── llm/                # LLM Client
│   ├── db.py               # Database Access
│   └── user_memory.py      # Long-Term Memory (pgvector)
├── genome_store/           # The "Mind"
│   ├── policies/
│   ├── prompts/
│   └── skills/
│       └── code/
│           └── game_strategy.py # Game Theory Logic
├── thinking_core/          # [NEW] Thinking Machine 2.0 Pipeline
│   ├── pipeline.py         # Main Orchestrator
│   ├── user_context/       # Context Assembler
│   ├── world_model/        # Simulator & Signals
│   ├── retrieval/          # RAG & Search
│   ├── reasoning/          # Multi-Agent Core
│   ├── reflection/         # Critic & Judge
│   └── output/             # Synthesizer
├── data/                   # Logs, Traces, Checkpoints
└── services/               # Microservices
    ├── api_gateway/        # Includes Admin API
    ├── core_agent/         # Calls thinking_core.pipeline
    ├── meta_agent/         # Includes Game Theory Proposer
    ├── orchestrator/
    ├── training_worker/
    ├── eval_judge/
    ├── safety_guard/
    └── monitor/
```
