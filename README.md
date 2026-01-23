# 🤖 Agent Orchestrator

> A powerful, service for orchestrating LLM-powered autonomous agents.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg) ![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)

**Agent Orchestrator** is a standalone service designed to manage the lifecycle, state, and execution of hierarchical LLM agents. It provides a robust HTTP API for creating, monitoring, and interacting with agents that can run independently or be extended for specialized domain-specific tasks.


---

## ✨ Features

- **Hierarchical Agents**: Native support for parent-child agent relationships and delegation.
- **State Management**: Robust persistent agent state backed by Redis.
- **LLM Agnostic**: Flexible model support (OpenAI, Gemini, DeepSeek) powered by `litellm`.
- **Extensible Tooling**: Plugin system for defining and registering agent capabilities.
- **Sandbox Integration**: Seamless optional integration with `sandbox-runtime` for secure, isolated tool execution.
- **Interactive CLI**: Rich terminal interface for real-time management and monitoring.

---

## 🚀 Quick Start

### Prerequisites

- **Docker** and **Docker Compose**
- **Python 3.11+**
- **Poetry** (for local development)
- **LLM API Key** (OpenAI, Gemini, or DeepSeek)

### 1. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API key
# e.g., GEMINI_API_KEY=your-key-here
```

### 2. Start Services

```bash
# Start Redis + Agent Manager (Standalone - No Sandbox)
make run

# View logs
make logs
```

### 3. Verify Deployment

```bash
# Check API docs availability
curl http://localhost:8083/docs

# Or open in browser: http://localhost:8083/docs
```

---

## 🔌 API Endpoints

The service exposes a comprehensive REST API.

| Method   | Endpoint              | Description                           |
| -------- | --------------------- | ------------------------------------- |
| `POST`   | `/agents`             | Create and start a new agent          |
| `GET`    | `/agents`             | List agents (paginated)               |
| `GET`    | `/agents/{id}/status` | Get agent status                      |
| `POST`   | `/agents/{id}/stop`   | Stop a running agent (keeps history)  |
| `DELETE` | `/agents/{id}`        | Stop agent and delete its data (wipe) |

### Example: Create an Agent

```bash
curl -X POST http://localhost:8083/agents \
  -H "Content-Type: application/json" \
  -d '{
    "agent_config": {
      "llm_config": {"model": "gemini/gemini-3-flash-preview"},
      "state": {
        "agent_id": "my-agent-001",
        "task": "Research the topic of quantum computing and create a summary"
      },
      "agent_hierarchy": []
    },
    "job_config": {
      "automatic": true
    }
  }'
```

---

## 🖥️ Interactive CLI

The project includes a rich **Command Line Interface (CLI)** for managing agents directly from the terminal.

### Usage

```bash
# Run the interactive CLI
poetry run python cli.py
```

### Capabilities

- **Create Agent**: Wizard-style creation of new agents.
- **Visual graph**: Visualize the agent hierarchy (Parent -> Child relationships).
- **Live Monitoring**: Watch real-time message logs and tool executions.
- **Intervention**: Send instructions to running agents on the fly.
- **Control**: Pause or stop agents instantly.

---

## 🛡️ Running with Sandbox

The `sandbox-runtime` provides secure isolated environments for tool execution (Browser, Python, Terminal).

> **Requirement:** The `sandbox-runtime` project must be present in the parent directory (sibling to `agent-orchestrator`) for the Docker build context.

To enable full sandbox support:

```bash
# Start with sandbox support
make run-with-sandbox
```

This sets `AGENT_SANDBOX_MODE=true`, enabling the registration of sensitive tools that require isolation.

---

## 🛠️ Local Development

```bash
# Install dependencies
make dev-install

# Start Redis in background
docker run -d -p 6379:6379 redis:7-alpine

# Run locally with hot reload
make run-local

# Run tests
make test
```

## 🏗️ Architecture

```
agent-orchestrator/
├── main.py                 # FastAPI service entry point
├── src/
│   └── agent_framework/    # Core agent library
│       ├── agents/         # Agent implementations
│       ├── llm/            # LLM abstraction layer
│       ├── prompts/        # Base prompt templates
│       ├── tools/          # Agent tools library
│       └── state/          # Redis state management
├── docker-compose.yml      # Standalone deployment
└── Makefile                # Development commands
```

## 🧩 Extension & Customization

The orchestrator is designed for extensibility.

### Custom Prompts via `AGENT_PROMPT_PATHS`

You can inject specialized prompts without modifying the core code:

```bash
# Mount custom prompts directory
export AGENT_PROMPT_PATHS=/path/to/custom/prompts
```

### Integration Patterns

1.  **Custom Prompts**: Tailor agent behavior for specific domains.
2.  **External API Layer**: Build application-specific APIs wrapping the orchestrator.
3.  **Custom Tools**: Register new capabilities in the `agent_framework`.

## ⚙️ Environment Variables

| Variable             | Description                | Default                         |
| -------------------- | -------------------------- | ------------------------------- |
| `AGENT_MODEL`        | LLM model (litellm format) | `gemini/gemini-3-flash-preview` |
| `OPENAI_API_KEY`     | OpenAI API key             | -                               |
| `GEMINI_API_KEY`     | Google Gemini API key      | -                               |
| `DEEPSEEK_API_KEY`   | DeepSeek API key           | -                               |
| `REDIS_HOST`         | Redis hostname             | `redis`                         |
| `AGENT_PROMPT_PATHS` | Extra prompt directories   | -                               |
| `AGENT_SANDBOX_URL`  | Sandbox service URL        | -                               |

## 📄 License

See repository root for license information.