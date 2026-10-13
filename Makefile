.PHONY: help install dev-install run run-with-sandbox run-local stop test logs clean create-agent

# Some tests reach across repositories: `benchmarks` and `abs` live in ../abs,
# `console_api` comes from arena-sensors (NOT intents' console_api -- an earlier
# one on the path shadows it) and `cyd_sensors` comes from cyd-sensors. The
# order below is load-bearing.
TEST_PYTHONPATH := src:$(abspath ../abs):$(abspath ../abs/src):$(abspath ../arena/cyd-sensors-and-algorithms/src):$(abspath ../arena/arena-sensors-and-algorithms/src)

help:
	@echo "Agent Orchestrator - Standalone Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install          - Install production dependencies"
	@echo "  dev-install      - Install all dependencies including dev"
	@echo ""
	@echo "Running (Docker):"
	@echo "  run              - Start Redis + Agent Manager"
	@echo "  run-with-sandbox - Start with sandbox-runtime (full stack)"
	@echo "  stop             - Stop all services"
	@echo "  logs             - Follow logs for all services"
	@echo ""
	@echo "Running (Local):"
	@echo "  run-local        - Run agent-manager directly (requires Redis)"
	@echo ""
	@echo "Testing:"
	@echo "  test             - Run pytest tests"
	@echo "  create-agent     - Create a test agent via API"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean            - Remove cache files"

# ============================================================================
# Setup
# ============================================================================

install:
	poetry install --only=main

dev-install:
	poetry install --with=dev

# ============================================================================
# Running with Docker
# ============================================================================

run:
	@echo "🚀 Starting Agent Orchestrator (Redis + Agent Manager)..."
	@mkdir -p logs
	docker compose up --build -d

run-with-sandbox:
	@echo "🚀 Starting Agent Orchestrator with Sandbox Runtime..."
	@mkdir -p logs
	AGENT_SANDBOX_MODE=true AGENT_SANDBOX_URL=http://sandbox-service:8000 docker compose --profile with-sandbox up --build -d

stop:
	@echo "🛑 Stopping services..."
	docker compose --profile with-sandbox down --remove-orphans -v

logs:
	docker compose logs -f --tail=100

# ============================================================================
# Running Locally (for development/debugging)
# ============================================================================

run-local:
	@echo "🔧 Running agent-manager locally (ensure Redis is running)..."
	@echo "   Start Redis with: docker run -d -p 6379:6379 redis:7-alpine"
	bash -c 'set -a; [ -f .env ] && source .env; set +a; \
		export REDIS_HOST=$${REDIS_HOST:-localhost}; \
		export PYTHONPATH=$$PYTHONPATH:$$(pwd)/src; \
		poetry run uvicorn main:app --host 0.0.0.0 --port 8083 --reload'

# ============================================================================
# Testing
# ============================================================================

# Tests are independent, so run them across all cores. The terminal tests are
# wall-clock bound rather than CPU bound, so the win is large: the suite drops
# from minutes to well under a minute. `test-serial` is kept for the rare case
# where a failure needs a readable, ordered transcript.
test:
	export PYTHONPATH=$$PYTHONPATH:$(TEST_PYTHONPATH) && poetry run pytest tests/ -n auto

test-serial:
	export PYTHONPATH=$$PYTHONPATH:$(TEST_PYTHONPATH) && poetry run pytest tests/ -v

test-cov:
	export PYTHONPATH=$$PYTHONPATH:$(TEST_PYTHONPATH) && poetry run pytest tests/ -n auto --cov=agent_framework --cov-report=term-missing

# Example: Create a simple test agent
create-agent:
	@echo "📤 Creating a test agent..."
	@curl -s -X POST "http://localhost:8083/agents" \
		-H "Content-Type: application/json" \
		-d '{ \
			"agent_config": { \
				"llm_config": {"model": "$${AGENT_MODEL:-openrouter/deepseek/deepseek-v4.1-flash}"}, \
				"state": { \
					"agent_id": "test-$$(date +%s)", \
					"task": "This is a test. Simply respond with: Test successful!" \
				}, \
				"agent_hierarchy": [] \
			}, \
			"job_config": {"automatic": true} \
		}' | jq .

# ============================================================================
# Maintenance
# ============================================================================

clean:
	@echo "🧹 Cleaning up cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
