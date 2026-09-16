FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy the manifest and lock before the source so the dependency layer is
# cached across source edits. Copying the source first invalidated this layer
# on every change and reinstalled the whole dependency set each build.
COPY llm-agents-orchestrator/pyproject.toml /app/pyproject.toml
COPY llm-agents-orchestrator/poetry.lock /app/poetry.lock

# Install Dependencies
WORKDIR /app
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main --no-root

# Now the source, and install just this project on top of the cached deps.
COPY llm-agents-orchestrator/src /app/src
COPY llm-agents-orchestrator/main.py /app/main.py
RUN poetry install --no-interaction --no-ansi --only main

ENV PYTHONPATH=/app:/app/src

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8083"]
