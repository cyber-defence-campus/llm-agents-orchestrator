FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy Source Code
COPY llm-agents-orchestrator/src /app/src
COPY llm-agents-orchestrator/main.py /app/main.py
COPY llm-agents-orchestrator/pyproject.toml /app/pyproject.toml
COPY llm-agents-orchestrator/poetry.lock /app/poetry.lock

# Install Dependencies
WORKDIR /app
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

ENV PYTHONPATH=/app:/app/src

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8083"]
