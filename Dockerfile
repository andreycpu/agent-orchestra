# Multi-stage Dockerfile for Agent Orchestra

# Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-dev.txt setup.py ./
COPY agent_orchestra/__init__.py agent_orchestra/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir wheel \
    && pip install --no-cache-dir -r requirements-dev.txt

# Copy the rest of the application
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r orchestra && useradd -r -g orchestra orchestra

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=orchestra:orchestra . .

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data \
    && chown -R orchestra:orchestra /app

# Switch to non-root user
USER orchestra

# Expose metrics port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "agent_orchestra.cli", "start", "--monitor"]

# Development stage
FROM builder as development

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    debugpy

# Set development environment
ENV PYTHONPATH=/app
ENV FLASK_ENV=development
ENV LOG_LEVEL=DEBUG

# Create development user
RUN groupadd -r dev && useradd -r -g dev -s /bin/bash dev
RUN mkdir -p /home/dev && chown -R dev:dev /home/dev

# Switch to dev user
USER dev

# Expose development ports
EXPOSE 8080 5678

# Development command with debugger
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m", "agent_orchestra.cli", "start", "--monitor"]

# Testing stage
FROM builder as testing

# Set test environment
ENV PYTHONTEST=1
ENV REDIS_URL=redis://redis:6379/0

# Run tests
RUN pytest tests/ -v --cov=agent_orchestra

# Linting and code quality
RUN black --check agent_orchestra tests examples \
    && isort --check-only agent_orchestra tests examples \
    && flake8 agent_orchestra \
    && mypy agent_orchestra --ignore-missing-imports

# CLI stage - minimal image for CLI usage
FROM python:3.11-slim as cli

# Install minimal dependencies
RUN pip install --no-cache-dir agent-orchestra

# Create CLI user
RUN groupadd -r cli && useradd -r -g cli cli

# Switch to CLI user
USER cli

# Set entrypoint to CLI
ENTRYPOINT ["python", "-m", "agent_orchestra.cli"]