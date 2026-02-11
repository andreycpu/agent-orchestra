# Multi-stage Dockerfile for Agent Orchestra

# Build stage
FROM python:3.11-slim as builder

# Install security updates and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Create virtual environment for better isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt pyproject.toml ./
COPY agent_orchestra/__init__.py agent_orchestra/

# Install Python dependencies with security scanning
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir safety \
    && safety check -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# Copy the rest of the application
COPY . .

# Install the package
RUN pip install --no-cache-dir -e . \
    && pip list --format=freeze > installed_packages.txt

# Production stage
FROM python:3.11-slim as production

# Install security updates and minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with specific UID/GID for security
RUN groupadd -r -g 1001 orchestra && \
    useradd -r -g orchestra -u 1001 -s /bin/false -c "Orchestra User" orchestra

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code and set ownership
COPY --chown=orchestra:orchestra agent_orchestra ./agent_orchestra
COPY --chown=orchestra:orchestra config ./config
COPY --chown=orchestra:orchestra examples ./examples
COPY --chown=orchestra:orchestra pyproject.toml setup.py ./

# Create directories for logs and data with proper permissions
RUN mkdir -p /app/logs /app/data /tmp/agent_orchestra \
    && chown -R orchestra:orchestra /app /tmp/agent_orchestra \
    && chmod 755 /app/logs /app/data \
    && chmod 1777 /tmp/agent_orchestra

# Add security labels
LABEL org.opencontainers.image.title="Agent Orchestra" \
      org.opencontainers.image.description="Multi-agent orchestration framework" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Agent Orchestra Team" \
      org.opencontainers.image.source="https://github.com/andreycpu/agent-orchestra" \
      org.opencontainers.image.licenses="MIT" \
      security.scan.type="production"

# Set security-focused environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TMPDIR=/tmp/agent_orchestra \
    HOME=/app

# Switch to non-root user
USER orchestra:orchestra

# Expose metrics port (non-privileged)
EXPOSE 8080

# Health check with better security
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# Default command with signal handling
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