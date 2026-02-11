# Makefile for Agent Orchestra

.PHONY: help install install-dev test lint format clean build docker docs

# Default target
help:
	@echo "Agent Orchestra - Make targets:"
	@echo "  help          Show this help message"
	@echo "  install       Install the package"
	@echo "  install-dev   Install in development mode with dev dependencies"
	@echo "  test          Run tests"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build package"
	@echo "  docker        Build Docker images"
	@echo "  docs          Generate documentation"
	@echo "  benchmark     Run performance benchmarks"
	@echo "  setup         Setup development environment"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e .
	pip install -r requirements-dev.txt

# Testing targets
test:
	pytest tests/ -v --cov=agent_orchestra --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -x --ff

test-integration:
	pytest tests/integration/ -v

# Code quality targets
lint:
	flake8 agent_orchestra tests examples
	mypy agent_orchestra --ignore-missing-imports
	bandit -r agent_orchestra

format:
	black agent_orchestra tests examples
	isort agent_orchestra tests examples

check-format:
	black --check agent_orchestra tests examples
	isort --check-only agent_orchestra tests examples

# Build targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

# Docker targets
docker:
	docker build -t agent-orchestra:latest .

docker-dev:
	docker build --target development -t agent-orchestra:dev .

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Documentation targets
docs:
	cd docs && make html

docs-serve:
	cd docs && python -m http.server 8000

# Performance targets
benchmark:
	python examples/benchmark.py

profile:
	python -m cProfile -o profile.stats examples/basic_usage.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Development setup
setup:
	chmod +x scripts/setup.sh
	./scripts/setup.sh

# Database/Redis targets
redis-start:
	docker run -d --name agent-orchestra-redis -p 6379:6379 redis:7-alpine

redis-stop:
	docker stop agent-orchestra-redis && docker rm agent-orchestra-redis

# Configuration targets
config-validate:
	python -m agent_orchestra.cli config validate config/sample.yaml

config-generate:
	python -m agent_orchestra.cli config generate config/generated.yaml

# Examples targets
example-basic:
	python examples/basic_usage.py

example-parallel:
	python examples/parallel_execution.py

example-worker:
	python examples/agent_worker.py --agent-id example-worker --capabilities text_processing math_processing

# Monitoring targets
metrics-export:
	python -m agent_orchestra.cli export metrics --format prometheus --output metrics.txt

dashboard-start:
	python -m agent_orchestra.cli monitor --continuous

# Release targets
release-check: clean lint test
	@echo "Release checks passed"

release-build: release-check build
	@echo "Release build complete"

# Git targets
git-tag:
	@read -p "Enter version tag: " tag; \
	git tag -a $$tag -m "Release $$tag" && \
	git push origin $$tag

# All-in-one targets
dev-setup: install-dev setup
	@echo "Development environment ready"

ci: install-dev lint test
	@echo "CI pipeline complete"

full-test: clean install-dev lint test benchmark
	@echo "Full test suite complete"