# Makefile for Agent Orchestra - Production-Ready Development Automation

.DEFAULT_GOAL := help
.PHONY: help install dev-install test lint format clean docs build security performance

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Project variables
PROJECT_NAME := agent-orchestra
PYTHON := python3
PIP := pip3
DOCKER_IMAGE := agent-orchestra
VERSION := $(shell grep '^__version__' agent_orchestra/__init__.py | cut -d'"' -f2)

##@ General
help: ## Display this help message
	@echo "$(CYAN)$(PROJECT_NAME) v$(VERSION) - Development Automation$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)

check-python: ## Check Python version
	@$(PYTHON) --version | grep -E "3\.(8|9|10|11|12)" > /dev/null || (echo "$(RED)Python 3.8+ required$(RESET)" && exit 1)

##@ Installation
install: check-python ## Install the package for production
	@echo "$(GREEN)Installing $(PROJECT_NAME)...$(RESET)"
	$(PIP) install -e .

dev-install: check-python ## Install development dependencies
	@echo "$(GREEN)Installing development environment...$(RESET)"
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements-dev.txt
	@$(MAKE) pre-commit-install

upgrade-deps: ## Upgrade all dependencies
	@echo "$(GREEN)Upgrading dependencies...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r requirements-dev.txt

##@ Testing
test: ## Run unit tests
	@echo "$(GREEN)Running unit tests...$(RESET)"
	pytest tests/ -v --tb=short -x

test-all: ## Run all tests (unit + integration)
	@echo "$(GREEN)Running all tests...$(RESET)"
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(RESET)"
	pytest tests/ -v --cov=agent_orchestra --cov-report=html --cov-report=term --cov-fail-under=80

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(RESET)"
	pytest tests/integration/ -v -m integration

test-performance: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(RESET)"
	pytest tests/performance/ -v --benchmark-only

test-watch: ## Run tests in watch mode
	@echo "$(GREEN)Running tests in watch mode...$(RESET)"
	pytest-watch tests/ -- -v --tb=short

##@ Code Quality
lint: ## Run all linting checks
	@echo "$(GREEN)Running linting checks...$(RESET)"
	flake8 agent_orchestra tests examples --max-line-length=88 --extend-ignore=E203,W503
	mypy agent_orchestra --ignore-missing-imports --strict
	bandit -r agent_orchestra -f json -o reports/bandit.json || true
	safety check || true

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(RESET)"
	black agent_orchestra tests examples --line-length=88
	isort agent_orchestra tests examples --profile=black

format-check: ## Check code formatting
	@echo "$(GREEN)Checking code formatting...$(RESET)"
	black --check agent_orchestra tests examples --line-length=88
	isort --check-only agent_orchestra tests examples --profile=black

type-check: ## Run type checking
	@echo "$(GREEN)Running type checks...$(RESET)"
	mypy agent_orchestra --ignore-missing-imports --html-report reports/mypy/

complexity-check: ## Check code complexity
	@echo "$(GREEN)Checking code complexity...$(RESET)"
	mccabe --min=10 agent_orchestra/

quality: lint format-check type-check complexity-check ## Run all code quality checks

##@ Security
security: ## Run security scans
	@echo "$(GREEN)Running security scans...$(RESET)"
	@mkdir -p reports
	safety check --json --output reports/safety.json || true
	bandit -r agent_orchestra -f json -o reports/bandit.json || true
	semgrep --config=auto --json --output=reports/semgrep.json agent_orchestra/ || true
	@echo "$(YELLOW)Security reports saved to reports/$(RESET)"

audit-dependencies: ## Audit Python dependencies for vulnerabilities
	@echo "$(GREEN)Auditing dependencies...$(RESET)"
	pip-audit --format=json --output=reports/pip-audit.json

##@ Performance
benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(RESET)"
	$(PYTHON) examples/benchmark.py

profile: ## Profile application performance
	@echo "$(GREEN)Profiling application...$(RESET)"
	$(PYTHON) -m cProfile -o reports/profile.stats examples/stress_test.py
	$(PYTHON) -c "import pstats; pstats.Stats('reports/profile.stats').sort_stats('cumulative').print_stats(20)"

memory-profile: ## Profile memory usage
	@echo "$(GREEN)Profiling memory usage...$(RESET)"
	mprof run examples/stress_test.py
	mprof plot -o reports/memory-profile.png

##@ Documentation
docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(RESET)"
	@mkdir -p docs/_build
	cd docs && $(MAKE) html

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8000$(RESET)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	cd docs && $(MAKE) clean

api-docs: ## Generate API documentation
	@echo "$(GREEN)Generating API documentation...$(RESET)"
	pdoc --html --output-dir docs/api agent_orchestra

##@ Build & Package
clean: ## Clean build artifacts
	@echo "$(GREEN)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf reports/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: clean ## Build distribution packages
	@echo "$(GREEN)Building distribution packages...$(RESET)"
	$(PYTHON) -m build

build-check: build ## Check built packages
	@echo "$(GREEN)Checking built packages...$(RESET)"
	twine check dist/*

##@ Docker
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(RESET)"
	docker build -t $(DOCKER_IMAGE):latest .

docker-dev: ## Build development Docker image
	@echo "$(GREEN)Building development Docker image...$(RESET)"
	docker build --target development -t $(DOCKER_IMAGE):dev .

docker-test: ## Build and run test Docker image
	@echo "$(GREEN)Building and testing Docker image...$(RESET)"
	docker build --target testing -t $(DOCKER_IMAGE):test .

docker-run: ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(RESET)"
	docker run -p 8080:8080 $(DOCKER_IMAGE):latest

docker-clean: ## Clean Docker images
	docker rmi $(DOCKER_IMAGE):latest $(DOCKER_IMAGE):dev $(DOCKER_IMAGE):test 2>/dev/null || true

##@ Release
release-check: build-check test-all security ## Check release readiness
	@echo "$(GREEN)Release readiness check completed$(RESET)"

release-test: build ## Release to test PyPI
	@echo "$(YELLOW)Uploading to test PyPI...$(RESET)"
	twine upload --repository testpypi dist/*

release: build ## Release to production PyPI
	@echo "$(RED)Uploading to production PyPI...$(RESET)"
	@read -p "Are you sure you want to release v$(VERSION)? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		twine upload dist/*; \
	else \
		echo "$(YELLOW)Release cancelled$(RESET)"; \
	fi

tag: ## Create and push git tag
	@echo "$(GREEN)Creating git tag v$(VERSION)...$(RESET)"
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	git push origin v$(VERSION)

##@ Development Tools
pre-commit-install: ## Install pre-commit hooks
	@echo "$(GREEN)Installing pre-commit hooks...$(RESET)"
	pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	@echo "$(GREEN)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

setup-dev: dev-install pre-commit-install ## Complete development environment setup
	@echo "$(GREEN)Development environment setup complete!$(RESET)"

env-info: ## Show environment information
	@echo "$(CYAN)Environment Information:$(RESET)"
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Pip version: $(shell $(PIP) --version)"
	@echo "Project version: $(VERSION)"
	@echo "Docker version: $(shell docker --version 2>/dev/null || echo 'Not installed')"

##@ CI/CD
ci-test: ## Run CI pipeline tests
	@echo "$(GREEN)Running CI pipeline tests...$(RESET)"
	@$(MAKE) clean
	@$(MAKE) dev-install
	@$(MAKE) quality
	@$(MAKE) test-cov
	@$(MAKE) security
	@$(MAKE) build-check

validate-config: ## Validate configuration files
	@echo "$(GREEN)Validating configuration files...$(RESET)"
	$(PYTHON) -m agent_orchestra.cli config validate config/schema.yaml

##@ Monitoring
health-check: ## Run health checks
	@echo "$(GREEN)Running health checks...$(RESET)"
	$(PYTHON) -m agent_orchestra.cli health check

metrics: ## Show system metrics
	@echo "$(GREEN)Collecting metrics...$(RESET)"
	$(PYTHON) -m agent_orchestra.cli metrics

##@ Examples
run-quickstart: ## Run quickstart example
	@echo "$(GREEN)Running quickstart example...$(RESET)"
	$(PYTHON) examples/quickstart.py

run-production: ## Run production example
	@echo "$(GREEN)Running production example...$(RESET)"
	$(PYTHON) examples/production_example.py

##@ Utilities
version: ## Show version information
	@echo "$(CYAN)$(PROJECT_NAME) v$(VERSION)$(RESET)"

update-version: ## Update version (requires VERSION=x.x.x)
ifndef VERSION
	$(error VERSION is required. Use: make update-version VERSION=1.2.3)
endif
	@echo "$(GREEN)Updating version to $(VERSION)...$(RESET)"
	sed -i 's/__version__ = "[^"]*"/__version__ = "$(VERSION)"/' agent_orchestra/__init__.py