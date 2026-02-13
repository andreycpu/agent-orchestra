# Development Guide

This guide covers development practices, architecture patterns, and best practices for contributing to Agent Orchestra.

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose (for integration testing)
- Git
- Make (for running build tasks)

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd agent-orchestra
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

5. Run tests to verify setup:
```bash
pytest tests/
```

## Project Structure

```
agent_orchestra/
├── __init__.py              # Package initialization
├── agent.py                 # Agent management
├── cli.py                   # Command-line interface
├── config.py                # Configuration management
├── exceptions.py            # Custom exceptions with retry logic
├── logging_utils.py         # Structured logging and observability
├── orchestra.py             # Main orchestration logic
├── security_utils.py        # Authentication and authorization
├── task_router.py           # Task routing and scheduling
├── testing_utils.py         # Test utilities and fixtures
├── types.py                 # Type definitions and models
├── utils.py                 # General utilities
└── validation.py            # Input validation and sanitization

tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_*.py                # Unit tests
├── integration/             # Integration tests
└── unit/                    # Additional unit tests

docs/
├── api.md                   # API documentation
├── architecture.md          # Architecture overview
├── development.md           # Development guide (this file)
├── security.md              # Security documentation
└── troubleshooting.md       # Common issues and solutions
```

## Architecture Overview

### Core Components

1. **Orchestra**: Main orchestration engine that manages task distribution and agent coordination
2. **Task Router**: Intelligent routing system that assigns tasks to appropriate agents
3. **Agent Manager**: Handles agent registration, health monitoring, and lifecycle
4. **State Manager**: Manages persistent state and task tracking
5. **Security Layer**: Authentication, authorization, and input validation

### Design Principles

1. **Modularity**: Each component has a single responsibility and clear interfaces
2. **Testability**: Comprehensive unit and integration tests with high coverage
3. **Observability**: Structured logging, metrics, and tracing throughout
4. **Security**: Defense in depth with input validation, authentication, and authorization
5. **Reliability**: Retry mechanisms, circuit breakers, and graceful degradation

## Development Practices

### Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security linting

Run all checks:
```bash
make lint
```

Auto-fix formatting issues:
```bash
make format
```

### Testing Strategy

#### Test Types

1. **Unit Tests** (`tests/test_*.py`): Test individual functions and classes
2. **Integration Tests** (`tests/integration/`): Test component interactions
3. **End-to-End Tests**: Full workflow testing with real components

#### Test Guidelines

- Use descriptive test names that explain the scenario
- Follow the Arrange-Act-Assert pattern
- Use fixtures for common setup (see `testing_utils.py`)
- Mock external dependencies appropriately
- Aim for >90% code coverage

#### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_validation.py

# With coverage
pytest --cov=agent_orchestra --cov-report=html

# Integration tests only
pytest tests/integration/

# Skip slow tests
pytest -m "not slow"
```

### Error Handling

Follow these patterns for robust error handling:

#### Exception Hierarchy

```python
from agent_orchestra.exceptions import (
    AgentOrchestraException,    # Base exception
    RetryableError,             # Can be retried
    PermanentError,             # Should not be retried
    ValidationError,            # Input validation failed
    SecurityError               # Security violation
)
```

#### Retry Logic

```python
from agent_orchestra.exceptions import handle_retry_logic

for attempt in range(1, max_attempts + 1):
    try:
        result = risky_operation()
        break
    except Exception as e:
        retry_info = handle_retry_logic(e, attempt, max_attempts)
        
        if not retry_info['should_retry']:
            raise
        
        time.sleep(retry_info['delay_seconds'])
```

#### Error Context

```python
from agent_orchestra.exceptions import ErrorContext, log_error_with_context

try:
    dangerous_operation()
except Exception as e:
    context = ErrorContext(
        operation="dangerous_operation",
        component="task_router",
        task_id=task.id
    )
    log_error_with_context(logger, e, context)
    raise
```

### Logging and Observability

#### Structured Logging

```python
from agent_orchestra.logging_utils import setup_logging, create_specialized_loggers

# Setup logging
logger = setup_logging(level='INFO', format_type='json')

# Use specialized loggers
loggers = create_specialized_loggers(logger)
loggers['performance'].log_operation_time('task_execution', duration)
loggers['audit'].log_authorization('user1', 'tasks', 'create', True)
```

#### Performance Monitoring

```python
from agent_orchestra.logging_utils import performance_logging

@performance_logging(threshold_seconds=1.0)
def slow_operation():
    # This will be logged if it takes > 1 second
    pass

# Or use context manager
with performance_logger.time_operation('database_query'):
    result = db.query(sql)
```

### Security Practices

#### Authentication & Authorization

```python
from agent_orchestra.security_utils import SecurityMiddleware, Permission

# Setup security
middleware = SecurityMiddleware(context_builder, auditor)

# Process requests
context = middleware.process_request(headers, source_ip)

# Check permissions
middleware.require_permission(context, Permission.CREATE_TASKS, 'tasks')
```

#### Input Validation

```python
from agent_orchestra.validation import validate_task_type, validate_timeout
from agent_orchestra.security_utils import InputSanitizer

# Validate and sanitize inputs
task_type = validate_task_type(user_input['type'])
timeout = validate_timeout(user_input.get('timeout'))
clean_data = InputSanitizer.validate_user_input(user_input)
```

## Testing Utilities

The `testing_utils.py` module provides comprehensive testing support:

### Mock Agents

```python
from agent_orchestra.testing_utils import MockAgent

# Create mock agent with specific behavior
agent = MockAgent(
    capabilities=['cpu_intensive', 'general'],
    failure_rate=0.1,
    processing_delay=0.5
)

# Use in tests
result = await agent.execute_task(task)
```

### Test Scenarios

```python
from agent_orchestra.testing_utils import create_test_scenario

# Create pre-configured test scenario
harness = create_test_scenario(
    scenario_name="load_test",
    num_agents=5,
    num_tasks=20,
    agent_failure_rate=0.05
)

# Access components
agents = harness.mock_agents
tasks = harness.submitted_tasks
```

### Configuration Testing

```python
from agent_orchestra.testing_utils import ConfigurationTestHelper

# Test with various configurations
config_helper = ConfigurationTestHelper()

# Invalid configuration
invalid_config = config_helper.create_invalid_config()

# Minimal valid configuration
minimal_config = config_helper.create_minimal_config()
```

## Performance Considerations

### Async/Await Patterns

- Use `asyncio` for I/O-bound operations
- Prefer `async`/`await` over threading for concurrency
- Use connection pooling for external services

### Resource Management

- Implement proper cleanup in `__enter__`/`__exit__` methods
- Use context managers for resource handling
- Monitor memory usage in long-running processes

### Caching

- Cache expensive computations and lookups
- Use TTL-based caching for dynamic data
- Implement cache invalidation strategies

## Debugging

### Logging Configuration

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use structured logging
from agent_orchestra.logging_utils import setup_logging
logger = setup_logging(level='DEBUG', format_type='json')
```

### Common Issues

1. **Agent Connection Issues**: Check network connectivity and authentication
2. **Task Routing Failures**: Verify agent capabilities match task requirements
3. **Performance Problems**: Use performance logging to identify bottlenecks
4. **Memory Leaks**: Monitor resource usage and implement proper cleanup

### Debugging Tools

- Use `pytest --pdb` to drop into debugger on test failures
- Enable verbose logging with environment variables
- Use the built-in health checks for system status

## Contributing

### Pull Request Process

1. Create a feature branch from `master`
2. Make your changes following the coding standards
3. Add/update tests for your changes
4. Update documentation as needed
5. Run the full test suite
6. Submit a pull request with clear description

### Commit Message Format

Use conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build process or auxiliary tool changes

Examples:
```
feat(validation): add email validation utility
fix(router): handle empty agent pool gracefully
test(security): add comprehensive auth tests
docs(api): update task creation examples
```

### Release Process

1. Update version in `__init__.py`
2. Update `CHANGELOG.md`
3. Create and push git tag
4. CI/CD will handle the rest

## Best Practices Summary

1. **Write Tests First**: TDD approach ensures better design
2. **Use Type Hints**: Improves code clarity and catches errors early
3. **Log Appropriately**: Use structured logging with appropriate levels
4. **Handle Errors Gracefully**: Distinguish between retryable and permanent errors
5. **Validate All Inputs**: Never trust user or external system input
6. **Monitor Performance**: Use timing and memory monitoring
7. **Document Decisions**: Update documentation for architectural changes
8. **Review Security**: Consider security implications of all changes

## Resources

- [Python Async Programming](https://docs.python.org/3/library/asyncio.html)
- [Testing with pytest](https://docs.pytest.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Security Best Practices](./security.md)

## Getting Help

- Check the [troubleshooting guide](./troubleshooting.md)
- Review existing issues in the issue tracker
- Ask questions in team channels
- Consult the API documentation