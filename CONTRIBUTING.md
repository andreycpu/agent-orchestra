# Contributing to Agent Orchestra

Thank you for your interest in contributing to Agent Orchestra! This document provides guidelines and information for contributors.

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Development Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/andreycpu/agent-orchestra.git
   cd agent-orchestra
   ```

2. **Set up development environment:**
   ```bash
   make dev-setup
   ```
   
   Or manually:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   pip install -r requirements-dev.txt
   ```

3. **Start Redis (required for full functionality):**
   ```bash
   make redis-start
   # Or manually: docker run -d --name redis -p 6379:6379 redis:7-alpine
   ```

4. **Run tests to verify setup:**
   ```bash
   make test
   ```

### Project Structure

```
agent-orchestra/
├── agent_orchestra/          # Main package
│   ├── __init__.py           # Package exports
│   ├── orchestra.py          # Main orchestration engine
│   ├── agent.py              # Agent implementation
│   ├── task_router.py        # Task routing logic
│   ├── state_manager.py      # State persistence
│   ├── failure_handler.py    # Error handling
│   ├── monitoring.py         # Observability
│   ├── events.py             # Event system
│   ├── plugins.py            # Plugin framework
│   ├── security.py           # Authentication/authorization
│   ├── cluster.py            # Distributed deployment
│   ├── config.py             # Configuration management
│   ├── migration.py          # Database migrations
│   ├── profiler.py           # Performance profiling
│   ├── exporters.py          # Metrics exporters
│   ├── utils.py              # Utility functions
│   ├── types.py              # Data models
│   ├── exceptions.py         # Exception classes
│   └── cli.py                # Command-line interface
├── tests/                    # Test suite
├── examples/                 # Example applications
├── docs/                     # Documentation
├── config/                   # Configuration files
├── scripts/                  # Utility scripts
└── .github/                  # GitHub workflows
```

## How to Contribute

### Reporting Issues

1. **Search existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide clear reproduction steps** for bugs
4. **Include environment information** (Python version, OS, etc.)

### Suggesting Features

1. **Check the roadmap** in issues or discussions
2. **Open a feature request** with:
   - Clear description of the proposed feature
   - Use cases and benefits
   - Possible implementation approach
   - Breaking changes (if any)

### Submitting Changes

#### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following coding standards
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Ensure all checks pass**
6. **Submit a pull request**

#### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

#### Commit Messages

Follow conventional commits:
- `feat: add new functionality`
- `fix: resolve bug in component`
- `docs: update API documentation`
- `test: add unit tests for router`
- `refactor: simplify error handling`
- `style: format code with black`
- `chore: update dependencies`

## Development Guidelines

### Code Style

- **Python**: Follow PEP 8, use `black` for formatting
- **Type hints**: Required for all public functions
- **Docstrings**: Use Google style docstrings
- **Line length**: 88 characters (black default)

#### Formatting Tools

```bash
make format          # Format code
make check-format    # Check formatting
make lint           # Run all linters
```

### Testing

#### Test Structure
- Unit tests in `tests/`
- Integration tests in `tests/integration/`
- Test files named `test_*.py`
- Test classes named `Test*`

#### Running Tests
```bash
make test           # Full test suite
make test-fast      # Fast tests only
pytest tests/test_agent.py -v  # Specific test file
```

#### Test Coverage
- Aim for >90% coverage
- Include both positive and negative test cases
- Test error conditions and edge cases

### Documentation

#### Code Documentation
- All public classes and functions must have docstrings
- Use Google style docstrings
- Include examples in docstrings when helpful

#### Project Documentation
- Update README.md for user-facing changes
- Update architecture docs for design changes
- Add examples for new features

### Performance

#### Guidelines
- Use async/await for I/O operations
- Profile performance-critical code
- Include benchmarks for significant changes
- Consider memory usage and leaks

#### Benchmarking
```bash
make benchmark      # Run performance benchmarks
make profile        # Profile example application
```

### Security

#### Guidelines
- Validate all inputs
- Use parameterized queries
- Follow principle of least privilege
- Review security implications

#### Security Checks
```bash
bandit -r agent_orchestra  # Security linting
```

## Plugin Development

### Creating Plugins

1. Inherit from appropriate plugin interface:
   ```python
   from agent_orchestra.plugins import TaskPlugin
   
   class MyPlugin(TaskPlugin):
       @property
       def metadata(self):
           return PluginMetadata(
               name="my_plugin",
               version="1.0.0",
               description="My custom plugin"
           )
   ```

2. Implement required methods
3. Register with plugin manager
4. Add tests and documentation

### Plugin Guidelines
- Follow single responsibility principle
- Handle errors gracefully
- Include configuration validation
- Document plugin capabilities

## Release Process

### Versioning
- Follow [Semantic Versioning](https://semver.org/)
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

### Release Checklist
1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test package
5. Create release tag
6. Publish to PyPI (maintainers only)

## Community

### Getting Help
- **Issues**: Bug reports and feature requests
- **Discussions**: Questions and community chat
- **Documentation**: Check docs/ directory
- **Examples**: See examples/ directory

### Communication
- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers get started
- Share knowledge and experiences

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Community discussions

Thank you for contributing to Agent Orchestra! 🎭✨