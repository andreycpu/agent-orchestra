# 🎭 Agent Orchestra

[![Build Status](https://github.com/andreycpu/agent-orchestra/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/andreycpu/agent-orchestra/actions)
[![Security Scan](https://github.com/andreycpu/agent-orchestra/workflows/Security%20Scanning/badge.svg)](https://github.com/andreycpu/agent-orchestra/actions)
[![Coverage Status](https://codecov.io/gh/andreycpu/agent-orchestra/branch/main/graph/badge.svg)](https://codecov.io/gh/andreycpu/agent-orchestra)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, multi-agent orchestration framework for coordinating distributed AI agents with enterprise-grade features including comprehensive monitoring, error recovery, and security.

## ✨ Key Features

### 🚀 **Core Orchestration**
- **Intelligent Task Routing** - Advanced algorithms for optimal agent selection
- **Parallel Execution** - Concurrent task processing with resource management
- **Dynamic Load Balancing** - Adaptive workload distribution across agents
- **Dependency Management** - Task dependencies and execution ordering

### 🛡️ **Enterprise Security**
- **JWT Authentication** - Secure token-based authentication
- **Role-Based Access Control** - Fine-grained permissions system
- **Rate Limiting** - Configurable request throttling
- **Security Auditing** - Comprehensive security event logging

### 📊 **Monitoring & Observability**
- **Real-time Metrics** - Prometheus-compatible metrics export
- **Health Checks** - Comprehensive system health monitoring
- **Performance Profiling** - Built-in performance analysis tools
- **Structured Logging** - JSON-formatted logs with context

### 🔧 **Reliability & Recovery**
- **Circuit Breaker Pattern** - Automatic failure detection and recovery
- **Adaptive Retry Logic** - Intelligent retry mechanisms with backoff
- **Graceful Degradation** - Continues operation during partial failures
- **State Persistence** - Redis-based state management

### ⚙️ **Developer Experience**
- **Rich CLI Interface** - Beautiful command-line tools with colors
- **OpenAPI Documentation** - Complete REST API specification
- **Docker Support** - Multi-stage Docker builds for all environments
- **Comprehensive Examples** - Production-ready example implementations

## 📦 Installation

### PyPI Installation
```bash
pip install agent-orchestra
```

### Development Installation
```bash
git clone https://github.com/andreycpu/agent-orchestra.git
cd agent-orchestra
pip install -e ".[dev]"
```

### Docker Installation
```bash
docker pull ghcr.io/andreycpu/agent-orchestra:latest
```

## 🚀 Quick Start

### Basic Example

```python
import asyncio
from agent_orchestra import Orchestra, Agent, setup_logging
from agent_orchestra.types import Task, TaskPriority

# Setup structured logging
setup_logging(level="INFO", format_type="console")

async def text_processor(task_data):
    """Simple text processing function"""
    text = task_data.get("text", "")
    return {
        "original": text,
        "uppercase": text.upper(),
        "word_count": len(text.split()),
        "processed": True
    }

async def main():
    # Create orchestra with async context manager
    async with Orchestra() as orchestra:
        print("🎭 Orchestra started!")
        
        # Create and register an agent
        agent = Agent(
            agent_id="text_worker_1",
            name="Text Processing Agent",
            capabilities=["text_processing"]
        )
        
        # Register task handler
        agent.register_handler("text_processing", text_processor)
        
        # Register agent with orchestra
        orchestra.register_agent(agent)
        
        # Submit a task
        task = Task(
            type="text_processing",
            data={"text": "Hello, Agent Orchestra!"},
            priority=TaskPriority.HIGH
        )
        
        task_id = await orchestra.submit_task(task.__dict__)
        print(f"📝 Submitted task: {task_id}")
        
        # Wait for result
        result = await orchestra.wait_for_task(task_id, timeout=30.0)
        
        if result.success:
            print("✅ Task completed successfully!")
            print(f"📊 Result: {result.result}")
        else:
            print(f"❌ Task failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Production Example

For a complete production-ready example with monitoring, health checks, and error recovery, see [`examples/production_example.py`](examples/production_example.py).

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client API    │    │   Agent 1       │    │   Agent N       │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Task Submit │ │    │ │Task Handler │ │    │ │Task Handler │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
     ┌────▼──────────────────────▼──────────────────────▼────┐
     │                Orchestra Core                         │
     │                                                       │
     │ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │
     │ │Task Router  │ │State Manager│ │Failure Handler  │   │
     │ └─────────────┘ └─────────────┘ └─────────────────┘   │
     │                                                       │
     │ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │
     │ │Event Bus    │ │Security     │ │Performance      │   │
     │ └─────────────┘ │Manager      │ │Profiler         │   │
     │                 └─────────────┘ └─────────────────┘   │
     └───────────────────────┬───────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Redis Store   │
                    │                 │
                    │ • Task State    │
                    │ • Agent Registry│
                    │ • Metrics       │
                    └─────────────────┘
```

### Core Components

- **🎭 Orchestra**: Main coordinator managing the entire system
- **🤖 Agent**: Individual workers with specific capabilities
- **🚦 TaskRouter**: Intelligent routing with performance optimization
- **💾 StateManager**: Redis-backed persistent state management
- **🛠️ FailureHandler**: Circuit breakers and retry mechanisms
- **📊 Monitor**: Real-time metrics and health monitoring
- **🔒 Security**: Authentication, authorization, and audit logging

## 📖 Documentation

### Configuration

Agent Orchestra supports both environment variables and configuration files:

```yaml
# config.yaml
version: "1.0.0"
environment: "production"

orchestra:
  max_concurrent_tasks: 100
  task_timeout_default: 300
  heartbeat_interval: 30

redis:
  url: "redis://localhost:6379/0"
  max_connections: 50

security:
  jwt:
    secret: "your-secret-key"
    expiry_hours: 24
  authentication:
    enabled: true
  rate_limiting:
    enabled: true
    requests_per_minute: 100

monitoring:
  enabled: true
  metrics:
    prometheus:
      enabled: true
      port: 9090
  logging:
    level: "INFO"
    format: "json"
```

### CLI Usage

```bash
# Start the orchestra with monitoring
agent-orchestra start --config config.yaml --monitor

# Check system health
agent-orchestra health check

# View system metrics
agent-orchestra metrics --format json

# Run performance analysis
agent-orchestra performance profile --duration 60

# Validate configuration
agent-orchestra config validate config.yaml

# Export system data
agent-orchestra export metrics --output metrics.json
```

### Health Checks

```python
from agent_orchestra import HealthCheckManager
from agent_orchestra.health_checks import SystemResourcesHealthCheck

health_manager = HealthCheckManager()

# Register system health check
health_manager.register_health_check(
    SystemResourcesHealthCheck(
        cpu_threshold=80.0,
        memory_threshold=85.0,
        disk_threshold=90.0
    )
)

# Run all health checks
results = await health_manager.run_all_health_checks()
overall_status = health_manager.get_overall_health()
```

### Performance Monitoring

```python
from agent_orchestra import profile, get_profiler

# Profile a function
@profile(name="data_processing", threshold=1.0)
async def process_data(data):
    # Your processing logic here
    return processed_data

# Get performance statistics
profiler = get_profiler()
stats = profiler.export_stats()
print(f"Total function calls: {stats['system_metrics']['total_function_calls']}")
```

## 🐳 Docker Usage

### Development
```bash
docker build --target development -t agent-orchestra:dev .
docker run -p 8080:8080 -p 5678:5678 agent-orchestra:dev
```

### Production
```bash
docker build --target production -t agent-orchestra:prod .
docker run -p 8080:8080 agent-orchestra:prod
```

### Using Docker Compose
```yaml
version: '3.8'
services:
  agent-orchestra:
    image: ghcr.io/andreycpu/agent-orchestra:latest
    ports:
      - "8080:8080"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent_orchestra --cov-report=html

# Run integration tests only
pytest -m integration

# Run performance tests
pytest tests/performance/ -v
```

## 🔧 Development

### Setup Development Environment
```bash
git clone https://github.com/andreycpu/agent-orchestra.git
cd agent-orchestra

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code quality checks
make lint
make type-check
make security-scan
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`make lint test`)
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📈 Performance

Agent Orchestra is designed for high-performance scenarios:

- **Throughput**: 10,000+ tasks/second with proper Redis setup
- **Latency**: Sub-millisecond task routing overhead
- **Scalability**: Horizontal scaling across multiple instances
- **Memory**: Efficient memory usage with configurable limits

## 🛡️ Security

Security is a top priority:

- **Authentication**: JWT-based with configurable expiry
- **Authorization**: Role-based access control
- **Rate Limiting**: Protection against abuse
- **Input Validation**: Pydantic-based validation
- **Audit Logging**: Comprehensive security event logging
- **Dependency Scanning**: Automated vulnerability detection

## 📊 Monitoring

### Metrics Available

- Task execution rates and latency
- Agent health and performance
- System resource utilization  
- Error rates and failure patterns
- Queue depths and processing times

### Integration

- **Prometheus**: Native metrics export
- **Grafana**: Pre-built dashboards available
- **ELK Stack**: Structured JSON logging
- **Custom**: Extensible metrics framework

## 🤝 Support

- **Documentation**: [Full docs](https://agent-orchestra.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/andreycpu/agent-orchestra/issues)
- **Discussions**: [GitHub Discussions](https://github.com/andreycpu/agent-orchestra/discussions)
- **Security**: [Security Policy](SECURITY.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [asyncio](https://docs.python.org/3/library/asyncio.html) for high-performance async operations
- Powered by [Redis](https://redis.io/) for reliable state management
- Monitored with [Prometheus](https://prometheus.io/) metrics
- Secured with industry-standard practices

---

<div align="center">

**[Website](https://agent-orchestra.dev) • [Documentation](https://docs.agent-orchestra.dev) • [Examples](examples/) • [Contributing](CONTRIBUTING.md)**

Made with ❤️ by the Agent Orchestra team

</div>