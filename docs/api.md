# API Documentation

## Core Classes

### Orchestra

Main orchestration engine.

```python
from agent_orchestra import Orchestra

orchestra = Orchestra(
    redis_url="redis://localhost:6379",
    max_concurrent_tasks=100,
    task_timeout_default=300
)
```

#### Methods

- `async start()`: Start the orchestra
- `async stop()`: Stop the orchestra  
- `register_agent(agent)`: Register an agent
- `unregister_agent(agent_id)`: Remove an agent
- `async submit_task(task_data)`: Submit a task for execution
- `async wait_for_task(task_id, timeout=None)`: Wait for task completion
- `async get_status()`: Get system status

### Agent

Individual worker that executes tasks.

```python
from agent_orchestra import Agent

agent = Agent("worker-1", capabilities=["text_processing"])
agent.register_task_handler("text_processing", handler_function)
```

#### Methods

- `register_task_handler(task_type, handler)`: Register task handler
- `can_handle_task(task)`: Check if agent can handle task
- `async execute_task(task)`: Execute a task
- `get_info()`: Get agent information

### Task Types

#### Task

```python
{
    "type": "text_processing",
    "data": {"text": "Hello World"},
    "priority": "normal",  # low, normal, high, urgent
    "timeout": 300,
    "max_retries": 3
}
```

#### ExecutionResult

```python
{
    "task_id": "uuid",
    "success": True,
    "result": {"processed": "data"},
    "execution_time": 1.23,
    "error": None
}
```

## Configuration

### YAML Configuration

```yaml
orchestra:
  max_concurrent_tasks: 100
  task_timeout_default: 300
  heartbeat_interval: 30

redis:
  host: "localhost"
  port: 6379
  db: 0

monitoring:
  enabled: true
  metrics_port: 8080
```

### Environment Variables

- `REDIS_URL`: Redis connection URL
- `LOG_LEVEL`: Logging level
- `MAX_CONCURRENT_TASKS`: Task limit
- `METRICS_ENABLED`: Enable metrics

## CLI Commands

### Start Orchestra

```bash
python -m agent_orchestra.cli start --config config.yaml
```

### Monitor System

```bash
python -m agent_orchestra.cli monitor --continuous
```

### Export Metrics

```bash
python -m agent_orchestra.cli export metrics --format prometheus
```

### Submit Task

```bash
python -m agent_orchestra.cli task text_processing --data '{"text": "hello"}'
```

## Events

Subscribe to system events:

```python
from agent_orchestra.events import EventBus

event_bus = EventBus()

@event_bus.subscribe("task.completed")
def on_task_completed(event):
    print(f"Task {event.data['task_id']} completed")
```

Event types:
- `task.submitted`
- `task.started`
- `task.completed`
- `task.failed`
- `agent.registered`
- `agent.status_changed`

## Error Handling

```python
from agent_orchestra.exceptions import TaskExecutionError

try:
    result = await orchestra.submit_task(task_data)
except TaskExecutionError as e:
    print(f"Task failed: {e}")
```

Exception types:
- `TaskExecutionError`
- `TaskTimeoutError`
- `AgentUnavailableError`
- `TaskRoutingError`