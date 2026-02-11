# Agent Orchestra

A multi-agent orchestration framework for coordinating distributed AI agents.

## Features

- Agent coordination and communication
- Intelligent task routing
- Parallel execution management
- Robust state management
- Comprehensive failure handling
- Real-time monitoring and observability

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from agent_orchestra import Orchestra, Agent

# Create an orchestra
orchestra = Orchestra()

# Register agents
orchestra.register_agent(Agent("worker-1", capabilities=["text_processing"]))
orchestra.register_agent(Agent("worker-2", capabilities=["image_processing"]))

# Execute a task
result = await orchestra.execute_task({
    "type": "text_processing",
    "data": "Hello, World!"
})
```

## Architecture

- **Orchestra**: Main coordinator that manages agents and routes tasks
- **Agent**: Individual worker that can execute specific types of tasks
- **TaskRouter**: Intelligent routing system for optimal task assignment
- **StateManager**: Persistent state management across the agent network
- **FailureHandler**: Robust error handling and recovery mechanisms