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

```bash
pip install agent-orchestra
```

```python
import asyncio
from agent_orchestra import Orchestra, Agent

async def text_processor(data):
    return {"processed": data["text"].upper()}

async def main():
    # Create an orchestra
    orchestra = Orchestra()
    
    # Create and register agents
    agent = Agent("worker-1", capabilities=["text_processing"])
    agent.register_task_handler("text_processing", text_processor)
    orchestra.register_agent(agent)
    
    await orchestra.start()
    
    # Execute a task
    task_id = await orchestra.submit_task({
        "type": "text_processing",
        "data": {"text": "Hello, World!"}
    })
    
    result = await orchestra.wait_for_task(task_id)
    print(result.result)  # {"processed": "HELLO, WORLD!"}
    
    await orchestra.stop()

asyncio.run(main())
```

## Architecture

- **Orchestra**: Main coordinator that manages agents and routes tasks
- **Agent**: Individual worker that can execute specific types of tasks
- **TaskRouter**: Intelligent routing system for optimal task assignment
- **StateManager**: Persistent state management across the agent network
- **FailureHandler**: Robust error handling and recovery mechanisms