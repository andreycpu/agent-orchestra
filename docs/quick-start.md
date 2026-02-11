# Quick Start Guide

Get up and running with Agent Orchestra in minutes!

## Installation

```bash
pip install agent-orchestra
```

## Basic Example

```python
import asyncio
from agent_orchestra import Orchestra, Agent

async def hello_task(data):
    name = data.get("name", "World")
    return {"message": f"Hello, {name}!"}

async def main():
    # Create orchestra
    orchestra = Orchestra()
    
    # Create and register agent
    agent = Agent("greeter", capabilities=["greeting"])
    agent.register_task_handler("greeting", hello_task)
    orchestra.register_agent(agent)
    
    # Start orchestra
    await orchestra.start()
    
    # Submit task
    task_id = await orchestra.submit_task({
        "type": "greeting",
        "data": {"name": "Agent Orchestra"}
    })
    
    # Get result
    result = await orchestra.wait_for_task(task_id)
    print(result.result)  # {"message": "Hello, Agent Orchestra!"}
    
    # Clean up
    await orchestra.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

- [Configuration Guide](configuration.md)
- [Examples](../examples/)
- [API Documentation](api.md)