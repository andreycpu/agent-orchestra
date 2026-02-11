#!/bin/bash
# Benchmark script for Agent Orchestra

set -e

echo "🎭 Agent Orchestra Benchmark Script"
echo "==================================="

# Check if Python environment is set up
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed"
    exit 1
fi

# Check if Agent Orchestra is installed
if ! python3 -c "import agent_orchestra" &> /dev/null; then
    echo "❌ Agent Orchestra is not installed"
    echo "Run: pip install -e ."
    exit 1
fi

# Check if Redis is running
if ! redis-cli ping &> /dev/null; then
    echo "⚠️ Redis is not running. Starting Redis with Docker..."
    docker run -d --name agent-orchestra-redis -p 6379:6379 redis:7-alpine
    sleep 2
fi

echo "✅ Prerequisites check passed"

# Run different benchmark scenarios
echo ""
echo "📊 Running Performance Benchmarks"
echo "---------------------------------"

echo "1. Basic throughput test..."
python3 -c "
import asyncio
from agent_orchestra import Orchestra, Agent

async def simple_handler(data):
    return {'result': 'processed', 'input': data}

async def run_basic():
    orchestra = Orchestra(max_concurrent_tasks=10)
    agent = Agent('bench-agent', capabilities=['simple_task'])
    agent.register_task_handler('simple_task', simple_handler)
    orchestra.register_agent(agent)
    
    await orchestra.start()
    
    # Submit 50 simple tasks
    import time
    start = time.time()
    
    tasks = []
    for i in range(50):
        task_id = await orchestra.submit_task({
            'type': 'simple_task',
            'data': {'index': i}
        })
        tasks.append(task_id)
    
    # Wait for completion
    for task_id in tasks:
        await orchestra.wait_for_task(task_id, timeout=30)
    
    duration = time.time() - start
    throughput = 50 / duration
    
    await orchestra.stop()
    
    print(f'✅ Basic throughput: {throughput:.2f} tasks/second')
    print(f'   Total time: {duration:.2f} seconds')

asyncio.run(run_basic())
"

echo ""
echo "2. Concurrent agents test..."
python3 -c "
import asyncio
from agent_orchestra import Orchestra, Agent

async def worker_handler(data):
    await asyncio.sleep(0.1)  # Simulate work
    return {'worker': data.get('worker_id'), 'processed': data}

async def run_concurrent():
    orchestra = Orchestra(max_concurrent_tasks=20)
    
    # Create multiple agents
    for i in range(3):
        agent = Agent(f'worker-{i}', capabilities=['work_task'])
        agent.register_task_handler('work_task', worker_handler)
        orchestra.register_agent(agent)
    
    await orchestra.start()
    
    import time
    start = time.time()
    
    # Submit tasks concurrently
    tasks = []
    for i in range(30):
        task_id = await orchestra.submit_task({
            'type': 'work_task',
            'data': {'task_id': i, 'worker_id': f'worker-{i % 3}'}
        })
        tasks.append(task_id)
    
    # Wait for all
    results = []
    for task_id in tasks:
        result = await orchestra.wait_for_task(task_id, timeout=30)
        results.append(result)
    
    duration = time.time() - start
    successful = sum(1 for r in results if r.success)
    throughput = successful / duration
    
    await orchestra.stop()
    
    print(f'✅ Concurrent test: {throughput:.2f} tasks/second')
    print(f'   Successful: {successful}/{len(tasks)}')
    print(f'   Total time: {duration:.2f} seconds')

asyncio.run(run_concurrent())
"

echo ""
echo "3. Running comprehensive benchmark..."
python3 examples/benchmark.py

echo ""
echo "🎉 Benchmark completed!"
echo ""
echo "💡 Tips for performance optimization:"
echo "  - Increase max_concurrent_tasks for higher throughput"
echo "  - Use Redis for distributed deployments"
echo "  - Profile your task handlers for bottlenecks"
echo "  - Monitor system resources during peak loads"