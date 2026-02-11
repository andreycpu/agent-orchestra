# Troubleshooting Guide

## Common Issues

### Installation Problems

#### ImportError: No module named 'agent_orchestra'

**Solution:**
```bash
pip install -e .
# or
pip install agent-orchestra
```

#### Redis Connection Failed

**Symptoms:** `ConnectionError: Error connecting to Redis`

**Solutions:**
1. Install and start Redis:
   ```bash
   # macOS
   brew install redis && brew services start redis
   
   # Ubuntu
   sudo apt-get install redis-server
   
   # Docker
   docker run -d -p 6379:6379 redis:7-alpine
   ```

2. Check Redis is running:
   ```bash
   redis-cli ping  # Should return PONG
   ```

### Runtime Issues

#### Tasks Not Executing

**Possible Causes:**
- No agents registered
- Agent capabilities don't match task type
- All agents are busy
- Circular task dependencies

**Debug Steps:**
```python
# Check registered agents
status = await orchestra.get_status()
print(f"Agents: {status['registered_agents']}")

# Check agent capabilities
agents = orchestra.get_agents()
for agent in agents:
    print(f"Agent {agent.id}: {[cap.name for cap in agent.capabilities]}")
```

#### High Memory Usage

**Debug:**
```python
from agent_orchestra.profiler import profiler

# Enable memory profiling
profiler.memory_profiler.enable()

# Take snapshots
profiler.memory_profiler.take_snapshot("before_tasks")
# ... run tasks ...
profiler.memory_profiler.take_snapshot("after_tasks")

# Check trends
trend = profiler.memory_profiler.get_memory_trend()
print(trend)
```

#### Performance Issues

**Use benchmarking:**
```bash
python examples/benchmark.py
python examples/stress_test.py --agents 5 --tasks 1000
```

**Enable profiling:**
```python
from agent_orchestra.profiler import profile

@profile("my_task_handler")
async def my_handler(data):
    # Your task logic
    return result
```

### Error Messages

#### "Circuit breaker is open"

**Meaning:** An agent has failed too many times and is temporarily disabled.

**Solution:**
```python
# Reset circuit breaker
orchestra.failure_handler.reset_circuit_breaker("agent-id")

# Or check failure statistics
stats = orchestra.failure_handler.get_failure_statistics()
print(stats)
```

#### "Task timed out"

**Solutions:**
1. Increase task timeout:
   ```python
   await orchestra.submit_task({
       "type": "slow_task",
       "timeout": 600  # 10 minutes
   })
   ```

2. Optimize task handler:
   - Use async/await properly
   - Avoid blocking operations
   - Break down large tasks

#### "Permission denied" in security mode

**Check user permissions:**
```python
# In a secure setup
from agent_orchestra.security import Permission

# Check if user has permission
if security_manager.check_permission(user, Permission.TASK_SUBMIT):
    # Submit task
    pass
```

## Debugging Tools

### Enable Debug Logging

```python
import structlog
import logging

# Set debug level
logging.basicConfig(level=logging.DEBUG)

# Or in configuration
{
    "logging": {
        "level": "DEBUG"
    }
}
```

### Monitor with CLI

```bash
# Continuous monitoring
python -m agent_orchestra.cli monitor --continuous

# Export metrics
python -m agent_orchestra.cli export metrics --format json
```

### Check System Resources

```bash
# Memory usage
python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'CPU: {psutil.cpu_percent()}%')
"

# Redis memory
redis-cli info memory
```

## Performance Optimization

### Tuning Parameters

```yaml
orchestra:
  max_concurrent_tasks: 100  # Increase for higher throughput
  task_timeout_default: 300  # Adjust based on task complexity
  heartbeat_interval: 30     # Reduce for faster failure detection
```

### Agent Optimization

- Keep task handlers lightweight
- Use connection pooling for external services
- Cache frequently accessed data
- Implement proper error handling

### System Optimization

- Use SSD storage
- Ensure sufficient RAM
- Monitor network latency
- Optimize Redis configuration

## Getting Help

If you're still experiencing issues:

1. Check the [FAQ](faq.md)
2. Enable debug logging
3. Collect system information:
   ```bash
   python -c "
   import platform
   import agent_orchestra
   print(f'OS: {platform.platform()}')
   print(f'Python: {platform.python_version()}')
   print(f'Agent Orchestra: {agent_orchestra.__version__}')
   "
   ```
4. Create an issue on GitHub with the information above