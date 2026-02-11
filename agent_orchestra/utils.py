"""
Utility functions for Agent Orchestra
"""
import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
import json
import hashlib
import structlog

logger = structlog.get_logger(__name__)


def generate_task_id(task_type: str, data: Dict[str, Any]) -> str:
    """Generate a deterministic task ID based on task type and data.
    
    Args:
        task_type: The type of task
        data: Task data dictionary
        
    Returns:
        A 16-character hexadecimal task ID
        
    Example:
        >>> generate_task_id("process_data", {"input": "test"})
        'a1b2c3d4e5f6g7h8'
    """
    content = f"{task_type}:{json.dumps(data, sort_keys=True)}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def calculate_task_hash(task_type: str, data: Dict[str, Any]) -> str:
    """Calculate a hash for task deduplication"""
    content = f"{task_type}:{json.dumps(data, sort_keys=True)}"
    return hashlib.md5(content.encode()).hexdigest()


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable duration string
        
    Examples:
        >>> format_duration(0.5)
        '500ms'
        >>> format_duration(65.0)
        '1.1m'
        >>> format_duration(3661.0)
        '1.0h'
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def parse_capability_string(capability: str) -> Dict[str, Any]:
    """Parse capability string into structured format"""
    # Simple parser for capabilities like "text_processing:cpu=2,memory=1GB"
    if ":" not in capability:
        return {"name": capability, "requirements": {}}
    
    name, requirements_str = capability.split(":", 1)
    requirements = {}
    
    for req in requirements_str.split(","):
        if "=" in req:
            key, value = req.split("=", 1)
            # Try to parse numeric values
            try:
                if value.isdigit():
                    value = int(value)
                elif "." in value and value.replace(".", "").isdigit():
                    value = float(value)
            except:
                pass
            requirements[key.strip()] = value.strip()
    
    return {"name": name.strip(), "requirements": requirements}


class AsyncRetry:
    """Async retry decorator with exponential backoff"""
    
    def __init__(
        self, 
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.exceptions = exceptions
    
    def __call__(self, func: Callable):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e
                    
                    if attempt == self.max_attempts - 1:
                        # Last attempt failed
                        raise e
                    
                    # Calculate delay
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    
                    logger.warning(
                        "Function failed, retrying",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=self.max_attempts,
                        delay=delay,
                        error=str(e)
                    )
                    
                    await asyncio.sleep(delay)
            
            # Should never reach here, but just in case
            raise last_exception
        
        return wrapper


class RateLimiter:
    """Simple token bucket rate limiter"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens, returns True if successful"""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1):
        """Wait until tokens are available"""
        while not await self.acquire(tokens):
            # Calculate how long to wait
            needed_tokens = tokens - self.tokens
            wait_time = needed_tokens / self.rate
            await asyncio.sleep(min(wait_time, 0.1))


class MetricsCollector:
    """Simple metrics collection utility"""
    
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._gauges: Dict[str, float] = {}
        self._start_time = time.time()
    
    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter"""
        key = self._make_key(name, tags)
        self._counters[key] = self._counters.get(key, 0) + value
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a value in histogram"""
        key = self._make_key(name, tags)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value"""
        key = self._make_key(name, tags)
        self._gauges[key] = value
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create metric key from name and tags"""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        uptime = time.time() - self._start_time
        
        # Calculate histogram statistics
        histogram_stats = {}
        for key, values in self._histograms.items():
            if values:
                values.sort()
                n = len(values)
                histogram_stats[key] = {
                    "count": n,
                    "sum": sum(values),
                    "avg": sum(values) / n,
                    "min": values[0],
                    "max": values[-1],
                    "p50": values[n // 2],
                    "p90": values[int(n * 0.9)],
                    "p99": values[int(n * 0.99)] if n > 100 else values[-1]
                }
        
        return {
            "uptime": uptime,
            "counters": self._counters.copy(),
            "histograms": histogram_stats,
            "gauges": self._gauges.copy()
        }
    
    def reset(self):
        """Reset all metrics"""
        self._counters.clear()
        self._histograms.clear()
        self._gauges.clear()
        self._start_time = time.time()


class TaskScheduler:
    """Simple task scheduler for delayed and recurring tasks"""
    
    def __init__(self):
        self._scheduled_tasks: Dict[str, Dict] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
    
    def schedule_once(
        self, 
        task_id: str, 
        func: Callable, 
        delay: float, 
        *args, 
        **kwargs
    ):
        """Schedule a task to run once after delay"""
        run_at = time.time() + delay
        
        self._scheduled_tasks[task_id] = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "run_at": run_at,
            "recurring": False,
            "interval": None
        }
    
    def schedule_recurring(
        self,
        task_id: str,
        func: Callable,
        interval: float,
        *args,
        **kwargs
    ):
        """Schedule a task to run repeatedly at interval"""
        run_at = time.time() + interval
        
        self._scheduled_tasks[task_id] = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "run_at": run_at,
            "recurring": True,
            "interval": interval
        }
    
    def cancel_task(self, task_id: str):
        """Cancel a scheduled task"""
        self._scheduled_tasks.pop(task_id, None)
    
    async def start(self):
        """Start the scheduler"""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop(self):
        """Stop the scheduler"""
        if not self._running:
            return
        
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            now = time.time()
            to_run = []
            
            # Find tasks ready to run
            for task_id, task_info in self._scheduled_tasks.items():
                if task_info["run_at"] <= now:
                    to_run.append((task_id, task_info))
            
            # Execute ready tasks
            for task_id, task_info in to_run:
                try:
                    func = task_info["func"]
                    args = task_info["args"]
                    kwargs = task_info["kwargs"]
                    
                    # Run the function
                    if asyncio.iscoroutinefunction(func):
                        await func(*args, **kwargs)
                    else:
                        func(*args, **kwargs)
                    
                    # Handle recurring tasks
                    if task_info["recurring"]:
                        # Schedule next run
                        task_info["run_at"] = now + task_info["interval"]
                    else:
                        # Remove one-time task
                        del self._scheduled_tasks[task_id]
                
                except Exception as e:
                    logger.error(
                        "Scheduled task failed",
                        task_id=task_id,
                        error=str(e)
                    )
                    
                    # Remove failed one-time tasks
                    if not task_info["recurring"]:
                        del self._scheduled_tasks[task_id]
            
            await asyncio.sleep(0.1)


def create_task_batch(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a batch of related tasks"""
    batch_id = generate_task_id("batch", {"timestamp": datetime.utcnow().isoformat()})
    
    for i, task in enumerate(tasks):
        task["metadata"] = task.get("metadata", {})
        task["metadata"]["batch_id"] = batch_id
        task["metadata"]["batch_index"] = i
        task["metadata"]["batch_size"] = len(tasks)
    
    return tasks