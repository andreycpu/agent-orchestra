"""
Utility functions for Agent Orchestra
"""
import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import deque
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


class AdaptiveRateLimiter:
    """Enhanced rate limiter with adaptive capacity and detailed metrics"""
    
    def __init__(self, initial_rate: float, initial_capacity: int, name: str = "adaptive"):
        self.rate = initial_rate
        self.capacity = initial_capacity
        self.tokens = initial_capacity
        self.last_refill = time.time()
        self.name = name
        self._lock = asyncio.Lock()
        
        # Adaptive parameters
        self.min_rate = initial_rate * 0.1
        self.max_rate = initial_rate * 10.0
        self.rate_adjustment_factor = 0.1
        
        # Metrics
        self._requests_allowed = 0
        self._requests_denied = 0
        self._total_wait_time = 0.0
        self._recent_success_rate = deque(maxlen=100)
        
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens with adaptive rate adjustment"""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                self._requests_allowed += 1
                self._recent_success_rate.append(1)
                return True
            
            self._requests_denied += 1
            self._recent_success_rate.append(0)
            self._adjust_rate()
            return False
    
    def _adjust_rate(self):
        """Adjust rate based on recent success rate"""
        if len(self._recent_success_rate) >= 20:
            success_rate = sum(self._recent_success_rate) / len(self._recent_success_rate)
            
            if success_rate < 0.8:  # Too many rejections, increase rate
                self.rate = min(self.max_rate, self.rate * (1 + self.rate_adjustment_factor))
            elif success_rate > 0.95:  # Very few rejections, can decrease rate
                self.rate = max(self.min_rate, self.rate * (1 - self.rate_adjustment_factor))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive rate limiter metrics"""
        total = self._requests_allowed + self._requests_denied
        return {
            "name": self.name,
            "current_rate": self.rate,
            "capacity": self.capacity,
            "tokens_available": self.tokens,
            "total_requests": total,
            "success_rate": self._requests_allowed / total if total > 0 else 0.0,
            "recent_success_rate": sum(self._recent_success_rate) / len(self._recent_success_rate) if self._recent_success_rate else 0.0
        }


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


def safe_dict_get(data: Dict[str, Any], keys: str, default: Any = None, separator: str = ".") -> Any:
    """Safely get nested dictionary value using dot notation
    
    Args:
        data: Dictionary to search
        keys: Dot-separated key path (e.g., "user.profile.name")
        default: Default value if key not found
        separator: Key separator character
        
    Returns:
        Value at the key path or default if not found
        
    Examples:
        >>> data = {"user": {"profile": {"name": "John"}}}
        >>> safe_dict_get(data, "user.profile.name")
        'John'
        >>> safe_dict_get(data, "user.profile.age", 25)
        25
    """
    if not isinstance(data, dict):
        return default
        
    key_list = keys.split(separator)
    current = data
    
    for key in key_list:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current


def validate_json_schema(data: Any, schema: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Simple JSON schema validation
    
    Args:
        data: Data to validate
        schema: Schema dictionary with type requirements
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> schema = {"name": str, "age": int, "active": bool}
        >>> validate_json_schema({"name": "John", "age": 30, "active": True}, schema)
        (True, [])
    """
    errors = []
    
    if not isinstance(data, dict):
        return False, ["Data must be a dictionary"]
    
    # Check required fields
    for field, expected_type in schema.items():
        if field not in data:
            errors.append(f"Missing required field: {field}")
            continue
            
        if not isinstance(data[field], expected_type):
            errors.append(
                f"Field '{field}' must be of type {expected_type.__name__}, "
                f"got {type(data[field]).__name__}"
            )
    
    return len(errors) == 0, errors


def sanitize_data(data: Dict[str, Any], sensitive_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Sanitize dictionary by redacting sensitive information
    
    Args:
        data: Dictionary to sanitize
        sensitive_keys: List of keys to redact (case-insensitive)
        
    Returns:
        Sanitized dictionary copy
        
    Example:
        >>> sanitize_data({"name": "John", "password": "secret123"})
        {'name': 'John', 'password': '[REDACTED]'}
    """
    if sensitive_keys is None:
        sensitive_keys = [
            "password", "token", "secret", "key", "auth", "credential",
            "api_key", "access_token", "refresh_token", "private_key"
        ]
    
    sensitive_keys_lower = [key.lower() for key in sensitive_keys]
    sanitized = {}
    
    for key, value in data.items():
        if key.lower() in sensitive_keys_lower:
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_data(value, sensitive_keys)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_data(item, sensitive_keys) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized


def calculate_memory_usage() -> Dict[str, int]:
    """Calculate current memory usage in bytes
    
    Returns:
        Dictionary with memory usage information
        
    Note:
        Requires psutil package for accurate measurements.
        Falls back to basic measurements if not available.
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available
        }
    except ImportError:
        # Fallback using tracemalloc if psutil not available
        try:
            import tracemalloc
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                return {
                    "current": current,
                    "peak": peak,
                    "rss": 0,
                    "vms": 0,
                    "percent": 0,
                    "available": 0
                }
        except:
            pass
        
        # Ultimate fallback
        return {
            "rss": 0,
            "vms": 0,
            "percent": 0,
            "available": 0,
            "current": 0,
            "peak": 0
        }


class RateLimiter:
    """Simple rate limiter using token bucket algorithm"""
    
    def __init__(self, max_tokens: int, refill_rate: float):
        """Initialize rate limiter
        
        Args:
            max_tokens: Maximum number of tokens in the bucket
            refill_rate: Number of tokens added per second
        """
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = max_tokens
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        self.last_refill = now
        
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
    
    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until tokens become available
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Time in seconds until tokens are available
        """
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries
    
    Args:
        dict1: First dictionary (base)
        dict2: Second dictionary (override values)
        
    Returns:
        Merged dictionary
        
    Example:
        >>> d1 = {"a": {"b": 1, "c": 2}}
        >>> d2 = {"a": {"c": 3, "d": 4}}
        >>> deep_merge_dicts(d1, d2)
        {'a': {'b': 1, 'c': 3, 'd': 4}}
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if (
            key in result 
            and isinstance(result[key], dict) 
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate string to maximum length with suffix
    
    Args:
        text: String to truncate
        max_length: Maximum allowed length including suffix
        suffix: Suffix to append when truncating
        
    Returns:
        Truncated string
        
    Example:
        >>> truncate_string("This is a very long text", 15)
        'This is a ve...'
    """
    if not isinstance(text, str):
        text = str(text)
    
    if len(text) <= max_length:
        return text
    
    if len(suffix) >= max_length:
        return suffix[:max_length]
    
    truncate_length = max_length - len(suffix)
    return text[:truncate_length] + suffix