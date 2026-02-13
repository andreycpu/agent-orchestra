"""
Centralized logging configuration for Agent Orchestra
"""
import logging
import sys
from typing import Dict, Any, Optional
import structlog
from structlog.typing import Processor
import json
import time
import hashlib
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Set, List, Callable
import asyncio
from concurrent.futures import ThreadPoolExecutor


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    enable_colors: bool = True,
    include_caller_info: bool = True
) -> None:
    """
    Setup structured logging for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Output format ('json' or 'console')
        enable_colors: Whether to use colored output for console format
        include_caller_info: Whether to include file/line information
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )
    
    # Setup processors chain
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    
    if include_caller_info:
        processors.append(structlog.processors.CallsiteParameterAdder(
            parameters=[structlog.processors.CallsiteParameter.FILENAME,
                       structlog.processors.CallsiteParameter.LINENO,
                       structlog.processors.CallsiteParameter.FUNC_NAME]
        ))
    
    processors.append(structlog.processors.TimeStamper(fmt="ISO"))
    
    if format_type == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        if enable_colors:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=False))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


class LoggingContext:
    """Context manager for adding structured logging context"""
    
    def __init__(self, **context):
        self.context = context
        self.previous_context = None
    
    def __enter__(self):
        self.previous_context = structlog.contextvars.get_contextvars()
        structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        structlog.contextvars.clear_contextvars()
        if self.previous_context:
            structlog.contextvars.bind_contextvars(**self.previous_context)


class SecurityAuditLogger:
    """Specialized logger for security events"""
    
    def __init__(self):
        self.logger = structlog.get_logger("security.audit")
    
    def log_authentication_attempt(
        self, 
        username: str, 
        success: bool, 
        ip_address: str = None,
        user_agent: str = None,
        additional_data: Dict[str, Any] = None
    ):
        """Log authentication attempts"""
        event_data = {
            "event_type": "authentication",
            "username": username,
            "success": success,
            "ip_address": ip_address,
            "user_agent": user_agent,
        }
        
        if additional_data:
            event_data.update(additional_data)
        
        if success:
            self.logger.info("Authentication successful", **event_data)
        else:
            self.logger.warning("Authentication failed", **event_data)
    
    def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
        reason: str = None
    ):
        """Log authorization failures"""
        self.logger.warning(
            "Authorization denied",
            event_type="authorization",
            user_id=user_id,
            resource=resource,
            action=action,
            reason=reason
        )
    
    def log_security_violation(
        self,
        violation_type: str,
        description: str,
        severity: str = "medium",
        user_id: str = None,
        ip_address: str = None,
        additional_data: Dict[str, Any] = None
    ):
        """Log security violations"""
        event_data = {
            "event_type": "security_violation",
            "violation_type": violation_type,
            "description": description,
            "severity": severity,
            "user_id": user_id,
            "ip_address": ip_address
        }
        
        if additional_data:
            event_data.update(additional_data)
        
        if severity == "critical":
            self.logger.critical("Critical security violation", **event_data)
        elif severity == "high":
            self.logger.error("High severity security violation", **event_data)
        else:
            self.logger.warning("Security violation", **event_data)


class PerformanceLogger:
    """Specialized logger for performance metrics"""
    
    def __init__(self):
        self.logger = structlog.get_logger("performance")
    
    def log_task_execution(
        self,
        task_id: str,
        task_type: str,
        execution_time: float,
        success: bool,
        agent_id: str = None,
        error: str = None
    ):
        """Log task execution performance"""
        self.logger.info(
            "Task execution completed",
            event_type="task_execution",
            task_id=task_id,
            task_type=task_type,
            execution_time_ms=execution_time * 1000,
            success=success,
            agent_id=agent_id,
            error=error
        )
    
    def log_system_metrics(
        self,
        cpu_usage: float = None,
        memory_usage: float = None,
        active_tasks: int = None,
        queue_depth: int = None,
        active_agents: int = None
    ):
        """Log system performance metrics"""
        self.logger.info(
            "System metrics snapshot",
            event_type="system_metrics",
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory_usage,
            active_tasks=active_tasks,
            queue_depth=queue_depth,
            active_agents=active_agents
        )
    
    def log_slow_operation(
        self,
        operation: str,
        duration: float,
        threshold: float,
        details: Dict[str, Any] = None
    ):
        """Log operations that exceed performance thresholds"""
        event_data = {
            "event_type": "slow_operation",
            "operation": operation,
            "duration_ms": duration * 1000,
            "threshold_ms": threshold * 1000,
            "slowness_ratio": duration / threshold
        }
        
        if details:
            event_data.update(details)
        
        self.logger.warning("Slow operation detected", **event_data)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured structlog logger instance"""
    return structlog.get_logger(name)


def configure_json_serialization():
    """Configure JSON serialization for complex types"""
    def json_serializer(obj):
        """Custom JSON serializer for structlog"""
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # Custom objects
            return obj.__dict__
        else:
            return str(obj)
    
    # Monkey patch json.dumps to use our serializer
    original_dumps = json.dumps
    
    def patched_dumps(*args, **kwargs):
        kwargs.setdefault('default', json_serializer)
        return original_dumps(*args, **kwargs)
    
    json.dumps = patched_dumps


class SensitiveDataFilter:
    """Filter sensitive data from log messages"""
    
    def __init__(self, sensitive_keys: Optional[Set[str]] = None, mask_char: str = '*'):
        """Initialize sensitive data filter
        
        Args:
            sensitive_keys: Set of keys to filter (case-insensitive)
            mask_char: Character to use for masking
        """
        self.sensitive_keys = sensitive_keys or {
            'password', 'token', 'secret', 'api_key', 'private_key',
            'auth', 'credential', 'session', 'cookie', 'authorization'
        }
        self.sensitive_keys_lower = {key.lower() for key in self.sensitive_keys}
        self.mask_char = mask_char
    
    def filter(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive data from event dictionary
        
        Args:
            event_dict: Log event dictionary
            
        Returns:
            Filtered event dictionary
        """
        return self._filter_recursive(event_dict.copy())
    
    def _filter_recursive(self, obj: Any) -> Any:
        """Recursively filter sensitive data"""
        if isinstance(obj, dict):
            filtered = {}
            for key, value in obj.items():
                if key.lower() in self.sensitive_keys_lower:
                    filtered[key] = self._mask_value(value)
                else:
                    filtered[key] = self._filter_recursive(value)
            return filtered
        elif isinstance(obj, list):
            return [self._filter_recursive(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._filter_recursive(item) for item in obj)
        else:
            return obj
    
    def _mask_value(self, value: Any) -> str:
        """Mask a sensitive value"""
        if value is None:
            return None
        
        value_str = str(value)
        if len(value_str) <= 4:
            return self.mask_char * len(value_str)
        else:
            # Show first 2 and last 2 characters
            return value_str[:2] + self.mask_char * (len(value_str) - 4) + value_str[-2:]


class LogSampler:
    """Sample log messages to reduce volume"""
    
    def __init__(self, sample_rate: float = 1.0, burst_limit: int = 10):
        """Initialize log sampler
        
        Args:
            sample_rate: Fraction of logs to keep (0.0 to 1.0)
            burst_limit: Maximum consecutive logs before sampling kicks in
        """
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        if burst_limit < 0:
            raise ValueError("burst_limit must be non-negative")
            
        self.sample_rate = sample_rate
        self.burst_limit = burst_limit
        self.burst_count = 0
        self.total_count = 0
        self.sample_count = 0
        self._lock = threading.Lock()
    
    def should_log(self, event_dict: Dict[str, Any]) -> bool:
        """Determine if this log event should be recorded
        
        Args:
            event_dict: Log event dictionary
            
        Returns:
            True if log should be recorded
        """
        with self._lock:
            self.total_count += 1
            
            # Always log during burst period
            if self.burst_count < self.burst_limit:
                self.burst_count += 1
                self.sample_count += 1
                return True
            
            # Sample based on rate
            import random
            if random.random() < self.sample_rate:
                self.sample_count += 1
                return True
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sampling statistics"""
        with self._lock:
            return {
                "total_logs": self.total_count,
                "sampled_logs": self.sample_count,
                "sample_rate_actual": self.sample_count / max(1, self.total_count),
                "sample_rate_configured": self.sample_rate,
                "burst_limit": self.burst_limit
            }


class LogAggregator:
    """Aggregate similar log messages to reduce noise"""
    
    def __init__(self, window_seconds: int = 60, max_duplicates: int = 5):
        """Initialize log aggregator
        
        Args:
            window_seconds: Time window for aggregation
            max_duplicates: Maximum duplicate messages before aggregating
        """
        self.window_seconds = window_seconds
        self.max_duplicates = max_duplicates
        self.message_counts = defaultdict(list)  # message_hash -> [timestamps]
        self.aggregated_messages = {}  # message_hash -> aggregated_event
        self._lock = threading.Lock()
    
    def process(self, event_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process log event for aggregation
        
        Args:
            event_dict: Log event dictionary
            
        Returns:
            Event dict to log, or None if aggregated
        """
        message = event_dict.get('event', event_dict.get('message', ''))
        message_hash = self._hash_message(message, event_dict.get('level', ''))
        current_time = time.time()
        
        with self._lock:
            # Clean old entries
            self._cleanup_old_entries(current_time)
            
            # Track this message
            timestamps = self.message_counts[message_hash]
            timestamps.append(current_time)
            
            # Check if we should aggregate
            recent_count = len([t for t in timestamps if current_time - t <= self.window_seconds])
            
            if recent_count <= self.max_duplicates:
                return event_dict  # Log normally
            
            # Aggregate the message
            if message_hash not in self.aggregated_messages:
                self.aggregated_messages[message_hash] = event_dict.copy()
                self.aggregated_messages[message_hash]['aggregated_count'] = recent_count
                self.aggregated_messages[message_hash]['first_occurrence'] = datetime.fromtimestamp(timestamps[0]).isoformat()
                self.aggregated_messages[message_hash]['last_occurrence'] = datetime.fromtimestamp(current_time).isoformat()
                return self.aggregated_messages[message_hash]
            else:
                # Update existing aggregation
                self.aggregated_messages[message_hash]['aggregated_count'] = recent_count
                self.aggregated_messages[message_hash]['last_occurrence'] = datetime.fromtimestamp(current_time).isoformat()
                return None  # Don't log, already aggregated
    
    def _hash_message(self, message: str, level: str) -> str:
        """Generate hash for message deduplication"""
        combined = f"{level}:{message}"
        return hashlib.md5(combined.encode()).hexdigest()[:8]
    
    def _cleanup_old_entries(self, current_time: float):
        """Clean up old message tracking entries"""
        cutoff_time = current_time - self.window_seconds
        
        # Clean message counts
        for message_hash in list(self.message_counts.keys()):
            timestamps = self.message_counts[message_hash]
            recent_timestamps = [t for t in timestamps if t > cutoff_time]
            
            if recent_timestamps:
                self.message_counts[message_hash] = recent_timestamps
            else:
                del self.message_counts[message_hash]
        
        # Clean aggregated messages
        for message_hash in list(self.aggregated_messages.keys()):
            if message_hash not in self.message_counts:
                del self.aggregated_messages[message_hash]


class AsyncLogHandler:
    """Asynchronous log handler for high-throughput applications"""
    
    def __init__(self, max_queue_size: int = 10000, flush_interval: float = 1.0, 
                 worker_count: int = 2):
        """Initialize async log handler
        
        Args:
            max_queue_size: Maximum log entries in queue
            flush_interval: How often to flush logs (seconds)
            worker_count: Number of worker threads
        """
        self.max_queue_size = max_queue_size
        self.flush_interval = flush_interval
        self.worker_count = worker_count
        
        self.log_queue = deque(maxlen=max_queue_size)
        self.executor = ThreadPoolExecutor(max_workers=worker_count)
        self.running = False
        self.flush_task = None
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "logs_queued": 0,
            "logs_processed": 0,
            "logs_dropped": 0,
            "queue_size": 0,
            "processing_errors": 0
        }
    
    def start(self):
        """Start the async log handler"""
        self.running = True
        if asyncio.get_event_loop().is_running():
            self.flush_task = asyncio.create_task(self._flush_worker())
        else:
            # Fallback to threading
            self.flush_task = threading.Timer(self.flush_interval, self._flush_worker_sync)
            self.flush_task.start()
    
    def stop(self):
        """Stop the async log handler"""
        self.running = False
        if self.flush_task:
            if hasattr(self.flush_task, 'cancel'):
                self.flush_task.cancel()
            else:
                self.flush_task.cancel()
        
        # Flush remaining logs
        self._flush_queue()
        self.executor.shutdown(wait=True)
    
    def enqueue(self, event_dict: Dict[str, Any], formatter: Callable = None):
        """Enqueue a log event for async processing
        
        Args:
            event_dict: Log event dictionary
            formatter: Optional formatter function
        """
        with self._lock:
            if len(self.log_queue) >= self.max_queue_size:
                self.stats["logs_dropped"] += 1
                return
            
            self.log_queue.append({
                "event": event_dict,
                "formatter": formatter,
                "timestamp": time.time()
            })
            self.stats["logs_queued"] += 1
            self.stats["queue_size"] = len(self.log_queue)
    
    def _flush_queue(self):
        """Flush queued logs"""
        batch = []
        with self._lock:
            while self.log_queue and len(batch) < 100:  # Process in batches
                batch.append(self.log_queue.popleft())
            self.stats["queue_size"] = len(self.log_queue)
        
        if batch:
            future = self.executor.submit(self._process_batch, batch)
            try:
                future.result(timeout=5.0)  # Wait max 5 seconds
            except Exception as e:
                self.stats["processing_errors"] += 1
                # In production, you'd want to log this error elsewhere
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of log events"""
        for log_entry in batch:
            try:
                event_dict = log_entry["event"]
                formatter = log_entry.get("formatter")
                
                if formatter:
                    formatted = formatter(event_dict)
                    print(formatted)  # Or write to file, send to service, etc.
                else:
                    print(json.dumps(event_dict))
                
                self.stats["logs_processed"] += 1
                
            except Exception as e:
                self.stats["processing_errors"] += 1
    
    async def _flush_worker(self):
        """Async worker to flush logs periodically"""
        while self.running:
            self._flush_queue()
            await asyncio.sleep(self.flush_interval)
    
    def _flush_worker_sync(self):
        """Sync worker to flush logs periodically"""
        if self.running:
            self._flush_queue()
            self.flush_task = threading.Timer(self.flush_interval, self._flush_worker_sync)
            self.flush_task.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        with self._lock:
            stats = self.stats.copy()
            stats["queue_size"] = len(self.log_queue)
            return stats


class StructuredLogProcessor:
    """Process log events with filtering, sampling, and aggregation"""
    
    def __init__(self):
        self.sensitive_filter = SensitiveDataFilter()
        self.sampler = LogSampler(sample_rate=1.0)  # No sampling by default
        self.aggregator = LogAggregator()
        self.async_handler = AsyncLogHandler()
        self.enabled_filters: List[str] = ["sensitive"]  # Default filters
        
        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        self.processed_count = 0
    
    def configure_sampling(self, sample_rate: float, burst_limit: int = 10):
        """Configure log sampling
        
        Args:
            sample_rate: Fraction of logs to keep (0.0 to 1.0)
            burst_limit: Maximum consecutive logs before sampling
        """
        self.sampler = LogSampler(sample_rate, burst_limit)
        if "sampling" not in self.enabled_filters:
            self.enabled_filters.append("sampling")
    
    def configure_aggregation(self, window_seconds: int = 60, max_duplicates: int = 5):
        """Configure log aggregation
        
        Args:
            window_seconds: Time window for aggregation
            max_duplicates: Maximum duplicate messages before aggregating
        """
        self.aggregator = LogAggregator(window_seconds, max_duplicates)
        if "aggregation" not in self.enabled_filters:
            self.enabled_filters.append("aggregation")
    
    def enable_async_processing(self, **kwargs):
        """Enable async log processing"""
        self.async_handler = AsyncLogHandler(**kwargs)
        self.async_handler.start()
        if "async" not in self.enabled_filters:
            self.enabled_filters.append("async")
    
    def __call__(self, logger, method_name, event_dict):
        """Process log event through configured filters
        
        Args:
            logger: Logger instance
            method_name: Log method name
            event_dict: Log event dictionary
            
        Returns:
            Processed event dict or raises DropEvent to suppress
        """
        start_time = time.time()
        
        try:
            # Apply sensitive data filtering
            if "sensitive" in self.enabled_filters:
                event_dict = self.sensitive_filter.filter(event_dict)
            
            # Apply sampling
            if "sampling" in self.enabled_filters:
                if not self.sampler.should_log(event_dict):
                    from structlog import DropEvent
                    raise DropEvent
            
            # Apply aggregation
            if "aggregation" in self.enabled_filters:
                processed_event = self.aggregator.process(event_dict)
                if processed_event is None:
                    from structlog import DropEvent
                    raise DropEvent
                event_dict = processed_event
            
            # Handle async processing
            if "async" in self.enabled_filters:
                self.async_handler.enqueue(event_dict)
                from structlog import DropEvent
                raise DropEvent  # Don't process synchronously
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.processed_count += 1
            
            return event_dict
            
        except Exception as e:
            if "DropEvent" in str(type(e)):
                raise  # Re-raise DropEvent
            
            # Log processing error (carefully to avoid recursion)
            error_event = {
                "event": "log_processing_error",
                "error": str(e),
                "original_event": str(event_dict)[:200]  # Truncate to avoid issues
            }
            return error_event
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get log processing performance statistics"""
        if not self.processing_times:
            return {"processed_count": 0}
        
        times = list(self.processing_times)
        return {
            "processed_count": self.processed_count,
            "avg_processing_time_ms": sum(times) / len(times) * 1000,
            "max_processing_time_ms": max(times) * 1000,
            "min_processing_time_ms": min(times) * 1000,
            "enabled_filters": self.enabled_filters.copy(),
            "sampler_stats": self.sampler.get_stats(),
            "async_handler_stats": self.async_handler.get_stats() if "async" in self.enabled_filters else {}
        }