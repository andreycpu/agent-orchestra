"""
Enhanced logging utilities for Agent Orchestra.

This module provides structured logging, performance tracking,
and observability tools for debugging and monitoring.
"""
import logging
import logging.handlers
import json
import time
import functools
from typing import Any, Dict, Optional, Union, Callable, List
from datetime import datetime, timedelta
from contextlib import contextmanager
from pathlib import Path
import threading
import queue
import sys
import traceback

from .types import TaskStatus, AgentStatus


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add thread information
        if hasattr(record, 'thread'):
            log_obj['thread_id'] = record.thread
            log_obj['thread_name'] = getattr(record, 'threadName', 'Unknown')
        
        # Add process information
        if hasattr(record, 'process'):
            log_obj['process_id'] = record.process
        
        # Add exception information
        if record.exc_info:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                extra_fields[key] = value
        
        if extra_fields:
            log_obj['extra'] = extra_fields
        
        return json.dumps(log_obj, default=str, separators=(',', ':'))


class PerformanceLogger:
    """Logger for performance metrics and timing information."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> str:
        """Start timing an operation."""
        timer_id = f"{operation}_{time.time()}"
        self.timers[timer_id] = time.time()
        return timer_id
    
    def stop_timer(self, timer_id: str, context: Optional[Dict[str, Any]] = None):
        """Stop timing and log the duration."""
        if timer_id not in self.timers:
            self.logger.warning(f"Timer {timer_id} not found")
            return
        
        duration = time.time() - self.timers[timer_id]
        del self.timers[timer_id]
        
        operation = timer_id.split('_')[0]
        log_context = {
            'operation': operation,
            'duration_ms': round(duration * 1000, 2),
            'duration_seconds': round(duration, 4)
        }
        
        if context:
            log_context.update(context)
        
        self.logger.info(f"Operation {operation} completed", extra=log_context)
    
    @contextmanager
    def time_operation(self, operation: str, context: Optional[Dict[str, Any]] = None):
        """Context manager for timing operations."""
        timer_id = self.start_timer(operation)
        try:
            yield
        finally:
            self.stop_timer(timer_id, context)
    
    def log_memory_usage(self, operation: str, context: Optional[Dict[str, Any]] = None):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            log_context = {
                'operation': operation,
                'memory_rss_mb': round(memory_info.rss / 1024 / 1024, 2),
                'memory_vms_mb': round(memory_info.vms / 1024 / 1024, 2),
                'memory_percent': round(process.memory_percent(), 2)
            }
            
            if context:
                log_context.update(context)
            
            self.logger.info(f"Memory usage for {operation}", extra=log_context)
        except ImportError:
            self.logger.debug("psutil not available for memory logging")
        except Exception as e:
            self.logger.warning(f"Failed to log memory usage: {e}")


class AuditLogger:
    """Logger for audit trails and security events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_authentication(self, user_id: str, success: bool, source_ip: Optional[str] = None):
        """Log authentication attempts."""
        event_data = {
            'event_type': 'authentication',
            'user_id': user_id,
            'success': success,
            'source_ip': source_ip,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        level = logging.INFO if success else logging.WARNING
        message = f"Authentication {'successful' if success else 'failed'} for user {user_id}"
        self.logger.log(level, message, extra=event_data)
    
    def log_authorization(self, user_id: str, action: str, resource: str, granted: bool):
        """Log authorization decisions."""
        event_data = {
            'event_type': 'authorization',
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'granted': granted,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        level = logging.INFO if granted else logging.WARNING
        message = f"Authorization {'granted' if granted else 'denied'} for {user_id} on {action}:{resource}"
        self.logger.log(level, message, extra=event_data)
    
    def log_data_access(self, user_id: str, data_type: str, operation: str, record_count: Optional[int] = None):
        """Log data access events."""
        event_data = {
            'event_type': 'data_access',
            'user_id': user_id,
            'data_type': data_type,
            'operation': operation,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if record_count is not None:
            event_data['record_count'] = record_count
        
        message = f"Data access: {user_id} performed {operation} on {data_type}"
        self.logger.info(message, extra=event_data)
    
    def log_configuration_change(self, user_id: str, component: str, changes: Dict[str, Any]):
        """Log configuration changes."""
        event_data = {
            'event_type': 'configuration_change',
            'user_id': user_id,
            'component': component,
            'changes': changes,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        message = f"Configuration changed by {user_id} in {component}"
        self.logger.warning(message, extra=event_data)


class TaskLogger:
    """Specialized logger for task lifecycle events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_task_created(self, task_id: str, task_type: str, priority: str, context: Optional[Dict[str, Any]] = None):
        """Log task creation."""
        event_data = {
            'event_type': 'task_created',
            'task_id': task_id,
            'task_type': task_type,
            'priority': priority,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if context:
            event_data.update(context)
        
        self.logger.info(f"Task created: {task_id} ({task_type})", extra=event_data)
    
    def log_task_assigned(self, task_id: str, agent_id: str, context: Optional[Dict[str, Any]] = None):
        """Log task assignment."""
        event_data = {
            'event_type': 'task_assigned',
            'task_id': task_id,
            'agent_id': agent_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if context:
            event_data.update(context)
        
        self.logger.info(f"Task assigned: {task_id} -> {agent_id}", extra=event_data)
    
    def log_task_started(self, task_id: str, agent_id: str, context: Optional[Dict[str, Any]] = None):
        """Log task start."""
        event_data = {
            'event_type': 'task_started',
            'task_id': task_id,
            'agent_id': agent_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if context:
            event_data.update(context)
        
        self.logger.info(f"Task started: {task_id} by {agent_id}", extra=event_data)
    
    def log_task_completed(self, task_id: str, agent_id: str, status: TaskStatus, 
                          duration: Optional[float] = None, context: Optional[Dict[str, Any]] = None):
        """Log task completion."""
        event_data = {
            'event_type': 'task_completed',
            'task_id': task_id,
            'agent_id': agent_id,
            'status': status.value,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if duration is not None:
            event_data['duration_seconds'] = round(duration, 4)
        
        if context:
            event_data.update(context)
        
        level = logging.INFO if status == TaskStatus.COMPLETED else logging.WARNING
        self.logger.log(level, f"Task {status.value}: {task_id} by {agent_id}", extra=event_data)
    
    def log_task_retry(self, task_id: str, attempt: int, max_retries: int, error: Optional[str] = None):
        """Log task retry attempts."""
        event_data = {
            'event_type': 'task_retry',
            'task_id': task_id,
            'attempt': attempt,
            'max_retries': max_retries,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if error:
            event_data['error'] = error
        
        self.logger.warning(f"Task retry: {task_id} (attempt {attempt}/{max_retries})", extra=event_data)


class AgentLogger:
    """Specialized logger for agent lifecycle events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_agent_registered(self, agent_id: str, capabilities: List[str], context: Optional[Dict[str, Any]] = None):
        """Log agent registration."""
        event_data = {
            'event_type': 'agent_registered',
            'agent_id': agent_id,
            'capabilities': capabilities,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if context:
            event_data.update(context)
        
        self.logger.info(f"Agent registered: {agent_id}", extra=event_data)
    
    def log_agent_status_change(self, agent_id: str, old_status: AgentStatus, new_status: AgentStatus,
                               context: Optional[Dict[str, Any]] = None):
        """Log agent status changes."""
        event_data = {
            'event_type': 'agent_status_change',
            'agent_id': agent_id,
            'old_status': old_status.value,
            'new_status': new_status.value,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if context:
            event_data.update(context)
        
        level = logging.WARNING if new_status == AgentStatus.ERROR else logging.INFO
        self.logger.log(level, f"Agent status changed: {agent_id} {old_status.value} -> {new_status.value}", 
                       extra=event_data)
    
    def log_agent_heartbeat(self, agent_id: str, healthy: bool, metrics: Optional[Dict[str, Any]] = None):
        """Log agent heartbeat."""
        event_data = {
            'event_type': 'agent_heartbeat',
            'agent_id': agent_id,
            'healthy': healthy,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if metrics:
            event_data['metrics'] = metrics
        
        level = logging.DEBUG if healthy else logging.WARNING
        self.logger.log(level, f"Agent heartbeat: {agent_id} ({'healthy' if healthy else 'unhealthy'})", 
                       extra=event_data)
    
    def log_agent_disconnected(self, agent_id: str, reason: Optional[str] = None, 
                              context: Optional[Dict[str, Any]] = None):
        """Log agent disconnection."""
        event_data = {
            'event_type': 'agent_disconnected',
            'agent_id': agent_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if reason:
            event_data['reason'] = reason
        
        if context:
            event_data.update(context)
        
        self.logger.warning(f"Agent disconnected: {agent_id}", extra=event_data)


def performance_logging(
    logger: Optional[logging.Logger] = None,
    threshold_seconds: float = 1.0,
    log_args: bool = False,
    log_result: bool = False
):
    """
    Decorator for automatic performance logging of function calls.
    
    Args:
        logger: Logger to use (defaults to function's module logger)
        threshold_seconds: Only log if execution takes longer than this
        log_args: Whether to log function arguments
        log_result: Whether to log function result
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_logger = logger or logging.getLogger(func.__module__)
            
            log_context = {
                'function': f"{func.__module__}.{func.__name__}",
                'operation': 'function_call'
            }
            
            if log_args and (args or kwargs):
                log_context['arguments'] = {
                    'args': args if log_args else '...', 
                    'kwargs': kwargs if log_args else '...'
                }
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if duration >= threshold_seconds:
                    log_context['duration_seconds'] = round(duration, 4)
                    log_context['success'] = True
                    
                    if log_result and result is not None:
                        log_context['result'] = str(result)[:500]  # Truncate long results
                    
                    func_logger.info(f"Slow function execution: {func.__name__}", extra=log_context)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                log_context['duration_seconds'] = round(duration, 4)
                log_context['success'] = False
                log_context['error'] = str(e)
                
                func_logger.error(f"Function execution failed: {func.__name__}", extra=log_context)
                raise
        
        return wrapper
    return decorator


def setup_logging(
    level: Union[str, int] = logging.INFO,
    format_type: str = 'json',
    log_file: Optional[Path] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        level: Logging level
        format_type: 'json' for structured logging, 'text' for human-readable
        log_file: Optional file path for logging
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        enable_console: Whether to enable console logging
    
    Returns:
        Configured root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Setup formatters
    if format_type == 'json':
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def create_specialized_loggers(base_logger: logging.Logger) -> Dict[str, Any]:
    """
    Create specialized logger instances for different components.
    
    Args:
        base_logger: Base logger to derive from
    
    Returns:
        Dictionary of specialized loggers
    """
    return {
        'performance': PerformanceLogger(base_logger.getChild('performance')),
        'audit': AuditLogger(base_logger.getChild('audit')),
        'tasks': TaskLogger(base_logger.getChild('tasks')),
        'agents': AgentLogger(base_logger.getChild('agents'))
    }