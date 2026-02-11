"""
Centralized logging configuration for Agent Orchestra
"""
import logging
import sys
from typing import Dict, Any, Optional
import structlog
from structlog.typing import Processor
import json


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