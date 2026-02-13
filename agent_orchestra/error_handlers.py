"""
Centralized error handling and recovery utilities for Agent Orchestra.

This module provides error handlers, recovery strategies, and utilities
for graceful error handling and system resilience.
"""
import logging
import traceback
import sys
from typing import Dict, List, Optional, Any, Callable, Type, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import threading

from .exceptions import *
from .logging_utils import log_error_with_context, ErrorContext


logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(str, Enum):
    """Recovery actions that can be taken."""
    IGNORE = "ignore"
    RETRY = "retry"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    SHUTDOWN = "shutdown"


@dataclass
class ErrorPattern:
    """Defines an error pattern and its handling strategy."""
    exception_type: Type[Exception]
    message_pattern: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    recovery_action: RecoveryAction = RecoveryAction.RETRY
    max_occurrences: Optional[int] = None
    time_window_seconds: Optional[int] = None
    custom_handler: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, exception: Exception) -> bool:
        """Check if exception matches this pattern."""
        if not isinstance(exception, self.exception_type):
            return False
        
        if self.message_pattern:
            import re
            return bool(re.search(self.message_pattern, str(exception), re.IGNORECASE))
        
        return True


@dataclass
class ErrorOccurrence:
    """Records an error occurrence."""
    exception: Exception
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def __post_init__(self):
        if self.stack_trace is None:
            self.stack_trace = traceback.format_exc()


class ErrorTracker:
    """Tracks error occurrences and patterns."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._occurrences: List[ErrorOccurrence] = []
        self._pattern_counts: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()
    
    def record_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorOccurrence:
        """Record an error occurrence."""
        occurrence = ErrorOccurrence(
            exception=exception,
            timestamp=datetime.utcnow(),
            context=context
        )
        
        with self._lock:
            self._occurrences.append(occurrence)
            
            # Keep only recent occurrences
            if len(self._occurrences) > self.max_history:
                self._occurrences = self._occurrences[-self.max_history:]
        
        return occurrence
    
    def record_pattern_match(self, pattern: ErrorPattern):
        """Record that a pattern was matched."""
        pattern_key = f"{pattern.exception_type.__name__}:{pattern.message_pattern or 'any'}"
        
        with self._lock:
            if pattern_key not in self._pattern_counts:
                self._pattern_counts[pattern_key] = []
            
            self._pattern_counts[pattern_key].append(datetime.utcnow())
            
            # Clean old entries if time window is specified
            if pattern.time_window_seconds:
                cutoff_time = datetime.utcnow() - timedelta(seconds=pattern.time_window_seconds)
                self._pattern_counts[pattern_key] = [
                    timestamp for timestamp in self._pattern_counts[pattern_key]
                    if timestamp >= cutoff_time
                ]
    
    def get_pattern_count(self, pattern: ErrorPattern, time_window_seconds: Optional[int] = None) -> int:
        """Get count of pattern matches in time window."""
        pattern_key = f"{pattern.exception_type.__name__}:{pattern.message_pattern or 'any'}"
        
        with self._lock:
            if pattern_key not in self._pattern_counts:
                return 0
            
            timestamps = self._pattern_counts[pattern_key]
            
            if time_window_seconds:
                cutoff_time = datetime.utcnow() - timedelta(seconds=time_window_seconds)
                return sum(1 for timestamp in timestamps if timestamp >= cutoff_time)
            else:
                return len(timestamps)
    
    def get_recent_errors(self, minutes: int = 60) -> List[ErrorOccurrence]:
        """Get errors from the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            return [
                occurrence for occurrence in self._occurrences
                if occurrence.timestamp >= cutoff_time
            ]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            total_errors = len(self._occurrences)
            
            if not self._occurrences:
                return {
                    'total_errors': 0,
                    'error_rate': 0.0,
                    'most_common_errors': [],
                    'recent_error_count': 0
                }
            
            # Count by exception type
            error_counts = {}
            for occurrence in self._occurrences:
                exc_type = type(occurrence.exception).__name__
                error_counts[exc_type] = error_counts.get(exc_type, 0) + 1
            
            # Get most common errors
            most_common = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Recent errors (last hour)
            recent_errors = self.get_recent_errors(60)
            
            return {
                'total_errors': total_errors,
                'error_types': len(error_counts),
                'most_common_errors': most_common,
                'recent_error_count': len(recent_errors),
                'oldest_error': self._occurrences[0].timestamp.isoformat() if self._occurrences else None,
                'newest_error': self._occurrences[-1].timestamp.isoformat() if self._occurrences else None
            }


class ErrorHandler:
    """Handles errors based on defined patterns and strategies."""
    
    def __init__(self):
        self.patterns: List[ErrorPattern] = []
        self.tracker = ErrorTracker()
        self.fallback_handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        
        # Register default patterns
        self._register_default_patterns()
    
    def _register_default_patterns(self):
        """Register common error patterns."""
        # Network errors - retry with backoff
        self.register_pattern(ErrorPattern(
            exception_type=NetworkError,
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.RETRY,
            max_occurrences=5,
            time_window_seconds=300
        ))
        
        # Database errors - retry with shorter time window
        self.register_pattern(ErrorPattern(
            exception_type=DatabaseError,
            severity=ErrorSeverity.HIGH,
            recovery_action=RecoveryAction.RETRY,
            max_occurrences=3,
            time_window_seconds=60
        ))
        
        # Validation errors - no retry
        self.register_pattern(ErrorPattern(
            exception_type=ValidationError,
            severity=ErrorSeverity.LOW,
            recovery_action=RecoveryAction.ESCALATE,
            max_occurrences=10,
            time_window_seconds=3600
        ))
        
        # Security errors - escalate immediately
        self.register_pattern(ErrorPattern(
            exception_type=SecurityError,
            severity=ErrorSeverity.CRITICAL,
            recovery_action=RecoveryAction.ESCALATE,
            max_occurrences=1,
            time_window_seconds=60
        ))
        
        # Permanent errors - escalate
        self.register_pattern(ErrorPattern(
            exception_type=PermanentError,
            severity=ErrorSeverity.HIGH,
            recovery_action=RecoveryAction.ESCALATE,
            max_occurrences=1
        ))
        
        # Memory errors - critical
        self.register_pattern(ErrorPattern(
            exception_type=MemoryError,
            severity=ErrorSeverity.CRITICAL,
            recovery_action=RecoveryAction.SHUTDOWN,
            max_occurrences=1
        ))
    
    def register_pattern(self, pattern: ErrorPattern):
        """Register an error pattern."""
        with self._lock:
            self.patterns.append(pattern)
        logger.info(f"Registered error pattern for {pattern.exception_type.__name__}")
    
    def register_fallback_handler(self, name: str, handler: Callable):
        """Register a fallback handler."""
        self.fallback_handlers[name] = handler
        logger.info(f"Registered fallback handler: {name}")
    
    def handle_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle an error according to registered patterns."""
        # Record the error
        occurrence = self.tracker.record_error(exception, context)
        
        # Find matching pattern
        pattern = self._find_matching_pattern(exception)
        
        if pattern:
            # Check if max occurrences exceeded
            if pattern.max_occurrences:
                count = self.tracker.get_pattern_count(pattern, pattern.time_window_seconds)
                if count >= pattern.max_occurrences:
                    logger.warning(
                        f"Error pattern limit exceeded: {pattern.exception_type.__name__} "
                        f"({count}/{pattern.max_occurrences})"
                    )
                    return self._escalate_error(exception, pattern, context, operation)
            
            # Record pattern match
            self.tracker.record_pattern_match(pattern)
            
            # Execute recovery action
            return self._execute_recovery_action(exception, pattern, context, operation)
        else:
            # No pattern matched - use default handling
            return self._handle_unknown_error(exception, context, operation)
    
    def _find_matching_pattern(self, exception: Exception) -> Optional[ErrorPattern]:
        """Find the first matching pattern for an exception."""
        with self._lock:
            for pattern in self.patterns:
                if pattern.matches(exception):
                    return pattern
        return None
    
    def _execute_recovery_action(
        self,
        exception: Exception,
        pattern: ErrorPattern,
        context: Optional[Dict[str, Any]],
        operation: Optional[str]
    ) -> Dict[str, Any]:
        """Execute the recovery action for a pattern."""
        action = pattern.recovery_action
        
        logger.info(
            f"Executing recovery action {action.value} for {pattern.exception_type.__name__}",
            extra={'pattern': pattern.exception_type.__name__, 'action': action.value}
        )
        
        if action == RecoveryAction.IGNORE:
            return self._ignore_error(exception, pattern)
        
        elif action == RecoveryAction.RETRY:
            return self._retry_error(exception, pattern)
        
        elif action == RecoveryAction.FALLBACK:
            return self._fallback_error(exception, pattern, context)
        
        elif action == RecoveryAction.ESCALATE:
            return self._escalate_error(exception, pattern, context, operation)
        
        elif action == RecoveryAction.SHUTDOWN:
            return self._shutdown_error(exception, pattern, context)
        
        else:
            return self._handle_unknown_error(exception, context, operation)
    
    def _ignore_error(self, exception: Exception, pattern: ErrorPattern) -> Dict[str, Any]:
        """Ignore the error."""
        logger.debug(f"Ignoring error: {exception}")
        return {
            'action': 'ignored',
            'success': True,
            'message': 'Error was ignored as configured'
        }
    
    def _retry_error(self, exception: Exception, pattern: ErrorPattern) -> Dict[str, Any]:
        """Indicate that error should be retried."""
        logger.info(f"Error should be retried: {exception}")
        return {
            'action': 'retry',
            'success': False,
            'retryable': True,
            'message': 'Error can be retried'
        }
    
    def _fallback_error(
        self,
        exception: Exception,
        pattern: ErrorPattern,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute fallback handling."""
        fallback_name = pattern.metadata.get('fallback_handler')
        
        if fallback_name and fallback_name in self.fallback_handlers:
            try:
                handler = self.fallback_handlers[fallback_name]
                result = handler(exception, context)
                logger.info(f"Fallback handler {fallback_name} executed successfully")
                return {
                    'action': 'fallback',
                    'success': True,
                    'handler': fallback_name,
                    'result': result
                }
            except Exception as e:
                logger.error(f"Fallback handler {fallback_name} failed: {e}")
        
        return {
            'action': 'fallback_failed',
            'success': False,
            'message': 'No fallback handler available or fallback failed'
        }
    
    def _escalate_error(
        self,
        exception: Exception,
        pattern: ErrorPattern,
        context: Optional[Dict[str, Any]],
        operation: Optional[str]
    ) -> Dict[str, Any]:
        """Escalate the error."""
        error_context = ErrorContext(
            operation=operation or 'unknown',
            component='error_handler',
            **context or {}
        )
        
        log_error_with_context(logger, exception, error_context, logging.ERROR)
        
        logger.error(
            f"Error escalated: {exception}",
            extra={
                'severity': pattern.severity.value,
                'exception_type': type(exception).__name__,
                'pattern_matched': True
            }
        )
        
        return {
            'action': 'escalated',
            'success': False,
            'severity': pattern.severity.value,
            'requires_attention': True,
            'message': f'Error escalated due to {pattern.severity.value} severity'
        }
    
    def _shutdown_error(
        self,
        exception: Exception,
        pattern: ErrorPattern,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle critical errors that require shutdown."""
        logger.critical(
            f"Critical error requiring shutdown: {exception}",
            extra={'exception_type': type(exception).__name__}
        )
        
        # In a real system, this might trigger graceful shutdown
        return {
            'action': 'shutdown_required',
            'success': False,
            'severity': 'critical',
            'requires_shutdown': True,
            'message': 'Critical error requires system shutdown'
        }
    
    def _handle_unknown_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]],
        operation: Optional[str]
    ) -> Dict[str, Any]:
        """Handle errors that don't match any pattern."""
        logger.warning(
            f"Unknown error pattern: {type(exception).__name__}: {exception}",
            extra={'exception_type': type(exception).__name__}
        )
        
        return {
            'action': 'unknown',
            'success': False,
            'retryable': isinstance(exception, RetryableError),
            'message': 'Error does not match any registered pattern'
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return self.tracker.get_error_statistics()


class ErrorRecovery:
    """Provides error recovery utilities."""
    
    @staticmethod
    def create_circuit_breaker_fallback(default_value: Any) -> Callable:
        """Create a fallback handler that returns a default value."""
        def fallback_handler(exception: Exception, context: Optional[Dict[str, Any]]) -> Any:
            logger.info(f"Circuit breaker fallback returning default value: {default_value}")
            return default_value
        return fallback_handler
    
    @staticmethod
    def create_cache_fallback(cache_key: str) -> Callable:
        """Create a fallback handler that attempts to return cached value."""
        def fallback_handler(exception: Exception, context: Optional[Dict[str, Any]]) -> Any:
            try:
                from .cache_manager import get_cache
                cache = get_cache("default")
                if cache:
                    cached_value = cache.get(cache_key)
                    if cached_value is not None:
                        logger.info(f"Fallback using cached value for key: {cache_key}")
                        return cached_value
            except Exception as cache_error:
                logger.warning(f"Cache fallback failed: {cache_error}")
            
            return None
        return fallback_handler
    
    @staticmethod
    def create_degraded_service_fallback(service_name: str) -> Callable:
        """Create a fallback handler for degraded service operation."""
        def fallback_handler(exception: Exception, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            logger.warning(f"Service {service_name} degraded, providing limited functionality")
            return {
                'status': 'degraded',
                'service': service_name,
                'message': 'Service operating in degraded mode',
                'full_functionality': False
            }
        return fallback_handler


@contextmanager
def error_handling(
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    handler: Optional[ErrorHandler] = None
):
    """Context manager for centralized error handling."""
    if handler is None:
        handler = global_error_handler
    
    try:
        yield handler
    except Exception as e:
        result = handler.handle_error(e, context, operation)
        
        if result.get('requires_shutdown'):
            logger.critical("System shutdown required due to critical error")
            # In production, this might trigger graceful shutdown
        
        if not result.get('success') and not result.get('retryable'):
            raise


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_error(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    operation: Optional[str] = None
) -> Dict[str, Any]:
    """Handle error using global error handler."""
    return global_error_handler.handle_error(exception, context, operation)


def register_error_pattern(pattern: ErrorPattern):
    """Register error pattern with global handler."""
    global_error_handler.register_pattern(pattern)


def register_fallback_handler(name: str, handler: Callable):
    """Register fallback handler with global handler."""
    global_error_handler.register_fallback_handler(name, handler)


def get_error_statistics() -> Dict[str, Any]:
    """Get error statistics from global handler."""
    return global_error_handler.get_statistics()