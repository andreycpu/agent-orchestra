"""
Custom exceptions for Agent Orchestra
"""
import traceback
from typing import Any, Dict, Optional, List, Union
from datetime import datetime
import logging


class AgentOrchestraException(Exception):
    """Base exception for all Agent Orchestra errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class AgentRegistrationError(AgentOrchestraException):
    """Raised when agent registration fails"""
    pass


class AgentNotFoundError(AgentOrchestraException):
    """Raised when an agent cannot be found"""
    pass


class AgentUnavailableError(AgentOrchestraException):
    """Raised when an agent is unavailable for task execution"""
    pass


class TaskRoutingError(AgentOrchestraException):
    """Raised when task routing fails"""
    pass


class TaskExecutionError(AgentOrchestraException):
    """Raised when task execution fails"""
    pass


class TaskTimeoutError(AgentOrchestraException):
    """Raised when a task times out"""
    pass


class StateManagementError(AgentOrchestraException):
    """Raised when state management operations fail"""
    pass


class ConcurrencyError(AgentOrchestraException):
    """Raised when concurrency limits are exceeded"""
    pass


class CircularDependencyError(AgentOrchestraException):
    """Raised when circular dependencies are detected in tasks"""
    pass


class ResourceExhaustionError(AgentOrchestraException):
    """Raised when system resources are exhausted"""
    pass


class ValidationError(AgentOrchestraException):
    """Raised when data validation fails"""
    pass


class ConfigurationError(AgentOrchestraException):
    """Raised when configuration is invalid or missing"""
    pass


class SecurityError(AgentOrchestraException):
    """Raised when security violations are detected"""
    pass


class NetworkError(AgentOrchestraException):
    """Raised when network operations fail"""
    pass


class HandlerRegistrationError(AgentOrchestraException):
    """Raised when task handler registration fails"""
    pass


class TaskValidationError(ValidationError):
    """Raised when task validation fails"""
    pass


class AgentCapabilityError(AgentOrchestraException):
    """Raised when agent capability issues occur"""
    pass


class HeartbeatError(AgentOrchestraException):
    """Raised when agent heartbeat operations fail"""
    pass


class StatusTransitionError(AgentOrchestraException):
    """Raised when invalid agent status transitions occur"""
    pass


class DuplicateHandlerError(HandlerRegistrationError):
    """Raised when attempting to register duplicate handlers"""
    pass


class IncompatibleTaskError(TaskExecutionError):
    """Raised when task is incompatible with agent capabilities"""
    pass


class RetryableError(AgentOrchestraException):
    """Base class for errors that can be retried"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.retry_after = retry_after  # seconds to wait before retry


class PermanentError(AgentOrchestraException):
    """Base class for permanent errors that should not be retried"""
    pass


class ThrottlingError(RetryableError):
    """Raised when operation is throttled and should be retried later"""
    pass


class TemporaryError(RetryableError):
    """Raised for temporary errors that can be retried immediately"""
    pass


class RateLimitError(ThrottlingError):
    """Raised when rate limits are exceeded"""
    pass


class BackpressureError(ThrottlingError):
    """Raised when system is under backpressure"""
    pass


class DatabaseError(RetryableError):
    """Raised when database operations fail"""
    pass


class ConnectionError(NetworkError):
    """Raised when connection establishment fails"""
    pass


class AuthenticationError(PermanentError):
    """Raised when authentication fails"""
    pass


class AuthorizationError(PermanentError):
    """Raised when authorization fails"""
    pass


class InvalidInputError(PermanentError):
    """Raised for invalid input that cannot be corrected"""
    pass


class ProtocolError(NetworkError):
    """Raised when protocol-level errors occur"""
    pass


class SerializationError(PermanentError):
    """Raised when serialization/deserialization fails"""
    pass


class CompatibilityError(PermanentError):
    """Raised when version or API compatibility issues occur"""
    pass


class MaintenanceError(TemporaryError):
    """Raised when system is in maintenance mode"""
    pass


# Error handling utilities

class ErrorContext:
    """Context information for error tracking and debugging"""
    
    def __init__(self, operation: str, component: str, **kwargs):
        self.operation = operation
        self.component = component
        self.timestamp = datetime.utcnow()
        self.context_data = kwargs
        self.stack_trace = traceback.format_stack()[:-1]  # Exclude current frame
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'component': self.component,
            'timestamp': self.timestamp.isoformat(),
            'context_data': self.context_data,
            'stack_trace': self.stack_trace
        }


class ErrorCollector:
    """Utility for collecting and managing multiple errors"""
    
    def __init__(self):
        self.errors: List[Exception] = []
        self.warnings: List[str] = []
    
    def add_error(self, error: Exception):
        """Add an error to the collection"""
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Add a warning to the collection"""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if any errors were collected"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if any warnings were collected"""
        return len(self.warnings) > 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected errors and warnings"""
        return {
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': [str(error) for error in self.errors],
            'warnings': self.warnings
        }
    
    def raise_if_errors(self, message: str = "Multiple errors occurred"):
        """Raise a composite error if any errors were collected"""
        if self.has_errors():
            error_details = {
                'errors': [str(error) for error in self.errors],
                'warnings': self.warnings
            }
            raise AgentOrchestraException(message, error_details)


def handle_retry_logic(
    error: Exception,
    attempt: int,
    max_attempts: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
) -> Dict[str, Any]:
    """
    Determine retry behavior based on error type and attempt count.
    
    Args:
        error: The exception that occurred
        attempt: Current attempt number (1-based)
        max_attempts: Maximum number of attempts allowed
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
    
    Returns:
        Dictionary with retry decision and delay information
    """
    should_retry = False
    delay = 0.0
    reason = "No retry needed"
    
    if attempt >= max_attempts:
        reason = "Maximum attempts reached"
    elif isinstance(error, PermanentError):
        reason = "Permanent error - no retry"
    elif isinstance(error, RetryableError):
        should_retry = True
        if hasattr(error, 'retry_after') and error.retry_after:
            delay = min(error.retry_after, max_delay)
        else:
            # Exponential backoff with jitter
            delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
            # Add some jitter (±10%)
            import random
            jitter = delay * 0.1 * (random.random() - 0.5)
            delay = max(0, delay + jitter)
        reason = f"Retryable error - waiting {delay:.2f}s"
    else:
        # For unknown errors, allow limited retries with backoff
        if attempt < min(3, max_attempts):
            should_retry = True
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            reason = f"Unknown error - limited retry after {delay:.2f}s"
        else:
            reason = "Unknown error - retry limit exceeded"
    
    return {
        'should_retry': should_retry,
        'delay_seconds': delay,
        'reason': reason,
        'attempt': attempt,
        'max_attempts': max_attempts
    }


def format_error_message(
    error: Exception,
    context: Optional[ErrorContext] = None,
    include_traceback: bool = False
) -> str:
    """
    Format an error message with optional context and traceback.
    
    Args:
        error: The exception to format
        context: Optional error context
        include_traceback: Whether to include stack trace
    
    Returns:
        Formatted error message
    """
    message_parts = [f"Error: {error.__class__.__name__}: {str(error)}"]
    
    if context:
        message_parts.append(f"Operation: {context.operation}")
        message_parts.append(f"Component: {context.component}")
        message_parts.append(f"Timestamp: {context.timestamp}")
        
        if context.context_data:
            message_parts.append(f"Context: {context.context_data}")
    
    if include_traceback:
        message_parts.append(f"Traceback: {traceback.format_exc()}")
    
    return "\n".join(message_parts)


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Optional[ErrorContext] = None,
    level: int = logging.ERROR
):
    """
    Log an error with structured context information.
    
    Args:
        logger: Logger instance to use
        error: Exception to log
        context: Optional error context
        level: Log level to use
    """
    message = format_error_message(error, context, include_traceback=False)
    
    extra_data = {
        'error_type': error.__class__.__name__,
        'error_message': str(error),
    }
    
    if context:
        extra_data.update(context.to_dict())
    
    if hasattr(error, 'details') and error.details:
        extra_data['error_details'] = error.details
    
    logger.log(level, message, extra=extra_data)


def create_error_response(
    error: Exception,
    request_id: Optional[str] = None,
    include_details: bool = True
) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary.
    
    Args:
        error: Exception to create response for
        request_id: Optional request identifier
        include_details: Whether to include error details
    
    Returns:
        Standardized error response
    """
    response = {
        'success': False,
        'error': {
            'type': error.__class__.__name__,
            'message': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
    }
    
    if request_id:
        response['request_id'] = request_id
    
    if include_details and hasattr(error, 'details') and error.details:
        response['error']['details'] = error.details
    
    # Add retry information for retryable errors
    if isinstance(error, RetryableError):
        response['error']['retryable'] = True
        if hasattr(error, 'retry_after') and error.retry_after:
            response['error']['retry_after'] = error.retry_after
    elif isinstance(error, PermanentError):
        response['error']['retryable'] = False
    
    return response