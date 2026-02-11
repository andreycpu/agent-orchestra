"""
Custom exceptions for Agent Orchestra
"""
from typing import Any, Dict, Optional


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