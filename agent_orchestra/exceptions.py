"""
Custom exceptions for Agent Orchestra
"""


class AgentOrchestraException(Exception):
    """Base exception for all Agent Orchestra errors"""
    pass


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