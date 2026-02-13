"""
Agent Orchestra - Multi-agent orchestration framework
"""

__version__ = "0.4.0"
__title__ = "Agent Orchestra"
__description__ = "A comprehensive multi-agent orchestration framework with advanced utilities"
__url__ = "https://github.com/andreycpu/agent-orchestra"
__author__ = "Agent Orchestra Team"

from .orchestra import Orchestra
from .agent import Agent
from .task_router import TaskRouter
from .state_manager import StateManager
from .failure_handler import FailureHandler
from .monitoring import OrchestrationMonitor
from .events import EventBus, Event, EventType
from .plugins import PluginManager, PluginInterface
from .config import ConfigurationManager
from .migration import MigrationManager
from .utils import AsyncRetry, RateLimiter, MetricsCollector, AdaptiveRateLimiter
from .recovery import ErrorRecoveryManager, CircuitBreaker, RecoveryStrategy
from .health_checks import HealthCheckManager, HealthCheck, HealthStatus
from .performance_profiler import PerformanceProfiler, get_profiler, profile
from .logging_config import setup_logging, get_logger, LoggingContext
from .env_config import get_config, get_env_var
from .migrations import MigrationManager as DatabaseMigrationManager

# Import new utilities (optional imports to avoid breaking existing code)
try:
    from .validation import validate_id, validate_task_type, validate_timeout
    from .security_utils import SecurityContext, Permission, Role
    from .cache_manager import create_cache, get_cache, cached
    from .performance_monitor import performance_manager, profile as perf_profile
    from .queue_manager import create_queue, get_queue, QueueType
    from .data_transformers import transform_data, DataFormat
    from .retry_mechanisms import with_retry, RetryConfig
    from .error_handlers import handle_error, ErrorPattern
    from .config_validator import ConfigurationLoader, OrchestraConfig
    from .http_client import HttpClient
    from .database_utils import DatabaseManager
    NEW_FEATURES_AVAILABLE = True
except ImportError:
    NEW_FEATURES_AVAILABLE = False

__all__ = [
    "Orchestra",
    "Agent", 
    "TaskRouter",
    "StateManager",
    "FailureHandler",
    "OrchestrationMonitor",
    "EventBus",
    "Event", 
    "EventType",
    "PluginManager",
    "PluginInterface",
    "ConfigurationManager",
    "MigrationManager",
    "AsyncRetry",
    "RateLimiter",
    "MetricsCollector",
    "AdaptiveRateLimiter",
    "ErrorRecoveryManager",
    "CircuitBreaker",
    "RecoveryStrategy",
    "HealthCheckManager",
    "HealthCheck", 
    "HealthStatus",
    "PerformanceProfiler",
    "get_profiler",
    "profile",
    "setup_logging",
    "get_logger",
    "LoggingContext",
    "get_config",
    "get_env_var",
    "DatabaseMigrationManager",
    "NEW_FEATURES_AVAILABLE"
]

# Add new features to __all__ if available
if NEW_FEATURES_AVAILABLE:
    __all__.extend([
        "validate_id",
        "validate_task_type", 
        "validate_timeout",
        "SecurityContext",
        "Permission",
        "Role",
        "create_cache",
        "get_cache",
        "cached",
        "performance_manager",
        "perf_profile",
        "create_queue",
        "get_queue", 
        "QueueType",
        "transform_data",
        "DataFormat",
        "with_retry",
        "RetryConfig", 
        "handle_error",
        "ErrorPattern",
        "ConfigurationLoader",
        "OrchestraConfig",
        "HttpClient",
        "DatabaseManager"
    ])