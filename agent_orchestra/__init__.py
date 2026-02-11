"""
Agent Orchestra - Multi-agent orchestration framework
"""

__version__ = "0.1.0"
__title__ = "Agent Orchestra"
__description__ = "A multi-agent orchestration framework"
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
from .utils import AsyncRetry, RateLimiter, MetricsCollector

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
    "MetricsCollector"
]