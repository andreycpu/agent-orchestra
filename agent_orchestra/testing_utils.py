"""
Testing utilities and fixtures for Agent Orchestra.

This module provides test helpers, mocks, fixtures, and utilities
to simplify testing of orchestration components.
"""
import asyncio
import uuid
import time
import random
from typing import Any, Dict, List, Optional, Callable, Union, AsyncGenerator
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, AsyncMock
from contextlib import asynccontextmanager, contextmanager
import tempfile
import json
from pathlib import Path

from .types import Task, TaskStatus, AgentInfo, AgentStatus, ExecutionResult, TaskPriority
from .exceptions import *


class MockAgent:
    """Mock agent for testing orchestration logic."""
    
    def __init__(
        self, 
        agent_id: str = None,
        name: str = None,
        capabilities: List[str] = None,
        status: AgentStatus = AgentStatus.IDLE,
        failure_rate: float = 0.0,
        processing_delay: float = 0.1
    ):
        self.id = agent_id or f"test-agent-{uuid.uuid4().hex[:8]}"
        self.name = name or f"Test Agent {self.id[-8:]}"
        self.capabilities = capabilities or ["test_task", "general"]
        self.status = status
        self.failure_rate = failure_rate
        self.processing_delay = processing_delay
        self.current_task = None
        self.task_history: List[str] = []
        self.last_heartbeat = datetime.utcnow()
        
    def can_handle_task(self, task_type: str) -> bool:
        """Check if agent can handle the given task type."""
        return task_type in self.capabilities or "general" in self.capabilities
    
    async def execute_task(self, task: Task) -> ExecutionResult:
        """Mock task execution with configurable behavior."""
        self.current_task = task.id
        self.task_history.append(task.id)
        
        # Simulate processing delay
        if self.processing_delay > 0:
            await asyncio.sleep(self.processing_delay)
        
        # Simulate random failures
        if random.random() < self.failure_rate:
            raise TaskExecutionError(f"Simulated failure for task {task.id}")
        
        # Generate mock result
        result = ExecutionResult(
            task_id=task.id,
            success=True,
            result={"status": "completed", "data": f"Result for {task.type}"},
            execution_time=self.processing_delay,
            metadata={"agent_id": self.id}
        )
        
        self.current_task = None
        return result
    
    def to_agent_info(self) -> AgentInfo:
        """Convert to AgentInfo object."""
        from .types import AgentCapability
        
        capabilities = [
            AgentCapability(name=cap, description=f"Test capability: {cap}")
            for cap in self.capabilities
        ]
        
        return AgentInfo(
            id=self.id,
            name=self.name,
            capabilities=capabilities,
            status=self.status,
            current_task=self.current_task,
            last_heartbeat=self.last_heartbeat,
            metadata={"test_agent": True}
        )


class TaskFactory:
    """Factory for creating test tasks with various configurations."""
    
    @staticmethod
    def create_simple_task(
        task_type: str = "test_task",
        priority: TaskPriority = TaskPriority.NORMAL,
        data: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a simple test task."""
        return Task(
            type=task_type,
            priority=priority,
            data=data or {"test": True, "value": 42}
        )
    
    @staticmethod
    def create_task_with_timeout(
        timeout: int = 30,
        task_type: str = "timeout_test",
        **kwargs
    ) -> Task:
        """Create a task with timeout."""
        return Task(
            type=task_type,
            timeout=timeout,
            data=kwargs.get("data", {"timeout_test": True})
        )
    
    @staticmethod
    def create_task_with_dependencies(
        dependencies: List[str],
        task_type: str = "dependent_task",
        **kwargs
    ) -> Task:
        """Create a task with dependencies."""
        return Task(
            type=task_type,
            dependencies=dependencies,
            data=kwargs.get("data", {"has_dependencies": True})
        )
    
    @staticmethod
    def create_task_chain(length: int = 3, prefix: str = "chain") -> List[Task]:
        """Create a chain of dependent tasks."""
        tasks = []
        prev_task_id = None
        
        for i in range(length):
            dependencies = [prev_task_id] if prev_task_id else []
            task = Task(
                type=f"{prefix}_task_{i}",
                dependencies=dependencies,
                data={"chain_position": i, "chain_length": length}
            )
            tasks.append(task)
            prev_task_id = task.id
        
        return tasks
    
    @staticmethod
    def create_task_batch(
        count: int = 5,
        task_type: str = "batch_task",
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> List[Task]:
        """Create a batch of independent tasks."""
        return [
            Task(
                type=f"{task_type}_{i}",
                priority=priority,
                data={"batch_index": i, "batch_size": count}
            )
            for i in range(count)
        ]


class OrchestrationTestHarness:
    """Test harness for orchestration scenarios."""
    
    def __init__(self):
        self.mock_agents: List[MockAgent] = []
        self.submitted_tasks: List[Task] = []
        self.completed_tasks: List[ExecutionResult] = []
        self.events: List[Dict[str, Any]] = []
        
    def add_agent(
        self,
        capabilities: List[str] = None,
        failure_rate: float = 0.0,
        processing_delay: float = 0.1
    ) -> MockAgent:
        """Add a mock agent to the test harness."""
        agent = MockAgent(
            capabilities=capabilities,
            failure_rate=failure_rate,
            processing_delay=processing_delay
        )
        self.mock_agents.append(agent)
        return agent
    
    def submit_task(self, task: Task):
        """Submit a task to the test harness."""
        self.submitted_tasks.append(task)
        self.events.append({
            "type": "task_submitted",
            "task_id": task.id,
            "timestamp": datetime.utcnow()
        })
    
    def complete_task(self, result: ExecutionResult):
        """Mark a task as completed."""
        self.completed_tasks.append(result)
        self.events.append({
            "type": "task_completed",
            "task_id": result.task_id,
            "success": result.success,
            "timestamp": datetime.utcnow()
        })
    
    def get_agent_by_capability(self, capability: str) -> Optional[MockAgent]:
        """Find an agent with the specified capability."""
        for agent in self.mock_agents:
            if agent.can_handle_task(capability):
                return agent
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get test execution statistics."""
        successful_tasks = sum(1 for result in self.completed_tasks if result.success)
        failed_tasks = len(self.completed_tasks) - successful_tasks
        
        return {
            "agents_count": len(self.mock_agents),
            "tasks_submitted": len(self.submitted_tasks),
            "tasks_completed": len(self.completed_tasks),
            "tasks_successful": successful_tasks,
            "tasks_failed": failed_tasks,
            "success_rate": successful_tasks / max(len(self.completed_tasks), 1),
            "events_count": len(self.events)
        }


class ConfigurationTestHelper:
    """Helper for testing configuration scenarios."""
    
    @staticmethod
    def create_temp_config_file(config_data: Dict[str, Any]) -> Path:
        """Create a temporary configuration file."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            delete=False
        )
        json.dump(config_data, temp_file, indent=2)
        temp_file.close()
        return Path(temp_file.name)
    
    @staticmethod
    def create_invalid_config() -> Dict[str, Any]:
        """Create an invalid configuration for testing error handling."""
        return {
            "invalid_field": "this should not be here",
            "timeout": -1,  # Invalid timeout
            "max_retries": "not a number"  # Invalid type
        }
    
    @staticmethod
    def create_minimal_config() -> Dict[str, Any]:
        """Create a minimal valid configuration."""
        return {
            "max_agents": 5,
            "task_timeout": 300,
            "heartbeat_interval": 30
        }
    
    @staticmethod
    def create_complete_config() -> Dict[str, Any]:
        """Create a complete configuration with all options."""
        return {
            "max_agents": 10,
            "task_timeout": 600,
            "heartbeat_interval": 15,
            "retry_policy": {
                "max_retries": 3,
                "base_delay": 1.0,
                "max_delay": 60.0
            },
            "security": {
                "enable_authentication": True,
                "token_expiry": 3600
            },
            "logging": {
                "level": "INFO",
                "format": "json"
            }
        }


class DatabaseTestHelper:
    """Helper for testing database operations."""
    
    def __init__(self):
        self.in_memory_data: Dict[str, Any] = {
            "tasks": {},
            "agents": {},
            "results": {}
        }
    
    def create_mock_database(self) -> MagicMock:
        """Create a mock database with basic operations."""
        db_mock = MagicMock()
        
        # Mock task operations
        db_mock.save_task.side_effect = self._save_task
        db_mock.get_task.side_effect = self._get_task
        db_mock.update_task_status.side_effect = self._update_task_status
        
        # Mock agent operations
        db_mock.save_agent.side_effect = self._save_agent
        db_mock.get_agent.side_effect = self._get_agent
        
        # Mock result operations
        db_mock.save_result.side_effect = self._save_result
        
        return db_mock
    
    def _save_task(self, task: Task):
        self.in_memory_data["tasks"][task.id] = task
    
    def _get_task(self, task_id: str) -> Optional[Task]:
        return self.in_memory_data["tasks"].get(task_id)
    
    def _update_task_status(self, task_id: str, status: TaskStatus):
        if task_id in self.in_memory_data["tasks"]:
            self.in_memory_data["tasks"][task_id].status = status
    
    def _save_agent(self, agent_info: AgentInfo):
        self.in_memory_data["agents"][agent_info.id] = agent_info
    
    def _get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        return self.in_memory_data["agents"].get(agent_id)
    
    def _save_result(self, result: ExecutionResult):
        self.in_memory_data["results"][result.task_id] = result
    
    def clear_data(self):
        """Clear all in-memory data."""
        for table in self.in_memory_data.values():
            table.clear()


@asynccontextmanager
async def mock_async_context():
    """Async context manager for testing async code."""
    try:
        yield
    finally:
        # Cleanup code here
        pass


@contextmanager
def capture_exceptions():
    """Context manager to capture and analyze exceptions during testing."""
    captured_exceptions = []
    
    original_excepthook = __import__('sys').excepthook
    
    def capture_hook(exc_type, exc_value, exc_traceback):
        captured_exceptions.append((exc_type, exc_value, exc_traceback))
        original_excepthook(exc_type, exc_value, exc_traceback)
    
    __import__('sys').excepthook = capture_hook
    
    try:
        yield captured_exceptions
    finally:
        __import__('sys').excepthook = original_excepthook


class PerformanceTestHelper:
    """Helper for performance testing."""
    
    def __init__(self):
        self.measurements: List[Dict[str, Any]] = []
    
    @contextmanager
    def measure_time(self, operation: str, **context):
        """Context manager to measure execution time."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            measurement = {
                "operation": operation,
                "duration": end_time - start_time,
                "start_memory": start_memory,
                "end_memory": end_memory,
                "memory_delta": end_memory - start_memory if start_memory and end_memory else None,
                "context": context,
                "timestamp": datetime.utcnow()
            }
            
            self.measurements.append(measurement)
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance testing summary."""
        if not self.measurements:
            return {"message": "No measurements taken"}
        
        durations = [m["duration"] for m in self.measurements]
        
        return {
            "total_measurements": len(self.measurements),
            "total_duration": sum(durations),
            "average_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "operations": [m["operation"] for m in self.measurements]
        }


def create_test_scenario(
    scenario_name: str,
    num_agents: int = 3,
    num_tasks: int = 10,
    agent_failure_rate: float = 0.1
) -> OrchestrationTestHarness:
    """
    Create a pre-configured test scenario.
    
    Args:
        scenario_name: Name of the scenario for logging
        num_agents: Number of mock agents to create
        num_tasks: Number of test tasks to create
        agent_failure_rate: Failure rate for agents
    
    Returns:
        Configured test harness
    """
    harness = OrchestrationTestHarness()
    
    # Create agents with different capabilities
    capabilities_sets = [
        ["cpu_intensive", "general"],
        ["io_bound", "database", "general"],
        ["network", "api_calls", "general"],
        ["ml_processing", "general"],
        ["file_processing", "general"]
    ]
    
    for i in range(num_agents):
        capabilities = capabilities_sets[i % len(capabilities_sets)]
        harness.add_agent(
            capabilities=capabilities,
            failure_rate=agent_failure_rate
        )
    
    # Create diverse test tasks
    task_types = ["cpu_intensive", "io_bound", "network", "ml_processing", "file_processing"]
    
    for i in range(num_tasks):
        task_type = task_types[i % len(task_types)]
        task = TaskFactory.create_simple_task(
            task_type=task_type,
            priority=random.choice(list(TaskPriority)),
            data={"scenario": scenario_name, "task_index": i}
        )
        harness.submit_task(task)
    
    return harness


# Pytest fixtures for common test setup
try:
    import pytest
    
    @pytest.fixture
    def mock_agent():
        """Pytest fixture for a single mock agent."""
        return MockAgent()
    
    @pytest.fixture
    def task_factory():
        """Pytest fixture for task factory."""
        return TaskFactory()
    
    @pytest.fixture
    def test_harness():
        """Pytest fixture for orchestration test harness."""
        return OrchestrationTestHarness()
    
    @pytest.fixture
    def db_helper():
        """Pytest fixture for database test helper."""
        return DatabaseTestHelper()
    
    @pytest.fixture
    def config_helper():
        """Pytest fixture for configuration test helper."""
        return ConfigurationTestHelper()
    
    @pytest.fixture
    def performance_helper():
        """Pytest fixture for performance test helper."""
        return PerformanceTestHelper()
    
except ImportError:
    # pytest not available, skip fixture definitions
    pass