"""
Tests for type definitions and validation
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from agent_orchestra.types import (
    Task, TaskStatus, TaskPriority, AgentStatus, AgentInfo, 
    AgentCapability, ExecutionResult
)


class TestTask:
    """Test cases for Task model"""
    
    def test_task_creation_with_defaults(self):
        """Test task creation with default values"""
        task = Task(type="test_task")
        
        assert task.type == "test_task"
        assert task.priority == TaskPriority.NORMAL
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert isinstance(task.created_at, datetime)
        assert len(task.id) > 0
    
    def test_task_type_validation(self):
        """Test task type validation"""
        with pytest.raises(ValidationError, match="Task type cannot be empty"):
            Task(type="")
            
        with pytest.raises(ValidationError, match="Task type cannot be empty"):
            Task(type="   ")
    
    def test_timeout_validation(self):
        """Test timeout validation"""
        with pytest.raises(ValidationError, match="Timeout must be positive"):
            Task(type="test_task", timeout=0)
            
        with pytest.raises(ValidationError, match="Timeout must be positive"):
            Task(type="test_task", timeout=-5)
            
        # Valid timeout should work
        task = Task(type="test_task", timeout=30)
        assert task.timeout == 30
    
    def test_max_retries_validation(self):
        """Test max retries validation"""
        with pytest.raises(ValidationError, match="Max retries cannot be negative"):
            Task(type="test_task", max_retries=-1)
            
        # Zero and positive should work
        task1 = Task(type="test_task", max_retries=0)
        assert task1.max_retries == 0
        
        task2 = Task(type="test_task", max_retries=5)
        assert task2.max_retries == 5


class TestAgentCapability:
    """Test cases for AgentCapability model"""
    
    def test_capability_creation(self):
        """Test capability creation"""
        capability = AgentCapability(
            name="test_capability",
            description="A test capability",
            resource_requirements={"cpu": 0.5, "memory": "512MB"}
        )
        
        assert capability.name == "test_capability"
        assert capability.description == "A test capability"
        assert capability.resource_requirements["cpu"] == 0.5


class TestAgentInfo:
    """Test cases for AgentInfo model"""
    
    def test_agent_info_creation(self):
        """Test agent info creation"""
        agent_info = AgentInfo(
            id="agent_1",
            name="Test Agent",
            status=AgentStatus.IDLE
        )
        
        assert agent_info.id == "agent_1"
        assert agent_info.name == "Test Agent"
        assert agent_info.status == AgentStatus.IDLE
        assert isinstance(agent_info.last_heartbeat, datetime)


class TestExecutionResult:
    """Test cases for ExecutionResult model"""
    
    def test_execution_result_creation(self):
        """Test execution result creation"""
        result = ExecutionResult(
            task_id="task_1",
            success=True,
            execution_time=1.5,
            result={"output": "success"}
        )
        
        assert result.task_id == "task_1"
        assert result.success is True
        assert result.execution_time == 1.5
        assert result.result["output"] == "success"