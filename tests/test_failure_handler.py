"""
Tests for FailureHandler class
"""
import pytest
from agent_orchestra.failure_handler import FailureHandler, FailureType, RecoveryStrategy
from agent_orchestra.types import Task
from agent_orchestra.exceptions import TaskExecutionError, TaskTimeoutError


class TestFailureHandler:
    """Test cases for FailureHandler class"""
    
    @pytest.fixture
    def failure_handler(self):
        """Create test failure handler"""
        return FailureHandler()
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task"""
        return Task(type="test_task", data={"test": "data"})
    
    @pytest.mark.asyncio
    async def test_failure_classification(self, failure_handler):
        """Test error classification"""
        # Test different error types
        timeout_error = TaskTimeoutError("Task timed out")
        classified = failure_handler._classify_failure(timeout_error)
        assert classified == FailureType.TASK_TIMEOUT
        
        exec_error = TaskExecutionError("Execution failed")
        classified = failure_handler._classify_failure(exec_error)
        assert classified == FailureType.EXECUTION_ERROR
        
        unknown_error = ValueError("Unknown error")
        classified = failure_handler._classify_failure(unknown_error)
        assert classified == FailureType.VALIDATION_ERROR
    
    @pytest.mark.asyncio
    async def test_handle_failure(self, failure_handler, sample_task):
        """Test failure handling"""
        error = TaskExecutionError("Test failure")
        
        strategy = await failure_handler.handle_failure(
            sample_task, "test-agent", error
        )
        
        assert strategy in RecoveryStrategy
        assert len(failure_handler._failure_history) == 1
    
    def test_circuit_breaker(self, failure_handler):
        """Test circuit breaker functionality"""
        agent_id = "test-agent"
        
        # Initially closed
        assert not failure_handler._is_circuit_breaker_open(agent_id)
        
        # Trigger failures
        for _ in range(3):
            failure_handler._update_circuit_breaker(agent_id, FailureType.EXECUTION_ERROR)
        
        # Should be open now
        assert failure_handler._is_circuit_breaker_open(agent_id)
        
        # Reset
        failure_handler.reset_circuit_breaker(agent_id)
        assert not failure_handler._is_circuit_breaker_open(agent_id)
    
    def test_failure_statistics(self, failure_handler, sample_task):
        """Test failure statistics collection"""
        # Initially empty
        stats = failure_handler.get_failure_statistics()
        assert stats["total_failures"] == 0
        
        # Add some failures
        for i in range(3):
            failure_handler._failure_history.append(
                failure_handler.FailureRecord(
                    task_id=f"task-{i}",
                    agent_id="test-agent",
                    failure_type=FailureType.EXECUTION_ERROR,
                    error_message="Test error"
                )
            )
        
        stats = failure_handler.get_failure_statistics()
        assert stats["total_failures"] == 3
    
    def test_backoff_calculation(self, failure_handler):
        """Test exponential backoff calculation"""
        task_id = "test-task"
        
        # No failures - base delay
        delay = failure_handler.calculate_backoff_delay(task_id, base_delay=1.0)
        assert delay == 1.0
        
        # Add failures
        for i in range(3):
            failure_handler._failure_history.append(
                failure_handler.FailureRecord(
                    task_id=task_id,
                    agent_id="test-agent",
                    failure_type=FailureType.EXECUTION_ERROR,
                    error_message="Test error"
                )
            )
        
        # Should have exponential backoff
        delay = failure_handler.calculate_backoff_delay(task_id, base_delay=1.0)
        assert delay > 1.0