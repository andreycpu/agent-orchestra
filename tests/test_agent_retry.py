"""
Tests for Agent retry functionality and enhanced error handling
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timedelta
from agent_orchestra.agent import Agent
from agent_orchestra.types import Task, AgentStatus
from agent_orchestra.exceptions import (
    ValidationError, StatusTransitionError, HeartbeatError, 
    IncompatibleTaskError, TaskValidationError
)


class TestAgentRetry:
    """Test cases for Agent retry functionality"""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent"""
        return Agent("retry-test-agent", capabilities=["retry_task"])
    
    @pytest.fixture
    def retry_task(self):
        """Create a task for retry testing"""
        return Task(type="retry_task", data={"value": 42})
    
    @pytest.mark.asyncio
    async def test_execute_task_with_retry_success_first_attempt(self, agent, retry_task):
        """Test successful execution on first attempt"""
        async def success_handler(data):
            return {"result": data["value"] * 2}
        
        agent.register_task_handler("retry_task", success_handler)
        
        result = await agent.execute_task_with_retry(retry_task, max_retries=3)
        
        assert result.success is True
        assert result.result == {"result": 84}
    
    @pytest.mark.asyncio
    async def test_execute_task_with_retry_success_after_failures(self, agent, retry_task):
        """Test successful execution after initial failures"""
        attempt_count = 0
        
        async def flaky_handler(data):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RuntimeError("Temporary failure")
            return {"result": "success after retries"}
        
        agent.register_task_handler("retry_task", flaky_handler)
        
        result = await agent.execute_task_with_retry(retry_task, max_retries=5)
        
        assert result.success is True
        assert result.result == {"result": "success after retries"}
        assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_task_with_retry_exhausted(self, agent, retry_task):
        """Test retry exhaustion with persistent failures"""
        attempt_count = 0
        
        async def failing_handler(data):
            nonlocal attempt_count
            attempt_count += 1
            raise RuntimeError("Persistent failure")
        
        agent.register_task_handler("retry_task", failing_handler)
        
        result = await agent.execute_task_with_retry(retry_task, max_retries=2)
        
        assert result.success is False
        assert "Task failed after 2 retries" in result.error
        assert "Persistent failure" in result.error
        assert attempt_count == 3  # Original attempt + 2 retries
    
    @pytest.mark.asyncio
    async def test_execute_task_with_retry_non_retryable_error(self, agent, retry_task):
        """Test that non-retryable errors are not retried"""
        attempt_count = 0
        
        async def validation_error_handler(data):
            nonlocal attempt_count
            attempt_count += 1
            raise ValidationError("Invalid input")
        
        agent.register_task_handler("retry_task", validation_error_handler)
        
        with pytest.raises(ValidationError):
            await agent.execute_task_with_retry(retry_task, max_retries=3)
        
        # Should only attempt once, no retries
        assert attempt_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_task_with_retry_invalid_parameters(self, agent, retry_task):
        """Test retry method with invalid parameters"""
        async def dummy_handler(data):
            return data
        
        agent.register_task_handler("retry_task", dummy_handler)
        
        # Negative max_retries
        with pytest.raises(ValidationError, match="max_retries must be non-negative"):
            await agent.execute_task_with_retry(retry_task, max_retries=-1)
        
        # Negative base_delay
        with pytest.raises(ValidationError, match="base_delay must be non-negative"):
            await agent.execute_task_with_retry(retry_task, base_delay=-1.0)
        
        # Invalid exponential_base
        with pytest.raises(ValidationError, match="exponential_base must be greater than 1"):
            await agent.execute_task_with_retry(retry_task, exponential_base=1.0)
    
    @pytest.mark.asyncio
    async def test_execute_task_with_retry_exponential_backoff(self, agent, retry_task):
        """Test that exponential backoff delays are applied"""
        attempt_times = []
        
        async def timing_handler(data):
            attempt_times.append(asyncio.get_event_loop().time())
            if len(attempt_times) < 3:
                raise RuntimeError("Temporary failure")
            return {"result": "success"}
        
        agent.register_task_handler("retry_task", timing_handler)
        
        start_time = asyncio.get_event_loop().time()
        result = await agent.execute_task_with_retry(
            retry_task, max_retries=3, base_delay=0.1, exponential_base=2.0
        )
        end_time = asyncio.get_event_loop().time()
        
        assert result.success is True
        assert len(attempt_times) == 3
        
        # Check that total time includes delays (0.1 + 0.2 = 0.3 seconds minimum)
        total_time = end_time - start_time
        assert total_time >= 0.3
    
    def test_is_retryable_error(self, agent):
        """Test error retryability classification"""
        # Retryable errors
        assert agent._is_retryable_error("Connection timeout") is True
        assert agent._is_retryable_error("Temporary service unavailable") is True
        assert agent._is_retryable_error(None) is True
        
        # Non-retryable errors
        assert agent._is_retryable_error("Validation error: invalid input") is False
        assert agent._is_retryable_error("Agent unavailable") is False
        assert agent._is_retryable_error("Cannot handle this task type") is False
        assert agent._is_retryable_error("Invalid parameter provided") is False


class TestAgentStatusTransition:
    """Test cases for Agent status transition validation"""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent"""
        return Agent("status-test-agent")
    
    def test_valid_status_transitions(self, agent):
        """Test valid status transitions"""
        # IDLE -> BUSY
        agent.set_status(AgentStatus.BUSY)
        assert agent.status == AgentStatus.BUSY
        
        # BUSY -> IDLE
        agent.set_status(AgentStatus.IDLE)
        assert agent.status == AgentStatus.IDLE
        
        # IDLE -> UNAVAILABLE
        agent.set_status(AgentStatus.UNAVAILABLE)
        assert agent.status == AgentStatus.UNAVAILABLE
        
        # UNAVAILABLE -> IDLE
        agent.set_status(AgentStatus.IDLE)
        assert agent.status == AgentStatus.IDLE
    
    def test_invalid_status_transitions(self, agent):
        """Test invalid status transitions"""
        # Start in IDLE state
        assert agent.status == AgentStatus.IDLE
        
        # Invalid transitions would need to be defined based on business logic
        # For now, the current implementation allows all transitions
        # This test documents the expected behavior
        
        # Test with invalid status type
        with pytest.raises(ValidationError, match="status must be an AgentStatus enum value"):
            agent.set_status("invalid_status")
    
    def test_status_transition_validation_method(self, agent):
        """Test the status transition validation logic"""
        # Test valid transitions
        assert agent._is_valid_status_transition(AgentStatus.IDLE, AgentStatus.BUSY) is True
        assert agent._is_valid_status_transition(AgentStatus.BUSY, AgentStatus.IDLE) is True
        assert agent._is_valid_status_transition(AgentStatus.IDLE, AgentStatus.UNAVAILABLE) is True
        assert agent._is_valid_status_transition(AgentStatus.UNAVAILABLE, AgentStatus.IDLE) is True


class TestAgentHeartbeat:
    """Test cases for Agent heartbeat functionality"""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent"""
        return Agent("heartbeat-test-agent")
    
    def test_update_heartbeat_success(self, agent):
        """Test successful heartbeat update"""
        original_heartbeat = agent._last_heartbeat
        
        # Wait a bit to ensure time difference
        import time
        time.sleep(0.01)
        
        agent.update_heartbeat()
        new_heartbeat = agent._last_heartbeat
        
        assert new_heartbeat > original_heartbeat
    
    def test_heartbeat_staleness_check(self, agent):
        """Test heartbeat staleness detection"""
        # Fresh heartbeat should not be stale
        assert agent.is_heartbeat_stale(max_age_seconds=300) is False
        
        # Manually set old heartbeat
        agent._last_heartbeat = datetime.utcnow() - timedelta(seconds=400)
        assert agent.is_heartbeat_stale(max_age_seconds=300) is True
        
        # Test with different threshold
        assert agent.is_heartbeat_stale(max_age_seconds=500) is False
    
    def test_heartbeat_staleness_invalid_parameter(self, agent):
        """Test heartbeat staleness check with invalid parameter"""
        with pytest.raises(ValidationError, match="max_age_seconds must be positive"):
            agent.is_heartbeat_stale(max_age_seconds=0)
        
        with pytest.raises(ValidationError, match="max_age_seconds must be positive"):
            agent.is_heartbeat_stale(max_age_seconds=-100)
    
    @patch('agent_orchestra.agent.datetime')
    def test_heartbeat_update_error(self, mock_datetime, agent):
        """Test heartbeat update error handling"""
        # Make datetime.utcnow() raise an exception
        mock_datetime.utcnow.side_effect = RuntimeError("Time service unavailable")
        
        with pytest.raises(HeartbeatError, match="Failed to update heartbeat"):
            agent.update_heartbeat()


class TestAgentIncompatibleTask:
    """Test cases for IncompatibleTaskError"""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent with specific capabilities"""
        return Agent("compat-test-agent", capabilities=["text_processing"])
    
    @pytest.mark.asyncio
    async def test_incompatible_task_error_details(self, agent):
        """Test that IncompatibleTaskError includes helpful details"""
        incompatible_task = Task(type="image_processing", data={})
        
        try:
            await agent.execute_task(incompatible_task)
            pytest.fail("Should have raised IncompatibleTaskError")
        except IncompatibleTaskError as e:
            assert "cannot handle task type" in str(e)
            assert e.details["agent_id"] == "compat-test-agent"
            assert e.details["task_type"] == "image_processing"
            assert "text_processing" in e.details["available_capabilities"]
    
    @pytest.mark.asyncio
    async def test_task_validation_error(self, agent):
        """Test TaskValidationError for invalid task"""
        # Create invalid task-like object
        class InvalidTask:
            def __init__(self):
                self.data = {}
                self.timeout = None
                self.id = "test-task"
                # Missing 'type' attribute
        
        invalid_task = InvalidTask()
        
        with pytest.raises(TaskValidationError, match="task must have a valid type"):
            await agent.execute_task(invalid_task)