"""
Tests for Agent class
"""
import asyncio
import pytest
from unittest.mock import AsyncMock
from agent_orchestra.agent import Agent
from agent_orchestra.types import Task, TaskStatus, AgentStatus
from agent_orchestra.exceptions import AgentUnavailableError, TaskExecutionError, ValidationError


class TestAgent:
    """Test cases for Agent class"""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent"""
        return Agent("test-agent", capabilities=["text_processing", "math"])
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task"""
        return Task(
            type="text_processing",
            data={"text": "hello world", "operation": "upper"}
        )
    
    def test_agent_creation(self):
        """Test agent creation with capabilities"""
        agent = Agent("test-1", name="Test Agent", capabilities=["task1", "task2"])
        
        assert agent.id == "test-1"
        assert agent.name == "Test Agent"
        assert len(agent.capabilities) == 2
        assert agent.status == AgentStatus.IDLE
        assert agent.current_task is None
    
    def test_agent_capability_handling(self, agent, sample_task):
        """Test agent capability checking"""
        # Register a handler
        async def text_handler(data):
            return {"processed": data["text"].upper()}
        
        agent.register_task_handler("text_processing", text_handler)
        
        assert agent.can_handle_task(sample_task) is True
        
        # Test with unknown task type
        unknown_task = Task(type="unknown_task", data={})
        assert agent.can_handle_task(unknown_task) is False
    
    @pytest.mark.asyncio
    async def test_successful_task_execution(self, agent, sample_task):
        """Test successful task execution"""
        # Register handler
        async def text_handler(data):
            return {"processed": data["text"].upper()}
        
        agent.register_task_handler("text_processing", text_handler)
        
        # Execute task
        result = await agent.execute_task(sample_task)
        
        assert result.success is True
        assert result.task_id == sample_task.id
        assert result.result == {"processed": "HELLO WORLD"}
        assert result.execution_time > 0
        assert agent.status == AgentStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_task_execution_failure(self, agent, sample_task):
        """Test task execution failure handling"""
        # Register handler that raises exception
        async def failing_handler(data):
            raise ValueError("Simulated failure")
        
        agent.register_task_handler("text_processing", failing_handler)
        
        # Execute task
        result = await agent.execute_task(sample_task)
        
        assert result.success is False
        assert result.task_id == sample_task.id
        assert "Simulated failure" in result.error
        assert result.execution_time > 0
        assert agent.status == AgentStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_agent_unavailable(self, agent, sample_task):
        """Test agent unavailable error"""
        agent.status = AgentStatus.BUSY
        
        with pytest.raises(AgentUnavailableError):
            await agent.execute_task(sample_task)
    
    @pytest.mark.asyncio
    async def test_unhandled_task_type(self, agent):
        """Test execution of unhandled task type"""
        unknown_task = Task(type="unknown_task", data={})
        
        with pytest.raises(TaskExecutionError):
            await agent.execute_task(unknown_task)
    
    @pytest.mark.asyncio
    async def test_task_timeout(self, agent):
        """Test task timeout handling"""
        # Register handler with long execution time
        async def slow_handler(data):
            await asyncio.sleep(2)
            return {"result": "completed"}
        
        agent.register_task_handler("slow_task", slow_handler)
        
        # Create task with short timeout
        timeout_task = Task(
            type="slow_task",
            data={},
            timeout=0.5
        )
        
        # Execute task
        result = await agent.execute_task(timeout_task)
        
        assert result.success is False
        assert "TimeoutError" in result.error or "timeout" in result.error.lower()
    
    def test_agent_info(self, agent):
        """Test agent info retrieval"""
        info = agent.get_info()
        
        assert info.id == agent.id
        assert info.name == agent.name
        assert info.status == agent.status
        assert len(info.capabilities) == len(agent.capabilities)
    
    def test_heartbeat_update(self, agent):
        """Test heartbeat functionality"""
        original_heartbeat = agent.get_info().last_heartbeat
        
        # Wait a bit and update heartbeat
        import time
        time.sleep(0.01)
        agent.update_heartbeat()
        
        new_heartbeat = agent.get_info().last_heartbeat
        assert new_heartbeat > original_heartbeat
    
    def test_status_update(self, agent):
        """Test agent status updates"""
        assert agent.status == AgentStatus.IDLE
        
        agent.set_status(AgentStatus.BUSY)
        assert agent.status == AgentStatus.BUSY
        
        agent.set_status(AgentStatus.UNAVAILABLE)
        assert agent.status == AgentStatus.UNAVAILABLE
    
    @pytest.mark.asyncio
    async def test_concurrent_task_prevention(self, agent):
        """Test that agent prevents concurrent task execution"""
        # Register a handler with delay
        async def slow_handler(data):
            await asyncio.sleep(0.5)
            return {"result": "completed"}
        
        agent.register_task_handler("slow_task", slow_handler)
        
        task1 = Task(type="slow_task", data={})
        task2 = Task(type="slow_task", data={})
        
        # Start first task
        task1_coroutine = agent.execute_task(task1)
        
        # Give it time to start
        await asyncio.sleep(0.1)
        
        # Try to start second task (should fail)
        with pytest.raises(AgentUnavailableError):
            await agent.execute_task(task2)
        
        # Wait for first task to complete
        result1 = await task1_coroutine
        assert result1.success is True
    
    def test_capability_registration(self, agent):
        """Test dynamic capability registration"""
        initial_capabilities = len(agent.capabilities)
        
        # Register new task handler
        async def new_handler(data):
            return {"result": "new task completed"}
        
        agent.register_task_handler("new_task_type", new_handler)
        
        # Should have added new capability
        assert len(agent.capabilities) == initial_capabilities + 1
        
        # Should be able to handle the new task type
        new_task = Task(type="new_task_type", data={})
        assert agent.can_handle_task(new_task) is True


class TestAgentValidation:
    """Test cases for Agent input validation"""
    
    def test_agent_creation_invalid_agent_id(self):
        """Test agent creation with invalid agent_id"""
        # Empty string
        with pytest.raises(ValidationError, match="agent_id must be a non-empty string"):
            Agent("")
        
        # None
        with pytest.raises(ValidationError, match="agent_id must be a non-empty string"):
            Agent(None)
        
        # Not a string
        with pytest.raises(ValidationError, match="agent_id must be a non-empty string"):
            Agent(123)
        
        # Whitespace only
        with pytest.raises(ValidationError, match="agent_id cannot be whitespace only"):
            Agent("   ")
    
    def test_agent_creation_invalid_name(self):
        """Test agent creation with invalid name"""
        # Not a string
        with pytest.raises(ValidationError, match="name must be a string"):
            Agent("test", name=123)
        
        # None should be ok (uses agent_id)
        agent = Agent("test", name=None)
        assert agent.name == "test"
    
    def test_agent_creation_invalid_capabilities(self):
        """Test agent creation with invalid capabilities"""
        # Not a list
        with pytest.raises(ValidationError, match="capabilities must be a list"):
            Agent("test", capabilities="invalid")
        
        # List with non-string items
        with pytest.raises(ValidationError, match="all capabilities must be strings"):
            Agent("test", capabilities=["valid", 123, "also_valid"])
    
    def test_agent_creation_invalid_metadata(self):
        """Test agent creation with invalid metadata"""
        # Not a dict
        with pytest.raises(ValidationError, match="metadata must be a dictionary"):
            Agent("test", metadata="invalid")
    
    def test_agent_creation_whitespace_handling(self):
        """Test agent creation handles whitespace correctly"""
        agent = Agent("  test-agent  ", name="  Test Agent  ")
        
        assert agent.id == "test-agent"
        assert agent.name == "Test Agent"
    
    def test_register_handler_invalid_task_type(self, agent):
        """Test registering handler with invalid task_type"""
        async def dummy_handler(data):
            return data
        
        # Empty string
        with pytest.raises(ValidationError, match="task_type must be a non-empty string"):
            agent.register_task_handler("", dummy_handler)
        
        # None
        with pytest.raises(ValidationError, match="task_type must be a non-empty string"):
            agent.register_task_handler(None, dummy_handler)
        
        # Not a string
        with pytest.raises(ValidationError, match="task_type must be a non-empty string"):
            agent.register_task_handler(123, dummy_handler)
        
        # Whitespace only
        with pytest.raises(ValidationError, match="task_type cannot be whitespace only"):
            agent.register_task_handler("   ", dummy_handler)
    
    def test_register_handler_invalid_handler(self, agent):
        """Test registering invalid handler"""
        # Not callable
        with pytest.raises(ValidationError, match="handler must be callable"):
            agent.register_task_handler("test_task", "not_callable")
        
        with pytest.raises(ValidationError, match="handler must be callable"):
            agent.register_task_handler("test_task", 123)
    
    def test_register_handler_whitespace_handling(self, agent):
        """Test handler registration handles whitespace correctly"""
        async def dummy_handler(data):
            return data
        
        agent.register_task_handler("  task_type  ", dummy_handler)
        
        # Should be registered without whitespace
        test_task = Task(type="task_type", data={})
        assert agent.can_handle_task(test_task) is True
    
    @pytest.mark.asyncio
    async def test_execute_task_invalid_task(self, agent):
        """Test executing invalid task"""
        # None task
        with pytest.raises(ValidationError, match="task cannot be None"):
            await agent.execute_task(None)
        
        # Not a Task instance
        with pytest.raises(ValidationError, match="task must be a Task instance"):
            await agent.execute_task("not_a_task")
    
    @pytest.mark.asyncio
    async def test_execute_task_invalid_task_type(self, agent):
        """Test executing task with invalid type"""
        # Create a mock task-like object without proper type
        class MockTask:
            def __init__(self, task_type=None):
                if task_type is not None:
                    self.type = task_type
                self.data = {}
                self.timeout = None
                self.id = "test-task"
        
        # No type attribute
        mock_task = MockTask()
        with pytest.raises(ValidationError, match="task must have a valid type"):
            await agent.execute_task(mock_task)
        
        # Empty type
        mock_task = MockTask("")
        with pytest.raises(ValidationError, match="task must have a valid type"):
            await agent.execute_task(mock_task)