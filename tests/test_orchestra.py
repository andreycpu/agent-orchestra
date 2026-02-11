"""
Tests for Orchestra class
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, Mock
from agent_orchestra.orchestra import Orchestra
from agent_orchestra.agent import Agent
from agent_orchestra.types import Task, TaskStatus, AgentStatus
from agent_orchestra.exceptions import AgentRegistrationError


class TestOrchestra:
    """Test cases for Orchestra class"""
    
    @pytest.fixture
    async def orchestra(self):
        """Create a test orchestra"""
        orchestra = Orchestra(max_concurrent_tasks=5, task_timeout_default=30)
        await orchestra.start()
        yield orchestra
        await orchestra.stop()
    
    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent"""
        agent = Agent("test-agent", capabilities=["test_task"])
        
        async def test_handler(data):
            return {"result": "test completed", "data": data}
        
        agent.register_task_handler("test_task", test_handler)
        return agent
    
    @pytest.mark.asyncio
    async def test_orchestra_creation(self):
        """Test orchestra creation"""
        orchestra = Orchestra()
        assert orchestra.state_manager is not None
        assert orchestra.task_router is not None
        assert orchestra.failure_handler is not None
        assert not orchestra._is_running
    
    @pytest.mark.asyncio
    async def test_orchestra_start_stop(self):
        """Test orchestra lifecycle"""
        orchestra = Orchestra()
        
        # Start orchestra
        await orchestra.start()
        assert orchestra._is_running
        assert orchestra._executor_task is not None
        assert orchestra._heartbeat_task is not None
        
        # Stop orchestra
        await orchestra.stop()
        assert not orchestra._is_running
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, orchestra, sample_agent):
        """Test agent registration"""
        # Register agent
        orchestra.register_agent(sample_agent)
        
        assert sample_agent.id in orchestra._agents
        
        # Test duplicate registration
        with pytest.raises(AgentRegistrationError):
            orchestra.register_agent(sample_agent)
    
    @pytest.mark.asyncio
    async def test_agent_unregistration(self, orchestra, sample_agent):
        """Test agent unregistration"""
        # Register then unregister
        orchestra.register_agent(sample_agent)
        assert sample_agent.id in orchestra._agents
        
        orchestra.unregister_agent(sample_agent.id)
        assert sample_agent.id not in orchestra._agents
    
    @pytest.mark.asyncio
    async def test_task_submission(self, orchestra, sample_agent):
        """Test task submission"""
        orchestra.register_agent(sample_agent)
        
        # Submit task
        task_id = await orchestra.submit_task({
            "type": "test_task",
            "data": {"message": "hello"},
            "priority": "normal"
        })
        
        assert task_id is not None
        assert len(task_id) > 0
    
    @pytest.mark.asyncio
    async def test_task_execution(self, orchestra, sample_agent):
        """Test complete task execution"""
        orchestra.register_agent(sample_agent)
        
        # Submit and wait for task
        task_id = await orchestra.submit_task({
            "type": "test_task",
            "data": {"message": "hello"},
            "priority": "normal"
        })
        
        # Wait for completion
        result = await orchestra.wait_for_task(task_id, timeout=10)
        
        assert result is not None
        assert result.success
        assert result.task_id == task_id
        assert "result" in result.result
    
    @pytest.mark.asyncio
    async def test_get_orchestra_status(self, orchestra):
        """Test status retrieval"""
        status = await orchestra.get_status()
        
        assert "is_running" in status
        assert "running_tasks" in status
        assert "max_concurrent_tasks" in status
        assert "registered_agents" in status
    
    @pytest.mark.asyncio
    async def test_get_agents(self, orchestra, sample_agent):
        """Test agent listing"""
        # Initially no agents
        agents = orchestra.get_agents()
        assert len(agents) == 0
        
        # Register agent
        orchestra.register_agent(sample_agent)
        agents = orchestra.get_agents()
        assert len(agents) == 1
        assert agents[0].id == sample_agent.id
    
    @pytest.mark.asyncio
    async def test_concurrent_task_limit(self, orchestra, sample_agent):
        """Test concurrent task limit enforcement"""
        orchestra.register_agent(sample_agent)
        
        # Create many tasks
        task_ids = []
        for i in range(10):
            task_id = await orchestra.submit_task({
                "type": "test_task",
                "data": {"index": i},
                "priority": "normal"
            })
            task_ids.append(task_id)
        
        # Should not exceed concurrent limit
        assert len(orchestra._running_tasks) <= orchestra._max_concurrent_tasks
    
    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, orchestra):
        """Test task timeout handling"""
        # Create agent with slow handler
        slow_agent = Agent("slow-agent", capabilities=["slow_task"])
        
        async def slow_handler(data):
            await asyncio.sleep(5)  # Longer than timeout
            return {"result": "slow completed"}
        
        slow_agent.register_task_handler("slow_task", slow_handler)
        orchestra.register_agent(slow_agent)
        
        # Submit task with short timeout
        task_id = await orchestra.submit_task({
            "type": "slow_task",
            "data": {},
            "timeout": 1  # 1 second timeout
        })
        
        # Should timeout
        result = await orchestra.wait_for_task(task_id, timeout=3)
        assert not result.success
        assert "timeout" in result.error.lower() or "timeerror" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_task_failure_handling(self, orchestra):
        """Test task failure handling"""
        # Create agent with failing handler
        failing_agent = Agent("failing-agent", capabilities=["failing_task"])
        
        async def failing_handler(data):
            raise ValueError("Simulated failure")
        
        failing_agent.register_task_handler("failing_task", failing_handler)
        orchestra.register_agent(failing_agent)
        
        # Submit failing task
        task_id = await orchestra.submit_task({
            "type": "failing_task",
            "data": {},
            "max_retries": 1
        })
        
        # Should fail after retries
        result = await orchestra.wait_for_task(task_id, timeout=5)
        assert not result.success
        assert "failure" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_no_suitable_agent(self, orchestra, sample_agent):
        """Test behavior when no suitable agent is available"""
        orchestra.register_agent(sample_agent)
        
        # Submit task that no agent can handle
        task_id = await orchestra.submit_task({
            "type": "unknown_task",
            "data": {},
            "priority": "normal"
        })
        
        # Task should remain pending
        await asyncio.sleep(1)  # Give time for routing attempt
        
        # Task should still be in queue or pending
        task = await orchestra.state_manager.get_task(task_id)
        assert task is None or task.status == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_multiple_agents(self, orchestra):
        """Test with multiple agents"""
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = Agent(f"agent-{i}", capabilities=["multi_task"])
            
            async def handler(data):
                return {"agent": agent.id, "data": data}
            
            agent.register_task_handler("multi_task", handler)
            agents.append(agent)
            orchestra.register_agent(agent)
        
        # Submit multiple tasks
        task_ids = []
        for i in range(5):
            task_id = await orchestra.submit_task({
                "type": "multi_task",
                "data": {"index": i}
            })
            task_ids.append(task_id)
        
        # Wait for all tasks
        results = []
        for task_id in task_ids:
            result = await orchestra.wait_for_task(task_id, timeout=10)
            results.append(result)
        
        # All should succeed
        assert all(r.success for r in results)
        
        # Should use different agents
        agent_ids = {r.result["agent"] for r in results}
        assert len(agent_ids) > 1  # Load should be distributed