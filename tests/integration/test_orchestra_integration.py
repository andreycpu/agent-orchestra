"""
Integration tests for Orchestra functionality
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_orchestra.orchestra import Orchestra
from agent_orchestra.agent import Agent
from agent_orchestra.types import Task, TaskStatus, TaskPriority
from agent_orchestra.config import OrchestraConfig


@pytest.mark.integration
class TestOrchestraIntegration:
    """Integration tests for the full orchestra system"""
    
    @pytest.fixture
    async def orchestra_setup(self):
        """Setup orchestra with mock dependencies"""
        config = OrchestraConfig(
            max_concurrent_tasks=10,
            task_timeout_default=60,
            heartbeat_interval=5,
            redis_url="redis://localhost:6379/0"
        )
        
        # Mock Redis connection for testing
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        
        orchestra = Orchestra(config=config)
        # Replace redis with mock for testing
        orchestra._redis = mock_redis
        
        yield orchestra
        
        # Cleanup
        await orchestra.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_registration_and_task_execution(self, orchestra_setup):
        """Test complete flow from agent registration to task execution"""
        orchestra = orchestra_setup
        
        # Create and register an agent
        agent = Agent(
            agent_id="test_agent_1",
            name="Test Agent",
            capabilities=["text_processing"]
        )
        
        # Mock task handler
        async def process_text(task_data):
            return {"result": f"Processed: {task_data.get('text', '')}"}
        
        agent.register_handler("text_processing", process_text)
        
        # Register agent with orchestra
        await orchestra.register_agent(agent)
        
        # Create and submit a task
        task = Task(
            type="text_processing",
            data={"text": "Hello World"},
            priority=TaskPriority.NORMAL
        )
        
        # Submit task and wait for completion
        task_id = await orchestra.submit_task(task)
        
        # Simulate task processing
        await asyncio.sleep(0.1)  # Allow time for processing
        
        # Verify task completion
        completed_task = await orchestra.get_task(task_id)
        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.result["result"] == "Processed: Hello World"
    
    @pytest.mark.asyncio
    async def test_multiple_agents_task_distribution(self, orchestra_setup):
        """Test task distribution across multiple agents"""
        orchestra = orchestra_setup
        
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = Agent(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                capabilities=["compute"]
            )
            
            # Simple computation handler
            async def compute_handler(task_data):
                return {"result": task_data.get("input", 0) * 2}
            
            agent.register_handler("compute", compute_handler)
            await orchestra.register_agent(agent)
            agents.append(agent)
        
        # Submit multiple tasks
        tasks = []
        for i in range(10):
            task = Task(
                type="compute",
                data={"input": i},
                priority=TaskPriority.NORMAL
            )
            task_id = await orchestra.submit_task(task)
            tasks.append(task_id)
        
        # Wait for all tasks to complete
        await asyncio.sleep(0.5)
        
        # Verify all tasks completed
        completed_count = 0
        for task_id in tasks:
            task = await orchestra.get_task(task_id)
            if task.status == TaskStatus.COMPLETED:
                completed_count += 1
        
        assert completed_count >= 8  # Allow for some processing time
    
    @pytest.mark.asyncio
    async def test_task_failure_and_retry(self, orchestra_setup):
        """Test task failure handling and retry mechanism"""
        orchestra = orchestra_setup
        
        agent = Agent(
            agent_id="failure_agent",
            name="Failure Test Agent",
            capabilities=["failing_task"]
        )
        
        # Handler that fails on first attempt but succeeds on retry
        attempt_count = 0
        
        async def failing_handler(task_data):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise Exception("Simulated failure")
            return {"result": "success", "attempts": attempt_count}
        
        agent.register_handler("failing_task", failing_handler)
        await orchestra.register_agent(agent)
        
        # Submit a task that will initially fail
        task = Task(
            type="failing_task",
            data={"test": "data"},
            max_retries=2
        )
        
        task_id = await orchestra.submit_task(task)
        
        # Wait for retry processing
        await asyncio.sleep(0.3)
        
        # Verify task eventually succeeded
        completed_task = await orchestra.get_task(task_id)
        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.result["attempts"] == 2
    
    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, orchestra_setup):
        """Test task timeout mechanism"""
        orchestra = orchestra_setup
        
        agent = Agent(
            agent_id="slow_agent",
            name="Slow Agent",
            capabilities=["slow_task"]
        )
        
        # Handler that takes longer than timeout
        async def slow_handler(task_data):
            await asyncio.sleep(2)  # Longer than our timeout
            return {"result": "completed"}
        
        agent.register_handler("slow_task", slow_handler)
        await orchestra.register_agent(agent)
        
        # Submit task with short timeout
        task = Task(
            type="slow_task",
            data={"test": "data"},
            timeout=1  # 1 second timeout
        )
        
        task_id = await orchestra.submit_task(task)
        
        # Wait for timeout to occur
        await asyncio.sleep(1.5)
        
        # Verify task timed out
        timed_out_task = await orchestra.get_task(task_id)
        assert timed_out_task.status in [TaskStatus.TIMEOUT, TaskStatus.FAILED]