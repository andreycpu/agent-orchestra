"""
Tests for StateManager class
"""
import pytest
import asyncio
from datetime import datetime
from agent_orchestra.state_manager import StateManager
from agent_orchestra.types import Task, AgentInfo, TaskStatus, AgentStatus, AgentCapability


class TestStateManager:
    """Test cases for StateManager class"""
    
    @pytest.fixture
    async def state_manager(self):
        """Create test state manager"""
        sm = StateManager()
        await sm.initialize()
        yield sm
        await sm.shutdown()
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task"""
        return Task(
            type="test_task",
            data={"test": "data"},
            status=TaskStatus.PENDING
        )
    
    @pytest.fixture
    def sample_agent_info(self):
        """Create sample agent info"""
        return AgentInfo(
            id="test-agent",
            name="Test Agent",
            capabilities=[AgentCapability(name="test_task")],
            status=AgentStatus.IDLE
        )
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_task(self, state_manager, sample_task):
        """Test task storage and retrieval"""
        # Store task
        await state_manager.store_task(sample_task)
        
        # Retrieve task
        retrieved_task = await state_manager.get_task(sample_task.id)
        
        assert retrieved_task is not None
        assert retrieved_task.id == sample_task.id
        assert retrieved_task.type == sample_task.type
        assert retrieved_task.data == sample_task.data
    
    @pytest.mark.asyncio
    async def test_update_task_status(self, state_manager, sample_task):
        """Test task status updates"""
        # Store initial task
        await state_manager.store_task(sample_task)
        
        # Update status
        await state_manager.update_task_status(
            sample_task.id, 
            TaskStatus.RUNNING,
            assigned_agent="test-agent"
        )
        
        # Retrieve updated task
        updated_task = await state_manager.get_task(sample_task.id)
        
        assert updated_task.status == TaskStatus.RUNNING
        assert updated_task.assigned_agent == "test-agent"
        assert updated_task.started_at is not None
    
    @pytest.mark.asyncio
    async def test_register_and_get_agent(self, state_manager, sample_agent_info):
        """Test agent registration and retrieval"""
        # Register agent
        await state_manager.register_agent(sample_agent_info)
        
        # Retrieve agent
        retrieved_agent = await state_manager.get_agent(sample_agent_info.id)
        
        assert retrieved_agent is not None
        assert retrieved_agent.id == sample_agent_info.id
        assert retrieved_agent.name == sample_agent_info.name
        assert retrieved_agent.status == sample_agent_info.status
    
    @pytest.mark.asyncio
    async def test_get_all_agents(self, state_manager):
        """Test getting all agents"""
        # Initially empty
        agents = await state_manager.get_all_agents()
        assert len(agents) == 0
        
        # Add multiple agents
        for i in range(3):
            agent_info = AgentInfo(
                id=f"agent-{i}",
                name=f"Agent {i}",
                capabilities=[AgentCapability(name="test_task")],
                status=AgentStatus.IDLE
            )
            await state_manager.register_agent(agent_info)
        
        # Retrieve all
        agents = await state_manager.get_all_agents()
        assert len(agents) == 3
    
    @pytest.mark.asyncio
    async def test_get_tasks_by_status(self, state_manager):
        """Test filtering tasks by status"""
        # Create tasks with different statuses
        tasks = []
        for i, status in enumerate([TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED]):
            task = Task(
                type="test_task",
                data={"index": i},
                status=status
            )
            tasks.append(task)
            await state_manager.store_task(task)
        
        # Get pending tasks
        pending_tasks = await state_manager.get_tasks_by_status(TaskStatus.PENDING)
        assert len(pending_tasks) == 1
        assert pending_tasks[0].status == TaskStatus.PENDING
        
        # Get running tasks
        running_tasks = await state_manager.get_tasks_by_status(TaskStatus.RUNNING)
        assert len(running_tasks) == 1
        assert running_tasks[0].status == TaskStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_update_agent_heartbeat(self, state_manager, sample_agent_info):
        """Test agent heartbeat updates"""
        # Register agent
        await state_manager.register_agent(sample_agent_info)
        original_heartbeat = sample_agent_info.last_heartbeat
        
        # Update heartbeat
        await state_manager.update_agent_heartbeat(sample_agent_info.id)
        
        # Retrieve and check
        updated_agent = await state_manager.get_agent(sample_agent_info.id)
        assert updated_agent.last_heartbeat > original_heartbeat
    
    @pytest.mark.asyncio
    async def test_nonexistent_task(self, state_manager):
        """Test retrieving nonexistent task"""
        result = await state_manager.get_task("nonexistent-task-id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_nonexistent_agent(self, state_manager):
        """Test retrieving nonexistent agent"""
        result = await state_manager.get_agent("nonexistent-agent-id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_global_state(self, state_manager, sample_agent_info):
        """Test global state summary"""
        # Add some data
        await state_manager.register_agent(sample_agent_info)
        
        task = Task(type="test", data={}, status=TaskStatus.PENDING)
        await state_manager.store_task(task)
        
        # Get global state
        global_state = await state_manager.get_global_state()
        
        assert "timestamp" in global_state
        assert "agents" in global_state
        assert "tasks" in global_state
        assert global_state["agents"]["total"] == 1
        assert global_state["tasks"]["pending"] == 1