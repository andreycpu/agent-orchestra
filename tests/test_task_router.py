"""
Tests for TaskRouter class
"""
import pytest
from datetime import datetime
from agent_orchestra.task_router import TaskRouter
from agent_orchestra.types import Task, AgentInfo, AgentCapability, AgentStatus, TaskPriority
from agent_orchestra.exceptions import CircularDependencyError


class TestTaskRouter:
    """Test cases for TaskRouter class"""
    
    @pytest.fixture
    def router(self):
        """Create a test task router"""
        return TaskRouter()
    
    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for testing"""
        return [
            AgentInfo(
                id="agent-1",
                name="Text Processor",
                capabilities=[AgentCapability(name="text_processing")],
                status=AgentStatus.IDLE
            ),
            AgentInfo(
                id="agent-2", 
                name="Math Processor",
                capabilities=[AgentCapability(name="math_processing")],
                status=AgentStatus.IDLE
            ),
            AgentInfo(
                id="agent-3",
                name="Multi Processor",
                capabilities=[
                    AgentCapability(name="text_processing"),
                    AgentCapability(name="data_processing")
                ],
                status=AgentStatus.IDLE
            ),
            AgentInfo(
                id="agent-4",
                name="Busy Agent",
                capabilities=[AgentCapability(name="text_processing")],
                status=AgentStatus.BUSY
            )
        ]
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing"""
        return [
            Task(
                type="text_processing",
                data={"text": "hello"},
                priority=TaskPriority.HIGH
            ),
            Task(
                type="math_processing",
                data={"numbers": [1, 2, 3]},
                priority=TaskPriority.NORMAL
            ),
            Task(
                type="data_processing",
                data={"data": "test"},
                priority=TaskPriority.LOW
            ),
            Task(
                type="urgent_task",
                data={},
                priority=TaskPriority.URGENT
            )
        ]
    
    def test_agent_registration(self, router, sample_agents):
        """Test agent registration and unregistration"""
        agent = sample_agents[0]
        
        # Register agent
        router.register_agent(agent)
        assert agent.id in router._agents
        
        # Unregister agent
        router.unregister_agent(agent.id)
        assert agent.id not in router._agents
    
    def test_agent_status_update(self, router, sample_agents):
        """Test agent status updates"""
        agent = sample_agents[0]
        router.register_agent(agent)
        
        # Update status
        updated_agent = AgentInfo(
            id=agent.id,
            name=agent.name,
            capabilities=agent.capabilities,
            status=AgentStatus.BUSY
        )
        
        router.update_agent_status(agent.id, updated_agent)
        assert router._agents[agent.id].status == AgentStatus.BUSY
    
    def test_task_addition_and_retrieval(self, router, sample_tasks):
        """Test task queue management"""
        # Add tasks with different priorities
        for task in sample_tasks:
            router.add_task(task)
        
        # Should retrieve highest priority task first
        next_task = router.get_next_task()
        assert next_task is not None
        assert next_task.priority == TaskPriority.URGENT
    
    def test_priority_ordering(self, router):
        """Test that tasks are returned in priority order"""
        # Create tasks with different priorities
        tasks = [
            Task(type="task", priority=TaskPriority.LOW),
            Task(type="task", priority=TaskPriority.URGENT),
            Task(type="task", priority=TaskPriority.HIGH),
            Task(type="task", priority=TaskPriority.NORMAL),
        ]
        
        # Add in random order
        for task in tasks:
            router.add_task(task)
        
        # Should get them back in priority order
        retrieved_tasks = []
        while True:
            task = router.get_next_task()
            if not task:
                break
            retrieved_tasks.append(task)
        
        expected_order = [TaskPriority.URGENT, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]
        actual_order = [task.priority for task in retrieved_tasks]
        assert actual_order == expected_order
    
    def test_suitable_agent_selection(self, router, sample_agents, sample_tasks):
        """Test agent selection for tasks"""
        # Register agents
        for agent in sample_agents:
            router.register_agent(agent)
        
        text_task = sample_tasks[0]  # text_processing task
        
        # Find suitable agent
        selected_agent = router.find_suitable_agent(text_task)
        
        # Should select an idle agent with text_processing capability
        assert selected_agent is not None
        assert selected_agent in ["agent-1", "agent-3"]  # Both can handle text
        
        # Should not select busy agent even if capable
        assert selected_agent != "agent-4"
    
    def test_no_suitable_agent(self, router, sample_agents):
        """Test case when no suitable agent is available"""
        # Register only math processing agent
        router.register_agent(sample_agents[1])
        
        # Try to find agent for text processing
        text_task = Task(type="text_processing", data={})
        selected_agent = router.find_suitable_agent(text_task)
        
        assert selected_agent is None
    
    def test_agent_scoring(self, router):
        """Test agent scoring for selection"""
        # Create agents with different last heartbeat times
        import time
        
        agent1 = AgentInfo(
            id="agent-1",
            name="Recent Agent",
            capabilities=[AgentCapability(name="test")],
            status=AgentStatus.IDLE,
            last_heartbeat=datetime.utcnow()
        )
        
        # Simulate older agent
        time.sleep(0.01)
        agent2 = AgentInfo(
            id="agent-2", 
            name="Older Agent",
            capabilities=[AgentCapability(name="test")],
            status=AgentStatus.IDLE,
            last_heartbeat=datetime.utcnow()
        )
        
        router.register_agent(agent1)
        router.register_agent(agent2)
        
        task = Task(type="test", data={})
        
        # The agent selection should prefer less recently used agents
        selected_agent = router.find_suitable_agent(task)
        
        # Both are suitable, but scoring should be deterministic
        assert selected_agent in ["agent-1", "agent-2"]
    
    def test_circular_dependency_detection(self, router):
        """Test circular dependency detection"""
        # Create tasks with circular dependencies
        task1 = Task(type="task1", dependencies=["task2"])
        task2 = Task(type="task2", dependencies=["task1"])
        
        # Adding task1 first should succeed
        router.add_task(task1)
        
        # Adding task2 should raise circular dependency error
        with pytest.raises(CircularDependencyError):
            router.add_task(task2)
    
    def test_complex_dependency_chain(self, router):
        """Test more complex dependency scenarios"""
        # Create a valid dependency chain: A -> B -> C
        task_c = Task(id="task-c", type="task", dependencies=[])
        task_b = Task(id="task-b", type="task", dependencies=["task-c"])
        task_a = Task(id="task-a", type="task", dependencies=["task-b"])
        
        # Should be able to add all tasks
        router.add_task(task_c)
        router.add_task(task_b)
        router.add_task(task_a)
        
        # Now try to create circular dependency: C -> A
        task_c_circular = Task(id="task-c-circular", type="task", dependencies=["task-a"])
        
        # This should detect the circular dependency
        with pytest.raises(CircularDependencyError):
            router.add_task(task_c_circular)
    
    def test_queue_status(self, router, sample_agents, sample_tasks):
        """Test queue status reporting"""
        # Register agents
        for agent in sample_agents:
            router.register_agent(agent)
        
        # Add tasks
        for task in sample_tasks:
            router.add_task(task)
        
        status = router.get_queue_status()
        
        assert status["total_tasks"] == len(sample_tasks)
        assert status["total_agents"] == len(sample_agents)
        assert status["available_agents"] == 3  # 3 idle agents
        assert status["busy_agents"] == 1  # 1 busy agent
    
    def test_dependency_tracking(self, router):
        """Test dependency graph management"""
        # Create tasks with dependencies
        task1 = Task(id="task-1", type="task", dependencies=[])
        task2 = Task(id="task-2", type="task", dependencies=["task-1"])
        
        router.add_task(task1)
        router.add_task(task2)
        
        # Check dependency graph
        assert "task-1" not in router._dependency_graph
        assert "task-2" in router._dependency_graph
        assert "task-1" in router._dependency_graph["task-2"]
        
        # Remove completed task
        router.remove_completed_task("task-1")
        
        # task-1 should be removed from dependency tracking
        # but task-2 should still have its dependencies recorded
        assert "task-1" not in router._dependency_graph
    
    def test_empty_queue_behavior(self, router):
        """Test behavior with empty task queue"""
        # Should return None when no tasks available
        assert router.get_next_task() is None
        
        # Status should show empty state
        status = router.get_queue_status()
        assert status["total_tasks"] == 0
    
    def test_agent_capability_matching(self, router):
        """Test precise capability matching"""
        # Agent with specific capabilities
        agent = AgentInfo(
            id="specialized-agent",
            capabilities=[
                AgentCapability(name="image_processing"),
                AgentCapability(name="video_processing")
            ],
            status=AgentStatus.IDLE
        )
        router.register_agent(agent)
        
        # Task that matches capability
        matching_task = Task(type="image_processing", data={})
        assert router.find_suitable_agent(matching_task) == "specialized-agent"
        
        # Task that doesn't match
        non_matching_task = Task(type="audio_processing", data={})
        assert router.find_suitable_agent(non_matching_task) is None
    
    def test_multiple_suitable_agents(self, router):
        """Test selection when multiple agents are suitable"""
        # Create multiple agents with same capability
        agents = [
            AgentInfo(
                id=f"agent-{i}",
                capabilities=[AgentCapability(name="common_task")],
                status=AgentStatus.IDLE
            )
            for i in range(3)
        ]
        
        for agent in agents:
            router.register_agent(agent)
        
        task = Task(type="common_task", data={})
        
        # Should select one of the suitable agents
        selected = router.find_suitable_agent(task)
        assert selected in [f"agent-{i}" for i in range(3)]