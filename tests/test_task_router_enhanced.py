"""
Tests for enhanced TaskRouter functionality
"""
import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from agent_orchestra.task_router import TaskRouter
from agent_orchestra.types import Task, AgentInfo, AgentCapability, TaskPriority, AgentStatus
from agent_orchestra.exceptions import ValidationError, CircularDependencyError


class TestTaskRouterValidation:
    """Test cases for TaskRouter input validation"""
    
    @pytest.fixture
    def router(self):
        """Create a TaskRouter instance"""
        return TaskRouter()
    
    @pytest.fixture
    def sample_agent_info(self):
        """Create sample agent info"""
        return AgentInfo(
            id="test-agent",
            name="Test Agent",
            capabilities=[AgentCapability(name="test_task", description="Test capability")],
            status=AgentStatus.IDLE,
            current_task=None,
            last_heartbeat=datetime.utcnow(),
            metadata={}
        )
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task"""
        return Task(
            type="test_task",
            data={"test": "data"},
            priority=TaskPriority.NORMAL
        )
    
    def test_register_agent_validation(self, router):
        """Test agent registration validation"""
        # None agent_info
        with pytest.raises(ValidationError, match="agent_info cannot be None"):
            router.register_agent(None)
        
        # Invalid agent_info type
        with pytest.raises(ValidationError, match="agent_info must be an AgentInfo instance"):
            router.register_agent("invalid")
        
        # Empty agent ID
        invalid_agent = Mock()
        invalid_agent.id = ""
        with pytest.raises(ValidationError, match="agent_info must have a valid id"):
            router.register_agent(invalid_agent)
        
        # No capabilities
        invalid_agent = Mock()
        invalid_agent.id = "test"
        invalid_agent.capabilities = []
        with pytest.raises(ValidationError, match="agent_info must have at least one capability"):
            router.register_agent(invalid_agent)
    
    def test_add_task_validation(self, router):
        """Test task addition validation"""
        # None task
        with pytest.raises(ValidationError, match="task cannot be None"):
            router.add_task(None)
        
        # Invalid task type
        with pytest.raises(ValidationError, match="task must be a Task instance"):
            router.add_task("invalid")
        
        # Empty task ID
        invalid_task = Mock()
        invalid_task.id = ""
        with pytest.raises(ValidationError, match="task must have a valid id"):
            router.add_task(invalid_task)
        
        # Empty task type
        invalid_task = Mock()
        invalid_task.id = "test-task"
        invalid_task.type = ""
        with pytest.raises(ValidationError, match="task must have a valid type"):
            router.add_task(invalid_task)
    
    def test_find_optimal_agent_validation(self, router):
        """Test optimal agent finding validation"""
        # None task
        with pytest.raises(ValidationError, match="task cannot be None"):
            router.find_optimal_agent_fast(None)
        
        # Task without type
        invalid_task = Mock()
        invalid_task.type = ""
        with pytest.raises(ValidationError, match="task must have a valid type"):
            router.find_optimal_agent_fast(invalid_task)


class TestTaskRouterMetrics:
    """Test cases for TaskRouter metrics functionality"""
    
    @pytest.fixture
    def router(self):
        """Create a TaskRouter instance"""
        return TaskRouter()
    
    @pytest.fixture
    def agent_info(self):
        """Create agent info for testing"""
        return AgentInfo(
            id="metrics-agent",
            name="Metrics Agent",
            capabilities=[AgentCapability(name="metrics_task", description="Metrics test")],
            status=AgentStatus.IDLE,
            current_task=None,
            last_heartbeat=datetime.utcnow(),
            metadata={}
        )
    
    @pytest.fixture
    def task(self):
        """Create task for testing"""
        return Task(
            type="metrics_task",
            data={"test": "data"},
            priority=TaskPriority.NORMAL
        )
    
    def test_get_routing_metrics_initial(self, router):
        """Test initial routing metrics"""
        metrics = router.get_routing_metrics()
        
        assert metrics["total_tasks_routed"] == 0
        assert metrics["successful_routes"] == 0
        assert metrics["failed_routes"] == 0
        assert metrics["cache_hits"] == 0
        assert metrics["cache_misses"] == 0
        assert metrics["queue_size"] == 0
        assert metrics["registered_agents"] == 0
        assert metrics["active_agents"] == 0
    
    def test_routing_metrics_after_agent_registration(self, router, agent_info):
        """Test metrics after agent registration"""
        router.register_agent(agent_info)
        
        metrics = router.get_routing_metrics()
        assert metrics["registered_agents"] == 1
        assert metrics["active_agents"] == 1
    
    def test_routing_metrics_after_task_routing(self, router, agent_info, task):
        """Test metrics after task routing"""
        router.register_agent(agent_info)
        
        # Route a task
        result = router.find_optimal_agent_fast(task)
        
        assert result == "metrics-agent"
        
        metrics = router.get_routing_metrics()
        assert metrics["total_tasks_routed"] == 1
        assert metrics["successful_routes"] == 1
        assert metrics["failed_routes"] == 0
        assert metrics["cache_misses"] == 1  # First time, should be cache miss
        
        # Route same type again (should be cache hit)
        task2 = Task(type="metrics_task", data={"test": "data2"})
        router.find_optimal_agent_fast(task2)
        
        metrics = router.get_routing_metrics()
        assert metrics["cache_hits"] == 1
    
    def test_agent_metrics_initial(self, router):
        """Test initial agent metrics"""
        metrics = router.get_agent_metrics()
        assert metrics == {}
        
        # Test specific agent that doesn't exist
        specific_metrics = router.get_agent_metrics("nonexistent")
        assert specific_metrics == {}
    
    def test_record_task_execution_validation(self, router):
        """Test task execution recording validation"""
        # Empty agent ID
        with pytest.raises(ValidationError, match="agent_id cannot be empty"):
            router.record_task_execution("", 1.0)
        
        # Negative execution time
        with pytest.raises(ValidationError, match="execution_time cannot be negative"):
            router.record_task_execution("agent", -1.0)
    
    def test_record_task_execution(self, router, agent_info):
        """Test task execution recording"""
        router.register_agent(agent_info)
        
        # Record some execution times
        router.record_task_execution("metrics-agent", 1.5)
        router.record_task_execution("metrics-agent", 2.0)
        router.record_task_execution("metrics-agent", 1.0)
        
        metrics = router.get_agent_metrics("metrics-agent")
        
        assert metrics["total_execution_time"] == 4.5
        assert metrics["avg_execution_time"] == 1.5
    
    def test_routing_time_tracking(self, router, agent_info, task):
        """Test routing time tracking"""
        router.register_agent(agent_info)
        
        # Route a task
        router.find_optimal_agent_fast(task)
        
        metrics = router.get_routing_metrics()
        assert len(metrics["routing_times"]) == 1
        assert metrics["routing_times"][0] > 0  # Should have some routing time
        assert metrics["avg_routing_time"] > 0


class TestTaskRouterLoadBalancing:
    """Test cases for TaskRouter load balancing functionality"""
    
    @pytest.fixture
    def router_with_agents(self):
        """Create router with multiple agents"""
        router = TaskRouter()
        
        # Register multiple agents
        for i in range(3):
            agent_info = AgentInfo(
                id=f"agent-{i}",
                name=f"Agent {i}",
                capabilities=[AgentCapability(name="test_task", description="Test capability")],
                status=AgentStatus.IDLE,
                current_task=None,
                last_heartbeat=datetime.utcnow(),
                metadata={}
            )
            router.register_agent(agent_info)
        
        return router
    
    def test_get_load_balancing_info(self, router_with_agents):
        """Test load balancing information"""
        info = router_with_agents.get_load_balancing_info()
        
        assert info["total_load"] == 0
        assert info["average_load"] == 0.0
        assert len(info["agent_loads"]) == 3
        
        # All agents should have zero load initially
        for agent_id, load_info in info["agent_loads"].items():
            assert load_info["current_load"] == 0
            assert load_info["status"] == "IDLE"
    
    def test_load_balancing_after_task_assignment(self, router_with_agents):
        """Test load balancing after assigning tasks"""
        # Simulate task assignments by manually updating load
        router_with_agents._agent_load["agent-0"] = 3
        router_with_agents._agent_load["agent-1"] = 1
        router_with_agents._agent_load["agent-2"] = 0
        
        info = router_with_agents.get_load_balancing_info()
        
        assert info["total_load"] == 4
        assert info["average_load"] == 4/3
        assert info["most_loaded_agent"]["agent_id"] == "agent-0"
        assert info["most_loaded_agent"]["load"] == 3
        assert info["least_loaded_agent"]["agent_id"] == "agent-2"
        assert info["least_loaded_agent"]["load"] == 0
    
    def test_rebalance_load_no_agents(self):
        """Test load rebalancing with no agents"""
        router = TaskRouter()
        result = router.rebalance_load()
        
        assert result["rebalanced"] is False
        assert "Insufficient agents" in result["reason"]
    
    def test_rebalance_load_balanced(self, router_with_agents):
        """Test rebalancing when load is already balanced"""
        result = router_with_agents.rebalance_load()
        
        assert result["rebalanced"] is False
        assert "Load already balanced" in result["reason"]
    
    def test_rebalance_load_needed(self, router_with_agents):
        """Test rebalancing when load variance is significant"""
        # Create significant load imbalance
        router_with_agents._agent_load["agent-0"] = 5
        router_with_agents._agent_load["agent-1"] = 1
        router_with_agents._agent_load["agent-2"] = 0
        
        # Add some cache entries to verify they get cleared
        router_with_agents._routing_cache["test_task"] = ("agent-0", time.time())
        
        result = router_with_agents.rebalance_load()
        
        assert result["rebalanced"] is True
        assert result["cache_cleared"] is True
        assert result["load_variance_before"] == 5
        assert len(router_with_agents._routing_cache) == 0  # Cache should be cleared


class TestTaskRouterDependencyGraph:
    """Test cases for TaskRouter dependency graph functionality"""
    
    @pytest.fixture
    def router(self):
        """Create a TaskRouter instance"""
        return TaskRouter()
    
    def test_get_dependency_graph_info_empty(self, router):
        """Test dependency graph info when empty"""
        info = router.get_dependency_graph_info()
        
        assert info["total_tasks_with_dependencies"] == 0
        assert info["total_dependencies"] == 0
        assert info["avg_dependencies_per_task"] == 0
        assert info["max_dependencies_task"] is None
    
    def test_get_dependency_graph_info_with_dependencies(self, router):
        """Test dependency graph info with dependencies"""
        # Manually add some dependencies
        router._dependency_graph = {
            "task1": {"dep1", "dep2"},
            "task2": {"dep1"},
            "task3": {"dep1", "dep2", "dep3", "dep4"}
        }
        
        info = router.get_dependency_graph_info()
        
        assert info["total_tasks_with_dependencies"] == 3
        assert info["total_dependencies"] == 7  # 2 + 1 + 4
        assert info["avg_dependencies_per_task"] == 7/3
        assert info["max_dependencies_task"]["task_id"] == "task3"
        assert info["max_dependencies_task"]["dependency_count"] == 4
    
    def test_cleanup_completed_tasks_validation(self, router):
        """Test cleanup validation"""
        with pytest.raises(ValidationError, match="completed_task_ids must be a list"):
            router.cleanup_completed_tasks("not_a_list")
    
    def test_cleanup_completed_tasks(self, router):
        """Test cleanup of completed tasks"""
        # Set up dependency graph
        router._dependency_graph = {
            "task1": {"dep1", "dep2"},
            "task2": {"dep1", "task3"},
            "task3": {"dep2"}
        }
        
        # Clean up task1 and dep1
        cleaned = router.cleanup_completed_tasks(["task1", "dep1"])
        
        assert cleaned == 1  # Only task1 was in the graph
        assert "task1" not in router._dependency_graph
        
        # dep1 should be removed from task2's dependencies
        assert "dep1" not in router._dependency_graph["task2"]
        assert router._dependency_graph["task2"] == {"task3"}
    
    def test_cleanup_nonexistent_tasks(self, router):
        """Test cleanup of nonexistent tasks"""
        router._dependency_graph = {"task1": {"dep1"}}
        
        cleaned = router.cleanup_completed_tasks(["nonexistent", "also_nonexistent"])
        
        assert cleaned == 0
        assert router._dependency_graph == {"task1": {"dep1"}}


class TestTaskRouterCaching:
    """Test cases for TaskRouter caching functionality"""
    
    @pytest.fixture
    def router_with_agent(self):
        """Create router with one agent"""
        router = TaskRouter()
        agent_info = AgentInfo(
            id="cache-agent",
            name="Cache Agent", 
            capabilities=[AgentCapability(name="cache_task", description="Cache test")],
            status=AgentStatus.IDLE,
            current_task=None,
            last_heartbeat=datetime.utcnow(),
            metadata={}
        )
        router.register_agent(agent_info)
        return router
    
    def test_cache_hit_after_first_route(self, router_with_agent):
        """Test cache hit behavior"""
        task1 = Task(type="cache_task", data={"test": "data1"})
        task2 = Task(type="cache_task", data={"test": "data2"})
        
        # First routing should be cache miss
        result1 = router_with_agent.find_optimal_agent_fast(task1)
        metrics1 = router_with_agent.get_routing_metrics()
        assert metrics1["cache_misses"] == 1
        assert metrics1["cache_hits"] == 0
        
        # Second routing of same type should be cache hit
        result2 = router_with_agent.find_optimal_agent_fast(task2)
        metrics2 = router_with_agent.get_routing_metrics()
        assert metrics2["cache_hits"] == 1
        
        assert result1 == result2 == "cache-agent"
    
    def test_cache_expiry(self, router_with_agent):
        """Test cache expiry functionality"""
        # Set very short cache TTL for testing
        router_with_agent._cache_ttl = 0.1
        
        task = Task(type="cache_task", data={"test": "data"})
        
        # First route
        router_with_agent.find_optimal_agent_fast(task)
        
        # Wait for cache to expire
        time.sleep(0.2)
        
        # Second route should be cache miss due to expiry
        router_with_agent.find_optimal_agent_fast(task)
        
        metrics = router_with_agent.get_routing_metrics()
        assert metrics["cache_misses"] == 2  # Both should be misses


class TestTaskRouterPerformance:
    """Test cases for TaskRouter performance features"""
    
    @pytest.fixture
    def router_with_multiple_agents(self):
        """Create router with agents having different performance histories"""
        router = TaskRouter()
        
        # Agent with good performance
        agent1 = AgentInfo(
            id="fast-agent",
            name="Fast Agent",
            capabilities=[AgentCapability(name="perf_task", description="Performance test")],
            status=AgentStatus.IDLE,
            current_task=None,
            last_heartbeat=datetime.utcnow(),
            metadata={}
        )
        
        # Agent with poor performance  
        agent2 = AgentInfo(
            id="slow-agent",
            name="Slow Agent",
            capabilities=[AgentCapability(name="perf_task", description="Performance test")],
            status=AgentStatus.IDLE,
            current_task=None,
            last_heartbeat=datetime.utcnow(),
            metadata={}
        )
        
        router.register_agent(agent1)
        router.register_agent(agent2)
        
        # Simulate performance history
        router._performance_history["fast-agent"].extend([0.5, 0.6, 0.4])  # Avg: 0.5
        router._performance_history["slow-agent"].extend([2.0, 2.5, 1.8])  # Avg: 2.1
        
        return router
    
    def test_performance_based_routing(self, router_with_multiple_agents):
        """Test that faster agent is selected based on performance"""
        task = Task(type="perf_task", data={"test": "data"})
        
        # Should select the faster agent
        selected = router_with_multiple_agents.find_optimal_agent_fast(task)
        assert selected == "fast-agent"
    
    def test_performance_score_calculation(self, router_with_multiple_agents):
        """Test performance score calculation"""
        fast_score = router_with_multiple_agents._get_agent_performance_score("fast-agent")
        slow_score = router_with_multiple_agents._get_agent_performance_score("slow-agent")
        
        assert fast_score < slow_score  # Lower is better
        assert fast_score == 0.5  # Average of [0.5, 0.6, 0.4]
        assert slow_score == 2.1  # Average of [2.0, 2.5, 1.8]
    
    def test_load_penalty_in_performance_score(self, router_with_multiple_agents):
        """Test that current load affects performance score"""
        # Add load to fast agent
        router_with_multiple_agents._agent_load["fast-agent"] = 5
        
        fast_score = router_with_multiple_agents._get_agent_performance_score("fast-agent")
        slow_score = router_with_multiple_agents._get_agent_performance_score("slow-agent")
        
        # Fast agent should now have higher score due to load penalty
        assert fast_score == 0.5 + (5 * 0.1)  # Base score + load penalty
        assert fast_score > slow_score  # Now slower due to load