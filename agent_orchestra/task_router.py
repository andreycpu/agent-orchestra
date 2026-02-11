"""
Task routing logic for Agent Orchestra
"""
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import heapq
from datetime import datetime, timedelta
import time
from collections import defaultdict, deque
import structlog

from .types import Task, AgentInfo, TaskPriority, AgentStatus
from .exceptions import TaskRoutingError, AgentNotFoundError, CircularDependencyError

logger = structlog.get_logger(__name__)


class TaskRouter:
    """
    Intelligent task routing system that assigns tasks to optimal agents
    """
    
    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._task_queue: List[Tuple[int, Task]] = []  # Priority queue (priority, task)
        self._dependency_graph: Dict[str, Set[str]] = {}
        
        # Performance optimization caches
        self._routing_cache: Dict[str, Tuple[str, float]] = {}  # task_type -> (agent_id, timestamp)
        self._capability_index: Dict[str, Set[str]] = defaultdict(set)  # capability -> agent_ids
        self._agent_load: Dict[str, int] = defaultdict(int)  # agent_id -> current task count
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))  # agent_id -> execution_times
        
        # Cache configuration
        self._cache_ttl = 300.0  # 5 minutes
        self._last_cache_cleanup = time.time()
        
    def register_agent(self, agent_info: AgentInfo):
        """Register an agent with the router"""
        self._agents[agent_info.id] = agent_info
        
        # Update capability index for fast lookups
        for capability in agent_info.capabilities:
            self._capability_index[capability.name].add(agent_info.id)
        
        # Initialize performance tracking
        self._agent_load[agent_info.id] = 0
        
        # Invalidate routing cache for this agent's capabilities
        self._invalidate_cache_for_capabilities([cap.name for cap in agent_info.capabilities])
        
        logger.info(
            "Agent registered with router",
            agent_id=agent_info.id,
            capabilities=[cap.name for cap in agent_info.capabilities]
        )
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the router"""
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info("Agent unregistered from router", agent_id=agent_id)
    
    def update_agent_status(self, agent_id: str, agent_info: AgentInfo):
        """Update agent information"""
        self._agents[agent_id] = agent_info
    
    def add_task(self, task: Task):
        """Add a task to the routing queue"""
        # Check for circular dependencies
        if task.dependencies:
            self._check_circular_dependencies(task.id, task.dependencies)
            self._dependency_graph[task.id] = set(task.dependencies)
        
        # Convert priority to numeric value for heap
        priority_value = self._priority_to_value(task.priority)
        heapq.heappush(self._task_queue, (priority_value, task.created_at, task))
        
        logger.info(
            "Task added to router queue",
            task_id=task.id,
            task_type=task.type,
            priority=task.priority
        )
    
    def _priority_to_value(self, priority: TaskPriority) -> int:
        """Convert TaskPriority to numeric value (lower = higher priority)"""
        priority_map = {
            TaskPriority.URGENT: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3
        }
        return priority_map[priority]
    
    def _check_circular_dependencies(self, task_id: str, dependencies: List[str]):
        """Check for circular dependencies in task graph"""
        def has_path(start: str, target: str, visited: Set[str]) -> bool:
            if start == target:
                return True
            if start in visited:
                return False
                
            visited.add(start)
            
            if start in self._dependency_graph:
                for dep in self._dependency_graph[start]:
                    if has_path(dep, target, visited):
                        return True
            
            return False
        
        for dep in dependencies:
            if has_path(dep, task_id, set()):
                raise CircularDependencyError(
                    f"Circular dependency detected: {task_id} -> {dep}"
                )
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute based on priority and dependencies"""
        # Try to find a task that can be executed (dependencies satisfied)
        available_tasks = []
        dependent_tasks = []
        
        while self._task_queue:
            priority, created_at, task = heapq.heappop(self._task_queue)
            
            if self._can_execute_task(task):
                # Put back remaining tasks and return this one
                for dep_task in dependent_tasks:
                    heapq.heappush(self._task_queue, dep_task)
                return task
            else:
                # Task has unsatisfied dependencies
                dependent_tasks.append((priority, created_at, task))
        
        # Put back all dependent tasks
        for dep_task in dependent_tasks:
            heapq.heappush(self._task_queue, dep_task)
        
        return None
    
    def _can_execute_task(self, task: Task) -> bool:
        """Check if a task's dependencies are satisfied"""
        if not task.dependencies:
            return True
        
        # For now, we assume dependencies are task IDs
        # In a real implementation, you'd check if dependent tasks are completed
        return True  # Simplified for initial implementation
    
    def find_suitable_agent(self, task: Task) -> Optional[str]:
        """Find the best agent for executing a task"""
        suitable_agents = []
        
        for agent_info in self._agents.values():
            # Check if agent can handle this task type
            if not any(cap.name == task.type for cap in agent_info.capabilities):
                continue
                
            # Check if agent is available
            if agent_info.status != AgentStatus.IDLE:
                continue
                
            # Calculate agent score (simple scoring for now)
            score = self._calculate_agent_score(agent_info, task)
            suitable_agents.append((score, agent_info.id))
        
        if not suitable_agents:
            return None
        
        # Return agent with highest score
        suitable_agents.sort(reverse=True)
        selected_agent_id = suitable_agents[0][1]
        
        logger.info(
            "Agent selected for task",
            task_id=task.id,
            task_type=task.type,
            selected_agent=selected_agent_id,
            candidates=len(suitable_agents)
        )
        
        return selected_agent_id
    
    def _calculate_agent_score(self, agent_info: AgentInfo, task: Task) -> float:
        """Calculate suitability score for an agent-task pair"""
        score = 1.0
        
        # Prefer agents that haven't worked recently (simple load balancing)
        time_since_heartbeat = (datetime.utcnow() - agent_info.last_heartbeat).total_seconds()
        score += min(time_since_heartbeat / 60.0, 10.0)  # Max 10 points for age
        
        # Could add more scoring logic here:
        # - Agent performance history
        # - Resource availability
        # - Locality/proximity
        # - Specialization level
        
        return score
    
    def get_queue_status(self) -> Dict:
        """Get current status of the task queue"""
        return {
            "total_tasks": len(self._task_queue),
            "available_agents": len([
                a for a in self._agents.values() 
                if a.status == AgentStatus.IDLE
            ]),
            "busy_agents": len([
                a for a in self._agents.values() 
                if a.status == AgentStatus.BUSY
            ]),
            "total_agents": len(self._agents)
        }
    
    def remove_completed_task(self, task_id: str):
        """Remove a completed task from dependency tracking"""
        if task_id in self._dependency_graph:
            del self._dependency_graph[task_id]
    
    def _invalidate_cache_for_capabilities(self, capabilities: List[str]):
        """Invalidate routing cache entries for specific capabilities"""
        keys_to_remove = []
        for task_type, (agent_id, _) in self._routing_cache.items():
            if any(cap in task_type for cap in capabilities):
                keys_to_remove.append(task_type)
        
        for key in keys_to_remove:
            del self._routing_cache[key]
    
    def _cleanup_expired_cache_entries(self):
        """Remove expired cache entries"""
        current_time = time.time()
        if current_time - self._last_cache_cleanup > self._cache_ttl:
            expired_keys = []
            for task_type, (_, timestamp) in self._routing_cache.items():
                if current_time - timestamp > self._cache_ttl:
                    expired_keys.append(task_type)
            
            for key in expired_keys:
                del self._routing_cache[key]
            
            self._last_cache_cleanup = current_time
    
    def update_agent_performance(self, agent_id: str, execution_time: float):
        """Update performance metrics for an agent"""
        self._performance_history[agent_id].append(execution_time)
        
        # Update load tracking
        if agent_id in self._agent_load:
            self._agent_load[agent_id] = max(0, self._agent_load[agent_id] - 1)
    
    def _get_agent_performance_score(self, agent_id: str) -> float:
        """Calculate performance score for an agent (lower is better)"""
        if agent_id not in self._performance_history or not self._performance_history[agent_id]:
            return 0.0  # No history, neutral score
        
        # Calculate average execution time
        avg_time = sum(self._performance_history[agent_id]) / len(self._performance_history[agent_id])
        
        # Factor in current load
        load_penalty = self._agent_load.get(agent_id, 0) * 0.1
        
        return avg_time + load_penalty
    
    def find_optimal_agent_fast(self, task: Task) -> Optional[str]:
        """Fast agent selection using performance caching and indexing"""
        self._cleanup_expired_cache_entries()
        
        # Check cache first
        if task.type in self._routing_cache:
            agent_id, _ = self._routing_cache[task.type]
            if agent_id in self._agents and self._agents[agent_id].status == AgentStatus.IDLE:
                return agent_id
        
        # Find capable agents using index
        capable_agents = self._capability_index.get(task.type, set())
        if not capable_agents:
            return None
        
        # Filter by availability
        available_agents = [
            agent_id for agent_id in capable_agents
            if agent_id in self._agents and self._agents[agent_id].status == AgentStatus.IDLE
        ]
        
        if not available_agents:
            return None
        
        # Select best agent based on performance
        best_agent = min(available_agents, key=self._get_agent_performance_score)
        
        # Update cache and load tracking
        self._routing_cache[task.type] = (best_agent, time.time())
        self._agent_load[best_agent] += 1
        
        return best_agent