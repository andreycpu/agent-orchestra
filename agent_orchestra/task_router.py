"""
Task routing logic for Agent Orchestra
"""
from typing import Dict, List, Optional, Set
import heapq
from datetime import datetime
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
        self._task_queue: List[tuple] = []  # Priority queue (priority, task)
        self._dependency_graph: Dict[str, Set[str]] = {}
        
    def register_agent(self, agent_info: AgentInfo):
        """Register an agent with the router"""
        self._agents[agent_info.id] = agent_info
        
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