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
from .exceptions import TaskRoutingError, AgentNotFoundError, CircularDependencyError, ValidationError

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
        
        # Metrics and monitoring
        self._routing_metrics = {
            "total_tasks_routed": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_routing_time": 0.0,
            "routing_times": deque(maxlen=1000)
        }
        self._agent_metrics = defaultdict(lambda: {
            "tasks_assigned": 0,
            "avg_execution_time": 0.0,
            "last_assigned": None,
            "total_execution_time": 0.0
        })
        
    def register_agent(self, agent_info: AgentInfo):
        """Register an agent with the router
        
        Args:
            agent_info: Agent information to register
            
        Raises:
            ValidationError: If agent_info is invalid
        """
        if agent_info is None:
            raise ValidationError("agent_info cannot be None")
        if not isinstance(agent_info, AgentInfo):
            raise ValidationError("agent_info must be an AgentInfo instance")
        if not agent_info.id:
            raise ValidationError("agent_info must have a valid id")
        if not agent_info.capabilities:
            raise ValidationError("agent_info must have at least one capability")
            
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
        """Add a task to the routing queue
        
        Args:
            task: Task to add to the queue
            
        Raises:
            ValidationError: If task is invalid
            CircularDependencyError: If circular dependencies detected
        """
        if task is None:
            raise ValidationError("task cannot be None")
        if not isinstance(task, Task):
            raise ValidationError("task must be a Task instance")
        if not task.id:
            raise ValidationError("task must have a valid id")
        if not task.type:
            raise ValidationError("task must have a valid type")
            
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
        """Fast agent selection using performance caching and indexing
        
        Args:
            task: Task to find an agent for
            
        Returns:
            Agent ID of the optimal agent, or None if no suitable agent found
            
        Raises:
            ValidationError: If task is invalid
        """
        if task is None:
            raise ValidationError("task cannot be None")
        if not hasattr(task, 'type') or not task.type:
            raise ValidationError("task must have a valid type")
            
        start_time = time.time()
        
        try:
            self._cleanup_expired_cache_entries()
            
            # Check cache first
            cache_hit = False
            if task.type in self._routing_cache:
                agent_id, _ = self._routing_cache[task.type]
                if agent_id in self._agents and self._agents[agent_id].status == AgentStatus.IDLE:
                    cache_hit = True
                    self._routing_metrics["cache_hits"] += 1
                    self._record_routing_metrics(start_time, success=True)
                    return agent_id
            
            if not cache_hit:
                self._routing_metrics["cache_misses"] += 1
            
            # Find capable agents using index
            capable_agents = self._capability_index.get(task.type, set())
            if not capable_agents:
                self._record_routing_metrics(start_time, success=False)
                return None
            
            # Filter by availability
            available_agents = [
                agent_id for agent_id in capable_agents
                if agent_id in self._agents and self._agents[agent_id].status == AgentStatus.IDLE
            ]
            
            if not available_agents:
                self._record_routing_metrics(start_time, success=False)
                return None
            
            # Select best agent based on performance
            best_agent = min(available_agents, key=self._get_agent_performance_score)
            
            # Update cache and load tracking
            self._routing_cache[task.type] = (best_agent, time.time())
            self._agent_load[best_agent] += 1
            
            # Update agent metrics
            agent_metrics = self._agent_metrics[best_agent]
            agent_metrics["tasks_assigned"] += 1
            agent_metrics["last_assigned"] = datetime.utcnow()
            
            self._record_routing_metrics(start_time, success=True)
            
            logger.debug(
                "Optimal agent found",
                task_id=task.id,
                task_type=task.type,
                selected_agent=best_agent,
                cache_hit=cache_hit,
                available_agents_count=len(available_agents)
            )
            
            return best_agent
            
        except Exception as e:
            self._record_routing_metrics(start_time, success=False)
            logger.error(
                "Agent routing failed",
                task_id=getattr(task, 'id', 'unknown'),
                task_type=task.type,
                error=str(e)
            )
            raise

    def _record_routing_metrics(self, start_time: float, success: bool):
        """Record routing performance metrics
        
        Args:
            start_time: Time when routing started
            success: Whether routing was successful
        """
        routing_time = time.time() - start_time
        
        self._routing_metrics["total_tasks_routed"] += 1
        self._routing_metrics["routing_times"].append(routing_time)
        
        if success:
            self._routing_metrics["successful_routes"] += 1
        else:
            self._routing_metrics["failed_routes"] += 1

    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics
        
        Returns:
            Dictionary containing routing metrics
        """
        metrics = self._routing_metrics.copy()
        
        # Calculate current averages
        if metrics["routing_times"]:
            metrics["avg_routing_time"] = sum(metrics["routing_times"]) / len(metrics["routing_times"])
        
        # Add queue information
        metrics["queue_size"] = len(self._task_queue)
        metrics["registered_agents"] = len(self._agents)
        metrics["active_agents"] = len([
            a for a in self._agents.values() 
            if a.status == AgentStatus.IDLE or a.status == AgentStatus.BUSY
        ])
        
        return metrics

    def get_agent_metrics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get agent performance metrics
        
        Args:
            agent_id: Specific agent ID, or None for all agents
            
        Returns:
            Dictionary containing agent metrics
        """
        if agent_id:
            if agent_id not in self._agent_metrics:
                return {}
            return self._agent_metrics[agent_id].copy()
        
        return {
            agent_id: metrics.copy()
            for agent_id, metrics in self._agent_metrics.items()
        }

    def record_task_execution(self, agent_id: str, execution_time: float):
        """Record task execution metrics
        
        Args:
            agent_id: ID of the agent that executed the task
            execution_time: Time taken to execute the task in seconds
            
        Raises:
            ValidationError: If parameters are invalid
        """
        if not agent_id:
            raise ValidationError("agent_id cannot be empty")
        if execution_time < 0:
            raise ValidationError("execution_time cannot be negative")
            
        # Update performance history
        self._performance_history[agent_id].append(execution_time)
        
        # Update agent metrics
        agent_metrics = self._agent_metrics[agent_id]
        agent_metrics["total_execution_time"] += execution_time
        
        # Calculate new average
        total_tasks = len(self._performance_history[agent_id])
        agent_metrics["avg_execution_time"] = (
            agent_metrics["total_execution_time"] / total_tasks
        )
        
        logger.debug(
            "Task execution recorded",
            agent_id=agent_id,
            execution_time=execution_time,
            avg_execution_time=agent_metrics["avg_execution_time"]
        )

    def get_load_balancing_info(self) -> Dict[str, Any]:
        """Get load balancing information
        
        Returns:
            Dictionary with load balancing details
        """
        agent_loads = {}
        total_load = 0
        
        for agent_id in self._agents.keys():
            load = self._agent_load.get(agent_id, 0)
            agent_loads[agent_id] = {
                "current_load": load,
                "status": self._agents[agent_id].status.value,
                "capabilities": [cap.name for cap in self._agents[agent_id].capabilities]
            }
            total_load += load
        
        avg_load = total_load / len(self._agents) if self._agents else 0
        
        # Find most and least loaded agents
        most_loaded = max(agent_loads.items(), key=lambda x: x[1]["current_load"], default=(None, {"current_load": 0}))
        least_loaded = min(agent_loads.items(), key=lambda x: x[1]["current_load"], default=(None, {"current_load": 0}))
        
        return {
            "total_load": total_load,
            "average_load": avg_load,
            "agent_loads": agent_loads,
            "most_loaded_agent": {
                "agent_id": most_loaded[0],
                "load": most_loaded[1]["current_load"]
            } if most_loaded[0] else None,
            "least_loaded_agent": {
                "agent_id": least_loaded[0], 
                "load": least_loaded[1]["current_load"]
            } if least_loaded[0] else None
        }

    def rebalance_load(self) -> Dict[str, Any]:
        """Attempt to rebalance load across agents
        
        Returns:
            Dictionary with rebalancing results
        """
        load_info = self.get_load_balancing_info()
        
        if not load_info["agent_loads"] or len(load_info["agent_loads"]) < 2:
            return {"rebalanced": False, "reason": "Insufficient agents for rebalancing"}
        
        # Calculate load distribution statistics
        loads = [info["current_load"] for info in load_info["agent_loads"].values()]
        max_load = max(loads)
        min_load = min(loads)
        load_variance = max_load - min_load
        
        # Only rebalance if variance is significant
        if load_variance <= 1:
            return {"rebalanced": False, "reason": "Load already balanced"}
        
        # Clear routing cache to force new routing decisions
        self._routing_cache.clear()
        
        logger.info(
            "Load rebalancing initiated",
            max_load=max_load,
            min_load=min_load,
            variance=load_variance,
            total_agents=len(self._agents)
        )
        
        return {
            "rebalanced": True,
            "cache_cleared": True,
            "load_variance_before": load_variance,
            "total_agents": len(self._agents)
        }

    def get_dependency_graph_info(self) -> Dict[str, Any]:
        """Get information about task dependency graph
        
        Returns:
            Dictionary with dependency graph statistics
        """
        total_tasks_with_deps = len(self._dependency_graph)
        total_dependencies = sum(len(deps) for deps in self._dependency_graph.values())
        
        # Find tasks with most dependencies
        max_deps_task = max(
            self._dependency_graph.items(), 
            key=lambda x: len(x[1]), 
            default=(None, set())
        )
        
        return {
            "total_tasks_with_dependencies": total_tasks_with_deps,
            "total_dependencies": total_dependencies,
            "avg_dependencies_per_task": (
                total_dependencies / total_tasks_with_deps 
                if total_tasks_with_deps > 0 else 0
            ),
            "max_dependencies_task": {
                "task_id": max_deps_task[0],
                "dependency_count": len(max_deps_task[1])
            } if max_deps_task[0] else None
        }

    def cleanup_completed_tasks(self, completed_task_ids: List[str]) -> int:
        """Clean up dependency graph for completed tasks
        
        Args:
            completed_task_ids: List of task IDs that have completed
            
        Returns:
            Number of tasks cleaned up
            
        Raises:
            ValidationError: If completed_task_ids is invalid
        """
        if not isinstance(completed_task_ids, list):
            raise ValidationError("completed_task_ids must be a list")
        
        cleaned_count = 0
        
        for task_id in completed_task_ids:
            if task_id in self._dependency_graph:
                del self._dependency_graph[task_id]
                cleaned_count += 1
                
            # Remove this task from other tasks' dependencies
            for deps in self._dependency_graph.values():
                deps.discard(task_id)
        
        logger.info(
            "Completed tasks cleaned up",
            cleaned_count=cleaned_count,
            remaining_dependency_tasks=len(self._dependency_graph)
        )
        
        return cleaned_count