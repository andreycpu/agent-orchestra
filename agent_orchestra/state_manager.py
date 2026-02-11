"""
State management for Agent Orchestra
"""
import json
import asyncio
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
import structlog

from .types import Task, AgentInfo, TaskStatus
from .exceptions import StateManagementError

logger = structlog.get_logger(__name__)


class StateManager:
    """
    Manages persistent state across the agent network
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self._redis_url = redis_url
        self._redis = None
        self._memory_store: Dict[str, Any] = {}
        self._task_history: Dict[str, Task] = {}
        self._agent_registry: Dict[str, AgentInfo] = {}
        self._execution_metrics: Dict[str, Dict] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        
    async def initialize(self):
        """Initialize the state manager"""
        if self._redis_url:
            try:
                import aioredis
                self._redis = await aioredis.from_url(self._redis_url)
                logger.info("Redis connection established", redis_url=self._redis_url)
            except ImportError:
                logger.warning("Redis not available, using in-memory storage")
            except Exception as e:
                logger.error("Failed to connect to Redis", error=str(e))
                logger.info("Falling back to in-memory storage")
        
        logger.info("StateManager initialized")
    
    async def shutdown(self):
        """Clean shutdown of state manager"""
        if self._redis:
            await self._redis.close()
            logger.info("Redis connection closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()
        return False  # Don't suppress exceptions
    
    async def store_task(self, task: Task):
        """
        Store a task in persistent state.
        
        Args:
            task: Task object to store
            
        Raises:
            StateManagementError: If storage operation fails
        """
        task_data = task.dict()
        
        async with self._get_lock(f"task:{task.id}"):
            if self._redis:
                await self._redis.hset(
                    "tasks",
                    task.id,
                    json.dumps(task_data, default=str)
                )
            else:
                self._task_history[task.id] = task
        
        logger.debug("Task stored", task_id=task.id, status=task.status)
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Retrieve a task from persistent state.
        
        Args:
            task_id: Unique identifier of the task
            
        Returns:
            Task object if found, None otherwise
            
        Raises:
            StateManagementError: If retrieval operation fails
        """
        async with self._get_lock(f"task:{task_id}"):
            if self._redis:
                task_data = await self._redis.hget("tasks", task_id)
                if task_data:
                    return Task(**json.loads(task_data))
            else:
                return self._task_history.get(task_id)
        
        return None
    
    async def update_task_status(self, task_id: str, status: TaskStatus, **kwargs):
        """Update task status and related fields"""
        async with self._get_lock(f"task:{task_id}"):
            task = await self.get_task(task_id)
            if not task:
                raise StateManagementError(f"Task {task_id} not found")
            
            task.status = status
            
            # Update specific fields
            if status == TaskStatus.RUNNING:
                task.started_at = datetime.utcnow()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                task.completed_at = datetime.utcnow()
            
            # Update additional fields from kwargs
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            await self.store_task(task)
        
        logger.info(
            "Task status updated",
            task_id=task_id,
            status=status,
            **kwargs
        )
    
    async def register_agent(self, agent_info: AgentInfo):
        """Register an agent in the state"""
        async with self._get_lock(f"agent:{agent_info.id}"):
            if self._redis:
                await self._redis.hset(
                    "agents",
                    agent_info.id,
                    json.dumps(agent_info.dict(), default=str)
                )
            else:
                self._agent_registry[agent_info.id] = agent_info
        
        logger.info("Agent registered in state", agent_id=agent_info.id)
    
    async def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent information"""
        async with self._get_lock(f"agent:{agent_id}"):
            if self._redis:
                agent_data = await self._redis.hget("agents", agent_id)
                if agent_data:
                    return AgentInfo(**json.loads(agent_data))
            else:
                return self._agent_registry.get(agent_id)
        
        return None
    
    async def get_all_agents(self) -> List[AgentInfo]:
        """Get all registered agents"""
        agents = []
        
        if self._redis:
            agent_data = await self._redis.hgetall("agents")
            for data in agent_data.values():
                agents.append(AgentInfo(**json.loads(data)))
        else:
            agents = list(self._agent_registry.values())
        
        return agents
    
    async def update_agent_heartbeat(self, agent_id: str):
        """Update agent heartbeat timestamp"""
        agent = await self.get_agent(agent_id)
        if agent:
            agent.last_heartbeat = datetime.utcnow()
            await self.register_agent(agent)
    
    async def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with a specific status"""
        tasks = []
        
        if self._redis:
            task_data = await self._redis.hgetall("tasks")
            for data in task_data.values():
                task = Task(**json.loads(data))
                if task.status == status:
                    tasks.append(task)
        else:
            tasks = [
                task for task in self._task_history.values()
                if task.status == status
            ]
        
        return tasks
    
    async def cleanup_old_tasks(self, older_than_days: int = 30):
        """Clean up old completed/failed tasks"""
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        
        if self._redis:
            task_data = await self._redis.hgetall("tasks")
            for task_id, data in task_data.items():
                task = Task(**json.loads(data))
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and
                    task.completed_at and task.completed_at < cutoff_date):
                    await self._redis.hdel("tasks", task_id)
        else:
            to_remove = []
            for task_id, task in self._task_history.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and
                    task.completed_at and task.completed_at < cutoff_date):
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self._task_history[task_id]
        
        logger.info(
            "Old tasks cleaned up",
            older_than_days=older_than_days,
            cutoff_date=cutoff_date
        )
    
    async def store_execution_metrics(self, agent_id: str, metrics: Dict[str, Any]):
        """Store execution metrics for an agent"""
        timestamp = datetime.utcnow().isoformat()
        metrics_key = f"metrics:{agent_id}:{timestamp}"
        
        if self._redis:
            await self._redis.set(
                metrics_key,
                json.dumps(metrics, default=str),
                ex=86400 * 7  # Keep for 7 days
            )
        else:
            if agent_id not in self._execution_metrics:
                self._execution_metrics[agent_id] = {}
            self._execution_metrics[agent_id][timestamp] = metrics
        
        logger.debug("Execution metrics stored", agent_id=agent_id, metrics=metrics)
    
    async def get_global_state(self) -> Dict[str, Any]:
        """Get global system state summary"""
        agents = await self.get_all_agents()
        
        pending_tasks = await self.get_tasks_by_status(TaskStatus.PENDING)
        running_tasks = await self.get_tasks_by_status(TaskStatus.RUNNING)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "agents": {
                "total": len(agents),
                "idle": len([a for a in agents if a.status.value == "idle"]),
                "busy": len([a for a in agents if a.status.value == "busy"]),
                "unavailable": len([a for a in agents if a.status.value == "unavailable"])
            },
            "tasks": {
                "pending": len(pending_tasks),
                "running": len(running_tasks)
            }
        }
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for a specific key"""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]