"""
Main Orchestra class that coordinates all components
"""
import asyncio
from typing import Dict, List, Optional, Any, Set, Union, Callable, Awaitable
from datetime import datetime, timedelta
import structlog

from .types import Task, AgentInfo, TaskStatus, AgentStatus, ExecutionResult, TaskPriority
from .agent import Agent
from .task_router import TaskRouter
from .state_manager import StateManager
from .failure_handler import FailureHandler, RecoveryStrategy
from .exceptions import (
    AgentRegistrationError, AgentNotFoundError, TaskRoutingError,
    TaskExecutionError, ConcurrencyError
)

logger = structlog.get_logger(__name__)


class Orchestra:
    """
    Main orchestration engine that coordinates agents and manages task execution
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        max_concurrent_tasks: int = 100,
        task_timeout_default: int = 300,
        heartbeat_interval: int = 30
    ):
        self.state_manager = StateManager(redis_url)
        self.task_router = TaskRouter()
        self.failure_handler = FailureHandler()
        
        self._agents: Dict[str, Agent] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._max_concurrent_tasks = max_concurrent_tasks
        self._task_timeout_default = task_timeout_default
        self._heartbeat_interval = heartbeat_interval
        
        self._is_running = False
        self._executor_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Setup recovery handlers
        self._setup_recovery_handlers()
        
        logger.info(
            "Orchestra initialized",
            max_concurrent_tasks=max_concurrent_tasks,
            task_timeout_default=task_timeout_default
        )
    
    def _setup_recovery_handlers(self):
        """Setup recovery handlers for failure handling"""
        self.failure_handler.register_recovery_handler(
            RecoveryStrategy.RETRY, self._handle_retry_recovery
        )
        self.failure_handler.register_recovery_handler(
            RecoveryStrategy.REASSIGN, self._handle_reassign_recovery
        )
        self.failure_handler.register_recovery_handler(
            RecoveryStrategy.ESCALATE, self._handle_escalate_recovery
        )
    
    async def start(self):
        """Start the orchestra"""
        if self._is_running:
            logger.warning("Orchestra is already running")
            return
        
        await self.state_manager.initialize()
        
        self._is_running = True
        
        # Start background tasks
        self._executor_task = asyncio.create_task(self._task_executor_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        logger.info("Orchestra started")
    
    async def stop(self):
        """Stop the orchestra"""
        if not self._is_running:
            logger.warning("Orchestra is not running")
            return
        
        self._is_running = False
        
        # Cancel background tasks
        if self._executor_task:
            self._executor_task.cancel()
            try:
                await self._executor_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Wait for running tasks to complete
        if self._running_tasks:
            logger.info(f"Waiting for {len(self._running_tasks)} tasks to complete")
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
        
        await self.state_manager.shutdown()
        
        logger.info("Orchestra stopped")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
        if exc_type:
            logger.error(
                "Orchestra context exited with exception",
                exception_type=exc_type.__name__,
                exception_message=str(exc_val)
            )
        return False  # Don't suppress exceptions
    
    def register_agent(self, agent: Agent):
        """Register an agent with the orchestra"""
        if agent.id in self._agents:
            raise AgentRegistrationError(f"Agent {agent.id} is already registered")
        
        self._agents[agent.id] = agent
        
        # Register with router and state manager
        agent_info = agent.get_info()
        self.task_router.register_agent(agent_info)
        
        if self._is_running:
            asyncio.create_task(self.state_manager.register_agent(agent_info))
        
        logger.info("Agent registered", agent_id=agent.id, name=agent.name)
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id not in self._agents:
            raise AgentNotFoundError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        
        # Check if agent is busy
        if agent.status == AgentStatus.BUSY:
            logger.warning(
                "Unregistering busy agent",
                agent_id=agent_id,
                current_task=agent.current_task
            )
        
        del self._agents[agent_id]
        self.task_router.unregister_agent(agent_id)
        
        logger.info("Agent unregistered", agent_id=agent_id)
    
    async def submit_task(self, task_data: Dict[str, Any]) -> str:
        """Submit a task for execution"""
        # Create task object
        task = Task(
            type=task_data["type"],
            data=task_data.get("data", {}),
            priority=TaskPriority(task_data.get("priority", "normal")),
            timeout=task_data.get("timeout", self._task_timeout_default),
            max_retries=task_data.get("max_retries", 3),
            dependencies=task_data.get("dependencies", [])
        )
        
        # Store task in state
        await self.state_manager.store_task(task)
        
        # Add to router queue
        self.task_router.add_task(task)
        
        logger.info(
            "Task submitted",
            task_id=task.id,
            task_type=task.type,
            priority=task.priority
        )
        
        return task.id
    
    async def get_task_result(self, task_id: str) -> Optional[ExecutionResult]:
        """Get the result of a completed task"""
        task = await self.state_manager.get_task(task_id)
        
        if not task:
            return None
        
        if task.status == TaskStatus.COMPLETED:
            return ExecutionResult(
                task_id=task.id,
                success=True,
                result=task.result,
                execution_time=(
                    task.completed_at - task.started_at
                ).total_seconds() if task.started_at and task.completed_at else 0
            )
        elif task.status == TaskStatus.FAILED:
            return ExecutionResult(
                task_id=task.id,
                success=False,
                error=task.error,
                execution_time=(
                    task.completed_at - task.started_at
                ).total_seconds() if task.started_at and task.completed_at else 0
            )
        
        return None
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> ExecutionResult:
        """Wait for a task to complete and return the result"""
        start_time = asyncio.get_event_loop().time()
        
        while True:
            result = await self.get_task_result(task_id)
            if result:
                return result
            
            # Check timeout
            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            await asyncio.sleep(0.5)
    
    async def _task_executor_loop(self):
        """Main task execution loop"""
        logger.info("Task executor started")
        
        while self._is_running:
            try:
                # Check if we can execute more tasks
                if len(self._running_tasks) >= self._max_concurrent_tasks:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next task from router
                task = self.task_router.get_next_task()
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                
                # Find suitable agent
                agent_id = self.task_router.find_suitable_agent(task)
                if not agent_id:
                    # No suitable agent available, put task back
                    self.task_router.add_task(task)
                    await asyncio.sleep(1.0)
                    continue
                
                # Execute task
                execution_task = asyncio.create_task(
                    self._execute_task(task, agent_id)
                )
                self._running_tasks[task.id] = execution_task
                
                # Set up task completion callback
                execution_task.add_done_callback(
                    lambda t, tid=task.id: self._running_tasks.pop(tid, None)
                )
                
            except Exception as e:
                logger.error("Error in task executor loop", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _execute_task(self, task: Task, agent_id: str):
        """Execute a single task"""
        agent = self._agents[agent_id]
        
        try:
            # Update task status
            task.assigned_agent = agent_id
            await self.state_manager.update_task_status(
                task.id, TaskStatus.RUNNING, assigned_agent=agent_id
            )
            
            logger.info(
                "Executing task",
                task_id=task.id,
                agent_id=agent_id,
                task_type=task.type
            )
            
            # Execute task on agent
            result = await agent.execute_task(task)
            
            if result.success:
                # Task completed successfully
                await self.state_manager.update_task_status(
                    task.id, TaskStatus.COMPLETED, result=result.result
                )
                self.failure_handler.reset_circuit_breaker(agent_id)
                
                logger.info(
                    "Task completed successfully",
                    task_id=task.id,
                    agent_id=agent_id,
                    execution_time=result.execution_time
                )
            else:
                # Task failed
                await self._handle_task_failure(task, agent_id, result.error)
        
        except Exception as e:
            # Execution error
            await self._handle_task_failure(task, agent_id, str(e))
    
    async def _handle_task_failure(self, task: Task, agent_id: str, error: str):
        """Handle task execution failure"""
        logger.error(
            "Task execution failed",
            task_id=task.id,
            agent_id=agent_id,
            error=error
        )
        
        # Create exception for failure handler
        exception = TaskExecutionError(error)
        
        # Handle failure and get recovery strategy
        recovery_strategy = await self.failure_handler.handle_failure(
            task, agent_id, exception
        )
        
        # Execute recovery strategy
        if recovery_strategy == RecoveryStrategy.RETRY:
            # Increment retry count and requeue
            task.retry_count += 1
            if task.retry_count <= task.max_retries:
                # Calculate backoff delay
                delay = self.failure_handler.calculate_backoff_delay(task.id)
                await asyncio.sleep(delay)
                
                # Reset assigned agent and requeue
                task.assigned_agent = None
                task.status = TaskStatus.PENDING
                await self.state_manager.store_task(task)
                self.task_router.add_task(task)
                
                logger.info(
                    "Task queued for retry",
                    task_id=task.id,
                    retry_count=task.retry_count,
                    delay=delay
                )
            else:
                # Max retries exceeded
                await self.state_manager.update_task_status(
                    task.id, TaskStatus.FAILED, error=f"Max retries exceeded: {error}"
                )
        
        elif recovery_strategy == RecoveryStrategy.REASSIGN:
            # Reset assigned agent and requeue
            task.assigned_agent = None
            task.status = TaskStatus.PENDING
            await self.state_manager.store_task(task)
            self.task_router.add_task(task)
            
            logger.info("Task queued for reassignment", task_id=task.id)
        
        else:
            # Abort task
            await self.state_manager.update_task_status(
                task.id, TaskStatus.FAILED, error=error
            )
    
    async def _heartbeat_loop(self):
        """Heartbeat loop to monitor agent health"""
        logger.info("Heartbeat monitor started")
        
        while self._is_running:
            try:
                for agent_id, agent in self._agents.items():
                    # Update heartbeat
                    agent.update_heartbeat()
                    
                    # Update in state manager
                    await self.state_manager.update_agent_heartbeat(agent_id)
                    
                    # Update router
                    self.task_router.update_agent_status(agent_id, agent.get_info())
                
                await asyncio.sleep(self._heartbeat_interval)
                
            except Exception as e:
                logger.error("Error in heartbeat loop", error=str(e))
                await asyncio.sleep(self._heartbeat_interval)
    
    async def _handle_retry_recovery(self, task: Task, failure_record):
        """Handle retry recovery strategy"""
        logger.info("Handling retry recovery", task_id=task.id)
        # Implementation handled in _handle_task_failure
    
    async def _handle_reassign_recovery(self, task: Task, failure_record):
        """Handle reassign recovery strategy"""
        logger.info("Handling reassign recovery", task_id=task.id)
        # Implementation handled in _handle_task_failure
    
    async def _handle_escalate_recovery(self, task: Task, failure_record):
        """Handle escalate recovery strategy"""
        logger.warning(
            "Task failure escalated",
            task_id=task.id,
            failure_type=failure_record.failure_type,
            error=failure_record.error_message
        )
        # Could implement notifications, alerts, etc.
    
    async def get_status(self) -> Dict[str, Any]:
        """Get orchestra status"""
        global_state = await self.state_manager.get_global_state()
        router_status = self.task_router.get_queue_status()
        failure_stats = self.failure_handler.get_failure_statistics()
        
        return {
            "is_running": self._is_running,
            "running_tasks": len(self._running_tasks),
            "max_concurrent_tasks": self._max_concurrent_tasks,
            "registered_agents": len(self._agents),
            "global_state": global_state,
            "router_status": router_status,
            "failure_statistics": failure_stats
        }
    
    def get_agents(self) -> List[AgentInfo]:
        """Get list of all registered agents"""
        return [agent.get_info() for agent in self._agents.values()]