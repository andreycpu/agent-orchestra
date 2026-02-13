"""
Agent implementation for the Agent Orchestra framework
"""
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Union, Awaitable
from datetime import datetime
import structlog

from .types import Task, AgentInfo, AgentCapability, AgentStatus, ExecutionResult
from .exceptions import TaskExecutionError, AgentUnavailableError, ValidationError

logger = structlog.get_logger(__name__)


class Agent:
    """
    An individual agent that can execute tasks within the orchestra
    """
    
    def __init__(
        self, 
        agent_id: str,
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        # Input validation
        if not agent_id or not isinstance(agent_id, str):
            raise ValidationError("agent_id must be a non-empty string")
        if not agent_id.strip():
            raise ValidationError("agent_id cannot be whitespace only")
        if name is not None and not isinstance(name, str):
            raise ValidationError("name must be a string")
        if capabilities is not None and not isinstance(capabilities, list):
            raise ValidationError("capabilities must be a list")
        if capabilities and not all(isinstance(cap, str) for cap in capabilities):
            raise ValidationError("all capabilities must be strings")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValidationError("metadata must be a dictionary")
            
        self.id = agent_id.strip()
        self.name = (name.strip() if name else agent_id.strip())
        self.capabilities = self._parse_capabilities(capabilities or [])
        self.status = AgentStatus.IDLE
        self.current_task: Optional[str] = None
        self.metadata = metadata or {}
        self._task_handlers: Dict[str, Callable] = {}
        self._last_heartbeat = datetime.utcnow()
        
        logger.info(
            "Agent initialized",
            agent_id=self.id,
            name=self.name,
            capabilities=[c.name for c in self.capabilities]
        )
    
    def _parse_capabilities(self, capabilities: List[str]) -> List[AgentCapability]:
        """Parse capability strings into AgentCapability objects"""
        return [
            AgentCapability(name=cap, description=f"Handler for {cap} tasks")
            for cap in capabilities
        ]
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler function for a specific task type
        
        Args:
            task_type: The type of task this handler can process
            handler: Callable that takes task data and returns a result
            
        Raises:
            ValidationError: If task_type is invalid or handler is not callable
        """
        if not task_type or not isinstance(task_type, str):
            raise ValidationError("task_type must be a non-empty string")
        if not task_type.strip():
            raise ValidationError("task_type cannot be whitespace only")
        if not callable(handler):
            raise ValidationError("handler must be callable")
            
        task_type = task_type.strip()
        self._task_handlers[task_type] = handler
        
        # Add capability if not already present
        if not any(cap.name == task_type for cap in self.capabilities):
            self.capabilities.append(
                AgentCapability(name=task_type, description=f"Handler for {task_type} tasks")
            )
        
        logger.info(
            "Task handler registered",
            agent_id=self.id,
            task_type=task_type
        )
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if this agent can handle the given task"""
        return task.type in self._task_handlers
    
    async def execute_task(self, task: Task) -> ExecutionResult:
        """Execute a task and return the result
        
        Args:
            task: The task to execute
            
        Returns:
            ExecutionResult: Result of task execution
            
        Raises:
            ValidationError: If task is invalid
            AgentUnavailableError: If agent is not available
            TaskExecutionError: If agent cannot handle task type
        """
        if task is None:
            raise ValidationError("task cannot be None")
        if not isinstance(task, Task):
            raise ValidationError("task must be a Task instance")
        if not hasattr(task, 'type') or not task.type:
            raise ValidationError("task must have a valid type")
            
        start_time = time.time()
        
        if self.status != AgentStatus.IDLE:
            raise AgentUnavailableError(f"Agent {self.id} is not available")
        
        if not self.can_handle_task(task):
            raise TaskExecutionError(f"Agent {self.id} cannot handle task type {task.type}")
        
        self.status = AgentStatus.BUSY
        self.current_task = task.id
        
        logger.info(
            "Starting task execution",
            agent_id=self.id,
            task_id=task.id,
            task_type=task.type
        )
        
        try:
            handler = self._task_handlers[task.type]
            result = await self._execute_with_timeout(handler, task)
            
            execution_time = time.time() - start_time
            
            logger.info(
                "Task completed successfully",
                agent_id=self.id,
                task_id=task.id,
                execution_time=execution_time
            )
            
            return ExecutionResult(
                task_id=task.id,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task execution failed: {str(e)}"
            
            logger.error(
                "Task execution failed",
                agent_id=self.id,
                task_id=task.id,
                error=error_msg,
                execution_time=execution_time
            )
            
            return ExecutionResult(
                task_id=task.id,
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
            
        finally:
            self.status = AgentStatus.IDLE
            self.current_task = None
            self._last_heartbeat = datetime.utcnow()
    
    async def _execute_with_timeout(self, handler: Callable, task: Task) -> Any:
        """Execute handler with optional timeout"""
        if task.timeout:
            return await asyncio.wait_for(
                handler(task.data),
                timeout=task.timeout
            )
        else:
            return await handler(task.data)
    
    def get_info(self) -> AgentInfo:
        """Get current agent information"""
        return AgentInfo(
            id=self.id,
            name=self.name,
            capabilities=self.capabilities,
            status=self.status,
            current_task=self.current_task,
            last_heartbeat=self._last_heartbeat,
            metadata=self.metadata
        )
    
    def update_heartbeat(self):
        """Update the last heartbeat timestamp"""
        self._last_heartbeat = datetime.utcnow()
        
    def set_status(self, status: AgentStatus):
        """Set the agent status"""
        old_status = self.status
        self.status = status
        
        logger.info(
            "Agent status changed",
            agent_id=self.id,
            old_status=old_status,
            new_status=status
        )