"""
Core types and data structures for Agent Orchestra
"""
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
import uuid


class TaskStatus(str, Enum):
    """Enumeration of possible task execution states
    
    Values:
        PENDING: Task has been created but not yet assigned to an agent
        RUNNING: Task is currently being executed by an agent
        COMPLETED: Task has been successfully completed
        FAILED: Task execution failed with an error
        CANCELLED: Task was cancelled before completion
        TIMEOUT: Task execution exceeded the specified timeout
    """
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class AgentStatus(str, Enum):
    """Enumeration of possible agent states
    
    Values:
        IDLE: Agent is available and waiting for tasks
        BUSY: Agent is currently executing a task
        UNAVAILABLE: Agent is offline or temporarily unavailable
        ERROR: Agent is in an error state and cannot process tasks
    """
    IDLE = "idle"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class TaskPriority(str, Enum):
    """Enumeration of task priority levels
    
    Values:
        LOW: Low priority task, executed when no higher priority tasks available
        NORMAL: Standard priority task, default for most operations
        HIGH: High priority task, processed before normal priority tasks
        URGENT: Critical priority task, processed immediately
    """
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Task(BaseModel):
    """Represents a task to be executed by an agent in the orchestra
    
    Attributes:
        id: Unique identifier for the task (auto-generated UUID)
        type: Type of task that determines which agents can handle it
        data: Task-specific data and parameters
        priority: Task priority level affecting execution order
        status: Current execution status of the task
        assigned_agent: ID of the agent currently handling this task
        created_at: Timestamp when the task was created
        started_at: Timestamp when task execution began
        completed_at: Timestamp when task execution finished
        timeout: Maximum execution time in seconds (None for no timeout)
        retry_count: Number of times this task has been retried
        max_retries: Maximum number of retry attempts allowed
        result: Task execution result data (populated on completion)
        error: Error message if task execution failed
        dependencies: List of task IDs that must complete before this task
        
    Example:
        >>> task = Task(
        ...     type="process_data",
        ...     data={"input_file": "data.csv", "operation": "analyze"},
        ...     priority=TaskPriority.HIGH,
        ...     timeout=300
        ... )
    """
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the task"
    )
    type: str = Field(
        ...,
        description="Type of task that determines agent compatibility"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific data and parameters"
    )
    priority: TaskPriority = Field(
        default=TaskPriority.NORMAL,
        description="Task priority level affecting execution order"
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current execution status of the task"
    )
    assigned_agent: Optional[str] = Field(
        default=None,
        description="ID of the agent currently handling this task"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the task was created"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when task execution began"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when task execution finished"
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Maximum execution time in seconds"
    )
    retry_count: int = Field(
        default=0,
        description="Number of times this task has been retried"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts allowed"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Task execution result data"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if task execution failed"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of task IDs that must complete before this task"
    )
    
    @validator('type')
    def validate_task_type(cls, v):
        if not v or not v.strip():
            raise ValueError('Task type cannot be empty')
        return v.strip()
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Timeout must be positive')
        return v
    
    @validator('max_retries')
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValueError('Max retries cannot be negative')
        return v


class AgentCapability(BaseModel):
    """Represents a capability that an agent can perform
    
    Attributes:
        name: Unique name identifier for the capability
        description: Human-readable description of what the capability does
        resource_requirements: Resource requirements for this capability (CPU, memory, etc.)
        
    Example:
        >>> capability = AgentCapability(
        ...     name="image_processing",
        ...     description="Process and analyze images using computer vision",
        ...     resource_requirements={"cpu": 2, "memory_gb": 4, "gpu": True}
        ... )
    """
    name: str = Field(
        ...,
        description="Unique name identifier for the capability"
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the capability"
    )
    resource_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resource requirements for this capability"
    )
    
    @validator('name')
    def validate_name(cls, v):
        """Validate that capability name is not empty"""
        if not v or not v.strip():
            raise ValueError('Capability name cannot be empty')
        return v.strip()
        
    @validator('resource_requirements')
    def validate_resource_requirements(cls, v):
        """Validate resource requirements format"""
        if not isinstance(v, dict):
            raise ValueError('Resource requirements must be a dictionary')
        
        # Validate numeric requirements
        for key, value in v.items():
            if key in ['cpu', 'memory_gb', 'disk_gb'] and not isinstance(value, (int, float)):
                raise ValueError(f'Resource requirement {key} must be numeric')
            if key in ['cpu', 'memory_gb', 'disk_gb'] and value <= 0:
                raise ValueError(f'Resource requirement {key} must be positive')
                
        return v


class AgentInfo(BaseModel):
    """Information about an agent in the orchestra
    
    Attributes:
        id: Unique identifier for the agent
        name: Human-readable name for the agent
        capabilities: List of capabilities this agent can perform
        status: Current status of the agent
        current_task: ID of task currently being executed (if any)
        last_heartbeat: Timestamp of the last heartbeat from this agent
        metadata: Additional agent metadata and configuration
        
    Example:
        >>> agent = AgentInfo(
        ...     id="agent-001",
        ...     name="Data Processing Agent",
        ...     capabilities=[
        ...         AgentCapability(name="data_analysis"),
        ...         AgentCapability(name="csv_processing")
        ...     ],
        ...     metadata={"version": "1.2.0", "location": "us-west-1"}
        ... )
    """
    id: str = Field(
        ...,
        description="Unique identifier for the agent"
    )
    name: str = Field(
        ...,
        description="Human-readable name for the agent"
    )
    capabilities: List[AgentCapability] = Field(
        default_factory=list,
        description="List of capabilities this agent can perform"
    )
    status: AgentStatus = Field(
        default=AgentStatus.IDLE,
        description="Current status of the agent"
    )
    current_task: Optional[str] = Field(
        default=None,
        description="ID of task currently being executed"
    )
    last_heartbeat: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the last heartbeat from this agent"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional agent metadata and configuration"
    )
    
    @validator('id')
    def validate_id(cls, v):
        """Validate that agent ID is not empty"""
        if not v or not v.strip():
            raise ValueError('Agent ID cannot be empty')
        return v.strip()
    
    @validator('name')
    def validate_name(cls, v):
        """Validate that agent name is not empty"""
        if not v or not v.strip():
            raise ValueError('Agent name cannot be empty')
        return v.strip()
    
    @validator('capabilities')
    def validate_capabilities(cls, v):
        """Validate capabilities list"""
        if not isinstance(v, list):
            raise ValueError('Capabilities must be a list')
        
        # Check for duplicate capability names
        capability_names = [cap.name for cap in v]
        if len(capability_names) != len(set(capability_names)):
            raise ValueError('Duplicate capability names are not allowed')
            
        return v


class ExecutionResult(BaseModel):
    """Result of task execution by an agent
    
    Attributes:
        task_id: ID of the task that was executed
        success: Whether the task execution was successful
        result: Task execution result data (if successful)
        error: Error message (if execution failed)
        execution_time: Time taken to execute the task in seconds
        metadata: Additional execution metadata and metrics
        
    Example:
        >>> result = ExecutionResult(
        ...     task_id="task-123",
        ...     success=True,
        ...     result={"processed_records": 1000, "output_file": "results.csv"},
        ...     execution_time=45.2,
        ...     metadata={"agent_version": "1.0.0", "memory_used": "256MB"}
        ... )
    """
    task_id: str = Field(
        ...,
        description="ID of the task that was executed"
    )
    success: bool = Field(
        ...,
        description="Whether the task execution was successful"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Task execution result data (if successful)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (if execution failed)"
    )
    execution_time: float = Field(
        ...,
        description="Time taken to execute the task in seconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata and metrics"
    )
    
    @validator('task_id')
    def validate_task_id(cls, v):
        """Validate that task ID is not empty"""
        if not v or not v.strip():
            raise ValueError('Task ID cannot be empty')
        return v.strip()
    
    @validator('execution_time')
    def validate_execution_time(cls, v):
        """Validate that execution time is non-negative"""
        if v < 0:
            raise ValueError('Execution time cannot be negative')
        return v
    
    @root_validator
    def validate_result_consistency(cls, values):
        """Validate consistency between success status and result/error fields"""
        success = values.get('success')
        result = values.get('result')
        error = values.get('error')
        
        if success and error:
            raise ValueError('Successful execution should not have an error message')
        if not success and not error:
            raise ValueError('Failed execution must have an error message')
            
        return values