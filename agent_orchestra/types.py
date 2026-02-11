"""
Core types and data structures for Agent Orchestra
"""
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class AgentStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout: Optional[int] = None  # seconds
    retry_count: int = 0
    max_retries: int = 3
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)


class AgentCapability(BaseModel):
    name: str
    description: Optional[str] = None
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)


class AgentInfo(BaseModel):
    id: str
    name: str
    capabilities: List[AgentCapability] = Field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    task_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)