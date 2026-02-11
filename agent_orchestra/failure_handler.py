"""
Failure handling and recovery mechanisms for Agent Orchestra
"""
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import structlog

from .types import Task, TaskStatus, AgentInfo, AgentStatus, ExecutionResult
from .exceptions import (
    TaskExecutionError, TaskTimeoutError, AgentUnavailableError,
    ResourceExhaustionError, AgentOrchestraException
)

logger = structlog.get_logger(__name__)


class FailureType(str, Enum):
    TASK_TIMEOUT = "task_timeout"
    AGENT_UNAVAILABLE = "agent_unavailable"
    EXECUTION_ERROR = "execution_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(str, Enum):
    RETRY = "retry"
    REASSIGN = "reassign"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    ABORT = "abort"


class FailureRecord:
    def __init__(
        self,
        task_id: str,
        agent_id: Optional[str],
        failure_type: FailureType,
        error_message: str,
        timestamp: datetime = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.task_id = task_id
        self.agent_id = agent_id
        self.failure_type = failure_type
        self.error_message = error_message
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata = metadata or {}


class FailureHandler:
    """
    Comprehensive failure handling and recovery system
    """
    
    def __init__(self):
        self._failure_history: List[FailureRecord] = []
        self._recovery_strategies: Dict[FailureType, RecoveryStrategy] = {
            FailureType.TASK_TIMEOUT: RecoveryStrategy.RETRY,
            FailureType.AGENT_UNAVAILABLE: RecoveryStrategy.REASSIGN,
            FailureType.EXECUTION_ERROR: RecoveryStrategy.RETRY,
            FailureType.RESOURCE_EXHAUSTION: RecoveryStrategy.ESCALATE,
            FailureType.NETWORK_ERROR: RecoveryStrategy.RETRY,
            FailureType.VALIDATION_ERROR: RecoveryStrategy.ABORT,
            FailureType.UNKNOWN_ERROR: RecoveryStrategy.ESCALATE
        }
        self._recovery_handlers: Dict[RecoveryStrategy, Callable] = {}
        self._circuit_breakers: Dict[str, Dict] = {}
        self._backoff_strategies: Dict[str, float] = {}
        
    def register_recovery_handler(self, strategy: RecoveryStrategy, handler: Callable):
        """Register a recovery handler for a specific strategy"""
        self._recovery_handlers[strategy] = handler
        logger.info("Recovery handler registered", strategy=strategy)
    
    def set_recovery_strategy(self, failure_type: FailureType, strategy: RecoveryStrategy):
        """Set the recovery strategy for a specific failure type"""
        self._recovery_strategies[failure_type] = strategy
        logger.info(
            "Recovery strategy updated",
            failure_type=failure_type,
            strategy=strategy
        )
    
    async def handle_failure(
        self,
        task: Task,
        agent_id: Optional[str],
        error: Exception,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RecoveryStrategy:
        """Handle a failure and determine recovery strategy"""
        
        # Classify the failure
        failure_type = self._classify_failure(error)
        
        # Record the failure
        failure_record = FailureRecord(
            task_id=task.id,
            agent_id=agent_id,
            failure_type=failure_type,
            error_message=str(error),
            metadata=metadata
        )
        self._failure_history.append(failure_record)
        
        logger.error(
            "Failure recorded",
            task_id=task.id,
            agent_id=agent_id,
            failure_type=failure_type,
            error=str(error)
        )
        
        # Check circuit breaker
        if agent_id and self._is_circuit_breaker_open(agent_id):
            logger.warning(
                "Circuit breaker open for agent",
                agent_id=agent_id,
                task_id=task.id
            )
            return RecoveryStrategy.REASSIGN
        
        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(task, failure_record)
        
        # Update circuit breaker state
        if agent_id:
            self._update_circuit_breaker(agent_id, failure_type)
        
        # Execute recovery if handler is available
        if strategy in self._recovery_handlers:
            try:
                await self._recovery_handlers[strategy](task, failure_record)
            except Exception as recovery_error:
                logger.error(
                    "Recovery handler failed",
                    strategy=strategy,
                    task_id=task.id,
                    error=str(recovery_error)
                )
                # Escalate if recovery fails
                strategy = RecoveryStrategy.ESCALATE
        
        return strategy
    
    def _classify_failure(self, error: Exception) -> FailureType:
        """Classify an exception into a failure type"""
        if isinstance(error, TaskTimeoutError):
            return FailureType.TASK_TIMEOUT
        elif isinstance(error, AgentUnavailableError):
            return FailureType.AGENT_UNAVAILABLE
        elif isinstance(error, TaskExecutionError):
            return FailureType.EXECUTION_ERROR
        elif isinstance(error, ResourceExhaustionError):
            return FailureType.RESOURCE_EXHAUSTION
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return FailureType.NETWORK_ERROR
        elif isinstance(error, (ValueError, TypeError)):
            return FailureType.VALIDATION_ERROR
        else:
            return FailureType.UNKNOWN_ERROR
    
    def _determine_recovery_strategy(
        self, 
        task: Task, 
        failure_record: FailureRecord
    ) -> RecoveryStrategy:
        """Determine the appropriate recovery strategy"""
        
        # Check if task has exceeded retry limit
        if task.retry_count >= task.max_retries:
            logger.info(
                "Task exceeded retry limit",
                task_id=task.id,
                retry_count=task.retry_count,
                max_retries=task.max_retries
            )
            return RecoveryStrategy.ABORT
        
        # Get base strategy for failure type
        base_strategy = self._recovery_strategies.get(
            failure_record.failure_type,
            RecoveryStrategy.ESCALATE
        )
        
        # Check failure history for this task
        task_failures = [
            f for f in self._failure_history 
            if f.task_id == task.id
        ]
        
        # If task has failed multiple times, escalate
        if len(task_failures) > 2:
            logger.info(
                "Task has multiple failures, escalating",
                task_id=task.id,
                failure_count=len(task_failures)
            )
            return RecoveryStrategy.ESCALATE
        
        return base_strategy
    
    def _is_circuit_breaker_open(self, agent_id: str) -> bool:
        """Check if circuit breaker is open for an agent"""
        if agent_id not in self._circuit_breakers:
            return False
        
        breaker = self._circuit_breakers[agent_id]
        
        # Check if breaker is in open state
        if breaker.get("state") == "open":
            # Check if timeout has passed
            timeout = breaker.get("timeout", datetime.utcnow())
            if datetime.utcnow() > timeout:
                # Move to half-open state
                breaker["state"] = "half-open"
                breaker["consecutive_failures"] = 0
                logger.info(
                    "Circuit breaker moved to half-open",
                    agent_id=agent_id
                )
                return False
            return True
        
        return False
    
    def _update_circuit_breaker(self, agent_id: str, failure_type: FailureType):
        """Update circuit breaker state based on failure"""
        if agent_id not in self._circuit_breakers:
            self._circuit_breakers[agent_id] = {
                "state": "closed",
                "consecutive_failures": 0,
                "last_failure": None,
                "timeout": None
            }
        
        breaker = self._circuit_breakers[agent_id]
        breaker["consecutive_failures"] += 1
        breaker["last_failure"] = datetime.utcnow()
        
        # Open circuit breaker after 3 consecutive failures
        if breaker["consecutive_failures"] >= 3:
            breaker["state"] = "open"
            breaker["timeout"] = datetime.utcnow() + timedelta(minutes=5)
            
            logger.warning(
                "Circuit breaker opened",
                agent_id=agent_id,
                consecutive_failures=breaker["consecutive_failures"]
            )
    
    def reset_circuit_breaker(self, agent_id: str):
        """Reset circuit breaker for an agent"""
        if agent_id in self._circuit_breakers:
            self._circuit_breakers[agent_id] = {
                "state": "closed",
                "consecutive_failures": 0,
                "last_failure": None,
                "timeout": None
            }
            logger.info("Circuit breaker reset", agent_id=agent_id)
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure statistics"""
        total_failures = len(self._failure_history)
        if total_failures == 0:
            return {"total_failures": 0}
        
        # Count failures by type
        failure_counts = {}
        for record in self._failure_history:
            failure_type = record.failure_type.value
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
        
        # Calculate failure rate over last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_failures = [
            f for f in self._failure_history
            if f.timestamp > one_hour_ago
        ]
        
        # Get agent failure statistics
        agent_failures = {}
        for record in self._failure_history:
            if record.agent_id:
                agent_failures[record.agent_id] = agent_failures.get(record.agent_id, 0) + 1
        
        return {
            "total_failures": total_failures,
            "failure_types": failure_counts,
            "recent_failures_1h": len(recent_failures),
            "agent_failures": agent_failures,
            "circuit_breakers": {
                agent_id: breaker["state"] 
                for agent_id, breaker in self._circuit_breakers.items()
            }
        }
    
    def cleanup_old_failures(self, older_than_days: int = 7):
        """Clean up old failure records"""
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        
        old_count = len(self._failure_history)
        self._failure_history = [
            f for f in self._failure_history
            if f.timestamp > cutoff_date
        ]
        
        cleaned_count = old_count - len(self._failure_history)
        logger.info(
            "Old failure records cleaned",
            cleaned_count=cleaned_count,
            remaining_count=len(self._failure_history)
        )
    
    def calculate_backoff_delay(self, task_id: str, base_delay: float = 1.0) -> float:
        """Calculate exponential backoff delay for task retry"""
        task_failures = [
            f for f in self._failure_history 
            if f.task_id == task_id
        ]
        
        failure_count = len(task_failures)
        delay = base_delay * (2 ** min(failure_count, 6))  # Max 64x base delay
        
        logger.debug(
            "Backoff delay calculated",
            task_id=task_id,
            failure_count=failure_count,
            delay=delay
        )
        
        return delay