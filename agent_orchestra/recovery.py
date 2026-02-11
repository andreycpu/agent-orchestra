"""
Error recovery and self-healing mechanisms for Agent Orchestra
"""
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
import structlog

from .types import Task, Agent, TaskStatus, AgentStatus
from .exceptions import (
    AgentOrchestraException, TaskExecutionError, AgentUnavailableError,
    ResourceExhaustionError, NetworkError
)

logger = structlog.get_logger(__name__)


class RecoveryStrategy(str, Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FAILOVER = "failover"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    RESTART = "restart"
    QUARANTINE = "quarantine"


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class CircuitBreaker:
    """Circuit breaker pattern implementation for error recovery"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout: float = 60.0,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.name = name
        
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._state = "closed"  # closed, open, half_open
        
        logger.info(
            "Circuit breaker initialized",
            name=name,
            failure_threshold=failure_threshold,
            timeout=timeout
        )
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        if self._state == "open":
            if self._should_attempt_reset():
                self._state = "half_open"
                logger.info("Circuit breaker half-open", name=self.name)
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful execution"""
        self._failure_count = 0
        
        if self._state == "half_open":
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = "closed"
                self._success_count = 0
                logger.info("Circuit breaker closed", name=self.name)
    
    async def _on_failure(self, exception: Exception):
        """Handle failed execution"""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = "open"
            self._success_count = 0
            logger.warning(
                "Circuit breaker opened",
                name=self.name,
                failure_count=self._failure_count,
                exception=str(exception)
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self._last_failure_time is None:
            return True
        
        return time.time() - self._last_failure_time >= self.timeout


class CircuitBreakerOpenError(AgentOrchestraException):
    """Raised when circuit breaker is open"""
    pass


class ErrorRecoveryManager:
    """Manages error recovery strategies and self-healing mechanisms"""
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._quarantined_agents: Set[str] = set()
        self._recovery_strategies: Dict[type, List[RecoveryStrategy]] = {
            TaskExecutionError: [RecoveryStrategy.RETRY, RecoveryStrategy.FAILOVER],
            AgentUnavailableError: [RecoveryStrategy.FAILOVER, RecoveryStrategy.RESTART],
            ResourceExhaustionError: [RecoveryStrategy.GRACEFUL_DEGRADATION, RecoveryStrategy.CIRCUIT_BREAKER],
            NetworkError: [RecoveryStrategy.RETRY, RecoveryStrategy.CIRCUIT_BREAKER],
        }
        self._health_checks: Dict[str, Callable] = {}
        self._recovery_handlers: Dict[RecoveryStrategy, Callable] = {
            RecoveryStrategy.RETRY: self._handle_retry,
            RecoveryStrategy.FAILOVER: self._handle_failover,
            RecoveryStrategy.CIRCUIT_BREAKER: self._handle_circuit_breaker,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._handle_degradation,
            RecoveryStrategy.RESTART: self._handle_restart,
            RecoveryStrategy.QUARANTINE: self._handle_quarantine,
        }
        
        logger.info("Error recovery manager initialized")
    
    def register_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout: float = 60.0
    ) -> CircuitBreaker:
        """Register a new circuit breaker"""
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            name=name
        )
        self._circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self._circuit_breakers.get(name)
    
    async def recover_from_error(
        self,
        exception: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Attempt to recover from an error using appropriate strategies
        
        Args:
            exception: The exception that occurred
            context: Context information about the error
            
        Returns:
            Recovery result with status and actions taken
        """
        exception_type = type(exception)
        strategies = self._recovery_strategies.get(exception_type, [RecoveryStrategy.RETRY])
        
        recovery_result = {
            "success": False,
            "strategies_attempted": [],
            "actions_taken": [],
            "final_strategy": None
        }
        
        logger.info(
            "Starting error recovery",
            exception_type=exception_type.__name__,
            exception_message=str(exception),
            strategies=strategies,
            context=context
        )
        
        for strategy in strategies:
            try:
                recovery_result["strategies_attempted"].append(strategy.value)
                
                handler = self._recovery_handlers.get(strategy)
                if handler:
                    result = await handler(exception, context)
                    recovery_result["actions_taken"].extend(result.get("actions", []))
                    
                    if result.get("success", False):
                        recovery_result["success"] = True
                        recovery_result["final_strategy"] = strategy.value
                        break
                        
            except Exception as recovery_error:
                logger.error(
                    "Recovery strategy failed",
                    strategy=strategy.value,
                    recovery_error=str(recovery_error)
                )
        
        logger.info(
            "Error recovery completed",
            success=recovery_result["success"],
            strategies_attempted=recovery_result["strategies_attempted"],
            actions_taken=recovery_result["actions_taken"]
        )
        
        return recovery_result
    
    async def _handle_retry(self, exception: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle retry recovery strategy"""
        max_retries = context.get("max_retries", 3)
        current_retry = context.get("current_retry", 0)
        
        if current_retry < max_retries:
            # Exponential backoff
            delay = min(2 ** current_retry, 30)
            await asyncio.sleep(delay)
            
            return {
                "success": True,
                "actions": [f"retry_after_{delay}_seconds"],
                "retry_count": current_retry + 1
            }
        
        return {"success": False, "actions": ["max_retries_exceeded"]}
    
    async def _handle_failover(self, exception: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle failover recovery strategy"""
        failed_agent = context.get("agent_id")
        available_agents = context.get("available_agents", [])
        
        if failed_agent in available_agents:
            available_agents.remove(failed_agent)
        
        if available_agents:
            # Add failed agent to quarantine temporarily
            self._quarantined_agents.add(failed_agent)
            
            return {
                "success": True,
                "actions": [f"failover_to_{available_agents[0]}", f"quarantine_{failed_agent}"],
                "target_agent": available_agents[0]
            }
        
        return {"success": False, "actions": ["no_agents_available"]}
    
    async def _handle_circuit_breaker(self, exception: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle circuit breaker recovery strategy"""
        service_name = context.get("service_name", "default")
        circuit_breaker = self.get_circuit_breaker(service_name)
        
        if not circuit_breaker:
            circuit_breaker = self.register_circuit_breaker(service_name)
        
        # The circuit breaker itself handles the logic
        return {
            "success": True,
            "actions": [f"circuit_breaker_activated_{service_name}"],
            "circuit_breaker": service_name
        }
    
    async def _handle_degradation(self, exception: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graceful degradation strategy"""
        return {
            "success": True,
            "actions": ["graceful_degradation_enabled", "reduced_functionality"],
            "mode": "degraded"
        }
    
    async def _handle_restart(self, exception: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle restart recovery strategy"""
        component = context.get("component", "unknown")
        
        return {
            "success": True,
            "actions": [f"restart_{component}"],
            "restart_target": component
        }
    
    async def _handle_quarantine(self, exception: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quarantine recovery strategy"""
        agent_id = context.get("agent_id")
        
        if agent_id:
            self._quarantined_agents.add(agent_id)
            
            return {
                "success": True,
                "actions": [f"quarantine_{agent_id}"],
                "quarantined_agent": agent_id
            }
        
        return {"success": False, "actions": ["no_agent_to_quarantine"]}
    
    def is_agent_quarantined(self, agent_id: str) -> bool:
        """Check if an agent is quarantined"""
        return agent_id in self._quarantined_agents
    
    def release_from_quarantine(self, agent_id: str) -> bool:
        """Release an agent from quarantine"""
        if agent_id in self._quarantined_agents:
            self._quarantined_agents.remove(agent_id)
            logger.info("Agent released from quarantine", agent_id=agent_id)
            return True
        return False
    
    def get_quarantined_agents(self) -> Set[str]:
        """Get list of quarantined agents"""
        return self._quarantined_agents.copy()
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        status = {}
        for name, cb in self._circuit_breakers.items():
            status[name] = {
                "state": cb._state,
                "failure_count": cb._failure_count,
                "success_count": cb._success_count,
                "last_failure": cb._last_failure_time
            }
        return status