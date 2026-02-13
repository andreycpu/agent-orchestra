"""
Advanced retry mechanisms and circuit breaker patterns for Agent Orchestra.

This module provides sophisticated retry strategies, circuit breakers,
bulkhead patterns, and resilience utilities for robust system operation.
"""
import asyncio
import time
import random
from typing import Any, Dict, Optional, Union, Callable, List, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
import threading

from .exceptions import RetryableError, PermanentError, CircuitBreakerError, BulkheadFullError


logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    """Retry strategies."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    JITTERED = "jittered"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [RetryableError])
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be positive")
        if self.multiplier <= 1:
            raise ValueError("multiplier must be greater than 1")


@dataclass
class RetryState:
    """State tracking for retry operations."""
    attempts: int = 0
    total_delay: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_exception: Optional[Exception] = None
    
    @property
    def elapsed_time(self) -> timedelta:
        """Get elapsed time since start."""
        return datetime.utcnow() - self.start_time


class RetryCalculator:
    """Calculates retry delays based on strategy."""
    
    @staticmethod
    def calculate_delay(
        attempt: int,
        config: RetryConfig,
        last_delay: float = 0.0
    ) -> float:
        """Calculate delay for retry attempt."""
        if config.strategy == RetryStrategy.FIXED:
            delay = config.base_delay
        
        elif config.strategy == RetryStrategy.LINEAR:
            delay = config.base_delay * attempt
        
        elif config.strategy == RetryStrategy.EXPONENTIAL:
            delay = config.base_delay * (config.multiplier ** (attempt - 1))
        
        elif config.strategy == RetryStrategy.FIBONACCI:
            delay = RetryCalculator._fibonacci_delay(attempt, config.base_delay)
        
        elif config.strategy == RetryStrategy.JITTERED:
            base_delay = config.base_delay * (config.multiplier ** (attempt - 1))
            jitter = random.uniform(-0.1, 0.1) * base_delay
            delay = base_delay + jitter
        
        else:
            delay = config.base_delay
        
        # Apply jitter if enabled
        if config.jitter and config.strategy != RetryStrategy.JITTERED:
            jitter_amount = delay * 0.1 * (random.random() - 0.5)
            delay += jitter_amount
        
        # Cap at maximum delay
        delay = min(delay, config.max_delay)
        
        return max(0.0, delay)
    
    @staticmethod
    def _fibonacci_delay(attempt: int, base_delay: float) -> float:
        """Calculate Fibonacci-based delay."""
        if attempt <= 1:
            return base_delay
        
        a, b = 1, 1
        for _ in range(2, attempt):
            a, b = b, a + b
        
        return base_delay * b


class RetryExecutor:
    """Executes operations with retry logic."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check if exception is retryable
        for exc_type in self.config.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        return False
    
    async def execute_async(self, operation: Callable[[], Any]) -> Any:
        """Execute async operation with retry."""
        state = RetryState()
        
        while state.attempts < self.config.max_attempts:
            state.attempts += 1
            
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return operation()
                    
            except Exception as e:
                state.last_exception = e
                
                logger.info(
                    f"Retry attempt {state.attempts}/{self.config.max_attempts} failed: {e}",
                    extra={
                        'operation': operation.__name__ if hasattr(operation, '__name__') else 'unknown',
                        'attempt': state.attempts,
                        'max_attempts': self.config.max_attempts,
                        'exception': str(e)
                    }
                )
                
                if not self.should_retry(e, state.attempts):
                    raise e
                
                # Calculate and apply delay
                delay = RetryCalculator.calculate_delay(
                    state.attempts,
                    self.config
                )
                
                if delay > 0:
                    state.total_delay += delay
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        if state.last_exception:
            raise state.last_exception
        else:
            raise RuntimeError("Operation failed without exception")
    
    def execute_sync(self, operation: Callable[[], Any]) -> Any:
        """Execute sync operation with retry."""
        state = RetryState()
        
        while state.attempts < self.config.max_attempts:
            state.attempts += 1
            
            try:
                return operation()
                    
            except Exception as e:
                state.last_exception = e
                
                logger.info(
                    f"Retry attempt {state.attempts}/{self.config.max_attempts} failed: {e}"
                )
                
                if not self.should_retry(e, state.attempts):
                    raise e
                
                # Calculate and apply delay
                delay = RetryCalculator.calculate_delay(
                    state.attempts,
                    self.config
                )
                
                if delay > 0:
                    state.total_delay += delay
                    time.sleep(delay)
        
        # All retries exhausted
        if state.last_exception:
            raise state.last_exception
        else:
            raise RuntimeError("Operation failed without exception")


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    expected_exception: Type[Exception] = Exception
    
    def __post_init__(self):
        """Validate configuration."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be at least 1")


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.utcnow()
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            
            elif self.state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has passed
                if (self.last_failure_time and 
                    datetime.utcnow() - self.last_failure_time >= 
                    timedelta(seconds=self.config.recovery_timeout)):
                    
                    self._transition_to_half_open()
                    return True
                
                return False
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                return True
            
            return False
    
    def record_success(self):
        """Record successful execution."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self, exception: Exception):
        """Record failed execution."""
        with self._lock:
            if isinstance(exception, self.config.expected_exception):
                self.failure_count += 1
                self.last_failure_time = datetime.utcnow()
                
                if self.state == CircuitBreakerState.CLOSED:
                    if self.failure_count >= self.config.failure_threshold:
                        self._transition_to_open()
                
                elif self.state == CircuitBreakerState.HALF_OPEN:
                    self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        logger.warning(f"Circuit breaker '{self.name}' transitioning to OPEN state")
        self.state = CircuitBreakerState.OPEN
        self.last_state_change = datetime.utcnow()
        self.success_count = 0
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN state")
        self.state = CircuitBreakerState.HALF_OPEN
        self.last_state_change = datetime.utcnow()
        self.success_count = 0
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED state")
        self.state = CircuitBreakerState.CLOSED
        self.last_state_change = datetime.utcnow()
        self.failure_count = 0
        self.success_count = 0
    
    async def execute_async(self, operation: Callable[[], Any]) -> Any:
        """Execute async operation through circuit breaker."""
        if not self.can_execute():
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is {self.state.value}"
            )
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = operation()
            
            self.record_success()
            return result
            
        except Exception as e:
            self.record_failure(e)
            raise
    
    def execute_sync(self, operation: Callable[[], Any]) -> Any:
        """Execute sync operation through circuit breaker."""
        if not self.can_execute():
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is {self.state.value}"
            )
        
        try:
            result = operation()
            self.record_success()
            return result
            
        except Exception as e:
            self.record_failure(e)
            raise
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get circuit breaker state information."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'last_state_change': self.last_state_change.isoformat(),
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'success_threshold': self.config.success_threshold
                }
            }


class Bulkhead:
    """Bulkhead pattern for resource isolation."""
    
    def __init__(self, name: str, max_concurrent: int):
        self.name = name
        self.max_concurrent = max_concurrent
        self.current_executions = 0
        self.total_executions = 0
        self.total_rejections = 0
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = threading.Lock()
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire bulkhead resource."""
        acquired = await self._semaphore.acquire()
        if not acquired:
            with self._lock:
                self.total_rejections += 1
            raise BulkheadFullError(f"Bulkhead '{self.name}' is full")
        
        try:
            with self._lock:
                self.current_executions += 1
                self.total_executions += 1
            yield
        finally:
            with self._lock:
                self.current_executions -= 1
            self._semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        with self._lock:
            return {
                'name': self.name,
                'max_concurrent': self.max_concurrent,
                'current_executions': self.current_executions,
                'total_executions': self.total_executions,
                'total_rejections': self.total_rejections,
                'rejection_rate': self.total_rejections / max(self.total_executions + self.total_rejections, 1)
            }


class ResilienceManager:
    """Manages retry, circuit breaker, and bulkhead patterns."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.retry_executors: Dict[str, RetryExecutor] = {}
    
    def create_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Create and register circuit breaker."""
        config = config or CircuitBreakerConfig()
        breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        return breaker
    
    def create_bulkhead(self, name: str, max_concurrent: int) -> Bulkhead:
        """Create and register bulkhead."""
        bulkhead = Bulkhead(name, max_concurrent)
        self.bulkheads[name] = bulkhead
        return bulkhead
    
    def create_retry_executor(
        self,
        name: str,
        config: Optional[RetryConfig] = None
    ) -> RetryExecutor:
        """Create and register retry executor."""
        config = config or RetryConfig()
        executor = RetryExecutor(config)
        self.retry_executors[name] = executor
        return executor
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def get_bulkhead(self, name: str) -> Optional[Bulkhead]:
        """Get bulkhead by name."""
        return self.bulkheads.get(name)
    
    def get_retry_executor(self, name: str) -> Optional[RetryExecutor]:
        """Get retry executor by name."""
        return self.retry_executors.get(name)
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get state of all resilience components."""
        return {
            'circuit_breakers': {
                name: breaker.get_state_info()
                for name, breaker in self.circuit_breakers.items()
            },
            'bulkheads': {
                name: bulkhead.get_stats()
                for name, bulkhead in self.bulkheads.items()
            }
        }


# Global resilience manager
resilience_manager = ResilienceManager()


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for adding retry logic to functions."""
    def decorator(func):
        executor = RetryExecutor(config or RetryConfig())
        
        async def async_wrapper(*args, **kwargs):
            return await executor.execute_async(lambda: func(*args, **kwargs))
        
        def sync_wrapper(*args, **kwargs):
            return executor.execute_sync(lambda: func(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for adding circuit breaker protection."""
    def decorator(func):
        breaker = resilience_manager.create_circuit_breaker(name, config)
        
        async def async_wrapper(*args, **kwargs):
            return await breaker.execute_async(lambda: func(*args, **kwargs))
        
        def sync_wrapper(*args, **kwargs):
            return breaker.execute_sync(lambda: func(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_bulkhead(name: str, max_concurrent: int):
    """Decorator for adding bulkhead isolation."""
    def decorator(func):
        bulkhead = resilience_manager.create_bulkhead(name, max_concurrent)
        
        async def async_wrapper(*args, **kwargs):
            async with bulkhead.acquire():
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        return async_wrapper
    
    return decorator