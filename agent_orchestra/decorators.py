"""
Utility decorators for Agent Orchestra.

This module provides common decorators for cross-cutting concerns
like caching, timing, retries, validation, and error handling.
"""
import functools
import time
import asyncio
from typing import Any, Callable, Optional, Dict, Union, List
from datetime import datetime, timedelta
import logging

from .exceptions import ValidationError, TimeoutError
from .validation import validate_timeout


logger = logging.getLogger(__name__)


def timeout(seconds: Union[int, float]):
    """
    Decorator to add timeout to function execution.
    
    Args:
        seconds: Timeout in seconds
        
    Raises:
        TimeoutError: If function execution exceeds timeout
    """
    validate_timeout(int(seconds))
    
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
                
                # Set timeout signal (Unix only)
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(seconds))
                
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Cancel alarm
                    return result
                except:
                    signal.alarm(0)  # Cancel alarm
                    raise
                finally:
                    signal.signal(signal.SIGALRM, old_handler)
            
            return sync_wrapper
    
    return decorator


def rate_limit(calls_per_second: float, burst: Optional[int] = None):
    """
    Decorator to rate limit function calls using token bucket algorithm.
    
    Args:
        calls_per_second: Maximum calls per second allowed
        burst: Maximum burst size (defaults to calls_per_second)
    """
    if calls_per_second <= 0:
        raise ValueError("calls_per_second must be positive")
    
    burst = burst or int(calls_per_second)
    tokens = burst
    last_update = time.time()
    
    def decorator(func: Callable) -> Callable:
        nonlocal tokens, last_update
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal tokens, last_update
            
            now = time.time()
            time_passed = now - last_update
            tokens = min(burst, tokens + time_passed * calls_per_second)
            last_update = now
            
            if tokens >= 1:
                tokens -= 1
                return func(*args, **kwargs)
            else:
                wait_time = (1 - tokens) / calls_per_second
                raise Exception(f"Rate limit exceeded. Try again in {wait_time:.2f} seconds")
        
        return wrapper
    
    return decorator


def validate_args(**validators):
    """
    Decorator to validate function arguments using provided validators.
    
    Args:
        **validators: Mapping of argument names to validation functions
    
    Example:
        @validate_args(
            user_id=lambda x: len(x) > 0,
            age=lambda x: 0 <= x <= 150
        )
        def create_user(user_id: str, age: int):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate arguments
            for arg_name, validator in validators.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    try:
                        if not validator(value):
                            raise ValidationError(f"Validation failed for argument '{arg_name}': {value}")
                    except ValidationError:
                        raise
                    except Exception as e:
                        raise ValidationError(f"Validation error for argument '{arg_name}': {e}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def trace_calls(logger_name: Optional[str] = None):
    """
    Decorator to trace function calls with entry/exit logging.
    
    Args:
        logger_name: Name of logger to use (defaults to function's module)
    """
    def decorator(func: Callable) -> Callable:
        func_logger = logging.getLogger(logger_name or func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Log entry
            func_logger.debug(
                f"ENTER {func.__name__}",
                extra={
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
            )
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful exit
                func_logger.debug(
                    f"EXIT {func.__name__} (success)",
                    extra={
                        'function': func.__name__,
                        'duration_seconds': duration,
                        'success': True
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log exception exit
                func_logger.debug(
                    f"EXIT {func.__name__} (exception)",
                    extra={
                        'function': func.__name__,
                        'duration_seconds': duration,
                        'success': False,
                        'exception_type': type(e).__name__,
                        'exception_message': str(e)
                    }
                )
                
                raise
        
        return wrapper
    
    return decorator


def deprecated(reason: str = "", version: str = "", removal_version: str = ""):
    """
    Decorator to mark functions as deprecated.
    
    Args:
        reason: Reason for deprecation
        version: Version when deprecated
        removal_version: Version when function will be removed
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message_parts = [f"Call to deprecated function {func.__name__}"]
            
            if version:
                message_parts.append(f"(deprecated since version {version})")
            
            if reason:
                message_parts.append(f"- {reason}")
                
            if removal_version:
                message_parts.append(f"Will be removed in version {removal_version}")
            
            warning_message = ". ".join(message_parts) + "."
            
            import warnings
            warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
            
            logger.warning(warning_message, extra={
                'deprecated_function': func.__name__,
                'deprecated_version': version,
                'removal_version': removal_version,
                'reason': reason
            })
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def singleton(cls):
    """
    Decorator to implement singleton pattern for classes.
    
    Example:
        @singleton
        class DatabaseConnection:
            pass
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def memoize(maxsize: int = 128, ttl_seconds: Optional[int] = None):
    """
    Decorator to memoize function results with optional TTL.
    
    Args:
        maxsize: Maximum number of cached results
        ttl_seconds: Time to live for cached results
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {} if ttl_seconds else None
        access_order = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = (args, tuple(sorted(kwargs.items())))
            
            # Check TTL if enabled
            if cache_times and key in cache_times:
                if time.time() - cache_times[key] > ttl_seconds:
                    # Expired, remove from cache
                    del cache[key]
                    del cache_times[key]
                    if key in access_order:
                        access_order.remove(key)
            
            # Return cached result if available
            if key in cache:
                # Update access order (move to end)
                if key in access_order:
                    access_order.remove(key)
                access_order.append(key)
                return cache[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[key] = result
            if cache_times:
                cache_times[key] = time.time()
            access_order.append(key)
            
            # Enforce maxsize (LRU eviction)
            while len(cache) > maxsize:
                oldest_key = access_order.pop(0)
                del cache[oldest_key]
                if cache_times and oldest_key in cache_times:
                    del cache_times[oldest_key]
            
            return result
        
        # Add cache inspection methods
        wrapper.cache = cache
        wrapper.cache_info = lambda: {
            'hits': 0,  # Would need to track this for real implementation
            'misses': 0,
            'maxsize': maxsize,
            'currsize': len(cache)
        }
        wrapper.cache_clear = lambda: cache.clear() or access_order.clear() or (cache_times.clear() if cache_times else None)
        
        return wrapper
    
    return decorator


def async_to_sync(func: Callable) -> Callable:
    """
    Decorator to convert async function to sync by running in event loop.
    
    Warning: Use carefully as this can block the current thread.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Try to get current event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is None:
            # No running loop, create new one
            return asyncio.run(func(*args, **kwargs))
        else:
            # Running in existing loop, use thread pool
            import concurrent.futures
            import threading
            
            def run_in_thread():
                return asyncio.run(func(*args, **kwargs))
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
    
    return wrapper


def sync_to_async(func: Callable) -> Callable:
    """
    Decorator to convert sync function to async by running in thread pool.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    return wrapper


def conditional(condition: Callable[..., bool]):
    """
    Decorator to conditionally execute a function.
    
    Args:
        condition: Function that takes same args as decorated function and returns bool
    
    Example:
        @conditional(lambda user: user.is_admin)
        def admin_only_function(user):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition(*args, **kwargs):
                return func(*args, **kwargs)
            else:
                logger.warning(f"Function {func.__name__} not executed due to failed condition")
                return None
        
        return wrapper
    
    return decorator


def ensure_types(**type_hints):
    """
    Decorator to ensure argument types at runtime.
    
    Args:
        **type_hints: Mapping of argument names to expected types
    
    Example:
        @ensure_types(name=str, age=int)
        def create_person(name, age):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for arg_name, expected_type in type_hints.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Argument '{arg_name}' must be of type {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def auto_str(cls):
    """
    Class decorator to automatically generate __str__ method.
    
    Example:
        @auto_str
        class Person:
            def __init__(self, name, age):
                self.name = name
                self.age = age
    """
    def __str__(self):
        attrs = []
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                attrs.append(f"{key}={value}")
        return f"{cls.__name__}({', '.join(attrs)})"
    
    cls.__str__ = __str__
    return cls


def property_cached(func: Callable) -> property:
    """
    Decorator to create a cached property that computes value once.
    
    Example:
        class MyClass:
            @property_cached
            def expensive_computation(self):
                # Expensive operation here
                return result
    """
    attr_name = f"_{func.__name__}_cached"
    
    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return wrapper