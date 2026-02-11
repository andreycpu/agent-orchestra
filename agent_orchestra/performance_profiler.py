"""
Performance profiling utilities for Agent Orchestra
"""
import time
import asyncio
import cProfile
import pstats
import io
import functools
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """Result of a performance profiling session"""
    function_name: str
    execution_time: float
    call_count: int
    avg_time_per_call: float
    stats_summary: str
    hotspots: List[Dict[str, Any]]


class PerformanceTimer:
    """Context manager for timing code execution"""
    
    def __init__(self, name: str, threshold: Optional[float] = None):
        self.name = name
        self.threshold = threshold  # Log warning if execution exceeds this
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.execution_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.execution_time = self.end_time - self.start_time
        
        # Log performance
        log_level = "debug"
        if self.threshold and self.execution_time > self.threshold:
            log_level = "warning"
        
        logger.log(
            log_level,
            "Performance timing",
            operation=self.name,
            execution_time=self.execution_time,
            threshold_exceeded=bool(self.threshold and self.execution_time > self.threshold)
        )


class AsyncPerformanceTimer:
    """Async context manager for timing async code execution"""
    
    def __init__(self, name: str, threshold: Optional[float] = None):
        self.name = name
        self.threshold = threshold
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.execution_time: Optional[float] = None
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.execution_time = self.end_time - self.start_time
        
        # Log performance
        log_level = "debug"
        if self.threshold and self.execution_time > self.threshold:
            log_level = "warning"
        
        logger.log(
            log_level,
            "Async performance timing",
            operation=self.name,
            execution_time=self.execution_time,
            threshold_exceeded=bool(self.threshold and self.execution_time > self.threshold)
        )


class PerformanceProfiler:
    """Comprehensive performance profiler with multiple collection strategies"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._profiling_sessions: List[ProfileResult] = []
        self._start_time = time.time()
    
    def record_metric(self, name: str, value: float, unit: str = "seconds", tags: Optional[Dict[str, str]] = None):
        """Record a performance metric"""
        if not self.enabled:
            return
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        self._metrics[name].append(metric)
    
    def time_function(self, name: Optional[str] = None, threshold: Optional[float] = None):
        """Decorator to time function execution"""
        def decorator(func):
            func_name = name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with AsyncPerformanceTimer(func_name, threshold) as timer:
                        result = await func(*args, **kwargs)
                    
                    if self.enabled:
                        self._timers[func_name].append(timer.execution_time)
                        self._call_counts[func_name] += 1
                        self.record_metric(f"{func_name}_execution_time", timer.execution_time)
                    
                    return result
                
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with PerformanceTimer(func_name, threshold) as timer:
                        result = func(*args, **kwargs)
                    
                    if self.enabled:
                        self._timers[func_name].append(timer.execution_time)
                        self._call_counts[func_name] += 1
                        self.record_metric(f"{func_name}_execution_time", timer.execution_time)
                    
                    return result
                
                return sync_wrapper
        
        return decorator
    
    @contextmanager
    def profile_code(self, name: str):
        """Context manager for profiling code with cProfile"""
        if not self.enabled:
            yield
            return
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            profiler.disable()
            
            # Analyze results
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            # Extract hotspots
            hotspots = []
            for func_info, (call_count, total_time, cumulative_time, per_call_time, per_call_cumulative) in stats.stats.items():
                filename, line_number, function_name = func_info
                hotspots.append({
                    'function': function_name,
                    'filename': filename,
                    'line_number': line_number,
                    'call_count': call_count,
                    'total_time': total_time,
                    'cumulative_time': cumulative_time,
                    'per_call_time': per_call_time
                })
            
            # Sort by cumulative time
            hotspots.sort(key=lambda x: x['cumulative_time'], reverse=True)
            
            profile_result = ProfileResult(
                function_name=name,
                execution_time=end_time - start_time,
                call_count=sum(info[0] for info in stats.stats.values()),
                avg_time_per_call=(end_time - start_time) / max(1, sum(info[0] for info in stats.stats.values())),
                stats_summary=stats_stream.getvalue(),
                hotspots=hotspots[:10]  # Top 10 hotspots
            )
            
            self._profiling_sessions.append(profile_result)
            
            logger.info(
                "Code profiling completed",
                name=name,
                execution_time=profile_result.execution_time,
                total_calls=profile_result.call_count,
                top_hotspot=hotspots[0]['function'] if hotspots else None
            )
    
    def get_function_stats(self, name: str) -> Dict[str, Any]:
        """Get performance statistics for a specific function"""
        if name not in self._timers:
            return {}
        
        timings = self._timers[name]
        if not timings:
            return {}
        
        return {
            'name': name,
            'call_count': self._call_counts[name],
            'total_time': sum(timings),
            'average_time': sum(timings) / len(timings),
            'min_time': min(timings),
            'max_time': max(timings),
            'last_10_avg': sum(timings[-10:]) / min(10, len(timings))
        }
    
    def get_all_function_stats(self) -> List[Dict[str, Any]]:
        """Get performance statistics for all tracked functions"""
        return [self.get_function_stats(name) for name in self._timers.keys()]
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""
        total_uptime = time.time() - self._start_time
        total_functions = len(self._timers)
        total_calls = sum(self._call_counts.values())
        
        return {
            'uptime_seconds': total_uptime,
            'total_functions_tracked': total_functions,
            'total_function_calls': total_calls,
            'calls_per_second': total_calls / total_uptime if total_uptime > 0 else 0,
            'profiling_sessions': len(self._profiling_sessions),
            'metrics_collected': sum(len(metrics) for metrics in self._metrics.values())
        }
    
    def get_slowest_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest functions by average execution time"""
        stats = self.get_all_function_stats()
        return sorted(stats, key=lambda x: x.get('average_time', 0), reverse=True)[:limit]
    
    def get_most_called_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most frequently called functions"""
        stats = self.get_all_function_stats()
        return sorted(stats, key=lambda x: x.get('call_count', 0), reverse=True)[:limit]
    
    def reset_stats(self):
        """Reset all collected statistics"""
        self._metrics.clear()
        self._timers.clear()
        self._call_counts.clear()
        self._profiling_sessions.clear()
        self._start_time = time.time()
        
        logger.info("Performance statistics reset")
    
    def export_stats(self) -> Dict[str, Any]:
        """Export all statistics for external analysis"""
        return {
            'system_metrics': self.get_system_metrics(),
            'function_stats': self.get_all_function_stats(),
            'profiling_sessions': [
                {
                    'function_name': session.function_name,
                    'execution_time': session.execution_time,
                    'call_count': session.call_count,
                    'avg_time_per_call': session.avg_time_per_call,
                    'top_hotspots': session.hotspots[:5]
                }
                for session in self._profiling_sessions
            ],
            'slowest_functions': self.get_slowest_functions(5),
            'most_called_functions': self.get_most_called_functions(5)
        }


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance"""
    return _global_profiler


def profile(name: Optional[str] = None, threshold: Optional[float] = None):
    """Convenience decorator for profiling functions"""
    return _global_profiler.time_function(name, threshold)


@contextmanager
def profile_block(name: str):
    """Convenience context manager for profiling code blocks"""
    with _global_profiler.profile_code(name):
        yield


def enable_profiling():
    """Enable performance profiling"""
    _global_profiler.enabled = True
    logger.info("Performance profiling enabled")


def disable_profiling():
    """Disable performance profiling"""
    _global_profiler.enabled = False
    logger.info("Performance profiling disabled")


def get_performance_report() -> Dict[str, Any]:
    """Get a comprehensive performance report"""
    return _global_profiler.export_stats()