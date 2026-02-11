"""
Performance profiler for Agent Orchestra
"""
import time
import asyncio
import functools
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ProfileEntry:
    """Single profiling entry"""
    name: str
    start_time: float
    end_time: float
    duration: float
    thread_id: int
    context: Dict[str, Any]


@dataclass
class ProfileSummary:
    """Summary of profiling data for a function/operation"""
    name: str
    call_count: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    percentile_95: float
    percentile_99: float


class AsyncProfiler:
    """Asynchronous performance profiler"""
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._entries: deque = deque(maxlen=max_entries)
        self._active_profiles: Dict[int, ProfileEntry] = {}
        self._lock = threading.Lock()
        self._enabled = True
    
    def enable(self):
        """Enable profiling"""
        self._enabled = True
        
    def disable(self):
        """Disable profiling"""
        self._enabled = False
    
    def start_profile(self, name: str, context: Dict[str, Any] = None) -> int:
        """Start profiling an operation"""
        if not self._enabled:
            return -1
        
        profile_id = id(threading.current_thread()) + int(time.time() * 1000000)
        thread_id = threading.get_ident()
        
        entry = ProfileEntry(
            name=name,
            start_time=time.perf_counter(),
            end_time=0,
            duration=0,
            thread_id=thread_id,
            context=context or {}
        )
        
        with self._lock:
            self._active_profiles[profile_id] = entry
        
        return profile_id
    
    def end_profile(self, profile_id: int):
        """End profiling an operation"""
        if not self._enabled or profile_id == -1:
            return
        
        end_time = time.perf_counter()
        
        with self._lock:
            if profile_id in self._active_profiles:
                entry = self._active_profiles[profile_id]
                entry.end_time = end_time
                entry.duration = end_time - entry.start_time
                
                self._entries.append(entry)
                del self._active_profiles[profile_id]
    
    def profile_function(self, name: Optional[str] = None, context: Dict[str, Any] = None):
        """Decorator to profile a function"""
        def decorator(func):
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    profile_id = self.start_profile(profile_name, context)
                    try:
                        return await func(*args, **kwargs)
                    finally:
                        self.end_profile(profile_id)
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    profile_id = self.start_profile(profile_name, context)
                    try:
                        return func(*args, **kwargs)
                    finally:
                        self.end_profile(profile_id)
                return sync_wrapper
        
        return decorator
    
    def profile_context(self, name: str, context: Dict[str, Any] = None):
        """Context manager for profiling"""
        return ProfileContext(self, name, context)
    
    def get_summary(self, name_filter: Optional[str] = None) -> List[ProfileSummary]:
        """Get profiling summary"""
        with self._lock:
            entries = list(self._entries)
        
        # Group by name
        groups = defaultdict(list)
        for entry in entries:
            if name_filter is None or name_filter in entry.name:
                groups[entry.name].append(entry.duration)
        
        # Create summaries
        summaries = []
        for name, durations in groups.items():
            if not durations:
                continue
            
            durations.sort()
            count = len(durations)
            
            summary = ProfileSummary(
                name=name,
                call_count=count,
                total_time=sum(durations),
                average_time=sum(durations) / count,
                min_time=durations[0],
                max_time=durations[-1],
                percentile_95=durations[int(count * 0.95)] if count > 1 else durations[0],
                percentile_99=durations[int(count * 0.99)] if count > 1 else durations[0]
            )
            summaries.append(summary)
        
        return sorted(summaries, key=lambda x: x.total_time, reverse=True)
    
    def get_recent_entries(self, count: int = 100) -> List[ProfileEntry]:
        """Get recent profile entries"""
        with self._lock:
            return list(self._entries)[-count:]
    
    def clear(self):
        """Clear all profiling data"""
        with self._lock:
            self._entries.clear()
            self._active_profiles.clear()
    
    def export_data(self) -> Dict[str, Any]:
        """Export profiling data"""
        with self._lock:
            return {
                "entries": [asdict(entry) for entry in self._entries],
                "active_profiles": len(self._active_profiles),
                "total_entries": len(self._entries),
                "enabled": self._enabled
            }


class ProfileContext:
    """Context manager for profiling"""
    
    def __init__(self, profiler: AsyncProfiler, name: str, context: Dict[str, Any] = None):
        self.profiler = profiler
        self.name = name
        self.context = context
        self.profile_id = None
    
    def __enter__(self):
        self.profile_id = self.profiler.start_profile(self.name, self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profile_id is not None:
            self.profiler.end_profile(self.profile_id)


class MemoryProfiler:
    """Memory usage profiler"""
    
    def __init__(self):
        self._snapshots: List[Dict[str, Any]] = []
        self._enabled = False
    
    def enable(self):
        """Enable memory profiling"""
        try:
            import psutil
            self._enabled = True
            logger.info("Memory profiler enabled")
        except ImportError:
            logger.warning("psutil not available, memory profiling disabled")
    
    def disable(self):
        """Disable memory profiling"""
        self._enabled = False
    
    def take_snapshot(self, name: str, context: Dict[str, Any] = None):
        """Take memory usage snapshot"""
        if not self._enabled:
            return
        
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot = {
                "name": name,
                "timestamp": time.time(),
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent(),
                "context": context or {}
            }
            
            self._snapshots.append(snapshot)
            
            # Keep only last 1000 snapshots
            if len(self._snapshots) > 1000:
                self._snapshots.pop(0)
        
        except Exception as e:
            logger.warning("Failed to take memory snapshot", error=str(e))
    
    def get_snapshots(self, name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get memory snapshots"""
        if name_filter:
            return [s for s in self._snapshots if name_filter in s["name"]]
        return list(self._snapshots)
    
    def get_memory_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """Get memory usage trend"""
        if len(self._snapshots) < 2:
            return {}
        
        recent_snapshots = self._snapshots[-window_size:]
        
        rss_values = [s["rss"] for s in recent_snapshots]
        vms_values = [s["vms"] for s in recent_snapshots]
        percent_values = [s["percent"] for s in recent_snapshots]
        
        return {
            "rss_trend": {
                "current": rss_values[-1],
                "average": sum(rss_values) / len(rss_values),
                "min": min(rss_values),
                "max": max(rss_values),
                "growth": (rss_values[-1] - rss_values[0]) / rss_values[0] if rss_values[0] > 0 else 0
            },
            "vms_trend": {
                "current": vms_values[-1],
                "average": sum(vms_values) / len(vms_values),
                "min": min(vms_values),
                "max": max(vms_values),
                "growth": (vms_values[-1] - vms_values[0]) / vms_values[0] if vms_values[0] > 0 else 0
            },
            "percent_trend": {
                "current": percent_values[-1],
                "average": sum(percent_values) / len(percent_values),
                "min": min(percent_values),
                "max": max(percent_values)
            }
        }


class OrchestrationProfiler:
    """Main profiler for Agent Orchestra"""
    
    def __init__(self):
        self.async_profiler = AsyncProfiler()
        self.memory_profiler = MemoryProfiler()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_interval = 30  # seconds
    
    def enable_all(self):
        """Enable all profilers"""
        self.async_profiler.enable()
        self.memory_profiler.enable()
    
    def disable_all(self):
        """Disable all profilers"""
        self.async_profiler.disable()
        self.memory_profiler.disable()
    
    async def start_monitoring(self):
        """Start background monitoring"""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Profiler monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Profiler monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        try:
            while True:
                # Take periodic memory snapshots
                self.memory_profiler.take_snapshot("periodic_check")
                
                # Log memory trend if growing significantly
                trend = self.memory_profiler.get_memory_trend()
                if trend and "rss_trend" in trend:
                    growth = trend["rss_trend"]["growth"]
                    if growth > 0.1:  # 10% growth
                        logger.warning("Memory usage growing",
                                     growth_rate=f"{growth:.2%}",
                                     current_mb=trend["rss_trend"]["current"] / 1024 / 1024)
                
                await asyncio.sleep(self._monitoring_interval)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Profiler monitoring failed", error=str(e))
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        async_summary = self.async_profiler.get_summary()
        memory_trend = self.memory_profiler.get_memory_trend()
        
        # Top slow operations
        slow_operations = [
            {
                "name": s.name,
                "avg_time": s.average_time,
                "total_time": s.total_time,
                "call_count": s.call_count
            }
            for s in async_summary[:10]
        ]
        
        return {
            "timestamp": time.time(),
            "async_profiling": {
                "total_operations": len(async_summary),
                "slow_operations": slow_operations
            },
            "memory_profiling": memory_trend,
            "recommendations": self._generate_recommendations(async_summary, memory_trend)
        }
    
    def _generate_recommendations(self, async_summary: List[ProfileSummary], memory_trend: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Check for slow operations
        if async_summary:
            slowest = async_summary[0]
            if slowest.average_time > 1.0:  # > 1 second average
                recommendations.append(
                    f"Operation '{slowest.name}' is slow (avg: {slowest.average_time:.2f}s). "
                    f"Consider optimization or asynchronous processing."
                )
        
        # Check for frequently called operations
        frequent_ops = [s for s in async_summary if s.call_count > 1000]
        if frequent_ops:
            recommendations.append(
                f"High-frequency operations detected: {[s.name for s in frequent_ops[:3]]}. "
                f"Consider caching or batching."
            )
        
        # Check memory growth
        if memory_trend and "rss_trend" in memory_trend:
            growth = memory_trend["rss_trend"]["growth"]
            if growth > 0.2:  # 20% growth
                recommendations.append(
                    f"Memory usage growing rapidly ({growth:.1%}). "
                    f"Check for memory leaks or consider garbage collection tuning."
                )
        
        return recommendations
    
    def profile_decorator(self, name: Optional[str] = None):
        """Convenient decorator for profiling"""
        return self.async_profiler.profile_function(name)
    
    def profile_context(self, name: str, context: Dict[str, Any] = None):
        """Convenient context manager for profiling"""
        return self.async_profiler.profile_context(name, context)
    
    def clear_all_data(self):
        """Clear all profiling data"""
        self.async_profiler.clear()
        self.memory_profiler._snapshots.clear()


# Global profiler instance
profiler = OrchestrationProfiler()


# Convenient decorators
def profile(name: Optional[str] = None):
    """Decorator to profile a function"""
    return profiler.profile_decorator(name)


def memory_snapshot(name: str, context: Dict[str, Any] = None):
    """Take a memory snapshot"""
    profiler.memory_profiler.take_snapshot(name, context)


# Context manager
profile_context = profiler.profile_context