"""
Code profiling and performance analysis utilities
"""
import cProfile
import pstats
import io
import time
import functools
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
from collections import defaultdict, deque
import structlog

from .exceptions import ValidationError

logger = structlog.get_logger(__name__)


class FunctionProfiler:
    """Profile function execution time and call frequency"""
    
    def __init__(self, enabled: bool = True):
        """Initialize function profiler
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.call_stats = defaultdict(lambda: {
            'total_calls': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'recent_times': deque(maxlen=100)
        })
        self._lock = threading.Lock()
    
    def profile(self, func: Optional[Callable] = None, *, name: Optional[str] = None):
        """Decorator to profile function execution
        
        Args:
            func: Function to profile (when used as @profile)
            name: Optional custom name for the profile
            
        Returns:
            Decorated function or decorator
            
        Example:
            @profiler.profile
            def my_function():
                pass
                
            @profiler.profile(name="custom_name")
            def another_function():
                pass
        """
        def decorator(f):
            profile_name = name or f.__name__
            
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return f(*args, **kwargs)
                
                start_time = time.time()
                try:
                    result = f(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self._record_execution(profile_name, execution_time, success=True)
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self._record_execution(profile_name, execution_time, success=False)
                    raise
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def _record_execution(self, name: str, execution_time: float, success: bool):
        """Record function execution statistics"""
        with self._lock:
            stats = self.call_stats[name]
            stats['total_calls'] += 1
            
            if success:
                stats['total_time'] += execution_time
                stats['min_time'] = min(stats['min_time'], execution_time)
                stats['max_time'] = max(stats['max_time'], execution_time)
                stats['recent_times'].append(execution_time)
    
    def get_stats(self, name: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Get profiling statistics
        
        Args:
            name: Specific function name, or None for all functions
            
        Returns:
            Statistics dictionary or dict of all stats
        """
        with self._lock:
            if name:
                if name not in self.call_stats:
                    return {}
                
                stats = self.call_stats[name].copy()
                recent_times = list(stats['recent_times'])
                
                if recent_times:
                    stats['avg_time'] = sum(recent_times) / len(recent_times)
                    stats['recent_min'] = min(recent_times)
                    stats['recent_max'] = max(recent_times)
                else:
                    stats['avg_time'] = 0.0
                    stats['recent_min'] = 0.0
                    stats['recent_max'] = 0.0
                
                del stats['recent_times']  # Don't include deque in output
                return stats
            
            # Return all stats
            all_stats = {}
            for func_name, stats in self.call_stats.items():
                all_stats[func_name] = self.get_stats(func_name)
            
            return all_stats
    
    def reset_stats(self, name: Optional[str] = None):
        """Reset profiling statistics
        
        Args:
            name: Specific function name, or None to reset all
        """
        with self._lock:
            if name:
                if name in self.call_stats:
                    del self.call_stats[name]
            else:
                self.call_stats.clear()
        
        logger.info("Profiling stats reset", function_name=name or "all")
    
    def enable(self):
        """Enable profiling"""
        self.enabled = True
        logger.info("Function profiling enabled")
    
    def disable(self):
        """Disable profiling"""
        self.enabled = False
        logger.info("Function profiling disabled")


class CodeProfiler:
    """Wrapper for cProfile with enhanced analysis capabilities"""
    
    def __init__(self):
        """Initialize code profiler"""
        self.profiler = None
        self.is_running = False
        self.results = None
    
    def start(self):
        """Start code profiling
        
        Raises:
            RuntimeError: If profiler is already running
        """
        if self.is_running:
            raise RuntimeError("Profiler is already running")
        
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.is_running = True
        
        logger.info("Code profiling started")
    
    def stop(self) -> pstats.Stats:
        """Stop code profiling and return results
        
        Returns:
            pstats.Stats object with profiling results
            
        Raises:
            RuntimeError: If profiler is not running
        """
        if not self.is_running:
            raise RuntimeError("Profiler is not running")
        
        self.profiler.disable()
        self.is_running = False
        
        # Create stats object
        stats_stream = io.StringIO()
        self.results = pstats.Stats(self.profiler, stream=stats_stream)
        
        logger.info("Code profiling stopped")
        return self.results
    
    def profile_block(self, func: Callable, *args, **kwargs) -> tuple[Any, pstats.Stats]:
        """Profile a specific code block
        
        Args:
            func: Function to profile
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Tuple of (function_result, profiling_stats)
        """
        if not callable(func):
            raise ValidationError("func must be callable")
        
        self.start()
        try:
            result = func(*args, **kwargs)
        finally:
            stats = self.stop()
        
        return result, stats
    
    def get_top_functions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N functions by cumulative time
        
        Args:
            n: Number of top functions to return
            
        Returns:
            List of function statistics
            
        Raises:
            RuntimeError: If no profiling results available
        """
        if self.results is None:
            raise RuntimeError("No profiling results available")
        
        # Create a copy to avoid modifying the original
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        # Extract top functions
        top_functions = []
        for i, (func_info, func_stats) in enumerate(stats.stats.items()):
            if i >= n:
                break
            
            filename, line_num, func_name = func_info
            cc, nc, tt, ct, callers = func_stats
            
            top_functions.append({
                'function': func_name,
                'filename': filename,
                'line_number': line_num,
                'total_calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / nc if nc > 0 else 0,
                'cumulative_per_call': ct / nc if nc > 0 else 0
            })
        
        return top_functions
    
    def save_stats(self, filename: str):
        """Save profiling stats to file
        
        Args:
            filename: File path to save stats
            
        Raises:
            RuntimeError: If no profiling results available
        """
        if self.results is None:
            raise RuntimeError("No profiling results available")
        
        self.results.dump_stats(filename)
        logger.info("Profiling stats saved", filename=filename)
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of profiling statistics
        
        Returns:
            Summary dictionary
            
        Raises:
            RuntimeError: If no profiling results available
        """
        if self.results is None:
            raise RuntimeError("No profiling results available")
        
        # Calculate summary statistics
        total_calls = self.results.total_calls
        total_time = self.results.total_tt
        
        # Get function count
        function_count = len(self.results.stats)
        
        # Get top function
        top_functions = self.get_top_functions(1)
        top_function = top_functions[0] if top_functions else None
        
        return {
            'total_calls': total_calls,
            'total_time_seconds': total_time,
            'function_count': function_count,
            'average_time_per_call': total_time / total_calls if total_calls > 0 else 0,
            'top_function': top_function
        }


class MemoryProfiler:
    """Simple memory usage profiler"""
    
    def __init__(self, sample_interval: float = 0.1):
        """Initialize memory profiler
        
        Args:
            sample_interval: Interval between memory samples in seconds
        """
        if sample_interval <= 0:
            raise ValidationError("sample_interval must be positive")
            
        self.sample_interval = sample_interval
        self.is_profiling = False
        self.memory_samples = []
        self._profiling_thread = None
    
    def start(self):
        """Start memory profiling
        
        Raises:
            RuntimeError: If profiling is already active
        """
        if self.is_profiling:
            raise RuntimeError("Memory profiling is already active")
        
        self.is_profiling = True
        self.memory_samples.clear()
        
        # Start background sampling thread
        self._profiling_thread = threading.Thread(target=self._sample_memory, daemon=True)
        self._profiling_thread.start()
        
        logger.info("Memory profiling started", sample_interval=self.sample_interval)
    
    def stop(self) -> List[Dict[str, Any]]:
        """Stop memory profiling and return samples
        
        Returns:
            List of memory samples
            
        Raises:
            RuntimeError: If profiling is not active
        """
        if not self.is_profiling:
            raise RuntimeError("Memory profiling is not active")
        
        self.is_profiling = False
        
        # Wait for sampling thread to finish
        if self._profiling_thread:
            self._profiling_thread.join(timeout=1.0)
        
        logger.info("Memory profiling stopped", sample_count=len(self.memory_samples))
        return self.memory_samples.copy()
    
    def _sample_memory(self):
        """Background thread function to sample memory usage"""
        from .utils import calculate_memory_usage
        
        while self.is_profiling:
            try:
                memory_info = calculate_memory_usage()
                sample = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'rss_mb': memory_info.get('rss', 0) / (1024 * 1024),
                    'vms_mb': memory_info.get('vms', 0) / (1024 * 1024),
                    'percent': memory_info.get('percent', 0)
                }
                self.memory_samples.append(sample)
            except Exception as e:
                logger.warning("Failed to sample memory", error=str(e))
            
            time.sleep(self.sample_interval)
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage from samples
        
        Returns:
            Dictionary with peak memory statistics
        """
        if not self.memory_samples:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
        
        peak_rss = max(sample['rss_mb'] for sample in self.memory_samples)
        peak_vms = max(sample['vms_mb'] for sample in self.memory_samples)
        peak_percent = max(sample['percent'] for sample in self.memory_samples)
        
        return {
            'rss_mb': peak_rss,
            'vms_mb': peak_vms,
            'percent': peak_percent
        }
    
    def get_memory_trend(self) -> Dict[str, Any]:
        """Get memory usage trend analysis
        
        Returns:
            Trend analysis dictionary
        """
        if len(self.memory_samples) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate trend from first and last samples
        first_sample = self.memory_samples[0]
        last_sample = self.memory_samples[-1]
        
        rss_change = last_sample['rss_mb'] - first_sample['rss_mb']
        change_percent = (rss_change / first_sample['rss_mb'] * 100) if first_sample['rss_mb'] > 0 else 0
        
        # Determine trend direction
        if abs(change_percent) < 5:
            trend = 'stable'
        elif change_percent > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'change_mb': rss_change,
            'change_percent': change_percent,
            'sample_count': len(self.memory_samples),
            'duration_seconds': len(self.memory_samples) * self.sample_interval
        }


# Global profiler instances for convenient use
function_profiler = FunctionProfiler()
code_profiler = CodeProfiler()


def profile_function(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """Convenience decorator for function profiling
    
    Args:
        func: Function to profile
        name: Optional custom name
        
    Returns:
        Decorated function or decorator
        
    Example:
        @profile_function
        def my_function():
            pass
    """
    return function_profiler.profile(func, name=name)


def profile_code_block(func: Callable, *args, **kwargs) -> tuple[Any, Dict[str, Any]]:
    """Profile a code block and return simplified results
    
    Args:
        func: Function to profile
        *args, **kwargs: Arguments to pass to function
        
    Returns:
        Tuple of (function_result, summary_stats)
    """
    profiler = CodeProfiler()
    result, stats = profiler.profile_block(func, *args, **kwargs)
    summary = profiler.get_stats_summary()
    
    return result, summary


def memory_profile(func: Callable, sample_interval: float = 0.1) -> tuple[Any, Dict[str, Any]]:
    """Profile memory usage of a function
    
    Args:
        func: Function to profile
        sample_interval: Memory sampling interval in seconds
        
    Returns:
        Tuple of (function_result, memory_stats)
    """
    profiler = MemoryProfiler(sample_interval)
    
    profiler.start()
    try:
        result = func()
    finally:
        samples = profiler.stop()
    
    peak_memory = profiler.get_peak_memory()
    trend = profiler.get_memory_trend()
    
    memory_stats = {
        'peak_memory': peak_memory,
        'trend': trend,
        'sample_count': len(samples)
    }
    
    return result, memory_stats