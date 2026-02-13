"""
Performance monitoring and profiling utilities for Agent Orchestra.

This module provides real-time performance tracking, resource monitoring,
and profiling tools for identifying bottlenecks and optimizing performance.
"""
import time
import threading
import psutil
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a single performance metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary representation."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    open_files: int = 0
    thread_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary representation."""
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'open_files': self.open_files,
            'thread_count': self.thread_count,
            'timestamp': self.timestamp.isoformat()
        }


class OperationTimer:
    """Timer for measuring operation performance."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        
    def start(self):
        """Start timing the operation."""
        self.start_time = time.perf_counter()
        
    def stop(self):
        """Stop timing the operation."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        
    @contextmanager
    def measure(self):
        """Context manager for measuring operation time."""
        self.start()
        try:
            yield self
        finally:
            self.stop()
    
    def get_duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        return self.duration * 1000 if self.duration is not None else None


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        with self.lock:
            self.metrics.append(metric)
    
    def record_duration(self, operation: str, duration_seconds: float, tags: Optional[Dict[str, str]] = None):
        """Record operation duration."""
        metric = PerformanceMetric(
            name=f"{operation}_duration",
            value=duration_seconds,
            unit="seconds",
            tags=tags or {}
        )
        
        with self.lock:
            self.metrics.append(metric)
            self.operation_stats[operation].append(duration_seconds)
            
            # Keep only recent measurements for stats
            if len(self.operation_stats[operation]) > 1000:
                self.operation_stats[operation] = self.operation_stats[operation][-500:]
    
    def get_operation_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get statistics for an operation."""
        with self.lock:
            durations = self.operation_stats.get(operation, [])
            
        if not durations:
            return None
        
        durations_sorted = sorted(durations)
        count = len(durations_sorted)
        
        return {
            'count': count,
            'min': durations_sorted[0],
            'max': durations_sorted[-1],
            'mean': sum(durations_sorted) / count,
            'median': durations_sorted[count // 2],
            'p95': durations_sorted[int(count * 0.95)],
            'p99': durations_sorted[int(count * 0.99)]
        }
    
    def get_recent_metrics(self, minutes: int = 5) -> List[PerformanceMetric]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self.lock:
            return [
                metric for metric in self.metrics
                if metric.timestamp >= cutoff_time
            ]
    
    def clear_metrics(self):
        """Clear all stored metrics."""
        with self.lock:
            self.metrics.clear()
            self.operation_stats.clear()


class SystemMonitor:
    """Monitors system resource usage."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.snapshots: deque = deque(maxlen=3600)  # Keep 1 hour at 1s intervals
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_network_stats = None
        
    def start_monitoring(self):
        """Start background monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
            except Exception as e:
                logger.error(f"Error taking system snapshot: {e}")
            
            time.sleep(self.sampling_interval)
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current system resources."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024 ** 3)
        
        # Network stats (delta since last measurement)
        network_bytes_sent = 0
        network_bytes_recv = 0
        try:
            net_stats = psutil.net_io_counters()
            if self._last_network_stats:
                network_bytes_sent = net_stats.bytes_sent - self._last_network_stats.bytes_sent
                network_bytes_recv = net_stats.bytes_recv - self._last_network_stats.bytes_recv
            self._last_network_stats = net_stats
        except Exception:
            pass
        
        # Process-specific stats
        try:
            process = psutil.Process()
            open_files = len(process.open_files())
            thread_count = process.num_threads()
        except Exception:
            open_files = 0
            thread_count = 0
        
        return ResourceSnapshot(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            open_files=open_files,
            thread_count=thread_count
        )
    
    def get_current_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        return self._take_snapshot()
    
    def get_average_usage(self, minutes: int = 5) -> Optional[Dict[str, float]]:
        """Get average resource usage over the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_snapshots = [
            snapshot for snapshot in self.snapshots
            if snapshot.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return None
        
        count = len(recent_snapshots)
        return {
            'cpu_percent': sum(s.cpu_percent for s in recent_snapshots) / count,
            'memory_percent': sum(s.memory_percent for s in recent_snapshots) / count,
            'disk_usage_percent': sum(s.disk_usage_percent for s in recent_snapshots) / count,
            'thread_count': sum(s.thread_count for s in recent_snapshots) / count,
            'sample_count': count
        }


class PerformanceProfiler:
    """Profiles function and method performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_timers: Dict[str, OperationTimer] = {}
        
    def profile_function(self, func: Callable, operation_name: Optional[str] = None) -> Callable:
        """Decorator to profile function execution time."""
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            timer = OperationTimer(operation_name)
            with timer.measure():
                result = func(*args, **kwargs)
            
            if timer.duration is not None:
                self.metrics_collector.record_duration(
                    operation_name,
                    timer.duration,
                    tags={'function': func.__name__}
                )
                
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    @contextmanager
    def profile_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for profiling operations."""
        timer = OperationTimer(operation_name)
        with timer.measure():
            yield timer
        
        if timer.duration is not None:
            self.metrics_collector.record_duration(operation_name, timer.duration, tags)


class AlertManager:
    """Manages performance alerts and thresholds."""
    
    def __init__(self):
        self.thresholds: Dict[str, float] = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'operation_duration': 10.0  # seconds
        }
        self.alert_callbacks: List[Callable] = []
        self.active_alerts: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(minutes=5)
    
    def set_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold for a metric."""
        self.thresholds[metric_name] = threshold
    
    def add_alert_callback(self, callback: Callable[[str, float, float], None]):
        """Add callback for alert notifications.
        
        Args:
            callback: Function that takes (metric_name, value, threshold)
        """
        self.alert_callbacks.append(callback)
    
    def check_snapshot(self, snapshot: ResourceSnapshot):
        """Check resource snapshot against thresholds."""
        metrics_to_check = [
            ('cpu_percent', snapshot.cpu_percent),
            ('memory_percent', snapshot.memory_percent),
            ('disk_usage_percent', snapshot.disk_usage_percent)
        ]
        
        for metric_name, value in metrics_to_check:
            threshold = self.thresholds.get(metric_name)
            if threshold and value > threshold:
                self._trigger_alert(metric_name, value, threshold)
    
    def check_operation_duration(self, operation_name: str, duration: float):
        """Check operation duration against threshold."""
        threshold = self.thresholds.get('operation_duration')
        if threshold and duration > threshold:
            self._trigger_alert(f"slow_operation_{operation_name}", duration, threshold)
    
    def _trigger_alert(self, alert_key: str, value: float, threshold: float):
        """Trigger an alert if not in cooldown."""
        now = datetime.utcnow()
        last_alert = self.active_alerts.get(alert_key)
        
        # Check cooldown
        if last_alert and (now - last_alert) < self.alert_cooldown:
            return
        
        # Record alert
        self.active_alerts[alert_key] = now
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_key, value, threshold)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")


class PerformanceManager:
    """Central manager for all performance monitoring."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor()
        self.profiler = PerformanceProfiler(self.metrics_collector)
        self.alert_manager = AlertManager()
        
        # Set up alert callback
        self.alert_manager.add_alert_callback(self._log_alert)
        
        # Monitor system snapshots for alerts
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Set up monitoring with alerts."""
        # This would ideally be a periodic task
        # For now, we'll check on demand
        pass
    
    def _log_alert(self, alert_key: str, value: float, threshold: float):
        """Default alert callback that logs alerts."""
        logger.warning(
            f"Performance alert: {alert_key} = {value:.2f} (threshold: {threshold:.2f})"
        )
    
    def start_monitoring(self):
        """Start all monitoring components."""
        self.system_monitor.start_monitoring()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.system_monitor.stop_monitoring()
        logger.info("Performance monitoring stopped")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        # Current system snapshot
        current_snapshot = self.system_monitor.get_current_snapshot()
        
        # Average usage
        avg_usage = self.system_monitor.get_average_usage(minutes=5)
        
        # Operation statistics
        operation_stats = {}
        for operation in self.metrics_collector.operation_stats.keys():
            stats = self.metrics_collector.get_operation_stats(operation)
            if stats:
                operation_stats[operation] = stats
        
        # Recent metrics
        recent_metrics = self.metrics_collector.get_recent_metrics(minutes=5)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'current_resources': current_snapshot.to_dict(),
            'average_usage_5min': avg_usage,
            'operation_statistics': operation_stats,
            'recent_metrics_count': len(recent_metrics),
            'active_alerts': list(self.alert_manager.active_alerts.keys())
        }
    
    def profile_function(self, func: Callable, operation_name: Optional[str] = None) -> Callable:
        """Convenience method to profile a function."""
        return self.profiler.profile_function(func, operation_name)
    
    def profile_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Convenience method to profile an operation."""
        return self.profiler.profile_operation(operation_name, tags)


# Global performance manager instance
performance_manager = PerformanceManager()


def start_performance_monitoring():
    """Start global performance monitoring."""
    performance_manager.start_monitoring()


def stop_performance_monitoring():
    """Stop global performance monitoring."""
    performance_manager.stop_monitoring()


def get_performance_report() -> Dict[str, Any]:
    """Get global performance report."""
    return performance_manager.get_performance_report()


def profile(operation_name: Optional[str] = None):
    """Decorator for profiling functions."""
    def decorator(func: Callable) -> Callable:
        return performance_manager.profile_function(func, operation_name)
    return decorator


@contextmanager
def measure_operation(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager for measuring operations."""
    with performance_manager.profile_operation(operation_name, tags) as timer:
        yield timer