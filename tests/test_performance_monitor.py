"""
Tests for performance monitoring utilities.
"""
import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock

from agent_orchestra.performance_monitor import (
    PerformanceMetric, ResourceSnapshot, OperationTimer,
    MetricsCollector, SystemMonitor, PerformanceProfiler,
    AlertManager, PerformanceManager, profile, measure_operation
)


class TestPerformanceMetric:
    """Test performance metric data structure."""
    
    def test_metric_creation(self):
        """Test metric creation and conversion."""
        metric = PerformanceMetric(
            name="cpu_usage",
            value=75.5,
            unit="percent",
            tags={"host": "server1", "component": "web"}
        )
        
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.unit == "percent"
        assert metric.tags["host"] == "server1"
        assert metric.timestamp is not None
        
        # Test dictionary conversion
        metric_dict = metric.to_dict()
        assert metric_dict["name"] == "cpu_usage"
        assert metric_dict["value"] == 75.5
        assert metric_dict["unit"] == "percent"
        assert "timestamp" in metric_dict
        assert metric_dict["tags"]["host"] == "server1"


class TestResourceSnapshot:
    """Test resource snapshot functionality."""
    
    def test_snapshot_creation(self):
        """Test resource snapshot creation."""
        snapshot = ResourceSnapshot(
            cpu_percent=45.2,
            memory_percent=60.8,
            memory_used_mb=2048.5,
            memory_available_mb=1500.2,
            disk_usage_percent=75.0,
            disk_free_gb=100.5,
            network_bytes_sent=1024000,
            network_bytes_recv=2048000,
            open_files=150,
            thread_count=25
        )
        
        assert snapshot.cpu_percent == 45.2
        assert snapshot.memory_percent == 60.8
        assert snapshot.disk_usage_percent == 75.0
        
        # Test dictionary conversion
        snapshot_dict = snapshot.to_dict()
        assert snapshot_dict["cpu_percent"] == 45.2
        assert snapshot_dict["memory_percent"] == 60.8
        assert "timestamp" in snapshot_dict


class TestOperationTimer:
    """Test operation timer functionality."""
    
    def test_timer_basic_operation(self):
        """Test basic timer start/stop operation."""
        timer = OperationTimer("test_operation")
        
        timer.start()
        time.sleep(0.1)
        timer.stop()
        
        assert timer.duration is not None
        assert timer.duration >= 0.1
        assert timer.get_duration_ms() >= 100
    
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        with OperationTimer("context_test").measure() as timer:
            time.sleep(0.05)
        
        assert timer.duration is not None
        assert timer.duration >= 0.05
    
    def test_timer_without_start(self):
        """Test timer stop without start raises error."""
        timer = OperationTimer("invalid_test")
        
        with pytest.raises(ValueError, match="Timer not started"):
            timer.stop()


class TestMetricsCollector:
    """Test metrics collector functionality."""
    
    def test_record_metric(self):
        """Test recording metrics."""
        collector = MetricsCollector(max_metrics=10)
        
        metric = PerformanceMetric("test_metric", 42.0, "units")
        collector.record_metric(metric)
        
        recent_metrics = collector.get_recent_metrics(minutes=1)
        assert len(recent_metrics) == 1
        assert recent_metrics[0].name == "test_metric"
        assert recent_metrics[0].value == 42.0
    
    def test_record_duration(self):
        """Test recording operation durations."""
        collector = MetricsCollector()
        
        collector.record_duration("db_query", 0.5, {"table": "users"})
        collector.record_duration("db_query", 1.5, {"table": "users"})
        collector.record_duration("db_query", 0.3, {"table": "orders"})
        
        stats = collector.get_operation_stats("db_query")
        assert stats is not None
        assert stats["count"] == 3
        assert stats["min"] == 0.3
        assert stats["max"] == 1.5
        assert stats["mean"] == 0.8  # (0.5 + 1.5 + 0.3) / 3
    
    def test_operation_stats_none_for_unknown(self):
        """Test that stats return None for unknown operations."""
        collector = MetricsCollector()
        
        stats = collector.get_operation_stats("unknown_operation")
        assert stats is None
    
    def test_metrics_limit_enforcement(self):
        """Test that metrics collector respects max limit."""
        collector = MetricsCollector(max_metrics=5)
        
        # Add more metrics than limit
        for i in range(10):
            metric = PerformanceMetric(f"metric_{i}", float(i), "count")
            collector.record_metric(metric)
        
        # Should only keep the last 5
        recent_metrics = collector.get_recent_metrics(minutes=60)
        assert len(recent_metrics) <= 5
    
    def test_clear_metrics(self):
        """Test clearing all metrics."""
        collector = MetricsCollector()
        
        collector.record_duration("test_op", 1.0)
        metric = PerformanceMetric("test_metric", 1.0, "unit")
        collector.record_metric(metric)
        
        collector.clear_metrics()
        
        assert len(collector.get_recent_metrics()) == 0
        assert collector.get_operation_stats("test_op") is None


class TestSystemMonitor:
    """Test system monitoring functionality."""
    
    @patch('agent_orchestra.performance_monitor.psutil')
    def test_take_snapshot(self, mock_psutil):
        """Test taking system resource snapshot."""
        # Setup mocks
        mock_psutil.cpu_percent.return_value = 45.5
        
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.used = 2048 * 1024 * 1024  # 2GB in bytes
        mock_memory.available = 1024 * 1024 * 1024  # 1GB in bytes
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.used = 50 * 1024**3  # 50GB in bytes
        mock_disk.total = 100 * 1024**3  # 100GB in bytes
        mock_disk.free = 50 * 1024**3  # 50GB in bytes
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_process = Mock()
        mock_process.open_files.return_value = ["file1", "file2"]  # 2 files
        mock_process.num_threads.return_value = 10
        mock_psutil.Process.return_value = mock_process
        
        monitor = SystemMonitor()
        snapshot = monitor._take_snapshot()
        
        assert snapshot.cpu_percent == 45.5
        assert snapshot.memory_percent == 60.0
        assert snapshot.memory_used_mb == 2048.0
        assert snapshot.disk_usage_percent == 50.0
        assert snapshot.open_files == 2
        assert snapshot.thread_count == 10
    
    def test_monitor_lifecycle(self):
        """Test starting and stopping monitoring."""
        monitor = SystemMonitor(sampling_interval=0.1)
        
        assert not monitor._monitoring
        
        monitor.start_monitoring()
        assert monitor._monitoring
        assert monitor._monitor_thread is not None
        
        time.sleep(0.2)  # Let it run briefly
        
        monitor.stop_monitoring()
        assert not monitor._monitoring
    
    @patch('agent_orchestra.performance_monitor.psutil')
    def test_get_average_usage(self, mock_psutil):
        """Test getting average resource usage."""
        # Setup basic mocks
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0, used=0, available=0)
        mock_psutil.disk_usage.return_value = Mock(used=0, total=1, free=0)
        mock_psutil.Process.return_value = Mock(
            open_files=lambda: [],
            num_threads=lambda: 5
        )
        
        monitor = SystemMonitor()
        
        # Add some sample snapshots
        for cpu in [40.0, 50.0, 60.0]:
            mock_psutil.cpu_percent.return_value = cpu
            snapshot = monitor._take_snapshot()
            monitor.snapshots.append(snapshot)
        
        avg_usage = monitor.get_average_usage(minutes=5)
        assert avg_usage is not None
        assert avg_usage["cpu_percent"] == 50.0  # Average of 40, 50, 60
        assert avg_usage["sample_count"] == 3


class TestPerformanceProfiler:
    """Test performance profiler functionality."""
    
    def test_profile_function_decorator(self):
        """Test function profiling decorator."""
        collector = MetricsCollector()
        profiler = PerformanceProfiler(collector)
        
        call_count = 0
        
        @profiler.profile_function
        def test_function(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.05)
            return x * 2
        
        result = test_function(5)
        
        assert result == 10
        assert call_count == 1
        
        # Check if metric was recorded
        stats = collector.get_operation_stats("test_performance_monitor.test_function")
        assert stats is not None
        assert stats["count"] == 1
        assert stats["mean"] >= 0.05
    
    def test_profile_operation_context(self):
        """Test operation profiling context manager."""
        collector = MetricsCollector()
        profiler = PerformanceProfiler(collector)
        
        with profiler.profile_operation("test_context", tags={"type": "unit_test"}):
            time.sleep(0.03)
        
        stats = collector.get_operation_stats("test_context")
        assert stats is not None
        assert stats["count"] == 1
        assert stats["mean"] >= 0.03


class TestAlertManager:
    """Test alert management functionality."""
    
    def test_threshold_configuration(self):
        """Test setting and getting thresholds."""
        alert_manager = AlertManager()
        
        alert_manager.set_threshold("cpu_percent", 90.0)
        assert alert_manager.thresholds["cpu_percent"] == 90.0
    
    def test_alert_callback_registration(self):
        """Test registering alert callbacks."""
        alert_manager = AlertManager()
        callback_called = False
        callback_args = None
        
        def test_callback(alert_key, value, threshold):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = (alert_key, value, threshold)
        
        alert_manager.add_alert_callback(test_callback)
        
        # Trigger alert by checking high CPU
        snapshot = ResourceSnapshot(
            cpu_percent=95.0,
            memory_percent=50.0,
            memory_used_mb=1000.0,
            memory_available_mb=1000.0,
            disk_usage_percent=50.0,
            disk_free_gb=100.0
        )
        
        alert_manager.check_snapshot(snapshot)
        
        assert callback_called
        assert callback_args[0] == "cpu_percent"
        assert callback_args[1] == 95.0
        assert callback_args[2] == 80.0  # Default threshold
    
    def test_alert_cooldown(self):
        """Test alert cooldown functionality."""
        alert_manager = AlertManager()
        alert_manager.alert_cooldown = timedelta(seconds=1)
        
        callback_count = 0
        
        def counting_callback(alert_key, value, threshold):
            nonlocal callback_count
            callback_count += 1
        
        alert_manager.add_alert_callback(counting_callback)
        
        # First alert should trigger
        alert_manager._trigger_alert("test_alert", 100.0, 80.0)
        assert callback_count == 1
        
        # Second immediate alert should be suppressed
        alert_manager._trigger_alert("test_alert", 100.0, 80.0)
        assert callback_count == 1
        
        # Wait for cooldown and try again
        time.sleep(1.1)
        alert_manager._trigger_alert("test_alert", 100.0, 80.0)
        assert callback_count == 2
    
    def test_operation_duration_check(self):
        """Test checking operation duration against threshold."""
        alert_manager = AlertManager()
        alert_manager.set_threshold("operation_duration", 5.0)
        
        callback_called = False
        
        def duration_callback(alert_key, value, threshold):
            nonlocal callback_called
            callback_called = True
        
        alert_manager.add_alert_callback(duration_callback)
        
        # Should not trigger alert
        alert_manager.check_operation_duration("fast_op", 2.0)
        assert not callback_called
        
        # Should trigger alert
        alert_manager.check_operation_duration("slow_op", 8.0)
        assert callback_called


class TestPerformanceManager:
    """Test performance manager integration."""
    
    def test_manager_initialization(self):
        """Test performance manager initialization."""
        manager = PerformanceManager()
        
        assert manager.metrics_collector is not None
        assert manager.system_monitor is not None
        assert manager.profiler is not None
        assert manager.alert_manager is not None
    
    @patch('agent_orchestra.performance_monitor.psutil')
    def test_get_performance_report(self, mock_psutil):
        """Test getting comprehensive performance report."""
        # Setup mocks
        mock_psutil.cpu_percent.return_value = 45.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=65.0, used=0, available=0
        )
        mock_psutil.disk_usage.return_value = Mock(used=0, total=1, free=0)
        mock_psutil.Process.return_value = Mock(
            open_files=lambda: [],
            num_threads=lambda: 8
        )
        
        manager = PerformanceManager()
        
        # Add some operation data
        manager.metrics_collector.record_duration("test_op", 1.5)
        
        report = manager.get_performance_report()
        
        assert "timestamp" in report
        assert "current_resources" in report
        assert "operation_statistics" in report
        assert report["current_resources"]["cpu_percent"] == 45.0
        assert "test_op" in report["operation_statistics"]


class TestGlobalFunctions:
    """Test global performance monitoring functions."""
    
    def test_profile_decorator(self):
        """Test global profile decorator."""
        call_count = 0
        
        @profile("global_test_operation")
        def test_function():
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        
        assert result == "result"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_measure_operation_context(self):
        """Test global measure_operation context manager."""
        async with measure_operation("async_test_operation", tags={"async": True}) as timer:
            await asyncio.sleep(0.02)
            assert timer.operation_name == "async_test_operation"
        
        assert timer.duration is not None
        assert timer.duration >= 0.02
    
    def test_performance_monitoring_lifecycle(self):
        """Test global performance monitoring start/stop."""
        from agent_orchestra.performance_monitor import (
            start_performance_monitoring, 
            stop_performance_monitoring,
            get_performance_report
        )
        
        # Should be able to start monitoring
        start_performance_monitoring()
        
        # Should be able to get a report
        report = get_performance_report()
        assert isinstance(report, dict)
        assert "timestamp" in report
        
        # Should be able to stop monitoring
        stop_performance_monitoring()