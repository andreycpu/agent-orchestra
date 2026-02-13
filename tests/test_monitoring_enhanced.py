"""
Tests for enhanced monitoring functionality
"""
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from agent_orchestra.monitoring import (
    AlertManager, PerformanceTrendAnalyzer, CustomMetricsCollector,
    Alert, HealthThreshold, PerformanceTrend
)
from agent_orchestra.exceptions import ValidationError


class TestHealthThreshold:
    """Test cases for HealthThreshold dataclass"""
    
    def test_health_threshold_creation(self):
        """Test HealthThreshold creation"""
        threshold = HealthThreshold(
            metric_name="cpu_usage",
            warning_threshold=75.0,
            critical_threshold=90.0,
            comparison="greater_than",
            description="CPU usage threshold"
        )
        
        assert threshold.metric_name == "cpu_usage"
        assert threshold.warning_threshold == 75.0
        assert threshold.critical_threshold == 90.0
        assert threshold.comparison == "greater_than"
        assert threshold.description == "CPU usage threshold"


class TestAlert:
    """Test cases for Alert dataclass"""
    
    def test_alert_creation(self):
        """Test Alert creation"""
        timestamp = datetime.utcnow()
        alert = Alert(
            alert_id="test-alert-1",
            severity="warning",
            title="High CPU Usage",
            message="CPU usage is 85%",
            metric_name="cpu_usage",
            current_value=85.0,
            threshold=75.0,
            timestamp=timestamp
        )
        
        assert alert.alert_id == "test-alert-1"
        assert alert.severity == "warning"
        assert alert.resolved is False
        assert alert.timestamp == timestamp


class TestAlertManager:
    """Test cases for AlertManager class"""
    
    @pytest.fixture
    def alert_manager(self):
        """Create AlertManager instance"""
        return AlertManager()
    
    def test_alert_manager_initialization(self, alert_manager):
        """Test AlertManager initialization"""
        assert len(alert_manager._thresholds) > 0  # Should have default thresholds
        assert len(alert_manager._active_alerts) == 0
        assert len(alert_manager._resolved_alerts) == 0
    
    def test_add_threshold_validation(self, alert_manager):
        """Test threshold addition validation"""
        # Invalid threshold type
        with pytest.raises(ValidationError, match="threshold must be a HealthThreshold instance"):
            alert_manager.add_threshold("invalid")
        
        # Empty metric name
        invalid_threshold = HealthThreshold("", 10.0, 20.0)
        with pytest.raises(ValidationError, match="threshold must have a metric_name"):
            alert_manager.add_threshold(invalid_threshold)
        
        # Invalid comparison
        invalid_threshold = HealthThreshold("test", 10.0, 20.0, "invalid_comparison")
        with pytest.raises(ValidationError, match="threshold comparison must be"):
            alert_manager.add_threshold(invalid_threshold)
        
        # Invalid threshold order
        invalid_threshold = HealthThreshold("test", 20.0, 10.0, "greater_than")
        with pytest.raises(ValidationError, match="critical_threshold must be greater than warning_threshold"):
            alert_manager.add_threshold(invalid_threshold)
    
    def test_add_valid_threshold(self, alert_manager):
        """Test adding valid threshold"""
        threshold = HealthThreshold(
            metric_name="custom_metric",
            warning_threshold=50.0,
            critical_threshold=80.0,
            comparison="greater_than",
            description="Custom test metric"
        )
        
        alert_manager.add_threshold(threshold)
        
        assert "custom_metric" in alert_manager._thresholds
        assert alert_manager._thresholds["custom_metric"] == threshold
    
    def test_check_metric_no_alert(self, alert_manager):
        """Test metric check that doesn't trigger alert"""
        # Check CPU below warning threshold
        alert = alert_manager.check_metric("cpu_usage", 50.0)
        
        assert alert is None
        assert len(alert_manager._active_alerts) == 0
    
    def test_check_metric_warning_alert(self, alert_manager):
        """Test metric check that triggers warning alert"""
        alert = alert_manager.check_metric("cpu_usage", 80.0)
        
        assert alert is not None
        assert alert.severity == "warning"
        assert alert.metric_name == "cpu_usage"
        assert alert.current_value == 80.0
        assert len(alert_manager._active_alerts) == 1
    
    def test_check_metric_critical_alert(self, alert_manager):
        """Test metric check that triggers critical alert"""
        alert = alert_manager.check_metric("cpu_usage", 95.0)
        
        assert alert is not None
        assert alert.severity == "critical"
        assert alert.metric_name == "cpu_usage"
        assert alert.current_value == 95.0
    
    def test_check_metric_less_than_threshold(self, alert_manager):
        """Test less_than comparison threshold"""
        # Add a threshold for available memory (alert when low)
        threshold = HealthThreshold(
            "available_memory",
            warning_threshold=20.0,
            critical_threshold=10.0,
            comparison="less_than"
        )
        alert_manager.add_threshold(threshold)
        
        # Should trigger critical alert when memory is very low
        alert = alert_manager.check_metric("available_memory", 5.0)
        
        assert alert is not None
        assert alert.severity == "critical"
    
    def test_alert_resolution(self, alert_manager):
        """Test automatic alert resolution"""
        # Trigger an alert
        alert_manager.check_metric("cpu_usage", 95.0)
        assert len(alert_manager._active_alerts) == 1
        
        # Check metric back to normal
        alert_manager.check_metric("cpu_usage", 50.0)
        
        # Alert should be resolved
        assert len(alert_manager._active_alerts) == 0
        assert len(alert_manager._resolved_alerts) == 1
        assert alert_manager._resolved_alerts[0].resolved is True
    
    def test_alert_handler(self, alert_manager):
        """Test custom alert handler"""
        handler_calls = []
        
        def test_handler(alert: Alert):
            handler_calls.append(alert)
        
        alert_manager.add_alert_handler(test_handler)
        
        # Trigger an alert
        alert_manager.check_metric("cpu_usage", 95.0)
        
        assert len(handler_calls) == 1
        assert handler_calls[0].severity == "critical"
    
    def test_invalid_alert_handler(self, alert_manager):
        """Test adding invalid alert handler"""
        with pytest.raises(ValidationError, match="handler must be callable"):
            alert_manager.add_alert_handler("not_callable")
    
    def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts"""
        # Trigger multiple alerts
        alert_manager.check_metric("cpu_usage", 95.0)
        alert_manager.check_metric("memory_usage", 85.0)
        
        alerts = alert_manager.get_active_alerts()
        assert len(alerts) == 2
        
        # Filter by severity
        critical_alerts = alert_manager.get_active_alerts(severity="critical")
        warning_alerts = alert_manager.get_active_alerts(severity="warning")
        
        assert len(critical_alerts) == 1
        assert len(warning_alerts) == 1
    
    def test_manual_alert_resolution(self, alert_manager):
        """Test manually resolving alerts"""
        # Trigger an alert
        alert = alert_manager.check_metric("cpu_usage", 95.0)
        alert_id = alert.alert_id
        
        # Manually resolve
        result = alert_manager.resolve_alert(alert_id)
        
        assert result is True
        assert len(alert_manager._active_alerts) == 0
        assert len(alert_manager._resolved_alerts) == 1
        
        # Try to resolve non-existent alert
        result = alert_manager.resolve_alert("nonexistent")
        assert result is False


class TestPerformanceTrendAnalyzer:
    """Test cases for PerformanceTrendAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self):
        """Create PerformanceTrendAnalyzer instance"""
        return PerformanceTrendAnalyzer(sample_size=50)
    
    def test_record_metric_validation(self, analyzer):
        """Test metric recording validation"""
        # Empty metric name
        with pytest.raises(ValidationError, match="metric_name cannot be empty"):
            analyzer.record_metric("", 10.0)
        
        # Non-numeric value
        with pytest.raises(ValidationError, match="value must be numeric"):
            analyzer.record_metric("test_metric", "not_numeric")
    
    def test_record_metric(self, analyzer):
        """Test metric recording"""
        timestamp = datetime.utcnow()
        analyzer.record_metric("response_time", 1.5, timestamp)
        
        assert "response_time" in analyzer._metric_history
        assert len(analyzer._metric_history["response_time"]) == 1
        
        entry = analyzer._metric_history["response_time"][0]
        assert entry["value"] == 1.5
        assert entry["timestamp"] == timestamp
    
    def test_analyze_trend_insufficient_data(self, analyzer):
        """Test trend analysis with insufficient data"""
        # No data
        trend = analyzer.analyze_trend("nonexistent_metric")
        assert trend is None
        
        # Only one data point
        analyzer.record_metric("test_metric", 10.0)
        trend = analyzer.analyze_trend("test_metric")
        assert trend is None
    
    def test_analyze_trend_improving(self, analyzer):
        """Test trend analysis showing improvement"""
        base_time = datetime.utcnow()
        
        # Record decreasing values (improving for response time)
        for i, value in enumerate([100.0, 90.0, 80.0, 70.0, 60.0]):
            timestamp = base_time + timedelta(minutes=i)
            analyzer.record_metric("response_time", value, timestamp)
        
        trend = analyzer.analyze_trend("response_time", window_minutes=10)
        
        assert trend is not None
        assert trend.metric_name == "response_time"
        assert trend.current_value == 60.0
        assert trend.trend_direction == "improving"  # Lower response time is better
        assert trend.change_percent < -5  # Significant decrease
    
    def test_analyze_trend_degrading(self, analyzer):
        """Test trend analysis showing degradation"""
        base_time = datetime.utcnow()
        
        # Record increasing values (degrading for response time)
        for i, value in enumerate([50.0, 60.0, 70.0, 80.0, 100.0]):
            timestamp = base_time + timedelta(minutes=i)
            analyzer.record_metric("response_time", value, timestamp)
        
        trend = analyzer.analyze_trend("response_time", window_minutes=10)
        
        assert trend is not None
        assert trend.trend_direction == "degrading"
        assert trend.change_percent > 5  # Significant increase
    
    def test_analyze_trend_stable(self, analyzer):
        """Test trend analysis showing stable performance"""
        base_time = datetime.utcnow()
        
        # Record stable values with small variations
        for i, value in enumerate([100.0, 101.0, 99.0, 102.0, 98.0]):
            timestamp = base_time + timedelta(minutes=i)
            analyzer.record_metric("cpu_usage", value, timestamp)
        
        trend = analyzer.analyze_trend("cpu_usage", window_minutes=10)
        
        assert trend is not None
        assert trend.trend_direction == "stable"
        assert abs(trend.change_percent) < 5  # Small change
    
    def test_get_all_trends(self, analyzer):
        """Test getting trends for all metrics"""
        base_time = datetime.utcnow()
        
        # Record data for multiple metrics
        metrics = ["cpu_usage", "memory_usage", "response_time"]
        for metric in metrics:
            for i in range(5):
                timestamp = base_time + timedelta(minutes=i)
                analyzer.record_metric(metric, 50.0 + i, timestamp)
        
        trends = analyzer.get_all_trends(window_minutes=10)
        
        assert len(trends) == 3
        for metric in metrics:
            assert metric in trends
            assert isinstance(trends[metric], PerformanceTrend)


class TestCustomMetricsCollector:
    """Test cases for CustomMetricsCollector class"""
    
    @pytest.fixture
    def collector(self):
        """Create CustomMetricsCollector instance"""
        return CustomMetricsCollector()
    
    def test_increment_counter_validation(self, collector):
        """Test counter increment validation"""
        # Empty name
        with pytest.raises(ValidationError, match="counter name cannot be empty"):
            collector.increment_counter("", 1.0)
        
        # Negative value
        with pytest.raises(ValidationError, match="counter increment value cannot be negative"):
            collector.increment_counter("test_counter", -1.0)
    
    def test_increment_counter(self, collector):
        """Test counter increment functionality"""
        collector.increment_counter("requests", 1.0)
        collector.increment_counter("requests", 2.0)
        
        assert collector.get_counter("requests") == 3.0
    
    def test_counter_with_tags(self, collector):
        """Test counter with tags"""
        tags1 = {"method": "GET", "status": "200"}
        tags2 = {"method": "POST", "status": "200"}
        
        collector.increment_counter("http_requests", 1.0, tags1)
        collector.increment_counter("http_requests", 1.0, tags1)
        collector.increment_counter("http_requests", 1.0, tags2)
        
        assert collector.get_counter("http_requests", tags1) == 2.0
        assert collector.get_counter("http_requests", tags2) == 1.0
    
    def test_set_gauge_validation(self, collector):
        """Test gauge set validation"""
        # Empty name
        with pytest.raises(ValidationError, match="gauge name cannot be empty"):
            collector.set_gauge("", 10.0)
        
        # Non-numeric value
        with pytest.raises(ValidationError, match="gauge value must be numeric"):
            collector.set_gauge("test_gauge", "not_numeric")
    
    def test_set_gauge(self, collector):
        """Test gauge functionality"""
        collector.set_gauge("temperature", 25.5)
        collector.set_gauge("temperature", 27.0)  # Overwrite previous value
        
        assert collector.get_gauge("temperature") == 27.0
    
    def test_record_histogram_validation(self, collector):
        """Test histogram recording validation"""
        # Empty name
        with pytest.raises(ValidationError, match="histogram name cannot be empty"):
            collector.record_histogram("", 1.0)
        
        # Non-numeric value
        with pytest.raises(ValidationError, match="histogram value must be numeric"):
            collector.record_histogram("test_histogram", "not_numeric")
    
    def test_record_histogram(self, collector):
        """Test histogram functionality"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        for value in values:
            collector.record_histogram("response_times", value)
        
        stats = collector.get_histogram_stats("response_times")
        
        assert stats["count"] == 10
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert stats["mean"] == 5.5
        assert stats["p50"] == 5.0  # Median
    
    def test_time_operation_validation(self, collector):
        """Test timer validation"""
        # Empty name
        with pytest.raises(ValidationError, match="timer name cannot be empty"):
            collector.time_operation("", 1.0)
        
        # Negative duration
        with pytest.raises(ValidationError, match="timer duration cannot be negative"):
            collector.time_operation("test_timer", -1.0)
    
    def test_time_operation(self, collector):
        """Test timer functionality"""
        durations = [0.1, 0.2, 0.3, 0.15, 0.25]
        
        for duration in durations:
            collector.time_operation("db_query", duration)
        
        stats = collector.get_timer_stats("db_query")
        
        assert stats["count"] == 5
        assert stats["min"] == 0.1
        assert stats["max"] == 0.3
        assert abs(stats["mean"] - 0.2) < 0.01  # Close to expected mean
    
    def test_empty_histogram_stats(self, collector):
        """Test histogram stats for non-existent metric"""
        stats = collector.get_histogram_stats("nonexistent")
        
        assert stats["count"] == 0
    
    def test_histogram_size_limit(self, collector):
        """Test histogram size limiting"""
        # Record more than 1000 values
        for i in range(1200):
            collector.record_histogram("large_histogram", float(i))
        
        # Should only keep last 1000 values
        key = "large_histogram"
        assert len(collector._histograms[key]) == 1000
        assert min(collector._histograms[key]) >= 200  # First 200 values should be removed
    
    def test_reset_metrics(self, collector):
        """Test metrics reset functionality"""
        # Add some metrics
        collector.increment_counter("test_counter", 1.0)
        collector.set_gauge("test_gauge", 10.0)
        collector.record_histogram("test_histogram", 5.0)
        collector.time_operation("test_timer", 0.1)
        
        # Reset all metrics
        collector.reset_metrics()
        
        assert collector.get_counter("test_counter") == 0.0
        assert collector.get_gauge("test_gauge") == 0.0
        assert collector.get_histogram_stats("test_histogram")["count"] == 0
        assert collector.get_timer_stats("test_timer")["count"] == 0
    
    def test_get_all_metrics(self, collector):
        """Test getting all metrics"""
        # Add some test metrics
        collector.increment_counter("requests", 10.0)
        collector.set_gauge("cpu_usage", 75.0)
        collector.record_histogram("response_time", 0.5)
        collector.time_operation("db_query", 0.1)
        
        all_metrics = collector.get_all_metrics()
        
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histogram_stats" in all_metrics
        assert "timer_stats" in all_metrics
        
        assert all_metrics["counters"]["requests"] == 10.0
        assert all_metrics["gauges"]["cpu_usage"] == 75.0
    
    def test_tag_parsing(self, collector):
        """Test tag parsing from metric keys"""
        tags = {"method": "GET", "status": "200"}
        key = collector._make_key("http_requests", tags)
        
        parsed_tags = collector._parse_tags_from_key(key)
        
        assert parsed_tags == tags
    
    def test_percentile_calculation(self, collector):
        """Test percentile calculation in histograms"""
        # Record values from 1 to 100
        for i in range(1, 101):
            collector.record_histogram("test_percentiles", float(i))
        
        stats = collector.get_histogram_stats("test_percentiles")
        
        # Check percentiles (approximately)
        assert abs(stats["p50"] - 50.0) <= 1.0  # 50th percentile around 50
        assert abs(stats["p95"] - 95.0) <= 1.0  # 95th percentile around 95
        assert abs(stats["p99"] - 99.0) <= 1.0  # 99th percentile around 99


class TestIntegrationMonitoring:
    """Integration tests for monitoring components"""
    
    def test_alert_manager_with_trend_analyzer(self):
        """Test AlertManager integration with PerformanceTrendAnalyzer"""
        alert_manager = AlertManager()
        trend_analyzer = PerformanceTrendAnalyzer()
        
        # Record trending data
        base_time = datetime.utcnow()
        for i, value in enumerate([50.0, 60.0, 70.0, 80.0, 90.0]):
            timestamp = base_time + timedelta(minutes=i)
            trend_analyzer.record_metric("cpu_usage", value, timestamp)
        
        # Analyze trend
        trend = trend_analyzer.analyze_trend("cpu_usage")
        
        # Check if current value would trigger alert
        alert = alert_manager.check_metric("cpu_usage", trend.current_value)
        
        assert trend.trend_direction == "degrading"
        assert alert is not None  # Should trigger alert at 90% CPU
        assert alert.severity == "critical"
    
    def test_custom_metrics_with_alert_manager(self):
        """Test CustomMetricsCollector integration with AlertManager"""
        collector = CustomMetricsCollector()
        alert_manager = AlertManager()
        
        # Add custom threshold for error rate
        threshold = HealthThreshold(
            "error_rate",
            warning_threshold=0.05,  # 5%
            critical_threshold=0.1,   # 10%
            comparison="greater_than"
        )
        alert_manager.add_threshold(threshold)
        
        # Simulate error tracking
        collector.increment_counter("total_requests", 100.0)
        collector.increment_counter("error_requests", 8.0)
        
        # Calculate error rate
        error_rate = collector.get_counter("error_requests") / collector.get_counter("total_requests")
        
        # Check against threshold
        alert = alert_manager.check_metric("error_rate", error_rate)
        
        assert alert is not None
        assert alert.severity == "warning"  # 8% is above 5% but below 10%