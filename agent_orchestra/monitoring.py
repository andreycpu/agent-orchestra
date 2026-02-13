"""
Monitoring and observability for Agent Orchestra
"""
import time
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import structlog

from .types import Task, TaskStatus, AgentStatus
from .utils import MetricsCollector
from .exceptions import ValidationError

logger = structlog.get_logger(__name__)


@dataclass
class SystemHealthMetrics:
    """System-wide health and resource metrics"""
    active_agents: int = 0
    total_queue_depth: int = 0
    system_cpu_percent: float = 0.0
    system_memory_percent: float = 0.0
    redis_connection_pool_size: int = 0
    redis_response_time_ms: float = 0.0
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    uptime_seconds: float = 0.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for tasks and agents"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    tasks_per_second: float = 0.0
    error_rate: float = 0.0


@dataclass
class AgentMetrics:
    """Metrics for individual agents"""
    agent_id: str
    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_task_time: float = 0.0
    uptime: float = 0.0
    current_status: str = "unknown"
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    queue_depth: int = 0
    last_heartbeat: Optional[datetime] = None
    error_count_24h: int = 0
    last_task_completed: Optional[datetime] = None
    error_rate: float = 0.0


@dataclass
class SystemHealth:
    """Overall system health indicators"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_agents: int = 0
    queued_tasks: int = 0
    healthy_agents: int = 0
    unhealthy_agents: int = 0
    system_load: float = 0.0


@dataclass 
class Alert:
    """Monitoring alert information"""
    alert_id: str
    severity: str  # 'critical', 'warning', 'info'
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class HealthThreshold:
    """Health check threshold configuration"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = 'greater_than'  # 'greater_than', 'less_than', 'equals'
    description: str = ""


@dataclass
class PerformanceTrend:
    """Performance trending information"""
    metric_name: str
    current_value: float
    previous_value: float
    change_percent: float
    trend_direction: str  # 'improving', 'degrading', 'stable'
    sample_count: int
    time_window_minutes: int


class TaskMonitor:
    """Monitor task execution and performance"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._task_history: deque = deque(maxlen=history_size)
        self._execution_times: Dict[str, List[float]] = defaultdict(list)
        self._task_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._start_time = time.time()
    
    def record_task_start(self, task: Task):
        """Record task start"""
        self._task_history.append({
            "task_id": task.id,
            "task_type": task.type,
            "status": "started",
            "timestamp": time.time(),
            "agent_id": task.assigned_agent
        })
    
    def record_task_completion(self, task: Task, execution_time: float, success: bool):
        """Record task completion"""
        self._task_history.append({
            "task_id": task.id,
            "task_type": task.type,
            "status": "completed" if success else "failed",
            "timestamp": time.time(),
            "execution_time": execution_time,
            "agent_id": task.assigned_agent,
            "success": success
        })
        
        # Update metrics
        self._execution_times[task.type].append(execution_time)
        if len(self._execution_times[task.type]) > 100:
            self._execution_times[task.type].pop(0)
        
        status_key = "completed" if success else "failed"
        self._task_counts[task.type][status_key] += 1
        self._task_counts[task.type]["total"] += 1
    
    def get_performance_metrics(self, task_type: Optional[str] = None) -> PerformanceMetrics:
        """Get performance metrics for all tasks or specific type"""
        if task_type:
            counts = self._task_counts[task_type]
            times = self._execution_times[task_type]
        else:
            # Aggregate all task types
            counts = defaultdict(int)
            times = []
            
            for type_counts in self._task_counts.values():
                for status, count in type_counts.items():
                    counts[status] += count
            
            for type_times in self._execution_times.values():
                times.extend(type_times)
        
        total = counts["total"]
        completed = counts["completed"]
        failed = counts["failed"]
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed
        )
        
        if times:
            metrics.average_execution_time = sum(times) / len(times)
            metrics.min_execution_time = min(times)
            metrics.max_execution_time = max(times)
        
        if total > 0:
            metrics.error_rate = failed / total
        
        # Calculate throughput
        uptime = time.time() - self._start_time
        if uptime > 0:
            metrics.tasks_per_second = completed / uptime
        
        return metrics
    
    def get_recent_tasks(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent task history"""
        return list(self._task_history)[-count:]
    
    def get_task_type_distribution(self) -> Dict[str, int]:
        """Get distribution of task types"""
        return {
            task_type: counts["total"]
            for task_type, counts in self._task_counts.items()
        }


class AgentMonitor:
    """Monitor agent health and performance"""
    
    def __init__(self):
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._agent_start_times: Dict[str, float] = {}
        self._heartbeat_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def register_agent(self, agent_id: str, agent_name: str = None):
        """Register an agent for monitoring"""
        self._agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
        self._agent_start_times[agent_id] = time.time()
        
        logger.info("Agent registered for monitoring", agent_id=agent_id)
    
    def update_agent_status(self, agent_id: str, status: AgentStatus):
        """Update agent status"""
        if agent_id in self._agent_metrics:
            self._agent_metrics[agent_id].current_status = status.value
    
    def record_agent_heartbeat(self, agent_id: str):
        """Record agent heartbeat"""
        now = time.time()
        self._heartbeat_history[agent_id].append(now)
        
        # Update uptime
        if agent_id in self._agent_metrics and agent_id in self._agent_start_times:
            self._agent_metrics[agent_id].uptime = now - self._agent_start_times[agent_id]
    
    def record_agent_task_completion(
        self, 
        agent_id: str, 
        success: bool, 
        execution_time: float
    ):
        """Record task completion by agent"""
        if agent_id not in self._agent_metrics:
            return
        
        metrics = self._agent_metrics[agent_id]
        metrics.total_tasks_processed += 1
        
        if success:
            metrics.successful_tasks += 1
        else:
            metrics.failed_tasks += 1
        
        # Update average task time (exponential moving average)
        if metrics.average_task_time == 0:
            metrics.average_task_time = execution_time
        else:
            alpha = 0.1  # Smoothing factor
            metrics.average_task_time = (
                alpha * execution_time + 
                (1 - alpha) * metrics.average_task_time
            )
        
        metrics.last_task_completed = datetime.utcnow()
        
        # Update error rate
        if metrics.total_tasks_processed > 0:
            metrics.error_rate = metrics.failed_tasks / metrics.total_tasks_processed
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get metrics for specific agent"""
        return self._agent_metrics.get(agent_id)
    
    def get_all_agent_metrics(self) -> List[AgentMetrics]:
        """Get metrics for all agents"""
        return list(self._agent_metrics.values())
    
    def is_agent_healthy(self, agent_id: str, heartbeat_threshold: float = 60.0) -> bool:
        """Check if agent is healthy based on recent heartbeats"""
        if agent_id not in self._heartbeat_history:
            return False
        
        heartbeats = self._heartbeat_history[agent_id]
        if not heartbeats:
            return False
        
        last_heartbeat = heartbeats[-1]
        return (time.time() - last_heartbeat) < heartbeat_threshold
    
    def get_healthy_agents(self) -> List[str]:
        """Get list of healthy agent IDs"""
        return [
            agent_id for agent_id in self._agent_metrics.keys()
            if self.is_agent_healthy(agent_id)
        ]
    
    def get_unhealthy_agents(self) -> List[str]:
        """Get list of unhealthy agent IDs"""
        return [
            agent_id for agent_id in self._agent_metrics.keys()
            if not self.is_agent_healthy(agent_id)
        ]


class SystemMonitor:
    """Monitor overall system health and performance"""
    
    def __init__(self):
        self._metrics_collector = MetricsCollector()
        self._system_stats: deque = deque(maxlen=1000)
        self._start_time = time.time()
    
    def record_system_stats(self, stats: Dict[str, Any]):
        """Record system statistics"""
        stats["timestamp"] = time.time()
        self._system_stats.append(stats)
        
        # Update metrics collector
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self._metrics_collector.set_gauge(f"system.{key}", value)
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health"""
        if not self._system_stats:
            return SystemHealth()
        
        latest_stats = self._system_stats[-1]
        
        return SystemHealth(
            cpu_usage=latest_stats.get("cpu_usage", 0.0),
            memory_usage=latest_stats.get("memory_usage", 0.0),
            active_agents=latest_stats.get("active_agents", 0),
            queued_tasks=latest_stats.get("queued_tasks", 0),
            healthy_agents=latest_stats.get("healthy_agents", 0),
            unhealthy_agents=latest_stats.get("unhealthy_agents", 0),
            system_load=latest_stats.get("system_load", 0.0)
        )
    
    def get_system_trends(self, duration_minutes: int = 60) -> Dict[str, List[float]]:
        """Get system trends over specified duration"""
        cutoff_time = time.time() - (duration_minutes * 60)
        
        trends = defaultdict(list)
        
        for stat in self._system_stats:
            if stat["timestamp"] < cutoff_time:
                continue
            
            for key, value in stat.items():
                if key != "timestamp" and isinstance(value, (int, float)):
                    trends[key].append(value)
        
        return dict(trends)
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all collected metrics"""
        return self._metrics_collector.get_metrics()


class OrchestrationMonitor:
    """Main monitoring interface that combines all monitors"""
    
    def __init__(self):
        self.task_monitor = TaskMonitor()
        self.agent_monitor = AgentMonitor()
        self.system_monitor = SystemMonitor()
        self._alerts: List[Dict[str, Any]] = []
        self._monitoring_active = False
    
    async def start_monitoring(self):
        """Start monitoring services"""
        self._monitoring_active = True
        logger.info("Orchestration monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring services"""
        self._monitoring_active = False
        logger.info("Orchestration monitoring stopped")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        performance = self.task_monitor.get_performance_metrics()
        agent_metrics = self.agent_monitor.get_all_agent_metrics()
        system_health = self.system_monitor.get_system_health()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "performance": asdict(performance),
            "agents": [asdict(metrics) for metrics in agent_metrics],
            "system_health": asdict(system_health),
            "task_distribution": self.task_monitor.get_task_type_distribution(),
            "recent_tasks": self.task_monitor.get_recent_tasks(10),
            "healthy_agents": self.agent_monitor.get_healthy_agents(),
            "unhealthy_agents": self.agent_monitor.get_unhealthy_agents(),
            "alerts": self._alerts[-10:]  # Last 10 alerts
        }
    
    def create_alert(self, level: str, message: str, details: Dict[str, Any] = None):
        """Create a system alert"""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "details": details or {}
        }
        
        self._alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self._alerts) > 1000:
            self._alerts.pop(0)
        
        logger.warning("System alert created", **alert)
    
    def check_health_and_alerts(self):
        """Check system health and create alerts if needed"""
        # Check agent health
        unhealthy_agents = self.agent_monitor.get_unhealthy_agents()
        if unhealthy_agents:
            self.create_alert(
                "warning",
                f"Unhealthy agents detected: {len(unhealthy_agents)}",
                {"agents": unhealthy_agents}
            )
        
        # Check error rates
        performance = self.task_monitor.get_performance_metrics()
        if performance.error_rate > 0.1:  # 10% error rate
            self.create_alert(
                "error",
                f"High error rate detected: {performance.error_rate:.2%}",
                {"error_rate": performance.error_rate}
            )
        
        # Check system resources
        system_health = self.system_monitor.get_system_health()
        if system_health.cpu_usage > 90:
            self.create_alert(
                "warning",
                f"High CPU usage: {system_health.cpu_usage}%",
                {"cpu_usage": system_health.cpu_usage}
            )
        
        if system_health.memory_usage > 90:
            self.create_alert(
                "warning", 
                f"High memory usage: {system_health.memory_usage}%",
                {"memory_usage": system_health.memory_usage}
            )


class AlertManager:
    """Advanced alerting system with configurable thresholds"""
    
    def __init__(self):
        self._thresholds: Dict[str, HealthThreshold] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._resolved_alerts: List[Alert] = []
        self._alert_handlers: List[callable] = []
        self._alert_counter = 0
        
        # Default thresholds
        self._setup_default_thresholds()
        
        logger.info("AlertManager initialized")
    
    def _setup_default_thresholds(self):
        """Setup default monitoring thresholds"""
        default_thresholds = [
            HealthThreshold("cpu_usage", 75.0, 90.0, "greater_than", "System CPU usage"),
            HealthThreshold("memory_usage", 80.0, 95.0, "greater_than", "System memory usage"), 
            HealthThreshold("error_rate", 0.05, 0.1, "greater_than", "Task error rate"),
            HealthThreshold("queue_depth", 100, 500, "greater_than", "Task queue depth"),
            HealthThreshold("agent_heartbeat_age", 300, 600, "greater_than", "Agent heartbeat staleness")
        ]
        
        for threshold in default_thresholds:
            self._thresholds[threshold.metric_name] = threshold
    
    def add_threshold(self, threshold: HealthThreshold):
        """Add or update a health check threshold
        
        Args:
            threshold: Threshold configuration
            
        Raises:
            ValidationError: If threshold is invalid
        """
        if not isinstance(threshold, HealthThreshold):
            raise ValidationError("threshold must be a HealthThreshold instance")
        if not threshold.metric_name:
            raise ValidationError("threshold must have a metric_name")
        if threshold.comparison not in ['greater_than', 'less_than', 'equals']:
            raise ValidationError("threshold comparison must be 'greater_than', 'less_than', or 'equals'")
        if threshold.critical_threshold <= threshold.warning_threshold and threshold.comparison == 'greater_than':
            raise ValidationError("critical_threshold must be greater than warning_threshold for 'greater_than' comparison")
            
        self._thresholds[threshold.metric_name] = threshold
        
        logger.info(
            "Health threshold configured",
            metric=threshold.metric_name,
            warning=threshold.warning_threshold,
            critical=threshold.critical_threshold
        )
    
    def check_metric(self, metric_name: str, value: float) -> Optional[Alert]:
        """Check a metric value against thresholds
        
        Args:
            metric_name: Name of the metric to check
            value: Current metric value
            
        Returns:
            Alert if threshold exceeded, None otherwise
        """
        if metric_name not in self._thresholds:
            return None
            
        threshold = self._thresholds[metric_name]
        alert_severity = None
        
        if threshold.comparison == 'greater_than':
            if value >= threshold.critical_threshold:
                alert_severity = 'critical'
            elif value >= threshold.warning_threshold:
                alert_severity = 'warning'
        elif threshold.comparison == 'less_than':
            if value <= threshold.critical_threshold:
                alert_severity = 'critical'
            elif value <= threshold.warning_threshold:
                alert_severity = 'warning'
        elif threshold.comparison == 'equals':
            if abs(value - threshold.critical_threshold) < 0.001:
                alert_severity = 'critical'
            elif abs(value - threshold.warning_threshold) < 0.001:
                alert_severity = 'warning'
        
        if alert_severity:
            return self._create_alert(metric_name, value, threshold, alert_severity)
        
        # Check if we should resolve existing alert
        self._try_resolve_alert(metric_name)
        return None
    
    def _create_alert(self, metric_name: str, value: float, threshold: HealthThreshold, severity: str) -> Alert:
        """Create a new alert"""
        self._alert_counter += 1
        alert_id = f"alert_{self._alert_counter}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=f"{metric_name.replace('_', ' ').title()} {severity.title()}",
            message=f"{threshold.description}: {value:.2f} exceeds {severity} threshold of {getattr(threshold, f'{severity}_threshold'):.2f}",
            metric_name=metric_name,
            current_value=value,
            threshold=getattr(threshold, f'{severity}_threshold'),
            timestamp=datetime.utcnow()
        )
        
        # Check if we already have an active alert for this metric
        if metric_name in self._active_alerts:
            existing = self._active_alerts[metric_name]
            # Update existing alert if severity is higher or same
            if (severity == 'critical') or (severity == 'warning' and existing.severity != 'critical'):
                self._active_alerts[metric_name] = alert
        else:
            self._active_alerts[metric_name] = alert
            
        # Trigger alert handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error("Alert handler failed", error=str(e), alert_id=alert_id)
        
        logger.warning(
            "Alert created",
            alert_id=alert_id,
            severity=severity,
            metric=metric_name,
            value=value,
            threshold=alert.threshold
        )
        
        return alert
    
    def _try_resolve_alert(self, metric_name: str):
        """Try to resolve an active alert if metric is back to normal"""
        if metric_name in self._active_alerts:
            alert = self._active_alerts[metric_name]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            
            self._resolved_alerts.append(alert)
            del self._active_alerts[metric_name]
            
            # Keep only last 1000 resolved alerts
            if len(self._resolved_alerts) > 1000:
                self._resolved_alerts.pop(0)
            
            logger.info("Alert resolved", alert_id=alert.alert_id, metric=metric_name)
    
    def add_alert_handler(self, handler: callable):
        """Add a custom alert handler function
        
        Args:
            handler: Function that accepts Alert object
        """
        if not callable(handler):
            raise ValidationError("handler must be callable")
            
        self._alert_handlers.append(handler)
        logger.info("Alert handler added")
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self._active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
            
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_resolved_alerts(self, limit: int = 100) -> List[Alert]:
        """Get recently resolved alerts"""
        return self._resolved_alerts[-limit:]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert
        
        Args:
            alert_id: ID of alert to resolve
            
        Returns:
            True if alert was resolved, False if not found
        """
        for metric_name, alert in self._active_alerts.items():
            if alert.alert_id == alert_id:
                self._try_resolve_alert(metric_name)
                return True
        
        return False


class PerformanceTrendAnalyzer:
    """Analyze performance trends over time"""
    
    def __init__(self, sample_size: int = 100):
        self._sample_size = sample_size
        self._metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=sample_size))
        self._last_analysis: Dict[str, datetime] = {}
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Record a metric value
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Optional timestamp (uses current time if None)
            
        Raises:
            ValidationError: If parameters are invalid
        """
        if not metric_name:
            raise ValidationError("metric_name cannot be empty")
        if not isinstance(value, (int, float)):
            raise ValidationError("value must be numeric")
            
        timestamp = timestamp or datetime.utcnow()
        
        self._metric_history[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
    
    def analyze_trend(self, metric_name: str, window_minutes: int = 30) -> Optional[PerformanceTrend]:
        """Analyze trend for a specific metric
        
        Args:
            metric_name: Metric to analyze
            window_minutes: Time window for analysis
            
        Returns:
            PerformanceTrend object or None if insufficient data
        """
        if metric_name not in self._metric_history:
            return None
            
        history = self._metric_history[metric_name]
        if len(history) < 2:
            return None
        
        # Filter to time window
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_data = [
            entry for entry in history 
            if entry['timestamp'] >= cutoff_time
        ]
        
        if len(recent_data) < 2:
            return None
        
        # Calculate trend
        values = [entry['value'] for entry in recent_data]
        current_value = values[-1]
        previous_value = sum(values[:-1]) / len(values[:-1])  # Average of previous values
        
        change_percent = ((current_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
        
        # Determine trend direction
        if abs(change_percent) < 5:  # Less than 5% change is stable
            trend_direction = 'stable'
        elif change_percent > 0:
            # For most metrics, higher is worse, but this could be configurable
            trend_direction = 'degrading'
        else:
            trend_direction = 'improving'
        
        return PerformanceTrend(
            metric_name=metric_name,
            current_value=current_value,
            previous_value=previous_value,
            change_percent=change_percent,
            trend_direction=trend_direction,
            sample_count=len(recent_data),
            time_window_minutes=window_minutes
        )
    
    def get_all_trends(self, window_minutes: int = 30) -> Dict[str, PerformanceTrend]:
        """Get trends for all tracked metrics"""
        trends = {}
        
        for metric_name in self._metric_history.keys():
            trend = self.analyze_trend(metric_name, window_minutes)
            if trend:
                trends[metric_name] = trend
        
        return trends


class CustomMetricsCollector:
    """Collect and manage custom application metrics"""
    
    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric
        
        Args:
            name: Counter name
            value: Amount to increment by
            tags: Optional tags for the metric
        """
        if not name:
            raise ValidationError("counter name cannot be empty")
        if value < 0:
            raise ValidationError("counter increment value cannot be negative")
            
        key = self._make_key(name, tags)
        self._counters[key] += value
        
        logger.debug("Counter incremented", name=name, value=value, total=self._counters[key])
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value
        
        Args:
            name: Gauge name  
            value: Value to set
            tags: Optional tags for the metric
        """
        if not name:
            raise ValidationError("gauge name cannot be empty")
        if not isinstance(value, (int, float)):
            raise ValidationError("gauge value must be numeric")
            
        key = self._make_key(name, tags)
        self._gauges[key] = value
        
        logger.debug("Gauge set", name=name, value=value)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a value in a histogram
        
        Args:
            name: Histogram name
            value: Value to record
            tags: Optional tags for the metric
        """
        if not name:
            raise ValidationError("histogram name cannot be empty")
        if not isinstance(value, (int, float)):
            raise ValidationError("histogram value must be numeric")
            
        key = self._make_key(name, tags)
        self._histograms[key].append(value)
        
        # Keep only last 1000 values to prevent memory issues
        if len(self._histograms[key]) > 1000:
            self._histograms[key].pop(0)
    
    def time_operation(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record operation timing
        
        Args:
            name: Timer name
            duration: Duration in seconds
            tags: Optional tags for the metric
        """
        if not name:
            raise ValidationError("timer name cannot be empty")
        if duration < 0:
            raise ValidationError("timer duration cannot be negative")
            
        key = self._make_key(name, tags)
        self._timers[key].append(duration)
        
        # Keep only last 1000 values
        if len(self._timers[key]) > 1000:
            self._timers[key].pop(0)
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric with tags"""
        if not tags:
            return name
            
        # Sort tags for consistent key generation
        tag_string = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_string}]"
    
    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value"""
        key = self._make_key(name, tags)
        return self._counters.get(key, 0.0)
    
    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value"""
        key = self._make_key(name, tags)
        return self._gauges.get(key, 0.0)
    
    def get_histogram_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        key = self._make_key(name, tags)
        values = self._histograms.get(key, [])
        
        if not values:
            return {"count": 0}
        
        sorted_values = sorted(values)
        count = len(values)
        
        return {
            "count": count,
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_timer_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics"""
        return self.get_histogram_stats(name, tags)  # Same calculation
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._timers.clear()
        
        logger.info("All metrics reset")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histogram_stats": {
                name: self.get_histogram_stats(name.split('[')[0], 
                    self._parse_tags_from_key(name) if '[' in name else None)
                for name in self._histograms.keys()
            },
            "timer_stats": {
                name: self.get_timer_stats(name.split('[')[0],
                    self._parse_tags_from_key(name) if '[' in name else None) 
                for name in self._timers.keys()
            }
        }
    
    def _parse_tags_from_key(self, key: str) -> Dict[str, str]:
        """Parse tags from a metric key"""
        if '[' not in key:
            return {}
            
        tag_part = key.split('[', 1)[1].rstrip(']')
        tags = {}
        
        for tag_pair in tag_part.split(','):
            if '=' in tag_pair:
                k, v = tag_pair.split('=', 1)
                tags[k] = v
        
        return tags