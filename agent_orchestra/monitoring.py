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

logger = structlog.get_logger(__name__)


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