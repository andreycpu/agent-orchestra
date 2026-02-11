"""
Metrics and data exporters for Agent Orchestra
"""
import json
import csv
import time
from typing import Dict, List, Any, Optional, TextIO
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)


class PrometheusExporter:
    """Export metrics in Prometheus format"""
    
    def __init__(self, prefix: str = "agent_orchestra"):
        self.prefix = prefix
    
    def format_metric(self, name: str, value: float, labels: Dict[str, str] = None,
                     help_text: str = None, metric_type: str = "gauge") -> str:
        """Format a single metric in Prometheus format"""
        lines = []
        
        # Add help text
        if help_text:
            lines.append(f"# HELP {self.prefix}_{name} {help_text}")
        
        # Add type
        lines.append(f"# TYPE {self.prefix}_{name} {metric_type}")
        
        # Format labels
        label_str = ""
        if labels:
            label_parts = [f'{k}="{v}"' for k, v in labels.items()]
            label_str = "{" + ",".join(label_parts) + "}"
        
        # Add metric line
        lines.append(f"{self.prefix}_{name}{label_str} {value}")
        
        return "\n".join(lines)
    
    def export_system_metrics(self, metrics: Dict[str, Any]) -> str:
        """Export system metrics in Prometheus format"""
        lines = []
        
        # Task metrics
        if "tasks" in metrics:
            task_metrics = metrics["tasks"]
            
            lines.append(self.format_metric(
                "tasks_total",
                task_metrics.get("total", 0),
                help_text="Total number of tasks"
            ))
            
            lines.append(self.format_metric(
                "tasks_completed",
                task_metrics.get("completed", 0),
                help_text="Number of completed tasks"
            ))
            
            lines.append(self.format_metric(
                "tasks_failed",
                task_metrics.get("failed", 0),
                help_text="Number of failed tasks"
            ))
            
            lines.append(self.format_metric(
                "tasks_pending",
                task_metrics.get("pending", 0),
                help_text="Number of pending tasks"
            ))
        
        # Agent metrics
        if "agents" in metrics:
            agent_metrics = metrics["agents"]
            
            lines.append(self.format_metric(
                "agents_total",
                agent_metrics.get("total", 0),
                help_text="Total number of registered agents"
            ))
            
            lines.append(self.format_metric(
                "agents_idle",
                agent_metrics.get("idle", 0),
                help_text="Number of idle agents"
            ))
            
            lines.append(self.format_metric(
                "agents_busy",
                agent_metrics.get("busy", 0),
                help_text="Number of busy agents"
            ))
        
        # Performance metrics
        if "performance" in metrics:
            perf_metrics = metrics["performance"]
            
            lines.append(self.format_metric(
                "tasks_per_second",
                perf_metrics.get("tasks_per_second", 0),
                help_text="Task processing rate"
            ))
            
            lines.append(self.format_metric(
                "average_execution_time",
                perf_metrics.get("avg_execution_time", 0),
                help_text="Average task execution time in seconds"
            ))
            
            lines.append(self.format_metric(
                "error_rate",
                perf_metrics.get("error_rate", 0),
                help_text="Task error rate (0-1)"
            ))
        
        return "\n".join(lines)


class JSONExporter:
    """Export data in JSON format"""
    
    def __init__(self, pretty_print: bool = True):
        self.pretty_print = pretty_print
    
    def export_tasks(self, tasks: List[Dict[str, Any]]) -> str:
        """Export tasks as JSON"""
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "tasks": tasks,
            "count": len(tasks)
        }
        
        if self.pretty_print:
            return json.dumps(export_data, indent=2, default=str)
        return json.dumps(export_data, default=str)
    
    def export_agents(self, agents: List[Dict[str, Any]]) -> str:
        """Export agents as JSON"""
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "agents": agents,
            "count": len(agents)
        }
        
        if self.pretty_print:
            return json.dumps(export_data, indent=2, default=str)
        return json.dumps(export_data, default=str)
    
    def export_metrics(self, metrics: Dict[str, Any]) -> str:
        """Export metrics as JSON"""
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }
        
        if self.pretty_print:
            return json.dumps(export_data, indent=2, default=str)
        return json.dumps(export_data, default=str)


class CSVExporter:
    """Export data in CSV format"""
    
    def write_tasks_csv(self, tasks: List[Dict[str, Any]], output_file: TextIO):
        """Write tasks to CSV file"""
        if not tasks:
            return
        
        # Get all possible fields
        all_fields = set()
        for task in tasks:
            all_fields.update(task.keys())
        
        fieldnames = sorted(all_fields)
        
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for task in tasks:
            # Flatten complex fields
            row = {}
            for field, value in task.items():
                if isinstance(value, (dict, list)):
                    row[field] = json.dumps(value, default=str)
                else:
                    row[field] = value
            
            writer.writerow(row)
    
    def write_agents_csv(self, agents: List[Dict[str, Any]], output_file: TextIO):
        """Write agents to CSV file"""
        if not agents:
            return
        
        fieldnames = [
            "id", "name", "status", "capabilities", "current_task",
            "last_heartbeat", "metadata"
        ]
        
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for agent in agents:
            row = {
                "id": agent.get("id", ""),
                "name": agent.get("name", ""),
                "status": agent.get("status", ""),
                "capabilities": json.dumps(agent.get("capabilities", [])),
                "current_task": agent.get("current_task", ""),
                "last_heartbeat": agent.get("last_heartbeat", ""),
                "metadata": json.dumps(agent.get("metadata", {}))
            }
            
            writer.writerow(row)


class InfluxDBExporter:
    """Export metrics to InfluxDB line protocol format"""
    
    def __init__(self, measurement_prefix: str = "agent_orchestra"):
        self.measurement_prefix = measurement_prefix
    
    def format_line_protocol(self, measurement: str, tags: Dict[str, str] = None,
                           fields: Dict[str, Any] = None, timestamp: Optional[int] = None) -> str:
        """Format data in InfluxDB line protocol"""
        if not fields:
            return ""
        
        # Measurement name
        line = f"{self.measurement_prefix}_{measurement}"
        
        # Tags
        if tags:
            tag_parts = []
            for key, value in sorted(tags.items()):
                # Escape special characters
                key = str(key).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")
                value = str(value).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")
                tag_parts.append(f"{key}={value}")
            
            line += "," + ",".join(tag_parts)
        
        # Fields
        field_parts = []
        for key, value in sorted(fields.items()):
            key = str(key).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")
            
            # Format value based on type
            if isinstance(value, bool):
                field_parts.append(f"{key}={str(value).lower()}")
            elif isinstance(value, int):
                field_parts.append(f"{key}={value}i")
            elif isinstance(value, float):
                field_parts.append(f"{key}={value}")
            else:
                # String values need quotes and escaping
                value = str(value).replace('"', '\\"')
                field_parts.append(f'{key}="{value}"')
        
        line += " " + ",".join(field_parts)
        
        # Timestamp (nanoseconds)
        if timestamp is None:
            timestamp = int(time.time() * 1_000_000_000)
        line += f" {timestamp}"
        
        return line
    
    def export_task_metrics(self, task_data: Dict[str, Any]) -> List[str]:
        """Export task metrics in line protocol format"""
        lines = []
        timestamp = int(time.time() * 1_000_000_000)
        
        # Task execution metrics
        lines.append(self.format_line_protocol(
            "task_execution",
            tags={
                "task_type": task_data.get("type", "unknown"),
                "status": task_data.get("status", "unknown"),
                "agent_id": task_data.get("assigned_agent", "unknown")
            },
            fields={
                "execution_time": task_data.get("execution_time", 0.0),
                "retry_count": task_data.get("retry_count", 0),
                "success": task_data.get("status") == "completed"
            },
            timestamp=timestamp
        ))
        
        return lines
    
    def export_agent_metrics(self, agent_data: Dict[str, Any]) -> List[str]:
        """Export agent metrics in line protocol format"""
        lines = []
        timestamp = int(time.time() * 1_000_000_000)
        
        # Agent status metrics
        lines.append(self.format_line_protocol(
            "agent_status",
            tags={
                "agent_id": agent_data.get("id", "unknown"),
                "status": agent_data.get("status", "unknown")
            },
            fields={
                "is_busy": agent_data.get("status") == "busy",
                "capability_count": len(agent_data.get("capabilities", [])),
                "uptime": 0.0  # Would calculate from heartbeat
            },
            timestamp=timestamp
        ))
        
        return lines


class ElasticSearchExporter:
    """Export data to Elasticsearch format"""
    
    def __init__(self, index_prefix: str = "agent-orchestra"):
        self.index_prefix = index_prefix
    
    def format_document(self, doc_type: str, data: Dict[str, Any], 
                       doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Format document for Elasticsearch"""
        document = {
            "index": {
                "_index": f"{self.index_prefix}-{doc_type}-{datetime.utcnow().strftime('%Y-%m-%d')}",
                "_type": "_doc"
            }
        }
        
        if doc_id:
            document["index"]["_id"] = doc_id
        
        # Add common fields
        data["@timestamp"] = datetime.utcnow().isoformat()
        data["doc_type"] = doc_type
        
        return document, data
    
    def export_task_logs(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Export task logs for Elasticsearch bulk API"""
        lines = []
        
        for task in tasks:
            index_action, doc_data = self.format_document("task", task, task.get("id"))
            
            lines.append(json.dumps(index_action))
            lines.append(json.dumps(doc_data, default=str))
        
        return lines
    
    def export_agent_logs(self, agents: List[Dict[str, Any]]) -> List[str]:
        """Export agent logs for Elasticsearch bulk API"""
        lines = []
        
        for agent in agents:
            index_action, doc_data = self.format_document("agent", agent, agent.get("id"))
            
            lines.append(json.dumps(index_action))
            lines.append(json.dumps(doc_data, default=str))
        
        return lines


class MetricsAggregator:
    """Aggregate and transform metrics for export"""
    
    def __init__(self, window_size: int = 300):  # 5 minutes
        self.window_size = window_size
        self._data_points: List[Dict[str, Any]] = []
    
    def add_data_point(self, data: Dict[str, Any]):
        """Add a data point to the aggregation window"""
        data_point = {
            "timestamp": time.time(),
            "data": data
        }
        
        self._data_points.append(data_point)
        
        # Remove old data points
        cutoff_time = time.time() - self.window_size
        self._data_points = [
            dp for dp in self._data_points 
            if dp["timestamp"] > cutoff_time
        ]
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics over the window"""
        if not self._data_points:
            return {}
        
        # Extract numeric values for aggregation
        numeric_fields = {}
        
        for data_point in self._data_points:
            data = data_point["data"]
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_fields:
                        numeric_fields[key] = []
                    numeric_fields[key].append(value)
        
        # Calculate aggregations
        aggregated = {}
        
        for field, values in numeric_fields.items():
            if values:
                aggregated[f"{field}_avg"] = sum(values) / len(values)
                aggregated[f"{field}_min"] = min(values)
                aggregated[f"{field}_max"] = max(values)
                aggregated[f"{field}_sum"] = sum(values)
                aggregated[f"{field}_count"] = len(values)
        
        # Add metadata
        aggregated["window_size_seconds"] = self.window_size
        aggregated["data_points"] = len(self._data_points)
        aggregated["start_time"] = self._data_points[0]["timestamp"]
        aggregated["end_time"] = self._data_points[-1]["timestamp"]
        
        return aggregated


class ExportManager:
    """Manage multiple exporters and export schedules"""
    
    def __init__(self):
        self.exporters: Dict[str, Any] = {}
        self._export_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
    
    def register_exporter(self, name: str, exporter: Any, 
                         interval_seconds: int = 60):
        """Register an exporter with a schedule"""
        self.exporters[name] = {
            "exporter": exporter,
            "interval": interval_seconds,
            "last_export": 0
        }
        
        logger.info("Exporter registered", 
                   name=name, 
                   interval=interval_seconds)
    
    async def start_exports(self):
        """Start scheduled exports"""
        self._running = True
        
        for name, config in self.exporters.items():
            self._export_tasks[name] = asyncio.create_task(
                self._export_loop(name, config)
            )
        
        logger.info("Export manager started", exporters=list(self.exporters.keys()))
    
    async def stop_exports(self):
        """Stop all exports"""
        self._running = False
        
        for task in self._export_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        logger.info("Export manager stopped")
    
    async def _export_loop(self, name: str, config: Dict[str, Any]):
        """Export loop for a specific exporter"""
        exporter = config["exporter"]
        interval = config["interval"]
        
        while self._running:
            try:
                # TODO: Get metrics from orchestra and export
                # This would be called with actual metrics data
                
                config["last_export"] = time.time()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error("Export failed", 
                           exporter=name, 
                           error=str(e))
                await asyncio.sleep(min(interval, 60))  # Retry with backoff