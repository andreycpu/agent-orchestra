"""
Health check system for Agent Orchestra
"""
import asyncio
import time
import psutil
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog

from .exceptions import ConfigurationError, NetworkError

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status levels"""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "execution_time": self.execution_time
        }


class HealthCheck:
    """Base class for health checks"""
    
    def __init__(self, name: str, timeout: float = 5.0, critical: bool = False):
        self.name = name
        self.timeout = timeout
        self.critical = critical  # If True, failure affects overall health
    
    async def check(self) -> HealthCheckResult:
        """Perform the health check"""
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(self._perform_check(), timeout=self.timeout)
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.FAIL,
                message=f"Health check timed out after {self.timeout}s",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.FAIL,
                message=f"Health check failed: {str(e)}",
                execution_time=execution_time
            )
    
    async def _perform_check(self) -> HealthCheckResult:
        """Implement the actual health check logic"""
        raise NotImplementedError


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity"""
    
    def __init__(self, connection_factory: Callable, name: str = "database"):
        super().__init__(name, timeout=10.0, critical=True)
        self.connection_factory = connection_factory
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check database connectivity"""
        try:
            async with self.connection_factory() as connection:
                if hasattr(connection, 'execute'):
                    # SQL database
                    await connection.execute("SELECT 1")
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.PASS,
                        message="Database connection successful"
                    )
                else:
                    # Redis or other NoSQL
                    await connection.ping()
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.PASS,
                        message="Database connection successful"
                    )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.FAIL,
                message=f"Database connection failed: {str(e)}"
            )


class RedisHealthCheck(HealthCheck):
    """Health check for Redis connectivity"""
    
    def __init__(self, redis_factory: Callable, name: str = "redis"):
        super().__init__(name, timeout=5.0, critical=True)
        self.redis_factory = redis_factory
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check Redis connectivity and basic operations"""
        try:
            redis = await self.redis_factory()
            
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            
            # Set a value
            await redis.set(test_key, "health_test", ex=60)
            
            # Get the value back
            value = await redis.get(test_key)
            
            # Clean up
            await redis.delete(test_key)
            
            if value == b"health_test":
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.PASS,
                    message="Redis connection and operations successful",
                    details={"operations_tested": ["SET", "GET", "DELETE"]}
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.FAIL,
                    message="Redis operations failed: value mismatch"
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.FAIL,
                message=f"Redis health check failed: {str(e)}"
            )


class SystemResourcesHealthCheck(HealthCheck):
    """Health check for system resources (CPU, memory, disk)"""
    
    def __init__(self, 
                 cpu_threshold: float = 90.0,
                 memory_threshold: float = 90.0,
                 disk_threshold: float = 90.0,
                 name: str = "system_resources"):
        super().__init__(name, timeout=5.0, critical=False)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check system resource utilization"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "thresholds": {
                    "cpu": self.cpu_threshold,
                    "memory": self.memory_threshold,
                    "disk": self.disk_threshold
                }
            }
            
            # Determine status
            if (cpu_percent > self.cpu_threshold or 
                memory_percent > self.memory_threshold or 
                disk_percent > self.disk_threshold):
                
                status = HealthStatus.FAIL
                issues = []
                if cpu_percent > self.cpu_threshold:
                    issues.append(f"CPU usage high: {cpu_percent:.1f}%")
                if memory_percent > self.memory_threshold:
                    issues.append(f"Memory usage high: {memory_percent:.1f}%")
                if disk_percent > self.disk_threshold:
                    issues.append(f"Disk usage high: {disk_percent:.1f}%")
                
                message = "System resources critical: " + ", ".join(issues)
                
            elif (cpu_percent > self.cpu_threshold * 0.8 or 
                  memory_percent > self.memory_threshold * 0.8 or 
                  disk_percent > self.disk_threshold * 0.8):
                
                status = HealthStatus.WARN
                message = "System resources elevated"
                
            else:
                status = HealthStatus.PASS
                message = "System resources normal"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.FAIL,
                message=f"System resource check failed: {str(e)}"
            )


class AgentHealthCheck(HealthCheck):
    """Health check for agent connectivity"""
    
    def __init__(self, agent_registry: Dict, name: str = "agents"):
        super().__init__(name, timeout=10.0, critical=True)
        self.agent_registry = agent_registry
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check agent health and availability"""
        try:
            total_agents = len(self.agent_registry)
            if total_agents == 0:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.WARN,
                    message="No agents registered",
                    details={"total_agents": 0}
                )
            
            healthy_agents = 0
            unhealthy_agents = []
            
            for agent_id, agent_info in self.agent_registry.items():
                # Check last heartbeat
                if hasattr(agent_info, 'last_heartbeat'):
                    time_since_heartbeat = datetime.utcnow() - agent_info.last_heartbeat
                    if time_since_heartbeat < timedelta(minutes=2):
                        healthy_agents += 1
                    else:
                        unhealthy_agents.append({
                            "agent_id": agent_id,
                            "last_heartbeat": agent_info.last_heartbeat.isoformat(),
                            "time_since_heartbeat": time_since_heartbeat.total_seconds()
                        })
                else:
                    unhealthy_agents.append({
                        "agent_id": agent_id,
                        "issue": "no_heartbeat_info"
                    })
            
            health_ratio = healthy_agents / total_agents
            
            details = {
                "total_agents": total_agents,
                "healthy_agents": healthy_agents,
                "unhealthy_agents": len(unhealthy_agents),
                "health_ratio": health_ratio,
                "unhealthy_details": unhealthy_agents[:5]  # Limit details
            }
            
            if health_ratio < 0.5:
                status = HealthStatus.FAIL
                message = f"Less than 50% of agents healthy ({healthy_agents}/{total_agents})"
            elif health_ratio < 0.8:
                status = HealthStatus.WARN
                message = f"Some agents unhealthy ({healthy_agents}/{total_agents})"
            else:
                status = HealthStatus.PASS
                message = f"All agents healthy ({healthy_agents}/{total_agents})"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.FAIL,
                message=f"Agent health check failed: {str(e)}"
            )


class CustomHealthCheck(HealthCheck):
    """Custom health check with user-provided function"""
    
    def __init__(self, name: str, check_function: Callable[[], Awaitable[HealthCheckResult]], 
                 timeout: float = 5.0, critical: bool = False):
        super().__init__(name, timeout, critical)
        self.check_function = check_function
    
    async def _perform_check(self) -> HealthCheckResult:
        """Execute custom health check function"""
        return await self.check_function()


class HealthCheckManager:
    """Manages and coordinates health checks"""
    
    def __init__(self):
        self._health_checks: Dict[str, HealthCheck] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._health_check_history: List[Dict[str, Any]] = []
        self.max_history = 100
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check"""
        self._health_checks[health_check.name] = health_check
        logger.info(f"Health check registered: {health_check.name}")
    
    def unregister_health_check(self, name: str):
        """Unregister a health check"""
        if name in self._health_checks:
            del self._health_checks[name]
            logger.info(f"Health check unregistered: {name}")
    
    async def run_health_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if name not in self._health_checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.FAIL,
                message=f"Health check '{name}' not found"
            )
        
        result = await self._health_checks[name].check()
        self._last_results[name] = result
        return result
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        # Run all health checks concurrently
        tasks = {
            name: self.run_health_check(name)
            for name in self._health_checks.keys()
        }
        
        completed_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for name, result in zip(tasks.keys(), completed_tasks):
            if isinstance(result, Exception):
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.FAIL,
                    message=f"Health check exception: {str(result)}"
                )
            else:
                results[name] = result
        
        # Store history
        self._store_health_check_history(results)
        
        return results
    
    def _store_health_check_history(self, results: Dict[str, HealthCheckResult]):
        """Store health check results in history"""
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "results": {name: result.to_dict() for name, result in results.items()}
        }
        
        self._health_check_history.append(history_entry)
        
        # Trim history to max size
        if len(self._health_check_history) > self.max_history:
            self._health_check_history = self._health_check_history[-self.max_history:]
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self._last_results:
            return {
                "status": HealthStatus.WARN.value,
                "message": "No health checks have been run",
                "details": {}
            }
        
        critical_failures = []
        warnings = []
        passes = []
        
        for name, result in self._last_results.items():
            health_check = self._health_checks.get(name)
            
            if result.status == HealthStatus.FAIL:
                if health_check and health_check.critical:
                    critical_failures.append(name)
                else:
                    warnings.append(name)
            elif result.status == HealthStatus.WARN:
                warnings.append(name)
            else:
                passes.append(name)
        
        # Determine overall status
        if critical_failures:
            overall_status = HealthStatus.FAIL
            message = f"Critical health checks failing: {', '.join(critical_failures)}"
        elif warnings:
            overall_status = HealthStatus.WARN
            message = f"Some health checks have warnings: {', '.join(warnings)}"
        else:
            overall_status = HealthStatus.PASS
            message = "All health checks passing"
        
        return {
            "status": overall_status.value,
            "message": message,
            "details": {
                "total_checks": len(self._last_results),
                "passing": len(passes),
                "warnings": len(warnings),
                "critical_failures": len(critical_failures),
                "last_check": max(
                    (result.timestamp for result in self._last_results.values()),
                    default=datetime.utcnow()
                ).isoformat()
            },
            "checks": {name: result.to_dict() for name, result in self._last_results.items()}
        }
    
    def get_health_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get health check history"""
        history = self._health_check_history
        if limit:
            history = history[-limit:]
        return history
    
    def clear_history(self):
        """Clear health check history"""
        self._health_check_history.clear()
        logger.info("Health check history cleared")