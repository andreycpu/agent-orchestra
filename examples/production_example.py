"""
Production-ready example of Agent Orchestra usage
"""
import asyncio
import logging
import signal
from typing import Dict, Any
from datetime import datetime

from agent_orchestra import (
    Orchestra, Agent, get_config, setup_logging, 
    HealthCheckManager, ErrorRecoveryManager, PerformanceProfiler,
    DatabaseHealthCheck, SystemResourcesHealthCheck, RedisHealthCheck
)
from agent_orchestra.types import Task, TaskPriority


class ProductionOrchestra:
    """Production-ready Orchestra wrapper with full monitoring and recovery"""
    
    def __init__(self):
        self.config = None
        self.orchestra = None
        self.health_manager = None
        self.recovery_manager = None
        self.profiler = None
        self.agents = []
        self.shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize all components"""
        # Load configuration
        self.config = get_config()
        
        # Setup structured logging
        setup_logging(
            level=self.config.monitoring.logging.level,
            format_type=self.config.monitoring.logging.format,
            enable_colors=not self.config.is_production()
        )
        
        logger = logging.getLogger(__name__)
        logger.info("Initializing production orchestra", environment=self.config.environment)
        
        # Initialize performance profiler
        self.profiler = PerformanceProfiler(enabled=self.config.debug)
        
        # Initialize error recovery manager
        self.recovery_manager = ErrorRecoveryManager()
        
        # Setup circuit breakers
        self.recovery_manager.register_circuit_breaker(
            "redis_operations", 
            failure_threshold=5, 
            timeout=60.0
        )
        
        # Initialize orchestra
        self.orchestra = Orchestra(
            redis_url=self.config.redis.url,
            max_concurrent_tasks=self.config.orchestra.max_concurrent_tasks
        )
        
        # Initialize health checks
        self.health_manager = HealthCheckManager()
        
        # Register database health check
        if hasattr(self.config, 'database'):
            db_check = DatabaseHealthCheck(
                connection_factory=self._create_db_connection,
                name="database"
            )
            self.health_manager.register_health_check(db_check)
        
        # Register Redis health check
        redis_check = RedisHealthCheck(
            redis_factory=self._create_redis_connection,
            name="redis"
        )
        self.health_manager.register_health_check(redis_check)
        
        # Register system resources health check
        system_check = SystemResourcesHealthCheck(
            cpu_threshold=80.0,
            memory_threshold=85.0,
            disk_threshold=90.0,
            name="system_resources"
        )
        self.health_manager.register_health_check(system_check)
        
        logger.info("Production orchestra initialized successfully")
    
    async def _create_db_connection(self):
        """Create database connection"""
        # Placeholder - implement based on your database
        pass
    
    async def _create_redis_connection(self):
        """Create Redis connection"""
        import aioredis
        return await aioredis.from_url(self.config.redis.url)
    
    async def register_sample_agents(self):
        """Register sample agents for demonstration"""
        logger = logging.getLogger(__name__)
        
        # Text processing agent
        text_agent = Agent(
            agent_id="text_processor_1",
            name="Text Processing Agent",
            capabilities=["text_analysis", "summarization", "translation"]
        )
        
        @self.profiler.time_function("text_processing", threshold=5.0)
        async def process_text(task_data: Dict[str, Any]) -> Dict[str, Any]:
            """Sample text processing function"""
            await asyncio.sleep(0.1)  # Simulate processing
            text = task_data.get("text", "")
            
            # Simple text analysis
            result = {
                "word_count": len(text.split()),
                "char_count": len(text),
                "processed_at": datetime.utcnow().isoformat(),
                "agent_id": "text_processor_1"
            }
            
            return result
        
        text_agent.register_handler("text_analysis", process_text)
        
        # Data processing agent
        data_agent = Agent(
            agent_id="data_processor_1", 
            name="Data Processing Agent",
            capabilities=["data_transformation", "validation", "enrichment"]
        )
        
        @self.profiler.time_function("data_processing", threshold=3.0)
        async def process_data(task_data: Dict[str, Any]) -> Dict[str, Any]:
            """Sample data processing function"""
            await asyncio.sleep(0.05)  # Simulate processing
            
            data = task_data.get("data", {})
            
            # Simple data transformation
            result = {
                "processed_data": {k: str(v).upper() if isinstance(v, str) else v 
                                 for k, v in data.items()},
                "validation_status": "valid",
                "processed_at": datetime.utcnow().isoformat(),
                "agent_id": "data_processor_1"
            }
            
            return result
        
        data_agent.register_handler("data_transformation", process_data)
        
        # Register agents with orchestra
        self.orchestra.register_agent(text_agent)
        self.orchestra.register_agent(data_agent)
        
        self.agents = [text_agent, data_agent]
        
        logger.info(f"Registered {len(self.agents)} sample agents")
    
    async def start(self):
        """Start the production orchestra"""
        logger = logging.getLogger(__name__)
        
        try:
            # Initialize components
            await self.initialize()
            
            # Start orchestra
            async with self.orchestra:
                logger.info("Orchestra started successfully")
                
                # Register sample agents
                await self.register_sample_agents()
                
                # Start health monitoring
                asyncio.create_task(self.health_monitoring_loop())
                
                # Start performance monitoring
                asyncio.create_task(self.performance_monitoring_loop())
                
                # Run example workload
                await self.run_example_workload()
                
                # Wait for shutdown signal
                logger.info("Orchestra running. Press Ctrl+C to shutdown...")
                await self.shutdown_event.wait()
                
        except Exception as e:
            logger.error(f"Failed to start orchestra: {e}")
            raise
        finally:
            logger.info("Orchestra shutdown complete")
    
    async def health_monitoring_loop(self):
        """Background health monitoring"""
        logger = logging.getLogger(__name__)
        
        while not self.shutdown_event.is_set():
            try:
                # Run health checks every 30 seconds
                await asyncio.sleep(30)
                
                results = await self.health_manager.run_all_health_checks()
                overall_health = self.health_manager.get_overall_health()
                
                if overall_health["status"] != "pass":
                    logger.warning(
                        "Health check issues detected",
                        status=overall_health["status"],
                        message=overall_health["message"]
                    )
                else:
                    logger.debug("All health checks passing")
                    
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def performance_monitoring_loop(self):
        """Background performance monitoring"""
        logger = logging.getLogger(__name__)
        
        while not self.shutdown_event.is_set():
            try:
                # Log performance stats every 60 seconds
                await asyncio.sleep(60)
                
                if self.profiler.enabled:
                    stats = self.profiler.export_stats()
                    
                    logger.info(
                        "Performance statistics",
                        total_functions=stats["system_metrics"]["total_functions_tracked"],
                        total_calls=stats["system_metrics"]["total_function_calls"],
                        calls_per_second=stats["system_metrics"]["calls_per_second"],
                        slowest_function=stats["slowest_functions"][0]["name"] if stats["slowest_functions"] else None
                    )
                    
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def run_example_workload(self):
        """Run example tasks to demonstrate the system"""
        logger = logging.getLogger(__name__)
        
        try:
            # Submit some example tasks
            tasks = []
            
            # Text analysis tasks
            for i in range(5):
                task = Task(
                    type="text_analysis",
                    data={
                        "text": f"This is sample text number {i} for analysis. " * (i + 1)
                    },
                    priority=TaskPriority.NORMAL
                )
                task_id = await self.orchestra.submit_task(task.__dict__)
                tasks.append(task_id)
            
            # Data transformation tasks
            for i in range(3):
                task = Task(
                    type="data_transformation",
                    data={
                        "data": {
                            "name": f"item_{i}",
                            "value": i * 10,
                            "description": f"Sample data item {i}"
                        }
                    },
                    priority=TaskPriority.HIGH
                )
                task_id = await self.orchestra.submit_task(task.__dict__)
                tasks.append(task_id)
            
            logger.info(f"Submitted {len(tasks)} example tasks")
            
            # Wait for tasks to complete
            completed_count = 0
            for task_id in tasks:
                try:
                    result = await self.orchestra.wait_for_task(task_id, timeout=30.0)
                    if result.success:
                        completed_count += 1
                        logger.debug(f"Task {task_id} completed successfully")
                    else:
                        logger.warning(f"Task {task_id} failed: {result.error}")
                except Exception as e:
                    logger.error(f"Task {task_id} error: {e}")
            
            logger.info(f"Completed {completed_count}/{len(tasks)} tasks successfully")
            
        except Exception as e:
            logger.error(f"Example workload error: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger = logging.getLogger(__name__)
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    production_orchestra = ProductionOrchestra()
    
    # Setup signal handlers
    production_orchestra.setup_signal_handlers()
    
    # Start the orchestra
    await production_orchestra.start()


if __name__ == "__main__":
    asyncio.run(main())