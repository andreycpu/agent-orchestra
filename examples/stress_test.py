#!/usr/bin/env python3
"""
Stress test for Agent Orchestra - tests system under heavy load
"""
import asyncio
import time
import random
import argparse
from typing import Dict, Any
import structlog

from agent_orchestra import Orchestra, Agent
from agent_orchestra.types import TaskPriority

logger = structlog.get_logger(__name__)


class StressTestRunner:
    """Stress test runner for Agent Orchestra"""
    
    def __init__(self, num_agents: int = 5, num_tasks: int = 1000, 
                 concurrent_limit: int = 50):
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.concurrent_limit = concurrent_limit
        self.orchestra = None
        self.results = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_time': 0,
            'throughput': 0,
            'error_rate': 0
        }
    
    async def setup(self):
        """Setup orchestra and agents for stress test"""
        self.orchestra = Orchestra(
            max_concurrent_tasks=self.concurrent_limit,
            task_timeout_default=30
        )
        
        # Create agents with different capabilities
        for i in range(self.num_agents):
            agent = Agent(f"stress-agent-{i}", capabilities=[
                "cpu_intensive", "io_intensive", "mixed_workload"
            ])
            
            # Register handlers
            agent.register_task_handler("cpu_intensive", self.cpu_intensive_task)
            agent.register_task_handler("io_intensive", self.io_intensive_task)
            agent.register_task_handler("mixed_workload", self.mixed_workload_task)
            
            self.orchestra.register_agent(agent)
        
        await self.orchestra.start()
        logger.info("Stress test setup complete", agents=self.num_agents)
    
    async def teardown(self):
        """Clean up resources"""
        if self.orchestra:
            await self.orchestra.stop()
    
    async def cpu_intensive_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate CPU-intensive work"""
        iterations = data.get("iterations", 10000)
        
        # CPU-bound work
        result = 0
        for i in range(iterations):
            result += i * i
        
        return {"result": result, "iterations": iterations}
    
    async def io_intensive_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate I/O-intensive work"""
        delay = data.get("delay", 0.1)
        
        # Simulate I/O wait
        await asyncio.sleep(delay)
        
        return {"delay": delay, "timestamp": time.time()}
    
    async def mixed_workload_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mixed CPU and I/O workload"""
        cpu_work = data.get("cpu_work", 1000)
        io_delay = data.get("io_delay", 0.05)
        
        # CPU work
        result = sum(i for i in range(cpu_work))
        
        # I/O work
        await asyncio.sleep(io_delay)
        
        return {"cpu_result": result, "io_delay": io_delay}
    
    async def run_stress_test(self) -> Dict[str, Any]:
        """Run the main stress test"""
        logger.info("Starting stress test", 
                   tasks=self.num_tasks, 
                   agents=self.num_agents)
        
        start_time = time.time()
        task_ids = []
        
        # Submit all tasks
        for i in range(self.num_tasks):
            task_type = random.choice(["cpu_intensive", "io_intensive", "mixed_workload"])
            
            # Vary task parameters
            if task_type == "cpu_intensive":
                data = {"iterations": random.randint(1000, 5000)}
            elif task_type == "io_intensive":
                data = {"delay": random.uniform(0.01, 0.2)}
            else:  # mixed_workload
                data = {
                    "cpu_work": random.randint(100, 1000),
                    "io_delay": random.uniform(0.01, 0.1)
                }
            
            priority = random.choice(list(TaskPriority))
            
            task_id = await self.orchestra.submit_task({
                "type": task_type,
                "data": data,
                "priority": priority
            })
            task_ids.append(task_id)
            
            # Small delay to avoid overwhelming the system
            if i % 100 == 0:
                await asyncio.sleep(0.1)
                logger.info("Submitted tasks", count=i+1, total=self.num_tasks)
        
        logger.info("All tasks submitted, waiting for completion")
        
        # Wait for all tasks to complete
        results = []
        for i, task_id in enumerate(task_ids):
            try:
                result = await self.orchestra.wait_for_task(task_id, timeout=60)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info("Tasks completed", count=i+1, total=len(task_ids))
                    
            except asyncio.TimeoutError:
                logger.warning("Task timed out", task_id=task_id)
                results.append(None)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate results
        successful = sum(1 for r in results if r and r.success)
        failed = len(results) - successful
        throughput = successful / total_time if total_time > 0 else 0
        error_rate = failed / len(results) if results else 0
        
        self.results = {
            'total_tasks': len(results),
            'successful_tasks': successful,
            'failed_tasks': failed,
            'total_time': total_time,
            'throughput': throughput,
            'error_rate': error_rate,
            'avg_execution_time': sum(r.execution_time for r in results if r and r.success) / successful if successful > 0 else 0
        }
        
        return self.results
    
    async def run_load_ramp_test(self) -> Dict[str, Any]:
        """Run load ramp test - gradually increase load"""
        logger.info("Starting load ramp test")
        
        ramp_stages = [10, 25, 50, 100, 200, 500]
        stage_results = []
        
        for stage_tasks in ramp_stages:
            if stage_tasks > self.num_tasks:
                break
            
            logger.info("Running ramp stage", tasks=stage_tasks)
            
            start_time = time.time()
            task_ids = []
            
            # Submit tasks for this stage
            for i in range(stage_tasks):
                task_id = await self.orchestra.submit_task({
                    "type": "mixed_workload",
                    "data": {"cpu_work": 500, "io_delay": 0.05}
                })
                task_ids.append(task_id)
            
            # Wait for completion
            results = []
            for task_id in task_ids:
                try:
                    result = await self.orchestra.wait_for_task(task_id, timeout=30)
                    results.append(result)
                except asyncio.TimeoutError:
                    results.append(None)
            
            duration = time.time() - start_time
            successful = sum(1 for r in results if r and r.success)
            throughput = successful / duration if duration > 0 else 0
            
            stage_result = {
                'stage_tasks': stage_tasks,
                'successful': successful,
                'duration': duration,
                'throughput': throughput
            }
            stage_results.append(stage_result)
            
            logger.info("Stage completed", **stage_result)
            
            # Brief cooldown between stages
            await asyncio.sleep(2)
        
        return {'ramp_stages': stage_results}


async def main():
    """Main stress test function"""
    parser = argparse.ArgumentParser(description="Agent Orchestra Stress Test")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--tasks", type=int, default=1000, help="Number of tasks")
    parser.add_argument("--concurrent", type=int, default=50, help="Concurrent task limit")
    parser.add_argument("--test-type", choices=["stress", "ramp"], default="stress", 
                       help="Type of test to run")
    
    args = parser.parse_args()
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Run stress test
    runner = StressTestRunner(
        num_agents=args.agents,
        num_tasks=args.tasks,
        concurrent_limit=args.concurrent
    )
    
    try:
        await runner.setup()
        
        if args.test_type == "stress":
            results = await runner.run_stress_test()
        else:
            results = await runner.run_load_ramp_test()
        
        # Print results
        print("\n" + "="*50)
        print("🎭 STRESS TEST RESULTS")
        print("="*50)
        
        if args.test_type == "stress":
            print(f"Total Tasks: {results['total_tasks']}")
            print(f"Successful: {results['successful_tasks']}")
            print(f"Failed: {results['failed_tasks']}")
            print(f"Total Time: {results['total_time']:.2f}s")
            print(f"Throughput: {results['throughput']:.2f} tasks/sec")
            print(f"Error Rate: {results['error_rate']:.2%}")
            print(f"Avg Execution Time: {results['avg_execution_time']:.3f}s")
        else:
            print("Load Ramp Test Results:")
            for stage in results['ramp_stages']:
                print(f"  {stage['stage_tasks']} tasks: {stage['throughput']:.2f} tasks/sec")
        
        print("\n💡 Performance Tips:")
        print("- Increase concurrent_limit for higher throughput")
        print("- Monitor system resources during test")
        print("- Use Redis for distributed setups")
        print("- Profile task handlers for optimization")
        
    finally:
        await runner.teardown()


if __name__ == "__main__":
    asyncio.run(main())