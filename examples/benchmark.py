"""
Performance benchmark for Agent Orchestra
"""
import asyncio
import time
import statistics
from typing import List, Dict, Any
import structlog
from agent_orchestra import Orchestra, Agent
from agent_orchestra.types import TaskPriority

logger = structlog.get_logger(__name__)


class BenchmarkSuite:
    """Comprehensive performance benchmarks for Agent Orchestra"""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        
    async def setup_orchestra(self, num_agents: int = 5) -> Orchestra:
        """Setup orchestra with test agents"""
        orchestra = Orchestra(max_concurrent_tasks=50)
        
        # Create diverse agent pool
        for i in range(num_agents):
            agent = Agent(
                f"benchmark-agent-{i}",
                capabilities=["cpu_task", "io_task", "mixed_task"]
            )
            
            # Register task handlers
            agent.register_task_handler("cpu_task", self.cpu_intensive_task)
            agent.register_task_handler("io_task", self.io_intensive_task)  
            agent.register_task_handler("mixed_task", self.mixed_task)
            
            orchestra.register_agent(agent)
        
        await orchestra.start()
        return orchestra
    
    async def cpu_intensive_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate CPU-intensive computation"""
        iterations = data.get("iterations", 10000)
        
        # Fibonacci calculation to simulate CPU work
        result = 0
        for i in range(iterations):
            result += i * i
        
        return {"result": result, "iterations": iterations}
    
    async def io_intensive_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate I/O-intensive operation"""
        delay = data.get("delay", 0.1)
        
        # Simulate network/disk I/O
        await asyncio.sleep(delay)
        
        return {"delay": delay, "timestamp": time.time()}
    
    async def mixed_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mixed CPU and I/O task"""
        cpu_work = data.get("cpu_work", 1000)
        io_delay = data.get("io_delay", 0.05)
        
        # CPU work
        result = sum(i * i for i in range(cpu_work))
        
        # I/O work
        await asyncio.sleep(io_delay)
        
        return {"cpu_result": result, "io_delay": io_delay}
    
    async def benchmark_throughput(self, orchestra: Orchestra, num_tasks: int = 100):
        """Benchmark task throughput"""
        logger.info("Starting throughput benchmark", num_tasks=num_tasks)
        
        start_time = time.time()
        
        # Submit tasks
        task_ids = []
        for i in range(num_tasks):
            task_id = await orchestra.submit_task({
                "type": "cpu_task",
                "data": {"iterations": 1000},
                "priority": "normal"
            })
            task_ids.append(task_id)
        
        # Wait for completion
        results = []
        for task_id in task_ids:
            result = await orchestra.wait_for_task(task_id, timeout=30)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        throughput = successful / total_time
        
        self.results["throughput"] = {
            "total_tasks": num_tasks,
            "successful": successful,
            "failed": failed,
            "total_time": total_time,
            "throughput": throughput,
            "avg_latency": statistics.mean([r.execution_time for r in results if r.success])
        }
        
        logger.info("Throughput benchmark completed",
                   throughput=f"{throughput:.2f} tasks/sec",
                   total_time=f"{total_time:.2f}s")
    
    async def benchmark_latency(self, orchestra: Orchestra, num_samples: int = 50):
        """Benchmark task latency distribution"""
        logger.info("Starting latency benchmark", num_samples=num_samples)
        
        latencies = []
        
        for i in range(num_samples):
            start_time = time.time()
            
            task_id = await orchestra.submit_task({
                "type": "io_task",
                "data": {"delay": 0.1},
                "priority": "normal"
            })
            
            result = await orchestra.wait_for_task(task_id, timeout=10)
            
            if result.success:
                end_time = time.time()
                latencies.append(end_time - start_time)
            
            # Small delay between requests
            await asyncio.sleep(0.01)
        
        self.results["latency"] = {
            "samples": len(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": self._percentile(latencies, 95),
            "p99": self._percentile(latencies, 99),
            "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
        
        logger.info("Latency benchmark completed",
                   mean_latency=f"{self.results['latency']['mean']:.3f}s")
    
    async def benchmark_concurrency(self, orchestra: Orchestra, max_concurrent: int = 20):
        """Benchmark concurrent task handling"""
        logger.info("Starting concurrency benchmark", max_concurrent=max_concurrent)
        
        concurrency_results = []
        
        for concurrency_level in [1, 5, 10, 15, max_concurrent]:
            logger.info("Testing concurrency level", level=concurrency_level)
            
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = []
            for i in range(concurrency_level):
                task_coroutine = self._submit_and_wait_task(
                    orchestra,
                    "mixed_task",
                    {"cpu_work": 500, "io_delay": 0.1}
                )
                tasks.append(task_coroutine)
            
            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            successful = sum(1 for r in results if not isinstance(r, Exception) and r.success)
            
            concurrency_results.append({
                "concurrency_level": concurrency_level,
                "total_time": total_time,
                "successful": successful,
                "failed": concurrency_level - successful,
                "throughput": successful / total_time
            })
        
        self.results["concurrency"] = concurrency_results
        
        logger.info("Concurrency benchmark completed")
    
    async def benchmark_priority_handling(self, orchestra: Orchestra):
        """Benchmark priority-based task scheduling"""
        logger.info("Starting priority benchmark")
        
        # Submit tasks with different priorities
        task_submissions = []
        priorities = ["low", "normal", "high", "urgent"]
        
        start_time = time.time()
        
        # Submit in reverse priority order
        for i, priority in enumerate(["low", "normal", "high", "urgent"] * 5):
            task_id = await orchestra.submit_task({
                "type": "cpu_task",
                "data": {"iterations": 100, "task_index": i},
                "priority": priority
            })
            task_submissions.append((task_id, priority, time.time()))
            
            # Small delay to ensure ordering
            await asyncio.sleep(0.001)
        
        # Wait for all tasks to complete
        completion_times = []
        for task_id, priority, submit_time in task_submissions:
            result = await orchestra.wait_for_task(task_id, timeout=30)
            if result.success:
                completion_times.append((priority, time.time() - submit_time))
        
        # Analyze priority handling
        priority_stats = {}
        for priority in priorities:
            priority_times = [t for p, t in completion_times if p == priority]
            if priority_times:
                priority_stats[priority] = {
                    "count": len(priority_times),
                    "avg_completion_time": statistics.mean(priority_times),
                    "min_completion_time": min(priority_times)
                }
        
        self.results["priority_handling"] = priority_stats
        
        logger.info("Priority benchmark completed")
    
    async def benchmark_failure_recovery(self, orchestra: Orchestra):
        """Benchmark failure handling and recovery"""
        logger.info("Starting failure recovery benchmark")
        
        # Create a failing task handler
        async def failing_task(data):
            failure_rate = data.get("failure_rate", 0.5)
            if time.time() % 1 < failure_rate:
                raise Exception("Simulated failure")
            return {"success": True}
        
        # Add failing capability to one agent
        agents = orchestra.get_agents()
        if agents:
            agent = orchestra._agents[agents[0].id]
            agent.register_task_handler("failing_task", failing_task)
        
        # Submit tasks that will fail
        task_ids = []
        for i in range(20):
            task_id = await orchestra.submit_task({
                "type": "failing_task",
                "data": {"failure_rate": 0.3},
                "priority": "normal",
                "max_retries": 2
            })
            task_ids.append(task_id)
        
        # Wait for completion
        results = []
        for task_id in task_ids:
            result = await orchestra.wait_for_task(task_id, timeout=30)
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        self.results["failure_recovery"] = {
            "total_tasks": len(task_ids),
            "successful_after_retry": successful,
            "permanently_failed": failed,
            "recovery_rate": successful / len(task_ids)
        }
        
        logger.info("Failure recovery benchmark completed",
                   recovery_rate=f"{self.results['failure_recovery']['recovery_rate']:.2%}")
    
    async def _submit_and_wait_task(self, orchestra: Orchestra, task_type: str, data: Dict[str, Any]):
        """Helper to submit and wait for a task"""
        task_id = await orchestra.submit_task({
            "type": task_type,
            "data": data,
            "priority": "normal"
        })
        return await orchestra.wait_for_task(task_id, timeout=30)
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def print_results(self):
        """Print benchmark results in a formatted way"""
        print("\n" + "=" * 60)
        print("🎭 AGENT ORCHESTRA PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        
        if "throughput" in self.results:
            tp = self.results["throughput"]
            print(f"\n📊 THROUGHPUT BENCHMARK")
            print(f"Tasks: {tp['successful']}/{tp['total_tasks']} successful")
            print(f"Time: {tp['total_time']:.2f}s")
            print(f"Throughput: {tp['throughput']:.2f} tasks/sec")
            print(f"Average latency: {tp['avg_latency']:.3f}s")
        
        if "latency" in self.results:
            lat = self.results["latency"]
            print(f"\n⏱️ LATENCY BENCHMARK")
            print(f"Samples: {lat['samples']}")
            print(f"Mean: {lat['mean']:.3f}s")
            print(f"Median: {lat['median']:.3f}s")
            print(f"P95: {lat['p95']:.3f}s")
            print(f"P99: {lat['p99']:.3f}s")
            print(f"Min/Max: {lat['min']:.3f}s / {lat['max']:.3f}s")
        
        if "concurrency" in self.results:
            print(f"\n🔄 CONCURRENCY BENCHMARK")
            for result in self.results["concurrency"]:
                print(f"Level {result['concurrency_level']:2d}: "
                      f"{result['throughput']:.2f} tasks/sec "
                      f"({result['successful']}/{result['concurrency_level']} successful)")
        
        if "priority_handling" in self.results:
            print(f"\n🎯 PRIORITY HANDLING BENCHMARK")
            for priority, stats in self.results["priority_handling"].items():
                print(f"{priority.upper():>8}: "
                      f"avg {stats['avg_completion_time']:.3f}s "
                      f"(min {stats['min_completion_time']:.3f}s)")
        
        if "failure_recovery" in self.results:
            fr = self.results["failure_recovery"]
            print(f"\n🛠️ FAILURE RECOVERY BENCHMARK")
            print(f"Recovery rate: {fr['recovery_rate']:.1%}")
            print(f"Successful after retry: {fr['successful_after_retry']}")
            print(f"Permanently failed: {fr['permanently_failed']}")
        
        print("\n" + "=" * 60)


async def run_full_benchmark():
    """Run complete benchmark suite"""
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
    
    benchmark = BenchmarkSuite()
    orchestra = await benchmark.setup_orchestra(num_agents=3)
    
    try:
        logger.info("Starting comprehensive benchmark suite")
        
        # Run all benchmarks
        await benchmark.benchmark_throughput(orchestra, num_tasks=50)
        await benchmark.benchmark_latency(orchestra, num_samples=30)
        await benchmark.benchmark_concurrency(orchestra, max_concurrent=15)
        await benchmark.benchmark_priority_handling(orchestra)
        await benchmark.benchmark_failure_recovery(orchestra)
        
        # Print results
        benchmark.print_results()
        
    finally:
        await orchestra.stop()


if __name__ == "__main__":
    asyncio.run(run_full_benchmark())