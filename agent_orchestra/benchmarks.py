"""
Performance benchmarking utilities for Agent Orchestra
"""
import time
import asyncio
import statistics
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import multiprocessing
import gc
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import structlog

from .utils import calculate_memory_usage
from .exceptions import ValidationError

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results of a performance benchmark"""
    name: str
    iterations: int
    duration_seconds: float
    operations_per_second: float
    min_time_ms: float
    max_time_ms: float
    mean_time_ms: float
    median_time_ms: float
    std_dev_ms: float
    percentile_95_ms: float
    percentile_99_ms: float
    memory_usage_mb: Optional[float] = None
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceUsage:
    """Resource usage during benchmark"""
    cpu_percent: float
    memory_mb: float
    gc_collections: int
    file_descriptors: int
    thread_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PerformanceBenchmark:
    """Performance benchmark utility for measuring operation performance"""
    
    def __init__(self, name: str, warmup_iterations: int = 100):
        """Initialize performance benchmark
        
        Args:
            name: Name of the benchmark
            warmup_iterations: Number of warmup iterations before measurement
        """
        if not name:
            raise ValidationError("benchmark name cannot be empty")
        if warmup_iterations < 0:
            raise ValidationError("warmup_iterations cannot be negative")
            
        self.name = name
        self.warmup_iterations = warmup_iterations
        self.execution_times: List[float] = []
        self.errors: List[Exception] = []
        self.resource_snapshots: List[ResourceUsage] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        logger.info("Benchmark initialized", name=name, warmup_iterations=warmup_iterations)
    
    def run_sync(self, func: Callable, iterations: int = 1000, *args, **kwargs) -> BenchmarkResult:
        """Run synchronous benchmark
        
        Args:
            func: Function to benchmark
            iterations: Number of iterations to run
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            BenchmarkResult with performance metrics
        """
        if not callable(func):
            raise ValidationError("func must be callable")
        if iterations <= 0:
            raise ValidationError("iterations must be positive")
            
        logger.info("Starting synchronous benchmark", name=self.name, iterations=iterations)
        
        # Warmup phase
        logger.debug("Running warmup phase", warmup_iterations=self.warmup_iterations)
        for _ in range(self.warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning("Warmup iteration failed", error=str(e))
        
        # Force garbage collection before benchmarking
        gc.collect()
        
        # Start memory tracking if available
        memory_start = None
        if tracemalloc.is_tracing():
            memory_start = tracemalloc.get_traced_memory()
        
        self.execution_times.clear()
        self.errors.clear()
        self.start_time = time.time()
        
        # Benchmark phase
        for i in range(iterations):
            iteration_start = time.time()
            
            try:
                func(*args, **kwargs)
                iteration_end = time.time()
                self.execution_times.append(iteration_end - iteration_start)
            except Exception as e:
                self.errors.append(e)
                logger.debug("Benchmark iteration failed", iteration=i, error=str(e))
            
            # Collect resource snapshots periodically
            if i % max(1, iterations // 10) == 0:
                self._collect_resource_snapshot()
        
        self.end_time = time.time()
        
        # Calculate memory usage
        memory_usage_mb = None
        if memory_start and tracemalloc.is_tracing():
            memory_end = tracemalloc.get_traced_memory()
            memory_usage_mb = (memory_end[0] - memory_start[0]) / (1024 * 1024)
        
        result = self._calculate_results(iterations, memory_usage_mb)
        
        logger.info(
            "Benchmark completed",
            name=self.name,
            ops_per_second=result.operations_per_second,
            mean_time_ms=result.mean_time_ms,
            error_count=result.error_count
        )
        
        return result
    
    async def run_async(self, func: Callable, iterations: int = 1000, *args, **kwargs) -> BenchmarkResult:
        """Run asynchronous benchmark
        
        Args:
            func: Async function to benchmark
            iterations: Number of iterations to run
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            BenchmarkResult with performance metrics
        """
        if not callable(func):
            raise ValidationError("func must be callable")
        if iterations <= 0:
            raise ValidationError("iterations must be positive")
            
        logger.info("Starting asynchronous benchmark", name=self.name, iterations=iterations)
        
        # Warmup phase
        for _ in range(self.warmup_iterations):
            try:
                await func(*args, **kwargs)
            except Exception as e:
                logger.warning("Async warmup iteration failed", error=str(e))
        
        gc.collect()
        
        memory_start = None
        if tracemalloc.is_tracing():
            memory_start = tracemalloc.get_traced_memory()
        
        self.execution_times.clear()
        self.errors.clear()
        self.start_time = time.time()
        
        # Benchmark phase
        for i in range(iterations):
            iteration_start = time.time()
            
            try:
                await func(*args, **kwargs)
                iteration_end = time.time()
                self.execution_times.append(iteration_end - iteration_start)
            except Exception as e:
                self.errors.append(e)
                logger.debug("Async benchmark iteration failed", iteration=i, error=str(e))
            
            if i % max(1, iterations // 10) == 0:
                self._collect_resource_snapshot()
        
        self.end_time = time.time()
        
        memory_usage_mb = None
        if memory_start and tracemalloc.is_tracing():
            memory_end = tracemalloc.get_traced_memory()
            memory_usage_mb = (memory_end[0] - memory_start[0]) / (1024 * 1024)
        
        result = self._calculate_results(iterations, memory_usage_mb)
        
        logger.info(
            "Async benchmark completed",
            name=self.name,
            ops_per_second=result.operations_per_second,
            mean_time_ms=result.mean_time_ms
        )
        
        return result
    
    def run_concurrent(
        self, 
        func: Callable, 
        iterations: int = 1000,
        concurrency: int = 10,
        executor_type: str = "thread",
        *args, 
        **kwargs
    ) -> BenchmarkResult:
        """Run concurrent benchmark
        
        Args:
            func: Function to benchmark
            iterations: Total number of iterations
            concurrency: Number of concurrent workers
            executor_type: "thread" or "process"
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            BenchmarkResult with performance metrics
        """
        if executor_type not in ["thread", "process"]:
            raise ValidationError("executor_type must be 'thread' or 'process'")
        if concurrency <= 0:
            raise ValidationError("concurrency must be positive")
        if iterations <= 0:
            raise ValidationError("iterations must be positive")
            
        logger.info(
            "Starting concurrent benchmark",
            name=self.name,
            iterations=iterations,
            concurrency=concurrency,
            executor_type=executor_type
        )
        
        # Warmup
        for _ in range(min(self.warmup_iterations, 10)):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        gc.collect()
        
        executor_class = ThreadPoolExecutor if executor_type == "thread" else ProcessPoolExecutor
        
        self.execution_times.clear()
        self.errors.clear()
        self.start_time = time.time()
        
        with executor_class(max_workers=concurrency) as executor:
            # Submit all tasks
            futures = []
            for _ in range(iterations):
                future = executor.submit(self._timed_execution, func, args, kwargs)
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    execution_time = future.result(timeout=30)  # 30 second timeout
                    if execution_time is not None:
                        self.execution_times.append(execution_time)
                except Exception as e:
                    self.errors.append(e)
                
                if i % max(1, iterations // 10) == 0:
                    self._collect_resource_snapshot()
        
        self.end_time = time.time()
        
        result = self._calculate_results(iterations)
        
        logger.info(
            "Concurrent benchmark completed",
            name=self.name,
            concurrency=concurrency,
            ops_per_second=result.operations_per_second
        )
        
        return result
    
    def _timed_execution(self, func: Callable, args: tuple, kwargs: dict) -> Optional[float]:
        """Execute function with timing"""
        start_time = time.time()
        try:
            func(*args, **kwargs)
            return time.time() - start_time
        except Exception as e:
            logger.debug("Timed execution failed", error=str(e))
            return None
    
    def _collect_resource_snapshot(self):
        """Collect current resource usage snapshot"""
        try:
            memory_info = calculate_memory_usage()
            cpu_percent = 0  # Placeholder - would need psutil for accurate CPU
            
            resource_usage = ResourceUsage(
                cpu_percent=cpu_percent,
                memory_mb=memory_info.get("rss", 0) / (1024 * 1024),
                gc_collections=sum(gc.get_stats()),
                file_descriptors=0,  # Placeholder
                thread_count=threading.active_count()
            )
            
            self.resource_snapshots.append(resource_usage)
            
        except Exception as e:
            logger.debug("Failed to collect resource snapshot", error=str(e))
    
    def _calculate_results(self, iterations: int, memory_usage_mb: Optional[float] = None) -> BenchmarkResult:
        """Calculate benchmark results"""
        if not self.execution_times:
            # Handle case where all iterations failed
            return BenchmarkResult(
                name=self.name,
                iterations=iterations,
                duration_seconds=self.end_time - self.start_time if self.start_time and self.end_time else 0,
                operations_per_second=0,
                min_time_ms=0,
                max_time_ms=0,
                mean_time_ms=0,
                median_time_ms=0,
                std_dev_ms=0,
                percentile_95_ms=0,
                percentile_99_ms=0,
                memory_usage_mb=memory_usage_mb,
                error_count=len(self.errors)
            )
        
        duration = self.end_time - self.start_time
        successful_iterations = len(self.execution_times)
        
        # Convert to milliseconds for reporting
        times_ms = [t * 1000 for t in self.execution_times]
        
        # Calculate statistics
        min_time = min(times_ms)
        max_time = max(times_ms)
        mean_time = statistics.mean(times_ms)
        median_time = statistics.median(times_ms)
        std_dev = statistics.stdev(times_ms) if len(times_ms) > 1 else 0
        
        # Calculate percentiles
        sorted_times = sorted(times_ms)
        percentile_95 = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
        percentile_99 = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
        
        ops_per_second = successful_iterations / duration if duration > 0 else 0
        
        return BenchmarkResult(
            name=self.name,
            iterations=iterations,
            duration_seconds=duration,
            operations_per_second=ops_per_second,
            min_time_ms=min_time,
            max_time_ms=max_time,
            mean_time_ms=mean_time,
            median_time_ms=median_time,
            std_dev_ms=std_dev,
            percentile_95_ms=percentile_95,
            percentile_99_ms=percentile_99,
            memory_usage_mb=memory_usage_mb,
            error_count=len(self.errors)
        )


class BenchmarkSuite:
    """Suite for running multiple related benchmarks"""
    
    def __init__(self, name: str):
        """Initialize benchmark suite
        
        Args:
            name: Name of the benchmark suite
        """
        if not name:
            raise ValidationError("suite name cannot be empty")
            
        self.name = name
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.results: Dict[str, BenchmarkResult] = {}
        self.suite_metadata: Dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
            "python_version": "",
            "platform": ""
        }
        
        logger.info("Benchmark suite initialized", name=name)
    
    def add_benchmark(self, benchmark: PerformanceBenchmark):
        """Add benchmark to suite
        
        Args:
            benchmark: PerformanceBenchmark instance
        """
        if not isinstance(benchmark, PerformanceBenchmark):
            raise ValidationError("benchmark must be a PerformanceBenchmark instance")
            
        self.benchmarks[benchmark.name] = benchmark
        logger.debug("Benchmark added to suite", benchmark_name=benchmark.name, suite_name=self.name)
    
    def run_all(self) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks in the suite
        
        Returns:
            Dictionary mapping benchmark names to results
        """
        logger.info("Running benchmark suite", suite_name=self.name, benchmark_count=len(self.benchmarks))
        
        start_time = time.time()
        
        for name, benchmark in self.benchmarks.items():
            logger.info("Running benchmark", benchmark_name=name)
            
            try:
                # This would need to be implemented based on how benchmarks are configured
                # For now, we'll create a placeholder result
                result = BenchmarkResult(
                    name=name,
                    iterations=0,
                    duration_seconds=0,
                    operations_per_second=0,
                    min_time_ms=0,
                    max_time_ms=0,
                    mean_time_ms=0,
                    median_time_ms=0,
                    std_dev_ms=0,
                    percentile_95_ms=0,
                    percentile_99_ms=0
                )
                self.results[name] = result
                
            except Exception as e:
                logger.error("Benchmark failed", benchmark_name=name, error=str(e))
                # Create error result
                self.results[name] = BenchmarkResult(
                    name=name,
                    iterations=0,
                    duration_seconds=0,
                    operations_per_second=0,
                    min_time_ms=0,
                    max_time_ms=0,
                    mean_time_ms=0,
                    median_time_ms=0,
                    std_dev_ms=0,
                    percentile_95_ms=0,
                    percentile_99_ms=0,
                    error_count=1,
                    metadata={"error": str(e)}
                )
        
        total_duration = time.time() - start_time
        
        logger.info(
            "Benchmark suite completed",
            suite_name=self.name,
            total_duration_seconds=total_duration,
            successful_benchmarks=len([r for r in self.results.values() if r.error_count == 0])
        )
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark suite results
        
        Returns:
            Summary dictionary with suite statistics
        """
        if not self.results:
            return {"error": "No benchmark results available"}
        
        total_operations = sum(r.operations_per_second for r in self.results.values())
        avg_ops_per_second = total_operations / len(self.results) if self.results else 0
        
        successful_benchmarks = [r for r in self.results.values() if r.error_count == 0]
        failed_benchmarks = [r for r in self.results.values() if r.error_count > 0]
        
        if successful_benchmarks:
            avg_latency = statistics.mean([r.mean_time_ms for r in successful_benchmarks])
            max_latency = max([r.max_time_ms for r in successful_benchmarks])
            min_latency = min([r.min_time_ms for r in successful_benchmarks])
        else:
            avg_latency = max_latency = min_latency = 0
        
        return {
            "suite_name": self.name,
            "total_benchmarks": len(self.results),
            "successful_benchmarks": len(successful_benchmarks),
            "failed_benchmarks": len(failed_benchmarks),
            "total_operations_per_second": total_operations,
            "average_operations_per_second": avg_ops_per_second,
            "average_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "benchmark_names": list(self.results.keys()),
            "metadata": self.suite_metadata
        }


def create_simple_benchmark(name: str, func: Callable, iterations: int = 1000, *args, **kwargs) -> BenchmarkResult:
    """Create and run a simple benchmark
    
    Args:
        name: Benchmark name
        func: Function to benchmark
        iterations: Number of iterations
        *args, **kwargs: Arguments for the function
        
    Returns:
        BenchmarkResult
    """
    benchmark = PerformanceBenchmark(name)
    return benchmark.run_sync(func, iterations, *args, **kwargs)


def compare_functions(name: str, functions: Dict[str, Callable], iterations: int = 1000, *args, **kwargs) -> Dict[str, BenchmarkResult]:
    """Compare performance of multiple functions
    
    Args:
        name: Comparison name
        functions: Dictionary mapping names to functions
        iterations: Number of iterations for each function
        *args, **kwargs: Arguments for the functions
        
    Returns:
        Dictionary mapping function names to benchmark results
    """
    results = {}
    
    logger.info("Starting function comparison", name=name, function_count=len(functions))
    
    for func_name, func in functions.items():
        benchmark_name = f"{name}_{func_name}"
        benchmark = PerformanceBenchmark(benchmark_name)
        results[func_name] = benchmark.run_sync(func, iterations, *args, **kwargs)
    
    # Log comparison summary
    best_func = min(results.keys(), key=lambda k: results[k].mean_time_ms)
    worst_func = max(results.keys(), key=lambda k: results[k].mean_time_ms)
    
    logger.info(
        "Function comparison completed",
        name=name,
        best_function=best_func,
        best_time_ms=results[best_func].mean_time_ms,
        worst_function=worst_func,
        worst_time_ms=results[worst_func].mean_time_ms
    )
    
    return results


def benchmark_with_memory_profiling(func: Callable, iterations: int = 100, *args, **kwargs) -> BenchmarkResult:
    """Benchmark function with detailed memory profiling
    
    Args:
        func: Function to benchmark
        iterations: Number of iterations
        *args, **kwargs: Arguments for the function
        
    Returns:
        BenchmarkResult with memory usage data
    """
    # Start memory tracing
    tracemalloc.start()
    
    try:
        benchmark = PerformanceBenchmark("memory_profiled_benchmark", warmup_iterations=10)
        result = benchmark.run_sync(func, iterations, *args, **kwargs)
        
        # Get memory statistics
        current, peak = tracemalloc.get_traced_memory()
        result.metadata["memory_current_mb"] = current / (1024 * 1024)
        result.metadata["memory_peak_mb"] = peak / (1024 * 1024)
        
        return result
        
    finally:
        tracemalloc.stop()


def create_load_test(func: Callable, max_concurrency: int = 100, duration_seconds: int = 60) -> Dict[str, BenchmarkResult]:
    """Create a load test with increasing concurrency
    
    Args:
        func: Function to load test
        max_concurrency: Maximum concurrency level to test
        duration_seconds: Duration of each concurrency level test
        
    Returns:
        Dictionary mapping concurrency levels to results
    """
    results = {}
    concurrency_levels = [1, 5, 10, 25, 50, max_concurrency]
    
    logger.info("Starting load test", max_concurrency=max_concurrency, duration_seconds=duration_seconds)
    
    for concurrency in concurrency_levels:
        if concurrency > max_concurrency:
            continue
            
        # Calculate iterations based on duration and expected performance
        estimated_iterations = max(100, duration_seconds * 10 * concurrency)
        
        benchmark = PerformanceBenchmark(f"load_test_concurrency_{concurrency}")
        result = benchmark.run_concurrent(
            func, 
            iterations=estimated_iterations,
            concurrency=concurrency,
            executor_type="thread"
        )
        
        results[f"concurrency_{concurrency}"] = result
        
        logger.info(
            "Load test level completed",
            concurrency=concurrency,
            ops_per_second=result.operations_per_second,
            error_rate=result.error_count / result.iterations if result.iterations > 0 else 0
        )
        
        # Stop if error rate gets too high
        error_rate = result.error_count / result.iterations if result.iterations > 0 else 0
        if error_rate > 0.1:  # 10% error rate
            logger.warning("Stopping load test due to high error rate", error_rate=error_rate)
            break
    
    return results