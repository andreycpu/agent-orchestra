"""
Tests for benchmarking functionality
"""
import pytest
import time
import asyncio
from unittest.mock import Mock, patch
from agent_orchestra.benchmarks import (
    PerformanceBenchmark, BenchmarkSuite, BenchmarkResult,
    create_simple_benchmark, compare_functions,
    benchmark_with_memory_profiling, create_load_test
)
from agent_orchestra.exceptions import ValidationError


class TestBenchmarkResult:
    """Test cases for BenchmarkResult dataclass"""
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult creation"""
        result = BenchmarkResult(
            name="test_benchmark",
            iterations=1000,
            duration_seconds=5.0,
            operations_per_second=200.0,
            min_time_ms=1.0,
            max_time_ms=10.0,
            mean_time_ms=5.0,
            median_time_ms=4.5,
            std_dev_ms=2.0,
            percentile_95_ms=8.0,
            percentile_99_ms=9.5
        )
        
        assert result.name == "test_benchmark"
        assert result.iterations == 1000
        assert result.operations_per_second == 200.0
        assert result.error_count == 0  # Default value


class TestPerformanceBenchmark:
    """Test cases for PerformanceBenchmark class"""
    
    def test_benchmark_creation(self):
        """Test PerformanceBenchmark creation"""
        benchmark = PerformanceBenchmark("test_benchmark", warmup_iterations=50)
        
        assert benchmark.name == "test_benchmark"
        assert benchmark.warmup_iterations == 50
        assert len(benchmark.execution_times) == 0
        assert len(benchmark.errors) == 0
    
    def test_benchmark_creation_validation(self):
        """Test PerformanceBenchmark creation validation"""
        # Empty name
        with pytest.raises(ValidationError, match="benchmark name cannot be empty"):
            PerformanceBenchmark("")
        
        # Negative warmup iterations
        with pytest.raises(ValidationError, match="warmup_iterations cannot be negative"):
            PerformanceBenchmark("test", warmup_iterations=-1)
    
    def test_run_sync_validation(self):
        """Test run_sync validation"""
        benchmark = PerformanceBenchmark("test")
        
        # Non-callable function
        with pytest.raises(ValidationError, match="func must be callable"):
            benchmark.run_sync("not_callable")
        
        # Invalid iterations
        with pytest.raises(ValidationError, match="iterations must be positive"):
            benchmark.run_sync(lambda: None, iterations=0)
        
        with pytest.raises(ValidationError, match="iterations must be positive"):
            benchmark.run_sync(lambda: None, iterations=-1)
    
    def test_run_sync_simple_function(self):
        """Test run_sync with simple function"""
        def simple_function():
            time.sleep(0.001)  # 1ms delay
            return "result"
        
        benchmark = PerformanceBenchmark("test", warmup_iterations=5)
        result = benchmark.run_sync(simple_function, iterations=10)
        
        assert result.name == "test"
        assert result.iterations == 10
        assert result.error_count == 0
        assert result.operations_per_second > 0
        assert result.mean_time_ms >= 1.0  # Should be at least 1ms due to sleep
        assert result.min_time_ms > 0
        assert result.max_time_ms >= result.min_time_ms
        assert len(benchmark.execution_times) == 10
    
    def test_run_sync_with_errors(self):
        """Test run_sync with function that sometimes fails"""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise ValueError("Test error")
            return "success"
        
        benchmark = PerformanceBenchmark("test", warmup_iterations=0)
        result = benchmark.run_sync(failing_function, iterations=10)
        
        assert result.error_count > 0
        assert result.error_count < result.iterations  # Some should succeed
        assert len(benchmark.errors) == result.error_count
    
    @pytest.mark.asyncio
    async def test_run_async_validation(self):
        """Test run_async validation"""
        benchmark = PerformanceBenchmark("test")
        
        # Non-callable function
        with pytest.raises(ValidationError, match="func must be callable"):
            await benchmark.run_async("not_callable")
        
        # Invalid iterations
        with pytest.raises(ValidationError, match="iterations must be positive"):
            await benchmark.run_async(lambda: None, iterations=0)
    
    @pytest.mark.asyncio
    async def test_run_async_simple_function(self):
        """Test run_async with simple async function"""
        async def async_function():
            await asyncio.sleep(0.001)  # 1ms delay
            return "result"
        
        benchmark = PerformanceBenchmark("async_test", warmup_iterations=3)
        result = await benchmark.run_async(async_function, iterations=5)
        
        assert result.name == "async_test"
        assert result.iterations == 5
        assert result.error_count == 0
        assert result.operations_per_second > 0
        assert result.mean_time_ms >= 1.0
        assert len(benchmark.execution_times) == 5
    
    @pytest.mark.asyncio
    async def test_run_async_with_errors(self):
        """Test run_async with function that fails"""
        async def failing_async_function():
            await asyncio.sleep(0.001)
            raise RuntimeError("Async test error")
        
        benchmark = PerformanceBenchmark("async_test", warmup_iterations=0)
        result = await benchmark.run_async(failing_async_function, iterations=5)
        
        assert result.error_count == 5  # All should fail
        assert len(benchmark.errors) == 5
        assert result.operations_per_second == 0  # No successful operations
    
    def test_run_concurrent_validation(self):
        """Test run_concurrent validation"""
        benchmark = PerformanceBenchmark("test")
        
        # Invalid executor type
        with pytest.raises(ValidationError, match="executor_type must be 'thread' or 'process'"):
            benchmark.run_concurrent(lambda: None, executor_type="invalid")
        
        # Invalid concurrency
        with pytest.raises(ValidationError, match="concurrency must be positive"):
            benchmark.run_concurrent(lambda: None, concurrency=0)
        
        # Invalid iterations
        with pytest.raises(ValidationError, match="iterations must be positive"):
            benchmark.run_concurrent(lambda: None, iterations=-1)
    
    def test_run_concurrent_thread_executor(self):
        """Test run_concurrent with thread executor"""
        def concurrent_function():
            time.sleep(0.01)  # 10ms delay
            return "result"
        
        benchmark = PerformanceBenchmark("concurrent_test", warmup_iterations=2)
        result = benchmark.run_concurrent(
            concurrent_function, 
            iterations=10, 
            concurrency=3, 
            executor_type="thread"
        )
        
        assert result.name == "concurrent_test"
        assert result.iterations == 10
        assert result.operations_per_second > 0
        # With concurrency, total time should be less than sequential
        assert result.duration_seconds < 0.1  # Should be much less than 10 * 0.01
    
    def test_timed_execution(self):
        """Test _timed_execution method"""
        benchmark = PerformanceBenchmark("test")
        
        # Successful execution
        def success_func():
            time.sleep(0.01)
            return "success"
        
        execution_time = benchmark._timed_execution(success_func, (), {})
        assert execution_time is not None
        assert execution_time >= 0.01
        
        # Failed execution
        def failing_func():
            raise ValueError("Test error")
        
        execution_time = benchmark._timed_execution(failing_func, (), {})
        assert execution_time is None
    
    def test_calculate_results_no_successful_iterations(self):
        """Test _calculate_results with no successful iterations"""
        benchmark = PerformanceBenchmark("test")
        benchmark.start_time = 1.0
        benchmark.end_time = 2.0
        benchmark.errors = [Exception("error1"), Exception("error2")]
        
        result = benchmark._calculate_results(10)
        
        assert result.operations_per_second == 0
        assert result.mean_time_ms == 0
        assert result.error_count == 2
        assert result.iterations == 10
    
    def test_calculate_results_with_successful_iterations(self):
        """Test _calculate_results with successful iterations"""
        benchmark = PerformanceBenchmark("test")
        benchmark.start_time = 1.0
        benchmark.end_time = 3.0  # 2 seconds duration
        benchmark.execution_times = [0.1, 0.2, 0.15, 0.25, 0.12]  # 5 successful iterations
        benchmark.errors = [Exception("error1")]  # 1 error
        
        result = benchmark._calculate_results(6)  # 6 total iterations
        
        assert result.operations_per_second == 2.5  # 5 successful / 2 seconds
        assert result.min_time_ms == 100.0  # 0.1 * 1000
        assert result.max_time_ms == 250.0  # 0.25 * 1000
        assert 100.0 <= result.mean_time_ms <= 250.0
        assert result.error_count == 1
        assert result.duration_seconds == 2.0
    
    @patch('agent_orchestra.benchmarks.calculate_memory_usage')
    def test_collect_resource_snapshot(self, mock_memory):
        """Test _collect_resource_snapshot method"""
        mock_memory.return_value = {"rss": 1024 * 1024 * 100}  # 100MB
        
        benchmark = PerformanceBenchmark("test")
        benchmark._collect_resource_snapshot()
        
        assert len(benchmark.resource_snapshots) == 1
        snapshot = benchmark.resource_snapshots[0]
        assert snapshot.memory_mb == 100.0
        assert snapshot.thread_count > 0


class TestBenchmarkSuite:
    """Test cases for BenchmarkSuite class"""
    
    def test_suite_creation(self):
        """Test BenchmarkSuite creation"""
        suite = BenchmarkSuite("test_suite")
        
        assert suite.name == "test_suite"
        assert len(suite.benchmarks) == 0
        assert len(suite.results) == 0
        assert "created_at" in suite.suite_metadata
    
    def test_suite_creation_validation(self):
        """Test BenchmarkSuite creation validation"""
        with pytest.raises(ValidationError, match="suite name cannot be empty"):
            BenchmarkSuite("")
    
    def test_add_benchmark(self):
        """Test adding benchmark to suite"""
        suite = BenchmarkSuite("test_suite")
        benchmark = PerformanceBenchmark("test_benchmark")
        
        suite.add_benchmark(benchmark)
        
        assert "test_benchmark" in suite.benchmarks
        assert suite.benchmarks["test_benchmark"] == benchmark
    
    def test_add_benchmark_validation(self):
        """Test add_benchmark validation"""
        suite = BenchmarkSuite("test_suite")
        
        with pytest.raises(ValidationError, match="benchmark must be a PerformanceBenchmark instance"):
            suite.add_benchmark("not_a_benchmark")
    
    def test_run_all_empty_suite(self):
        """Test running empty benchmark suite"""
        suite = BenchmarkSuite("empty_suite")
        results = suite.run_all()
        
        assert len(results) == 0
    
    def test_get_summary_no_results(self):
        """Test get_summary with no results"""
        suite = BenchmarkSuite("test_suite")
        summary = suite.get_summary()
        
        assert summary["error"] == "No benchmark results available"
    
    def test_get_summary_with_results(self):
        """Test get_summary with mock results"""
        suite = BenchmarkSuite("test_suite")
        
        # Add mock results
        suite.results = {
            "bench1": BenchmarkResult(
                name="bench1", iterations=100, duration_seconds=1.0,
                operations_per_second=100.0, min_time_ms=5.0, max_time_ms=15.0,
                mean_time_ms=10.0, median_time_ms=9.0, std_dev_ms=2.0,
                percentile_95_ms=14.0, percentile_99_ms=15.0, error_count=0
            ),
            "bench2": BenchmarkResult(
                name="bench2", iterations=100, duration_seconds=2.0,
                operations_per_second=50.0, min_time_ms=10.0, max_time_ms=30.0,
                mean_time_ms=20.0, median_time_ms=18.0, std_dev_ms=5.0,
                percentile_95_ms=28.0, percentile_99_ms=30.0, error_count=2
            )
        }
        
        summary = suite.get_summary()
        
        assert summary["suite_name"] == "test_suite"
        assert summary["total_benchmarks"] == 2
        assert summary["successful_benchmarks"] == 1  # bench1 has no errors
        assert summary["failed_benchmarks"] == 1  # bench2 has errors
        assert summary["total_operations_per_second"] == 150.0  # 100 + 50
        assert summary["average_operations_per_second"] == 75.0  # 150 / 2
        assert summary["average_latency_ms"] == 10.0  # Only successful benchmark
        assert summary["min_latency_ms"] == 5.0
        assert summary["max_latency_ms"] == 15.0


class TestBenchmarkUtilities:
    """Test cases for benchmark utility functions"""
    
    def test_create_simple_benchmark(self):
        """Test create_simple_benchmark utility"""
        def simple_func():
            time.sleep(0.001)
            return "result"
        
        result = create_simple_benchmark("simple_test", simple_func, iterations=5)
        
        assert result.name == "simple_test"
        assert result.iterations == 5
        assert result.operations_per_second > 0
        assert result.error_count == 0
    
    def test_compare_functions(self):
        """Test compare_functions utility"""
        def fast_func():
            time.sleep(0.001)
            return "fast"
        
        def slow_func():
            time.sleep(0.005)
            return "slow"
        
        functions = {
            "fast": fast_func,
            "slow": slow_func
        }
        
        results = compare_functions("speed_test", functions, iterations=5)
        
        assert "fast" in results
        assert "slow" in results
        assert results["fast"].mean_time_ms < results["slow"].mean_time_ms
        assert results["fast"].operations_per_second > results["slow"].operations_per_second
    
    def test_benchmark_with_memory_profiling(self):
        """Test benchmark_with_memory_profiling utility"""
        def memory_func():
            # Create some objects to use memory
            data = [i for i in range(1000)]
            return sum(data)
        
        result = benchmark_with_memory_profiling(memory_func, iterations=5)
        
        assert result.name == "memory_profiled_benchmark"
        assert result.iterations == 5
        assert "memory_current_mb" in result.metadata
        assert "memory_peak_mb" in result.metadata
        assert result.metadata["memory_current_mb"] >= 0
        assert result.metadata["memory_peak_mb"] >= 0
    
    def test_create_load_test(self):
        """Test create_load_test utility"""
        def load_func():
            time.sleep(0.001)
            return "loaded"
        
        # Small load test for testing
        results = create_load_test(load_func, max_concurrency=10, duration_seconds=1)
        
        assert len(results) > 0
        assert all("concurrency_" in key for key in results.keys())
        
        # Check that results exist for different concurrency levels
        concurrency_levels = [int(key.split("_")[1]) for key in results.keys()]
        assert min(concurrency_levels) == 1
        assert max(concurrency_levels) <= 10
        
        # Results should show increasing throughput with concurrency (up to a point)
        for key, result in results.items():
            assert result.operations_per_second > 0
            assert isinstance(result, BenchmarkResult)


class TestBenchmarkPerformance:
    """Performance tests for benchmark framework itself"""
    
    def test_benchmark_overhead(self):
        """Test that benchmark overhead is minimal"""
        def minimal_func():
            return 42
        
        # Time the function directly
        direct_start = time.time()
        for _ in range(1000):
            minimal_func()
        direct_time = time.time() - direct_start
        
        # Time the function through benchmark
        benchmark = PerformanceBenchmark("overhead_test", warmup_iterations=0)
        benchmark_start = time.time()
        result = benchmark.run_sync(minimal_func, iterations=1000)
        benchmark_time = time.time() - benchmark_start
        
        # Benchmark overhead should be reasonable (less than 10x direct execution)
        overhead_ratio = benchmark_time / max(direct_time, 0.001)  # Avoid division by zero
        assert overhead_ratio < 10, f"Benchmark overhead too high: {overhead_ratio}x"
        
        # Should have accurate results
        assert result.error_count == 0
        assert result.iterations == 1000
    
    def test_concurrent_benchmark_scaling(self):
        """Test that concurrent benchmarks scale appropriately"""
        def io_bound_func():
            time.sleep(0.01)  # 10ms simulated I/O
        
        # Sequential benchmark
        sequential_benchmark = PerformanceBenchmark("sequential", warmup_iterations=0)
        sequential_result = sequential_benchmark.run_sync(io_bound_func, iterations=10)
        
        # Concurrent benchmark
        concurrent_benchmark = PerformanceBenchmark("concurrent", warmup_iterations=0)
        concurrent_result = concurrent_benchmark.run_concurrent(
            io_bound_func, iterations=10, concurrency=5, executor_type="thread"
        )
        
        # Concurrent should be faster than sequential for I/O bound tasks
        assert concurrent_result.duration_seconds < sequential_result.duration_seconds
        
        # Both should have same number of successful operations
        successful_sequential = sequential_result.iterations - sequential_result.error_count
        successful_concurrent = concurrent_result.iterations - concurrent_result.error_count
        assert successful_sequential == successful_concurrent


class TestBenchmarkErrorHandling:
    """Test error handling in benchmarks"""
    
    def test_benchmark_with_timeout_errors(self):
        """Test benchmark behavior with timeout errors"""
        def slow_func():
            time.sleep(1.0)  # Very slow function
            return "result"
        
        benchmark = PerformanceBenchmark("timeout_test", warmup_iterations=0)
        
        # This should complete but take a while
        result = benchmark.run_sync(slow_func, iterations=3)
        
        assert result.iterations == 3
        assert result.mean_time_ms >= 1000  # Should be at least 1000ms
    
    def test_benchmark_with_memory_errors(self):
        """Test benchmark behavior with memory allocation"""
        def memory_hungry_func():
            # Allocate and immediately release memory
            data = bytearray(1024 * 1024)  # 1MB
            del data
            return "done"
        
        benchmark = PerformanceBenchmark("memory_test", warmup_iterations=0)
        result = benchmark.run_sync(memory_hungry_func, iterations=10)
        
        assert result.error_count == 0  # Should not fail
        assert result.operations_per_second > 0
    
    def test_benchmark_exception_isolation(self):
        """Test that exceptions in one iteration don't affect others"""
        iteration_count = 0
        
        def sometimes_failing_func():
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count == 5:  # Fail on 5th iteration
                raise ValueError(f"Planned failure at iteration {iteration_count}")
            return f"success_{iteration_count}"
        
        benchmark = PerformanceBenchmark("isolation_test", warmup_iterations=0)
        result = benchmark.run_sync(sometimes_failing_func, iterations=10)
        
        assert result.error_count == 1  # Only one failure
        assert len(benchmark.execution_times) == 9  # 9 successful iterations
        assert result.operations_per_second > 0  # Should still calculate ops/sec