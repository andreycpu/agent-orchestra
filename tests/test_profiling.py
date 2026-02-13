"""
Tests for profiling functionality
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch
from agent_orchestra.profiling import (
    FunctionProfiler, CodeProfiler, MemoryProfiler,
    profile_function, profile_code_block, memory_profile,
    function_profiler
)
from agent_orchestra.exceptions import ValidationError


class TestFunctionProfiler:
    """Test cases for FunctionProfiler class"""
    
    @pytest.fixture
    def profiler(self):
        """Create FunctionProfiler instance"""
        return FunctionProfiler()
    
    def test_profiler_creation(self, profiler):
        """Test FunctionProfiler creation"""
        assert profiler.enabled is True
        assert len(profiler.call_stats) == 0
    
    def test_profiler_creation_disabled(self):
        """Test FunctionProfiler creation disabled"""
        profiler = FunctionProfiler(enabled=False)
        assert profiler.enabled is False
    
    def test_profile_decorator_basic(self, profiler):
        """Test basic profile decorator functionality"""
        @profiler.profile
        def test_function():
            time.sleep(0.01)  # 10ms delay
            return "result"
        
        # Call function multiple times
        for _ in range(3):
            result = test_function()
            assert result == "result"
        
        # Check stats
        stats = profiler.get_stats("test_function")
        assert stats["total_calls"] == 3
        assert stats["avg_time"] >= 0.01  # Should be at least 10ms
        assert stats["min_time"] <= stats["max_time"]
        assert stats["total_time"] > 0
    
    def test_profile_decorator_with_custom_name(self, profiler):
        """Test profile decorator with custom name"""
        @profiler.profile(name="custom_name")
        def some_function():
            return "custom"
        
        some_function()
        
        stats = profiler.get_stats("custom_name")
        assert stats["total_calls"] == 1
        
        # Original function name should not be in stats
        original_stats = profiler.get_stats("some_function")
        assert original_stats == {}
    
    def test_profile_decorator_with_exception(self, profiler):
        """Test profile decorator when function raises exception"""
        @profiler.profile
        def failing_function():
            time.sleep(0.005)  # 5ms delay
            raise ValueError("Test error")
        
        # Function should still raise exception
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
        
        # But stats should still be recorded
        stats = profiler.get_stats("failing_function")
        assert stats["total_calls"] == 1
        assert stats["avg_time"] >= 0.005
    
    def test_profiler_disabled(self):
        """Test profiler when disabled"""
        profiler = FunctionProfiler(enabled=False)
        
        @profiler.profile
        def disabled_function():
            time.sleep(0.01)
            return "result"
        
        result = disabled_function()
        assert result == "result"
        
        # No stats should be recorded
        stats = profiler.get_stats("disabled_function")
        assert stats == {}
    
    def test_enable_disable_profiler(self, profiler):
        """Test enabling and disabling profiler"""
        @profiler.profile
        def toggle_function():
            return "toggle"
        
        # Initially enabled
        toggle_function()
        assert profiler.get_stats("toggle_function")["total_calls"] == 1
        
        # Disable profiler
        profiler.disable()
        toggle_function()
        assert profiler.get_stats("toggle_function")["total_calls"] == 1  # Should not increase
        
        # Re-enable profiler
        profiler.enable()
        toggle_function()
        assert profiler.get_stats("toggle_function")["total_calls"] == 2  # Should increase
    
    def test_get_all_stats(self, profiler):
        """Test getting stats for all functions"""
        @profiler.profile
        def func1():
            pass
        
        @profiler.profile
        def func2():
            pass
        
        func1()
        func2()
        func2()
        
        all_stats = profiler.get_stats()
        assert "func1" in all_stats
        assert "func2" in all_stats
        assert all_stats["func1"]["total_calls"] == 1
        assert all_stats["func2"]["total_calls"] == 2
    
    def test_reset_stats_specific(self, profiler):
        """Test resetting stats for specific function"""
        @profiler.profile
        def reset_test():
            pass
        
        reset_test()
        assert profiler.get_stats("reset_test")["total_calls"] == 1
        
        profiler.reset_stats("reset_test")
        assert profiler.get_stats("reset_test") == {}
    
    def test_reset_stats_all(self, profiler):
        """Test resetting all stats"""
        @profiler.profile
        def func_a():
            pass
        
        @profiler.profile
        def func_b():
            pass
        
        func_a()
        func_b()
        
        assert len(profiler.get_stats()) == 2
        
        profiler.reset_stats()
        assert len(profiler.get_stats()) == 0
    
    def test_thread_safety(self, profiler):
        """Test thread safety of profiler"""
        @profiler.profile
        def concurrent_function():
            time.sleep(0.001)
            return threading.current_thread().ident
        
        results = []
        
        def worker():
            for _ in range(10):
                results.append(concurrent_function())
        
        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Check that all calls were recorded
        stats = profiler.get_stats("concurrent_function")
        assert stats["total_calls"] == 30  # 3 threads * 10 calls each
        assert len(results) == 30


class TestCodeProfiler:
    """Test cases for CodeProfiler class"""
    
    @pytest.fixture
    def profiler(self):
        """Create CodeProfiler instance"""
        return CodeProfiler()
    
    def test_profiler_creation(self, profiler):
        """Test CodeProfiler creation"""
        assert profiler.profiler is None
        assert profiler.is_running is False
        assert profiler.results is None
    
    def test_start_stop_profiler(self, profiler):
        """Test starting and stopping profiler"""
        assert profiler.is_running is False
        
        profiler.start()
        assert profiler.is_running is True
        assert profiler.profiler is not None
        
        stats = profiler.stop()
        assert profiler.is_running is False
        assert stats is not None
        assert profiler.results is not None
    
    def test_start_already_running(self, profiler):
        """Test starting profiler when already running"""
        profiler.start()
        
        with pytest.raises(RuntimeError, match="Profiler is already running"):
            profiler.start()
        
        profiler.stop()  # Cleanup
    
    def test_stop_not_running(self, profiler):
        """Test stopping profiler when not running"""
        with pytest.raises(RuntimeError, match="Profiler is not running"):
            profiler.stop()
    
    def test_profile_block(self, profiler):
        """Test profiling a code block"""
        def test_function(x, y):
            time.sleep(0.01)
            return x + y
        
        result, stats = profiler.profile_block(test_function, 5, 10)
        
        assert result == 15
        assert stats is not None
        assert profiler.results is not None
    
    def test_profile_block_with_exception(self, profiler):
        """Test profiling block that raises exception"""
        def failing_function():
            time.sleep(0.01)
            raise ValueError("Test exception")
        
        with pytest.raises(ValueError, match="Test exception"):
            profiler.profile_block(failing_function)
        
        # Profiling should still have results
        assert profiler.results is not None
    
    def test_profile_block_validation(self, profiler):
        """Test profile_block validation"""
        with pytest.raises(ValidationError, match="func must be callable"):
            profiler.profile_block("not_callable")
    
    def test_get_top_functions_no_results(self, profiler):
        """Test get_top_functions with no results"""
        with pytest.raises(RuntimeError, match="No profiling results available"):
            profiler.get_top_functions()
    
    def test_get_top_functions(self, profiler):
        """Test getting top functions"""
        def complex_function():
            # Create some nested calls
            def inner1():
                time.sleep(0.005)
            
            def inner2():
                time.sleep(0.003)
            
            inner1()
            inner2()
            return "done"
        
        profiler.profile_block(complex_function)
        top_functions = profiler.get_top_functions(5)
        
        assert isinstance(top_functions, list)
        assert len(top_functions) > 0
        
        for func_info in top_functions:
            assert "function" in func_info
            assert "total_calls" in func_info
            assert "total_time" in func_info
            assert "cumulative_time" in func_info
    
    def test_get_stats_summary_no_results(self, profiler):
        """Test get_stats_summary with no results"""
        with pytest.raises(RuntimeError, match="No profiling results available"):
            profiler.get_stats_summary()
    
    def test_get_stats_summary(self, profiler):
        """Test getting stats summary"""
        def summary_test():
            for i in range(100):
                pass
            return "summary"
        
        profiler.profile_block(summary_test)
        summary = profiler.get_stats_summary()
        
        assert "total_calls" in summary
        assert "total_time_seconds" in summary
        assert "function_count" in summary
        assert "average_time_per_call" in summary
        assert "top_function" in summary
        
        assert summary["total_calls"] > 0
        assert summary["total_time_seconds"] >= 0
        assert summary["function_count"] > 0


class TestMemoryProfiler:
    """Test cases for MemoryProfiler class"""
    
    @pytest.fixture
    def profiler(self):
        """Create MemoryProfiler instance"""
        return MemoryProfiler(sample_interval=0.05)  # Fast sampling for tests
    
    def test_profiler_creation(self, profiler):
        """Test MemoryProfiler creation"""
        assert profiler.sample_interval == 0.05
        assert profiler.is_profiling is False
        assert len(profiler.memory_samples) == 0
    
    def test_profiler_creation_validation(self):
        """Test MemoryProfiler creation validation"""
        with pytest.raises(ValidationError, match="sample_interval must be positive"):
            MemoryProfiler(sample_interval=0)
        
        with pytest.raises(ValidationError, match="sample_interval must be positive"):
            MemoryProfiler(sample_interval=-1)
    
    def test_start_stop_profiling(self, profiler):
        """Test starting and stopping memory profiling"""
        assert profiler.is_profiling is False
        
        profiler.start()
        assert profiler.is_profiling is True
        
        # Let it collect some samples
        time.sleep(0.2)
        
        samples = profiler.stop()
        assert profiler.is_profiling is False
        assert len(samples) > 0
        
        # Check sample structure
        for sample in samples:
            assert "timestamp" in sample
            assert "rss_mb" in sample
            assert "vms_mb" in sample
            assert "percent" in sample
            assert sample["rss_mb"] >= 0
    
    def test_start_already_profiling(self, profiler):
        """Test starting profiler when already running"""
        profiler.start()
        
        with pytest.raises(RuntimeError, match="Memory profiling is already active"):
            profiler.start()
        
        profiler.stop()  # Cleanup
    
    def test_stop_not_profiling(self, profiler):
        """Test stopping profiler when not running"""
        with pytest.raises(RuntimeError, match="Memory profiling is not active"):
            profiler.stop()
    
    def test_get_peak_memory_no_samples(self, profiler):
        """Test get_peak_memory with no samples"""
        peak = profiler.get_peak_memory()
        
        assert peak["rss_mb"] == 0
        assert peak["vms_mb"] == 0
        assert peak["percent"] == 0
    
    @patch('agent_orchestra.profiling.calculate_memory_usage')
    def test_get_peak_memory_with_samples(self, mock_memory, profiler):
        """Test get_peak_memory with sample data"""
        # Mock memory usage to return increasing values
        memory_values = [
            {"rss": 100 * 1024 * 1024, "vms": 200 * 1024 * 1024, "percent": 10},
            {"rss": 150 * 1024 * 1024, "vms": 250 * 1024 * 1024, "percent": 15},
            {"rss": 120 * 1024 * 1024, "vms": 220 * 1024 * 1024, "percent": 12},
        ]
        
        mock_memory.side_effect = memory_values
        
        profiler.start()
        time.sleep(0.2)  # Let it sample
        profiler.stop()
        
        peak = profiler.get_peak_memory()
        
        # Should find the peak values
        assert peak["rss_mb"] == 150.0  # Peak RSS
        assert peak["vms_mb"] == 250.0  # Peak VMS
        assert peak["percent"] == 15.0  # Peak percent
    
    def test_get_memory_trend_insufficient_data(self, profiler):
        """Test get_memory_trend with insufficient data"""
        # No samples
        trend = profiler.get_memory_trend()
        assert trend["trend"] == "insufficient_data"
        
        # One sample
        profiler.memory_samples = [{"rss_mb": 100, "vms_mb": 200, "percent": 10}]
        trend = profiler.get_memory_trend()
        assert trend["trend"] == "insufficient_data"
    
    def test_get_memory_trend_stable(self, profiler):
        """Test get_memory_trend with stable memory usage"""
        profiler.memory_samples = [
            {"rss_mb": 100, "vms_mb": 200, "percent": 10},
            {"rss_mb": 102, "vms_mb": 204, "percent": 10.2}  # Small change
        ]
        
        trend = profiler.get_memory_trend()
        assert trend["trend"] == "stable"
        assert trend["change_mb"] == 2.0
        assert abs(trend["change_percent"] - 2.0) < 0.1
    
    def test_get_memory_trend_increasing(self, profiler):
        """Test get_memory_trend with increasing memory usage"""
        profiler.memory_samples = [
            {"rss_mb": 100, "vms_mb": 200, "percent": 10},
            {"rss_mb": 120, "vms_mb": 240, "percent": 12}  # 20% increase
        ]
        
        trend = profiler.get_memory_trend()
        assert trend["trend"] == "increasing"
        assert trend["change_mb"] == 20.0
        assert abs(trend["change_percent"] - 20.0) < 0.1
    
    def test_get_memory_trend_decreasing(self, profiler):
        """Test get_memory_trend with decreasing memory usage"""
        profiler.memory_samples = [
            {"rss_mb": 100, "vms_mb": 200, "percent": 10},
            {"rss_mb": 80, "vms_mb": 160, "percent": 8}  # 20% decrease
        ]
        
        trend = profiler.get_memory_trend()
        assert trend["trend"] == "decreasing"
        assert trend["change_mb"] == -20.0
        assert abs(trend["change_percent"] + 20.0) < 0.1  # Should be negative


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_profile_function_decorator(self):
        """Test profile_function utility decorator"""
        @profile_function
        def utility_test():
            time.sleep(0.01)
            return "utility"
        
        result = utility_test()
        assert result == "utility"
        
        # Check that global profiler recorded it
        stats = function_profiler.get_stats("utility_test")
        assert stats["total_calls"] == 1
        assert stats["avg_time"] >= 0.01
    
    def test_profile_function_with_name(self):
        """Test profile_function with custom name"""
        @profile_function(name="custom_utility")
        def named_utility():
            return "named"
        
        result = named_utility()
        assert result == "named"
        
        stats = function_profiler.get_stats("custom_utility")
        assert stats["total_calls"] == 1
    
    def test_profile_code_block(self):
        """Test profile_code_block utility"""
        def block_test():
            # Simulate some work
            for i in range(1000):
                pass
            return "block_result"
        
        result, summary = profile_code_block(block_test)
        
        assert result == "block_result"
        assert "total_calls" in summary
        assert "total_time_seconds" in summary
        assert "function_count" in summary
        assert summary["total_calls"] > 0
    
    @patch('agent_orchestra.profiling.calculate_memory_usage')
    def test_memory_profile_utility(self, mock_memory):
        """Test memory_profile utility function"""
        # Mock memory usage
        mock_memory.return_value = {"rss": 100 * 1024 * 1024, "vms": 200 * 1024 * 1024, "percent": 10}
        
        def memory_test():
            # Simulate memory allocation
            data = [i for i in range(1000)]
            return len(data)
        
        result, memory_stats = memory_profile(memory_test, sample_interval=0.05)
        
        assert result == 1000
        assert "peak_memory" in memory_stats
        assert "trend" in memory_stats
        assert "sample_count" in memory_stats
        
        assert memory_stats["peak_memory"]["rss_mb"] == 100.0
        assert memory_stats["sample_count"] >= 0


class TestProfilerIntegration:
    """Integration tests for profilers"""
    
    def test_function_and_code_profiler_together(self):
        """Test using function profiler and code profiler together"""
        @profile_function
        def integrated_test():
            time.sleep(0.01)
            return "integrated"
        
        code_profiler = CodeProfiler()
        result, stats = code_profiler.profile_block(integrated_test)
        
        assert result == "integrated"
        
        # Function profiler should have recorded the call
        func_stats = function_profiler.get_stats("integrated_test")
        assert func_stats["total_calls"] == 1
        
        # Code profiler should have detailed stats
        summary = code_profiler.get_stats_summary()
        assert summary["total_calls"] > 0
    
    def test_profiler_with_exception_handling(self):
        """Test profiler behavior with exception handling"""
        @profile_function
        def exception_test():
            time.sleep(0.005)
            raise RuntimeError("Profiler test exception")
        
        # Function should still raise exception
        with pytest.raises(RuntimeError, match="Profiler test exception"):
            exception_test()
        
        # But profiler should record the call
        stats = function_profiler.get_stats("exception_test")
        assert stats["total_calls"] == 1
        assert stats["avg_time"] >= 0.005
    
    def test_nested_profiled_functions(self):
        """Test nested profiled functions"""
        @profile_function
        def outer_function():
            inner_function()
            return "outer"
        
        @profile_function
        def inner_function():
            time.sleep(0.01)
            return "inner"
        
        result = outer_function()
        assert result == "outer"
        
        # Both functions should be recorded
        outer_stats = function_profiler.get_stats("outer_function")
        inner_stats = function_profiler.get_stats("inner_function")
        
        assert outer_stats["total_calls"] == 1
        assert inner_stats["total_calls"] == 1
        assert outer_stats["avg_time"] >= inner_stats["avg_time"]  # Outer includes inner time