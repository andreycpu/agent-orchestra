"""
Unit tests for utility functions
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock
import time

from agent_orchestra.utils import (
    generate_task_id, calculate_task_hash, format_duration,
    parse_capability_string, AsyncRetry, RateLimiter, 
    AdaptiveRateLimiter, MetricsCollector
)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_generate_task_id(self):
        """Test task ID generation"""
        task_id1 = generate_task_id("test_task", {"key": "value"})
        task_id2 = generate_task_id("test_task", {"key": "value"})
        
        # Same input should generate same ID (deterministic)
        assert task_id1 == task_id2
        assert len(task_id1) == 16
        assert task_id1.isalnum()
        
        # Different input should generate different ID
        task_id3 = generate_task_id("test_task", {"key": "different"})
        assert task_id1 != task_id3
    
    def test_calculate_task_hash(self):
        """Test task hash calculation"""
        hash1 = calculate_task_hash("test_task", {"key": "value"})
        hash2 = calculate_task_hash("test_task", {"key": "value"})
        
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
        
    def test_format_duration(self):
        """Test duration formatting"""
        assert format_duration(0.5) == "500ms"
        assert format_duration(1.5) == "1.5s"
        assert format_duration(65.0) == "1.1m"
        assert format_duration(3661.0) == "1.0h"
    
    def test_parse_capability_string(self):
        """Test capability string parsing"""
        # Simple capability
        result = parse_capability_string("text_processing")
        assert result == {"name": "text_processing", "requirements": {}}
        
        # Complex capability with requirements
        result = parse_capability_string("compute:cpu=2,memory=1GB,gpu=1")
        expected = {
            "name": "compute",
            "requirements": {
                "cpu": 2,
                "memory": "1GB",
                "gpu": 1
            }
        }
        assert result == expected


class TestAsyncRetry:
    """Test async retry decorator"""
    
    @pytest.mark.asyncio
    async def test_successful_retry(self):
        """Test retry with eventual success"""
        attempt_count = 0
        
        @AsyncRetry(max_attempts=3, base_delay=0.01)
        async def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = await flaky_function()
        assert result == "success"
        assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test retry with all attempts failing"""
        @AsyncRetry(max_attempts=2, base_delay=0.01)
        async def always_fail():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            await always_fail()


class TestRateLimiter:
    """Test basic rate limiter"""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_basic(self):
        """Test basic rate limiting functionality"""
        rate_limiter = RateLimiter(rate=10.0, capacity=5)  # 10 tokens/sec, max 5
        
        # Should be able to acquire initial tokens
        assert await rate_limiter.acquire(3) is True
        assert await rate_limiter.acquire(2) is True
        
        # Should fail to acquire more tokens
        assert await rate_limiter.acquire(1) is False
    
    @pytest.mark.asyncio
    async def test_rate_limiting_refill(self):
        """Test token refill over time"""
        rate_limiter = RateLimiter(rate=100.0, capacity=5)  # Fast refill for testing
        
        # Use all tokens
        await rate_limiter.acquire(5)
        assert await rate_limiter.acquire(1) is False
        
        # Wait for refill
        await asyncio.sleep(0.1)  # Should refill 10 tokens
        assert await rate_limiter.acquire(1) is True


class TestAdaptiveRateLimiter:
    """Test enhanced adaptive rate limiter"""
    
    @pytest.mark.asyncio
    async def test_adaptive_rate_adjustment(self):
        """Test rate adaptation based on success rate"""
        limiter = AdaptiveRateLimiter(initial_rate=10.0, initial_capacity=5)
        initial_rate = limiter.rate
        
        # Simulate many rejections to trigger rate increase
        for _ in range(25):
            await limiter.acquire(10)  # This will fail and trigger adjustment
        
        # Rate should have increased due to rejections
        assert limiter.rate > initial_rate
    
    @pytest.mark.asyncio
    async def test_adaptive_metrics(self):
        """Test metrics collection in adaptive limiter"""
        limiter = AdaptiveRateLimiter(initial_rate=10.0, initial_capacity=5, name="test")
        
        # Perform some operations
        await limiter.acquire(2)
        await limiter.acquire(10)  # This will fail
        
        metrics = limiter.get_metrics()
        assert metrics["name"] == "test"
        assert metrics["total_requests"] == 2
        assert metrics["success_rate"] == 0.5  # 1 success out of 2 requests


class TestMetricsCollector:
    """Test metrics collection utility"""
    
    def test_counter_increment(self):
        """Test counter functionality"""
        collector = MetricsCollector()
        
        collector.increment("test_counter")
        collector.increment("test_counter", 5)
        
        counters = collector.get_counters()
        assert counters["test_counter"] == 6
    
    def test_histogram_recording(self):
        """Test histogram functionality"""
        collector = MetricsCollector()
        
        collector.record("response_time", 1.5)
        collector.record("response_time", 2.0)
        collector.record("response_time", 1.0)
        
        histograms = collector.get_histograms()
        assert len(histograms["response_time"]) == 3
        assert 1.5 in histograms["response_time"]
    
    def test_gauge_setting(self):
        """Test gauge functionality"""
        collector = MetricsCollector()
        
        collector.set_gauge("cpu_usage", 45.2)
        collector.set_gauge("cpu_usage", 50.0)  # Should overwrite
        
        gauges = collector.get_gauges()
        assert gauges["cpu_usage"] == 50.0
    
    def test_metrics_with_tags(self):
        """Test metrics with tags"""
        collector = MetricsCollector()
        
        collector.increment("requests", tags={"method": "GET", "status": "200"})
        collector.increment("requests", tags={"method": "POST", "status": "200"})
        
        counters = collector.get_counters()
        assert "requests|method=GET,status=200" in counters
        assert "requests|method=POST,status=200" in counters