"""
Tests for cache manager utilities.
"""
import pytest
import time
from unittest.mock import Mock, patch

from agent_orchestra.cache_manager import (
    InMemoryCache, CacheManager, CacheDecorator,
    EvictionPolicy, CacheStats, CacheEntry,
    create_cache, get_cache, cached, clear_all_caches
)
from agent_orchestra.exceptions import CacheError


class TestCacheEntry:
    """Test cache entry functionality."""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation."""
        entry = CacheEntry(
            value="test_data",
            created_at=time.time()
        )
        
        assert entry.value == "test_data"
        assert entry.access_count == 0
        assert not entry.is_expired
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        import datetime
        past_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=1)
        
        entry = CacheEntry(
            value="test_data",
            created_at=datetime.datetime.utcnow(),
            expires_at=past_time
        )
        
        assert entry.is_expired
    
    def test_cache_entry_touch(self):
        """Test cache entry access tracking."""
        entry = CacheEntry(value="test_data", created_at=time.time())
        
        initial_count = entry.access_count
        initial_time = entry.last_accessed
        
        time.sleep(0.01)  # Small delay to ensure time difference
        entry.touch()
        
        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_time


class TestInMemoryCache:
    """Test in-memory cache implementation."""
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = InMemoryCache(max_size=10)
        
        # Set and get
        assert cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Non-existent key
        assert cache.get("nonexistent") is None
        assert cache.get("nonexistent", "default") == "default"
        
        # Key existence
        assert cache.exists("key1")
        assert not cache.exists("nonexistent")
        
        # Delete
        assert cache.delete("key1")
        assert not cache.exists("key1")
        assert not cache.delete("nonexistent")
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = InMemoryCache()
        
        # Set with TTL
        cache.set("temp_key", "temp_value", ttl=1)
        assert cache.get("temp_key") == "temp_value"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("temp_key") is None
        assert not cache.exists("temp_key")
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = InMemoryCache(max_size=3, eviction_policy=EvictionPolicy.LRU)
        
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2") 
        cache.set("key3", "value3")
        
        # Access key1 to make it more recently used
        cache.get("key1")
        
        # Add new key, should evict key2 (least recently used)
        cache.set("key4", "value4")
        
        assert cache.exists("key1")  # Recently accessed
        assert not cache.exists("key2")  # Should be evicted
        assert cache.exists("key3")
        assert cache.exists("key4")
    
    def test_lfu_eviction(self):
        """Test LFU eviction policy."""
        cache = InMemoryCache(max_size=3, eviction_policy=EvictionPolicy.LFU)
        
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 multiple times
        for _ in range(5):
            cache.get("key1")
        
        # Access key2 once
        cache.get("key2")
        
        # key3 has no accesses, should be evicted first
        cache.set("key4", "value4")
        
        assert cache.exists("key1")  # Most frequently used
        assert cache.exists("key2")  # Some access
        assert not cache.exists("key3")  # Least frequently used
        assert cache.exists("key4")
    
    def test_memory_limit(self):
        """Test memory-based eviction."""
        cache = InMemoryCache(max_memory_bytes=100, eviction_policy=EvictionPolicy.LRU)
        
        # Add items that exceed memory limit
        large_value = "x" * 50  # 50 bytes roughly
        
        cache.set("key1", large_value)
        cache.set("key2", large_value)
        cache.set("key3", large_value)  # This should trigger eviction
        
        # Should have evicted some entries due to memory limit
        assert cache.size() < 3
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = InMemoryCache()
        
        # Perform operations
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        cache.delete("key2")
        
        stats = cache.stats()
        
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.sets == 2
        assert stats.deletes == 1
        assert stats.hit_rate == 0.5
    
    def test_clear(self):
        """Test cache clearing."""
        cache = InMemoryCache()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert cache.size() == 2
        
        cache.clear()
        
        assert cache.size() == 0
        assert cache.get("key1") is None


class TestCacheManager:
    """Test cache manager functionality."""
    
    def test_cache_creation(self):
        """Test creating named caches."""
        manager = CacheManager()
        
        cache = manager.create_cache(
            "test_cache",
            max_size=100,
            default_ttl=60
        )
        
        assert cache is not None
        assert cache.max_size == 100
        assert cache.default_ttl == 60
        
        # Get cache by name
        retrieved_cache = manager.get_cache("test_cache")
        assert retrieved_cache is cache
    
    def test_default_cache(self):
        """Test default cache functionality."""
        manager = CacheManager()
        
        cache1 = manager.create_cache("first_cache")
        assert manager.get_default_cache() is cache1
        
        cache2 = manager.create_cache("second_cache")
        manager.set_default_cache("second_cache")
        assert manager.get_default_cache() is cache2
    
    def test_cache_warmup(self):
        """Test cache warmup functionality."""
        manager = CacheManager()
        cache = manager.create_cache("warmup_cache")
        
        warmup_called = False
        
        def warmup_function(cache):
            nonlocal warmup_called
            warmup_called = True
            cache.set("preloaded", "data")
        
        manager.register_warmup_task("warmup_cache", warmup_function)
        
        # Run warmup
        import asyncio
        asyncio.run(manager.warmup_cache("warmup_cache"))
        
        assert warmup_called
        assert cache.get("preloaded") == "data"
    
    def test_all_stats(self):
        """Test getting statistics for all caches."""
        manager = CacheManager()
        
        cache1 = manager.create_cache("cache1")
        cache2 = manager.create_cache("cache2")
        
        cache1.set("key1", "value1")
        cache2.set("key2", "value2")
        
        all_stats = manager.get_all_stats()
        
        assert "cache1" in all_stats
        assert "cache2" in all_stats
        assert all_stats["cache1"]["stats"]["size"] == 1
        assert all_stats["cache2"]["stats"]["size"] == 1
    
    def test_clear_all_caches(self):
        """Test clearing all caches."""
        manager = CacheManager()
        
        cache1 = manager.create_cache("cache1")
        cache2 = manager.create_cache("cache2")
        
        cache1.set("key1", "value1")
        cache2.set("key2", "value2")
        
        manager.clear_all_caches()
        
        assert cache1.size() == 0
        assert cache2.size() == 0


class TestCacheDecorator:
    """Test cache decorator functionality."""
    
    def test_function_caching(self):
        """Test function result caching."""
        cache = InMemoryCache()
        call_count = 0
        
        @CacheDecorator(cache=cache, ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
        
        # Different argument should call function again
        result3 = expensive_function(3)
        assert result3 == 6
        assert call_count == 2
    
    def test_cache_key_building(self):
        """Test cache key building for different arguments."""
        cache = InMemoryCache()
        decorator = CacheDecorator(cache=cache)
        
        def test_func(a, b, c=None):
            return a + b
        
        # Test key generation
        key1 = decorator._default_key_builder(test_func, (1, 2), {})
        key2 = decorator._default_key_builder(test_func, (1, 2), {"c": 3})
        key3 = decorator._default_key_builder(test_func, (2, 1), {})
        
        assert key1 != key2  # Different kwargs
        assert key1 != key3  # Different args
        assert key2 != key3  # Different everything
    
    def test_long_key_hashing(self):
        """Test that long cache keys are hashed."""
        cache = InMemoryCache()
        decorator = CacheDecorator(cache=cache)
        
        def test_func():
            return "result"
        
        # Create very long arguments
        long_args = tuple("x" * 100 for _ in range(10))
        key = decorator._default_key_builder(test_func, long_args, {})
        
        # Key should be hashed (shorter than original)
        assert len(key) == 64  # SHA256 hex length


class TestGlobalCacheFunctions:
    """Test global cache management functions."""
    
    def test_create_and_get_cache(self):
        """Test global cache creation and retrieval."""
        cache = create_cache(
            "global_test",
            max_size=50,
            default_ttl=120
        )
        
        assert cache is not None
        
        retrieved = get_cache("global_test")
        assert retrieved is cache
        
        # Non-existent cache
        assert get_cache("nonexistent") is None
    
    def test_cached_decorator(self):
        """Test global cached decorator."""
        call_count = 0
        
        @cached(cache_name="default", ttl=60)
        def test_function(x):
            nonlocal call_count
            call_count += 1
            return x ** 2
        
        # Create the default cache first
        create_cache("default")
        
        # First call
        result1 = test_function(4)
        assert result1 == 16
        assert call_count == 1
        
        # Second call should use cache
        result2 = test_function(4)
        assert result2 == 16
        assert call_count == 1
    
    def test_clear_all_caches_global(self):
        """Test global cache clearing."""
        cache1 = create_cache("global1")
        cache2 = create_cache("global2")
        
        cache1.set("key1", "value1")
        cache2.set("key2", "value2")
        
        clear_all_caches()
        
        assert cache1.size() == 0
        assert cache2.size() == 0


class TestCacheLock:
    """Test cache locking functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_lock_acquisition(self):
        """Test cache lock acquisition and release."""
        from agent_orchestra.cache_manager import cache_lock
        
        cache = InMemoryCache()
        
        # Test successful lock acquisition
        async with cache_lock(cache, "test_resource", timeout=5):
            # Lock should be acquired
            assert cache.exists("lock:test_resource")
        
        # Lock should be released after context
        assert not cache.exists("lock:test_resource")
    
    @pytest.mark.asyncio
    async def test_cache_lock_timeout(self):
        """Test cache lock timeout."""
        from agent_orchestra.cache_manager import cache_lock
        from agent_orchestra.exceptions import CacheError
        
        cache = InMemoryCache()
        
        # Acquire lock manually
        cache.set("lock:test_resource", "manual_lock", ttl=10)
        
        # Should timeout trying to acquire same lock
        with pytest.raises(CacheError, match="Failed to acquire cache lock"):
            async with cache_lock(cache, "test_resource", timeout=1):
                pass


class TestEvictionPolicies:
    """Test different eviction policies comprehensively."""
    
    def test_fifo_eviction(self):
        """Test FIFO eviction policy."""
        cache = InMemoryCache(max_size=3, eviction_policy=EvictionPolicy.FIFO)
        
        cache.set("first", "1")
        cache.set("second", "2") 
        cache.set("third", "3")
        
        # Access items in different order
        cache.get("third")
        cache.get("first")
        cache.get("second")
        
        # Add new item - should evict "first" (first in)
        cache.set("fourth", "4")
        
        assert not cache.exists("first")
        assert cache.exists("second")
        assert cache.exists("third")
        assert cache.exists("fourth")
    
    def test_random_eviction(self):
        """Test random eviction policy."""
        cache = InMemoryCache(max_size=2, eviction_policy=EvictionPolicy.RANDOM)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Add third item - should evict randomly
        cache.set("key3", "value3")
        
        # Should have exactly 2 items
        assert cache.size() == 2
        # One of the original items should be gone
        assert not (cache.exists("key1") and cache.exists("key2") and cache.exists("key3"))