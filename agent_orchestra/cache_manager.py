"""
Cache management utilities for Agent Orchestra.

This module provides various caching strategies including in-memory,
Redis-based, and distributed caching with TTL, eviction policies,
and cache warming capabilities.
"""
import asyncio
import time
import json
import pickle
import hashlib
import threading
from typing import Any, Dict, Optional, Union, Callable, List, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
import logging

from .exceptions import CacheError, ConfigurationError
from .validation import validate_timeout


logger = logging.getLogger(__name__)


class EvictionPolicy(str, Enum):
    """Cache eviction policies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In, First Out
    RANDOM = "random"     # Random eviction
    TTL = "ttl"           # Time To Live based


class CacheBackend(str, Enum):
    """Cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    FILE = "file"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return self.expires_at and datetime.utcnow() > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    sets: int = 0
    deletes: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate miss rate."""
        return 1.0 - self.hit_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'expirations': self.expirations,
            'sets': self.sets,
            'deletes': self.deletes,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'size_bytes': self.size_bytes,
            'entry_count': self.entry_count
        }


class InMemoryCache:
    """Thread-safe in-memory cache with various eviction policies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_bytes: Optional[int] = None,
        default_ttl: Optional[int] = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_bytes or (100 * 1024 * 1024)  # 100MB default
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()  # For LRU
        self._access_frequency = defaultdict(int)  # For LFU
        self._insertion_order = OrderedDict()  # For FIFO
        self._lock = threading.RLock()
        self._stats = CacheStats()
    
    def _calculate_size(self, obj: Any) -> int:
        """Calculate approximate size of object in bytes."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback to string representation
            return len(str(obj).encode('utf-8'))
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key, count_as_expiration=True)
    
    def _remove_entry(self, key: str, count_as_expiration: bool = False):
        """Remove entry from cache and all tracking structures."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats.size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
            
            if count_as_expiration:
                self._stats.expirations += 1
            else:
                self._stats.evictions += 1
            
            # Clean up tracking structures
            self._access_order.pop(key, None)
            self._insertion_order.pop(key, None)
            if key in self._access_frequency:
                del self._access_frequency[key]
    
    def _evict_if_needed(self):
        """Evict entries if cache limits are exceeded."""
        # Clean up expired entries first
        self._cleanup_expired()
        
        # Check size limits
        while (len(self._cache) > self.max_size or 
               self._stats.size_bytes > self.max_memory_bytes):
            
            if not self._cache:
                break
            
            key_to_evict = self._select_eviction_candidate()
            if key_to_evict:
                self._remove_entry(key_to_evict)
            else:
                break
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select key for eviction based on policy."""
        if not self._cache:
            return None
        
        if self.eviction_policy == EvictionPolicy.LRU:
            return next(iter(self._access_order))
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            min_frequency = min(self._access_frequency.values())
            for key in self._cache:
                if self._access_frequency[key] == min_frequency:
                    return key
        
        elif self.eviction_policy == EvictionPolicy.FIFO:
            return next(iter(self._insertion_order))
        
        elif self.eviction_policy == EvictionPolicy.RANDOM:
            import random
            return random.choice(list(self._cache.keys()))
        
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Evict entry with earliest expiration
            earliest_expiry = None
            earliest_key = None
            for key, entry in self._cache.items():
                if entry.expires_at:
                    if earliest_expiry is None or entry.expires_at < earliest_expiry:
                        earliest_expiry = entry.expires_at
                        earliest_key = key
            return earliest_key
        
        return None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats.misses += 1
                return default
            
            if entry.is_expired:
                self._remove_entry(key, count_as_expiration=True)
                self._stats.misses += 1
                return default
            
            # Update access tracking
            entry.touch()
            self._access_order.move_to_end(key)
            self._access_frequency[key] += 1
            
            self._stats.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self._lock:
            # Calculate expiry time
            expires_at = None
            ttl_to_use = ttl if ttl is not None else self.default_ttl
            if ttl_to_use:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl_to_use)
            
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Create entry
            entry = CacheEntry(
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                size_bytes=size_bytes
            )
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.size_bytes -= old_entry.size_bytes
            else:
                self._stats.entry_count += 1
            
            # Add new entry
            self._cache[key] = entry
            self._stats.size_bytes += size_bytes
            self._stats.sets += 1
            
            # Update tracking structures
            self._access_order[key] = True
            self._insertion_order[key] = True
            self._access_frequency[key] = 1
            
            # Evict if necessary
            self._evict_if_needed()
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                self._stats.deletes += 1
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            
            if entry.is_expired:
                self._remove_entry(key, count_as_expiration=True)
                return False
            
            return True
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._insertion_order.clear()
            self._access_frequency.clear()
            self._stats = CacheStats()
    
    def keys(self) -> List[str]:
        """Get all non-expired keys."""
        with self._lock:
            self._cleanup_expired()
            return list(self._cache.keys())
    
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            # Update current stats
            self._stats.entry_count = len(self._cache)
            return self._stats
    
    def info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        with self._lock:
            return {
                'max_size': self.max_size,
                'max_memory_bytes': self.max_memory_bytes,
                'default_ttl': self.default_ttl,
                'eviction_policy': self.eviction_policy.value,
                'stats': self.stats().to_dict()
            }


class CacheManager:
    """Central cache management with multiple backends."""
    
    def __init__(self):
        self._caches: Dict[str, InMemoryCache] = {}
        self._default_cache: Optional[InMemoryCache] = None
        self._warmup_tasks: Dict[str, Callable] = {}
    
    def create_cache(
        self,
        name: str,
        backend: CacheBackend = CacheBackend.MEMORY,
        **config
    ) -> InMemoryCache:
        """Create a named cache."""
        if backend == CacheBackend.MEMORY:
            cache = InMemoryCache(**config)
        else:
            # Other backends would be implemented here
            raise NotImplementedError(f"Cache backend {backend} not yet implemented")
        
        self._caches[name] = cache
        
        if self._default_cache is None:
            self._default_cache = cache
        
        logger.info(f"Created cache '{name}' with backend {backend.value}")
        return cache
    
    def get_cache(self, name: str) -> Optional[InMemoryCache]:
        """Get cache by name."""
        return self._caches.get(name)
    
    def get_default_cache(self) -> Optional[InMemoryCache]:
        """Get default cache."""
        return self._default_cache
    
    def set_default_cache(self, name: str):
        """Set default cache by name."""
        cache = self._caches.get(name)
        if cache:
            self._default_cache = cache
        else:
            raise CacheError(f"Cache '{name}' not found")
    
    def register_warmup_task(self, cache_name: str, warmup_func: Callable):
        """Register cache warmup task."""
        self._warmup_tasks[cache_name] = warmup_func
    
    async def warmup_cache(self, cache_name: str):
        """Warm up a specific cache."""
        if cache_name not in self._warmup_tasks:
            logger.warning(f"No warmup task registered for cache '{cache_name}'")
            return
        
        cache = self.get_cache(cache_name)
        if not cache:
            logger.error(f"Cache '{cache_name}' not found")
            return
        
        logger.info(f"Starting warmup for cache '{cache_name}'")
        
        try:
            warmup_func = self._warmup_tasks[cache_name]
            if asyncio.iscoroutinefunction(warmup_func):
                await warmup_func(cache)
            else:
                warmup_func(cache)
            
            logger.info(f"Cache '{cache_name}' warmup completed")
            
        except Exception as e:
            logger.error(f"Cache warmup failed for '{cache_name}': {e}")
    
    async def warmup_all_caches(self):
        """Warm up all registered caches."""
        tasks = [
            self.warmup_cache(cache_name)
            for cache_name in self._warmup_tasks.keys()
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            name: cache.info()
            for name, cache in self._caches.items()
        }
    
    def clear_all_caches(self):
        """Clear all caches."""
        for cache in self._caches.values():
            cache.clear()


class CacheDecorator:
    """Decorator for caching function results."""
    
    def __init__(
        self,
        cache: Optional[InMemoryCache] = None,
        cache_name: str = "default",
        ttl: Optional[int] = None,
        key_builder: Optional[Callable] = None
    ):
        self.cache = cache
        self.cache_name = cache_name
        self.ttl = ttl
        self.key_builder = key_builder or self._default_key_builder
    
    def _default_key_builder(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Default cache key builder."""
        key_parts = [func.__module__, func.__name__]
        
        # Add args
        if args:
            key_parts.extend(str(arg) for arg in args)
        
        # Add kwargs
        if kwargs:
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
        
        key = ":".join(key_parts)
        
        # Hash long keys
        if len(key) > 250:
            key = hashlib.sha256(key.encode()).hexdigest()
        
        return key
    
    def __call__(self, func: Callable) -> Callable:
        """Apply caching to function."""
        def wrapper(*args, **kwargs):
            # Get cache instance
            cache = self.cache
            if cache is None:
                cache = cache_manager.get_cache(self.cache_name)
                if cache is None:
                    # No cache available, execute function normally
                    return func(*args, **kwargs)
            
            # Build cache key
            cache_key = self.key_builder(func, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=self.ttl)
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


@asynccontextmanager
async def cache_lock(cache: InMemoryCache, key: str, timeout: int = 60):
    """Distributed cache lock context manager."""
    lock_key = f"lock:{key}"
    lock_value = f"{time.time()}:{threading.get_ident()}"
    acquired = False
    
    try:
        # Try to acquire lock
        start_time = time.time()
        while time.time() - start_time < timeout:
            if cache.set(lock_key, lock_value, ttl=timeout):
                # Check if we actually got the lock (avoid race conditions)
                if cache.get(lock_key) == lock_value:
                    acquired = True
                    break
            
            await asyncio.sleep(0.1)
        
        if not acquired:
            raise CacheError(f"Failed to acquire cache lock for key: {key}")
        
        yield
        
    finally:
        # Release lock
        if acquired:
            current_value = cache.get(lock_key)
            if current_value == lock_value:
                cache.delete(lock_key)


# Global cache manager instance
cache_manager = CacheManager()


def get_cache(name: str = "default") -> Optional[InMemoryCache]:
    """Get cache instance by name."""
    return cache_manager.get_cache(name)


def cached(
    cache_name: str = "default",
    ttl: Optional[int] = None,
    key_builder: Optional[Callable] = None
):
    """Cache decorator for functions."""
    return CacheDecorator(
        cache_name=cache_name,
        ttl=ttl,
        key_builder=key_builder
    )


def create_cache(
    name: str = "default",
    max_size: int = 1000,
    max_memory_mb: int = 100,
    default_ttl: Optional[int] = None,
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
) -> InMemoryCache:
    """Create and register a new cache."""
    return cache_manager.create_cache(
        name=name,
        backend=CacheBackend.MEMORY,
        max_size=max_size,
        max_memory_bytes=max_memory_mb * 1024 * 1024,
        default_ttl=default_ttl,
        eviction_policy=eviction_policy
    )


def clear_all_caches():
    """Clear all registered caches."""
    cache_manager.clear_all_caches()


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches."""
    return cache_manager.get_all_stats()