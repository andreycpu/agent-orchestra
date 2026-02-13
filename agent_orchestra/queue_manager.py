"""
Queue management utilities for Agent Orchestra.

This module provides priority queues, task queues, and queue management
utilities for handling task distribution and processing.
"""
import asyncio
import heapq
import threading
import time
from typing import Any, Dict, List, Optional, Callable, Generic, TypeVar, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import logging

from .types import Task, TaskPriority
from .exceptions import QueueError, ConfigurationError


logger = logging.getLogger(__name__)

T = TypeVar('T')


class QueueType(str, Enum):
    """Queue types."""
    FIFO = "fifo"
    LIFO = "lifo"
    PRIORITY = "priority"
    DELAY = "delay"
    ROUND_ROBIN = "round_robin"


@dataclass
class QueueItem(Generic[T]):
    """Queue item with metadata."""
    data: T
    priority: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    delay_until: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare items for priority queue (higher priority first)."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Reverse for max-heap behavior
        return self.timestamp < other.timestamp


@dataclass
class QueueStats:
    """Queue statistics."""
    size: int = 0
    enqueued_total: int = 0
    dequeued_total: int = 0
    failed_total: int = 0
    retried_total: int = 0
    average_wait_time_ms: float = 0.0
    max_size_reached: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'size': self.size,
            'enqueued_total': self.enqueued_total,
            'dequeued_total': self.dequeued_total,
            'failed_total': self.failed_total,
            'retried_total': self.retried_total,
            'throughput': self.dequeued_total / max(1, self.enqueued_total),
            'average_wait_time_ms': self.average_wait_time_ms,
            'max_size_reached': self.max_size_reached
        }


class BaseQueue(Generic[T]):
    """Base queue implementation with common functionality."""
    
    def __init__(self, max_size: Optional[int] = None, name: str = "queue"):
        self.max_size = max_size
        self.name = name
        self._lock = threading.RLock()
        self._stats = QueueStats()
        self._created_at = datetime.utcnow()
        self._wait_times: deque = deque(maxlen=1000)  # Keep last 1000 wait times
    
    def size(self) -> int:
        """Get current queue size."""
        raise NotImplementedError
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self.max_size is not None and self.size() >= self.max_size
    
    def put(self, item: T, **kwargs) -> bool:
        """Put item in queue."""
        raise NotImplementedError
    
    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get item from queue."""
        raise NotImplementedError
    
    def peek(self) -> Optional[T]:
        """Peek at next item without removing it."""
        raise NotImplementedError
    
    def clear(self):
        """Clear all items from queue."""
        raise NotImplementedError
    
    def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        with self._lock:
            self._stats.size = self.size()
            if self._wait_times:
                self._stats.average_wait_time_ms = sum(self._wait_times) / len(self._wait_times)
            return self._stats
    
    def _update_enqueue_stats(self):
        """Update enqueue statistics."""
        with self._lock:
            self._stats.enqueued_total += 1
            current_size = self.size()
            self._stats.max_size_reached = max(self._stats.max_size_reached, current_size)
    
    def _update_dequeue_stats(self, wait_time_ms: float):
        """Update dequeue statistics."""
        with self._lock:
            self._stats.dequeued_total += 1
            self._wait_times.append(wait_time_ms)


class FIFOQueue(BaseQueue[T]):
    """First-In-First-Out queue."""
    
    def __init__(self, max_size: Optional[int] = None, name: str = "fifo_queue"):
        super().__init__(max_size, name)
        self._queue: deque = deque()
    
    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)
    
    def put(self, item: T, **kwargs) -> bool:
        """Put item in queue."""
        with self._lock:
            if self.is_full():
                return False
            
            queue_item = QueueItem(data=item, **kwargs)
            self._queue.append(queue_item)
            self._update_enqueue_stats()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get item from queue."""
        start_time = time.time()
        
        with self._lock:
            if self._queue:
                queue_item = self._queue.popleft()
                wait_time_ms = (time.time() - start_time) * 1000
                self._update_dequeue_stats(wait_time_ms)
                return queue_item.data
        
        return None
    
    def peek(self) -> Optional[T]:
        """Peek at next item without removing it."""
        with self._lock:
            if self._queue:
                return self._queue[0].data
        return None
    
    def clear(self):
        """Clear all items from queue."""
        with self._lock:
            self._queue.clear()


class LIFOQueue(BaseQueue[T]):
    """Last-In-First-Out queue (stack)."""
    
    def __init__(self, max_size: Optional[int] = None, name: str = "lifo_queue"):
        super().__init__(max_size, name)
        self._queue: List[QueueItem[T]] = []
    
    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)
    
    def put(self, item: T, **kwargs) -> bool:
        """Put item in queue."""
        with self._lock:
            if self.is_full():
                return False
            
            queue_item = QueueItem(data=item, **kwargs)
            self._queue.append(queue_item)
            self._update_enqueue_stats()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get item from queue."""
        start_time = time.time()
        
        with self._lock:
            if self._queue:
                queue_item = self._queue.pop()
                wait_time_ms = (time.time() - start_time) * 1000
                self._update_dequeue_stats(wait_time_ms)
                return queue_item.data
        
        return None
    
    def peek(self) -> Optional[T]:
        """Peek at next item without removing it."""
        with self._lock:
            if self._queue:
                return self._queue[-1].data
        return None
    
    def clear(self):
        """Clear all items from queue."""
        with self._lock:
            self._queue.clear()


class PriorityQueue(BaseQueue[T]):
    """Priority queue using heap."""
    
    def __init__(self, max_size: Optional[int] = None, name: str = "priority_queue"):
        super().__init__(max_size, name)
        self._heap: List[QueueItem[T]] = []
    
    def size(self) -> int:
        """Get current queue size."""
        return len(self._heap)
    
    def put(self, item: T, priority: int = 0, **kwargs) -> bool:
        """Put item in queue with priority."""
        with self._lock:
            if self.is_full():
                return False
            
            queue_item = QueueItem(data=item, priority=priority, **kwargs)
            heapq.heappush(self._heap, queue_item)
            self._update_enqueue_stats()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get highest priority item from queue."""
        start_time = time.time()
        
        with self._lock:
            if self._heap:
                queue_item = heapq.heappop(self._heap)
                wait_time_ms = (time.time() - start_time) * 1000
                self._update_dequeue_stats(wait_time_ms)
                return queue_item.data
        
        return None
    
    def peek(self) -> Optional[T]:
        """Peek at highest priority item without removing it."""
        with self._lock:
            if self._heap:
                return self._heap[0].data
        return None
    
    def clear(self):
        """Clear all items from queue."""
        with self._lock:
            self._heap.clear()


class DelayQueue(BaseQueue[T]):
    """Queue with delayed delivery of items."""
    
    def __init__(self, max_size: Optional[int] = None, name: str = "delay_queue"):
        super().__init__(max_size, name)
        self._heap: List[QueueItem[T]] = []
    
    def size(self) -> int:
        """Get current queue size."""
        return len(self._heap)
    
    def put(self, item: T, delay_seconds: float = 0, **kwargs) -> bool:
        """Put item in queue with delay."""
        with self._lock:
            if self.is_full():
                return False
            
            delay_until = datetime.utcnow() + timedelta(seconds=delay_seconds)
            queue_item = QueueItem(data=item, delay_until=delay_until, **kwargs)
            
            # Use delay time as priority (earlier times have higher priority)
            queue_item.priority = -int(delay_until.timestamp())
            
            heapq.heappush(self._heap, queue_item)
            self._update_enqueue_stats()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get item that is ready for delivery."""
        start_time = time.time()
        now = datetime.utcnow()
        
        with self._lock:
            # Check if any items are ready
            while self._heap:
                queue_item = self._heap[0]
                if queue_item.delay_until and queue_item.delay_until <= now:
                    queue_item = heapq.heappop(self._heap)
                    wait_time_ms = (time.time() - start_time) * 1000
                    self._update_dequeue_stats(wait_time_ms)
                    return queue_item.data
                else:
                    break
        
        return None
    
    def peek(self) -> Optional[T]:
        """Peek at next ready item without removing it."""
        now = datetime.utcnow()
        
        with self._lock:
            if self._heap:
                queue_item = self._heap[0]
                if not queue_item.delay_until or queue_item.delay_until <= now:
                    return queue_item.data
        
        return None
    
    def get_next_ready_time(self) -> Optional[datetime]:
        """Get time when next item will be ready."""
        with self._lock:
            if self._heap:
                return self._heap[0].delay_until
        return None
    
    def clear(self):
        """Clear all items from queue."""
        with self._lock:
            self._heap.clear()


class RoundRobinQueue(BaseQueue[T]):
    """Round-robin queue distributing items across multiple sub-queues."""
    
    def __init__(self, num_queues: int = 3, max_size: Optional[int] = None, name: str = "round_robin_queue"):
        super().__init__(max_size, name)
        self.num_queues = num_queues
        self._queues: List[deque] = [deque() for _ in range(num_queues)]
        self._current_queue = 0
        self._get_index = 0
    
    def size(self) -> int:
        """Get total size across all sub-queues."""
        return sum(len(q) for q in self._queues)
    
    def put(self, item: T, **kwargs) -> bool:
        """Put item in next queue in round-robin fashion."""
        with self._lock:
            if self.is_full():
                return False
            
            queue_item = QueueItem(data=item, **kwargs)
            self._queues[self._current_queue].append(queue_item)
            self._current_queue = (self._current_queue + 1) % self.num_queues
            self._update_enqueue_stats()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get item from queues in round-robin fashion."""
        start_time = time.time()
        
        with self._lock:
            # Try each queue starting from current get index
            for _ in range(self.num_queues):
                queue = self._queues[self._get_index]
                if queue:
                    queue_item = queue.popleft()
                    self._get_index = (self._get_index + 1) % self.num_queues
                    wait_time_ms = (time.time() - start_time) * 1000
                    self._update_dequeue_stats(wait_time_ms)
                    return queue_item.data
                
                self._get_index = (self._get_index + 1) % self.num_queues
        
        return None
    
    def peek(self) -> Optional[T]:
        """Peek at next item that would be returned by get."""
        with self._lock:
            for i in range(self.num_queues):
                queue_index = (self._get_index + i) % self.num_queues
                queue = self._queues[queue_index]
                if queue:
                    return queue[0].data
        return None
    
    def clear(self):
        """Clear all sub-queues."""
        with self._lock:
            for queue in self._queues:
                queue.clear()
    
    def get_queue_sizes(self) -> List[int]:
        """Get sizes of all sub-queues."""
        with self._lock:
            return [len(q) for q in self._queues]


class QueueManager:
    """Manages multiple named queues."""
    
    def __init__(self):
        self._queues: Dict[str, BaseQueue] = {}
        self._lock = threading.Lock()
    
    def create_queue(
        self,
        name: str,
        queue_type: QueueType,
        max_size: Optional[int] = None,
        **queue_options
    ) -> BaseQueue:
        """Create a new named queue."""
        with self._lock:
            if name in self._queues:
                raise QueueError(f"Queue '{name}' already exists")
            
            if queue_type == QueueType.FIFO:
                queue = FIFOQueue(max_size, name)
            elif queue_type == QueueType.LIFO:
                queue = LIFOQueue(max_size, name)
            elif queue_type == QueueType.PRIORITY:
                queue = PriorityQueue(max_size, name)
            elif queue_type == QueueType.DELAY:
                queue = DelayQueue(max_size, name)
            elif queue_type == QueueType.ROUND_ROBIN:
                num_queues = queue_options.get('num_queues', 3)
                queue = RoundRobinQueue(num_queues, max_size, name)
            else:
                raise ConfigurationError(f"Unknown queue type: {queue_type}")
            
            self._queues[name] = queue
            logger.info(f"Created queue '{name}' of type {queue_type.value}")
            return queue
    
    def get_queue(self, name: str) -> Optional[BaseQueue]:
        """Get queue by name."""
        return self._queues.get(name)
    
    def delete_queue(self, name: str) -> bool:
        """Delete a queue."""
        with self._lock:
            if name in self._queues:
                del self._queues[name]
                logger.info(f"Deleted queue '{name}'")
                return True
            return False
    
    def list_queues(self) -> List[str]:
        """List all queue names."""
        return list(self._queues.keys())
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all queues."""
        return {
            name: queue.get_stats().to_dict()
            for name, queue in self._queues.items()
        }
    
    def clear_all_queues(self):
        """Clear all queues."""
        for queue in self._queues.values():
            queue.clear()
    
    def total_size(self) -> int:
        """Get total size across all queues."""
        return sum(queue.size() for queue in self._queues.values())


class AsyncQueue:
    """Async wrapper for queue operations."""
    
    def __init__(self, queue: BaseQueue[T]):
        self.queue = queue
        self._condition = asyncio.Condition()
    
    async def put(self, item: T, **kwargs) -> bool:
        """Put item in queue (async)."""
        result = self.queue.put(item, **kwargs)
        if result:
            async with self._condition:
                self._condition.notify_all()
        return result
    
    async def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get item from queue (async)."""
        async with self._condition:
            end_time = time.time() + timeout if timeout else None
            
            while self.queue.is_empty():
                if end_time and time.time() >= end_time:
                    return None
                
                wait_time = min(0.1, end_time - time.time()) if end_time else 0.1
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=wait_time)
                except asyncio.TimeoutError:
                    if end_time and time.time() >= end_time:
                        return None
            
            return self.queue.get()
    
    async def get_nowait(self) -> Optional[T]:
        """Get item from queue without waiting."""
        return self.queue.get()
    
    def size(self) -> int:
        """Get queue size."""
        return self.queue.size()


# Global queue manager
queue_manager = QueueManager()


def create_queue(
    name: str,
    queue_type: QueueType = QueueType.FIFO,
    max_size: Optional[int] = None,
    **options
) -> BaseQueue:
    """Create a named queue."""
    return queue_manager.create_queue(name, queue_type, max_size, **options)


def get_queue(name: str) -> Optional[BaseQueue]:
    """Get queue by name."""
    return queue_manager.get_queue(name)


def get_queue_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all queues."""
    return queue_manager.get_all_stats()