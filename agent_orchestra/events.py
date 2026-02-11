"""
Event system for Agent Orchestra - pub/sub messaging and event handling
"""
import asyncio
import json
from typing import Dict, List, Any, Callable, Optional, Set, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import structlog

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Standard event types in the system"""
    TASK_SUBMITTED = "task.submitted"
    TASK_STARTED = "task.started" 
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_TIMEOUT = "task.timeout"
    TASK_CANCELLED = "task.cancelled"
    
    AGENT_REGISTERED = "agent.registered"
    AGENT_UNREGISTERED = "agent.unregistered"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    AGENT_HEARTBEAT = "agent.heartbeat"
    AGENT_ERROR = "agent.error"
    
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_ERROR = "system.error"
    SYSTEM_ALERT = "system.alert"
    
    METRICS_COLLECTED = "metrics.collected"
    THRESHOLD_EXCEEDED = "threshold.exceeded"


@dataclass
class Event:
    """Represents a single event in the system"""
    type: str
    source: str
    data: Dict[str, Any]
    timestamp: datetime = None
    id: str = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.id is None:
            import uuid
            self.id = str(uuid.uuid4())


class EventHandler:
    """Base class for event handlers"""
    
    async def handle(self, event: Event) -> bool:
        """Handle an event. Return True if handled successfully."""
        raise NotImplementedError
    
    def should_handle(self, event: Event) -> bool:
        """Check if this handler should process the event"""
        return True


class AsyncEventHandler(EventHandler):
    """Event handler that wraps an async function"""
    
    def __init__(self, handler_func: Callable[[Event], Any], event_types: Set[str] = None):
        self.handler_func = handler_func
        self.event_types = event_types or set()
    
    async def handle(self, event: Event) -> bool:
        """Execute the handler function"""
        try:
            if asyncio.iscoroutinefunction(self.handler_func):
                await self.handler_func(event)
            else:
                self.handler_func(event)
            return True
        except Exception as e:
            logger.error("Event handler failed", 
                        event_type=event.type,
                        handler=self.handler_func.__name__,
                        error=str(e))
            return False
    
    def should_handle(self, event: Event) -> bool:
        """Check if this handler should process the event"""
        if not self.event_types:
            return True
        return event.type in self.event_types


class EventBus:
    """Central event bus for pub/sub messaging"""
    
    def __init__(self, redis_client=None):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._event_history: List[Event] = []
        self._redis = redis_client
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._running = False
        self._redis_subscriber_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the event bus"""
        self._running = True
        
        # Start Redis subscriber if available
        if self._redis:
            self._redis_subscriber_task = asyncio.create_task(
                self._redis_subscriber_loop()
            )
        
        logger.info("Event bus started")
    
    async def stop(self):
        """Stop the event bus"""
        self._running = False
        
        if self._redis_subscriber_task:
            self._redis_subscriber_task.cancel()
            try:
                await self._redis_subscriber_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event bus stopped")
    
    def subscribe(self, event_type: str, handler: Union[EventHandler, Callable]):
        """Subscribe a handler to specific event type"""
        if not isinstance(handler, EventHandler):
            # Wrap function in AsyncEventHandler
            handler = AsyncEventHandler(handler, {event_type})
        
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        self._handlers[event_type].append(handler)
        
        logger.info("Handler subscribed", 
                   event_type=event_type,
                   handler=handler.__class__.__name__)
    
    def subscribe_all(self, handler: Union[EventHandler, Callable]):
        """Subscribe a handler to all events"""
        if not isinstance(handler, EventHandler):
            handler = AsyncEventHandler(handler)
        
        self._global_handlers.append(handler)
        
        logger.info("Global handler subscribed",
                   handler=handler.__class__.__name__)
    
    def unsubscribe(self, event_type: str, handler: EventHandler):
        """Unsubscribe a handler from event type"""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                logger.info("Handler unsubscribed", 
                           event_type=event_type,
                           handler=handler.__class__.__name__)
            except ValueError:
                pass
    
    async def publish(self, event: Event):
        """Publish an event to all subscribers"""
        if not self._running:
            return
        
        # Store in history (limited size)
        self._event_history.append(event)
        if len(self._event_history) > 1000:
            self._event_history.pop(0)
        
        logger.debug("Event published", 
                    event_type=event.type,
                    source=event.source,
                    event_id=event.id)
        
        # Publish to Redis if available
        if self._redis:
            await self._publish_to_redis(event)
        
        # Notify local handlers
        await self._notify_handlers(event)
        
        # Notify queue subscribers
        await self._notify_queue_subscribers(event)
    
    async def emit(self, event_type: str, source: str, data: Dict[str, Any], **kwargs):
        """Convenient method to create and publish an event"""
        event = Event(
            type=event_type,
            source=source,
            data=data,
            **kwargs
        )
        await self.publish(event)
    
    async def _notify_handlers(self, event: Event):
        """Notify all relevant handlers of an event"""
        handlers_to_notify = []
        
        # Add specific handlers
        if event.type in self._handlers:
            handlers_to_notify.extend(
                h for h in self._handlers[event.type]
                if h.should_handle(event)
            )
        
        # Add global handlers
        handlers_to_notify.extend(
            h for h in self._global_handlers
            if h.should_handle(event)
        )
        
        # Notify all handlers concurrently
        if handlers_to_notify:
            await asyncio.gather(
                *[handler.handle(event) for handler in handlers_to_notify],
                return_exceptions=True
            )
    
    async def _notify_queue_subscribers(self, event: Event):
        """Notify queue-based subscribers"""
        # Notify specific event type subscribers
        if event.type in self._subscribers:
            for queue in self._subscribers[event.type]:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning("Subscriber queue full, dropping event",
                                 event_type=event.type)
        
        # Notify wildcard subscribers
        if "*" in self._subscribers:
            for queue in self._subscribers["*"]:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning("Subscriber queue full, dropping event",
                                 event_type=event.type)
    
    async def _publish_to_redis(self, event: Event):
        """Publish event to Redis for distributed subscribers"""
        try:
            event_data = {
                "type": event.type,
                "source": event.source,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
                "id": event.id,
                "correlation_id": event.correlation_id
            }
            
            # Publish to specific channel and wildcard channel
            await self._redis.publish(
                f"events:{event.type}",
                json.dumps(event_data, default=str)
            )
            await self._redis.publish(
                "events:*",
                json.dumps(event_data, default=str)
            )
            
        except Exception as e:
            logger.error("Failed to publish to Redis", error=str(e))
    
    async def _redis_subscriber_loop(self):
        """Background task to handle Redis subscriptions"""
        try:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe("events:*")
            
            while self._running:
                message = await pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    await self._handle_redis_message(message)
                    
        except Exception as e:
            logger.error("Redis subscriber loop failed", error=str(e))
        finally:
            try:
                await pubsub.close()
            except:
                pass
    
    async def _handle_redis_message(self, message):
        """Handle incoming Redis message"""
        try:
            event_data = json.loads(message["data"].decode())
            
            event = Event(
                type=event_data["type"],
                source=event_data["source"], 
                data=event_data["data"],
                timestamp=datetime.fromisoformat(event_data["timestamp"]),
                id=event_data["id"],
                correlation_id=event_data.get("correlation_id")
            )
            
            # Only notify local handlers, not Redis again
            await self._notify_handlers(event)
            await self._notify_queue_subscribers(event)
            
        except Exception as e:
            logger.error("Failed to handle Redis message", error=str(e))
    
    async def create_subscriber(self, event_type: str = "*", queue_size: int = 100) -> asyncio.Queue:
        """Create a queue-based subscriber for events"""
        queue = asyncio.Queue(maxsize=queue_size)
        
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        
        self._subscribers[event_type].add(queue)
        
        logger.info("Queue subscriber created", 
                   event_type=event_type,
                   queue_size=queue_size)
        
        return queue
    
    def remove_subscriber(self, event_type: str, queue: asyncio.Queue):
        """Remove a queue-based subscriber"""
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(queue)
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get recent event history"""
        if event_type:
            events = [e for e in self._event_history if e.type == event_type]
        else:
            events = self._event_history
        
        return events[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        handler_count = sum(len(handlers) for handlers in self._handlers.values())
        
        return {
            "event_types": list(self._handlers.keys()),
            "total_handlers": handler_count + len(self._global_handlers),
            "specific_handlers": handler_count,
            "global_handlers": len(self._global_handlers),
            "queue_subscribers": sum(len(subs) for subs in self._subscribers.values()),
            "events_in_history": len(self._event_history),
            "redis_enabled": self._redis is not None
        }


class EventLogger(EventHandler):
    """Event handler that logs all events"""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level.upper()
    
    async def handle(self, event: Event) -> bool:
        """Log the event"""
        log_data = {
            "event_type": event.type,
            "source": event.source,
            "event_id": event.id,
            "timestamp": event.timestamp.isoformat()
        }
        
        if self.log_level == "DEBUG":
            log_data["data"] = event.data
        
        logger.info("Event occurred", **log_data)
        return True


class MetricsCollectorHandler(EventHandler):
    """Event handler that collects metrics from events"""
    
    def __init__(self):
        self.event_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
    
    async def handle(self, event: Event) -> bool:
        """Collect metrics from event"""
        # Count events by type
        self.event_counts[event.type] = self.event_counts.get(event.type, 0) + 1
        
        # Track errors
        if "error" in event.type or "failed" in event.type:
            source = event.source
            self.error_counts[source] = self.error_counts.get(source, 0) + 1
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        return {
            "event_counts": self.event_counts.copy(),
            "error_counts": self.error_counts.copy(),
            "total_events": sum(self.event_counts.values()),
            "total_errors": sum(self.error_counts.values())
        }


class AlertHandler(EventHandler):
    """Event handler that generates alerts based on conditions"""
    
    def __init__(self, alert_callback: Callable[[str, Event], None]):
        self.alert_callback = alert_callback
        self.alert_conditions: Dict[str, Callable[[Event], bool]] = {}
    
    def add_condition(self, name: str, condition: Callable[[Event], bool]):
        """Add an alert condition"""
        self.alert_conditions[name] = condition
    
    async def handle(self, event: Event) -> bool:
        """Check alert conditions"""
        for condition_name, condition_func in self.alert_conditions.items():
            try:
                if condition_func(event):
                    self.alert_callback(condition_name, event)
            except Exception as e:
                logger.error("Alert condition failed",
                           condition=condition_name,
                           error=str(e))
        
        return True


def create_standard_event_handlers(event_bus: EventBus) -> Dict[str, EventHandler]:
    """Create standard event handlers"""
    handlers = {}
    
    # Event logger
    event_logger = EventLogger("INFO")
    event_bus.subscribe_all(event_logger)
    handlers["logger"] = event_logger
    
    # Metrics collector
    metrics_collector = MetricsCollectorHandler()
    event_bus.subscribe_all(metrics_collector)
    handlers["metrics"] = metrics_collector
    
    # Alert handler with common conditions
    def alert_callback(condition: str, event: Event):
        logger.warning("Alert triggered",
                      condition=condition,
                      event_type=event.type,
                      event_source=event.source)
    
    alert_handler = AlertHandler(alert_callback)
    
    # Add common alert conditions
    alert_handler.add_condition(
        "high_failure_rate",
        lambda e: e.type == EventType.TASK_FAILED
    )
    
    alert_handler.add_condition(
        "agent_error",
        lambda e: e.type == EventType.AGENT_ERROR
    )
    
    alert_handler.add_condition(
        "system_error",
        lambda e: e.type == EventType.SYSTEM_ERROR
    )
    
    event_bus.subscribe_all(alert_handler)
    handlers["alerts"] = alert_handler
    
    return handlers