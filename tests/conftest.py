"""
Pytest configuration and shared fixtures for Agent Orchestra tests
"""
import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock
import structlog

# Configure structured logging for tests
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.testing.LogCapture(),
    ],
    wrapper_class=structlog.testing.LogCapture,
    logger_factory=structlog.testing.TestingLoggerFactory(),
    cache_logger_on_first_use=True,
)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    redis_mock = Mock()
    
    # Mock Redis methods
    redis_mock.hset = Mock(return_value=asyncio.coroutine(lambda: True)())
    redis_mock.hget = Mock(return_value=asyncio.coroutine(lambda: None)())
    redis_mock.hgetall = Mock(return_value=asyncio.coroutine(lambda: {})())
    redis_mock.set = Mock(return_value=asyncio.coroutine(lambda: True)())
    redis_mock.get = Mock(return_value=asyncio.coroutine(lambda: None)())
    redis_mock.delete = Mock(return_value=asyncio.coroutine(lambda: 1)())
    redis_mock.expire = Mock(return_value=asyncio.coroutine(lambda: True)())
    redis_mock.close = Mock(return_value=asyncio.coroutine(lambda: None)())
    
    return redis_mock


@pytest.fixture
def test_config():
    """Test configuration dictionary"""
    return {
        "orchestra": {
            "max_concurrent_tasks": 10,
            "task_timeout_default": 30,
            "heartbeat_interval": 5
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 15  # Use different DB for tests
        },
        "logging": {
            "level": "DEBUG"
        }
    }


@pytest.fixture
def sample_task_data():
    """Sample task data for testing"""
    return {
        "type": "test_task",
        "data": {"message": "test data", "number": 42},
        "priority": "normal",
        "timeout": 30
    }


@pytest.fixture
def sample_agent_config():
    """Sample agent configuration"""
    return {
        "id": "test-agent",
        "name": "Test Agent",
        "capabilities": ["test_task", "another_task"]
    }


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "redis_required: mark test as requiring Redis"
    )


# Skip Redis tests if Redis is not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle Redis requirements"""
    import socket
    
    def is_redis_available():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 6379))
            sock.close()
            return result == 0
        except:
            return False
    
    redis_available = is_redis_available()
    
    for item in items:
        if "redis_required" in item.keywords and not redis_available:
            item.add_marker(pytest.mark.skip(reason="Redis not available"))


class AsyncTestHelper:
    """Helper class for async testing utilities"""
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
        """Wait for a condition to become true"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            await asyncio.sleep(interval)
        
        return False
    
    @staticmethod
    async def collect_events(event_bus, event_type, count=1, timeout=5.0):
        """Collect events from event bus for testing"""
        events = []
        
        def event_handler(event):
            events.append(event)
        
        event_bus.subscribe(event_type, event_handler)
        
        # Wait for events
        start_time = asyncio.get_event_loop().time()
        while len(events) < count and (asyncio.get_event_loop().time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        return events


@pytest.fixture
def async_helper():
    """Provide async testing helper"""
    return AsyncTestHelper()


# Environment setup for tests
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    # Set test-specific environment variables
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    
    yield
    
    # Cleanup
    os.environ.pop("TESTING", None)