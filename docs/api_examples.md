# API Examples

This document provides practical examples for using Agent Orchestra's APIs and utilities.

## Basic Task Creation and Execution

### Creating a Simple Task

```python
from agent_orchestra.types import Task, TaskPriority

# Create a basic task
task = Task(
    type="data_processing",
    data={
        "input_file": "data.csv",
        "output_format": "json",
        "filters": ["active_only", "recent"]
    },
    priority=TaskPriority.HIGH,
    timeout=300  # 5 minutes
)

print(f"Created task {task.id} with priority {task.priority}")
```

### Task with Dependencies

```python
from agent_orchestra.types import Task

# Create parent task
parent_task = Task(
    type="data_extraction",
    data={"source": "database", "table": "users"}
)

# Create dependent task
child_task = Task(
    type="data_transformation",
    data={"format": "json"},
    dependencies=[parent_task.id]
)

print(f"Task {child_task.id} depends on {parent_task.id}")
```

## Using the Validation System

### Input Validation

```python
from agent_orchestra.validation import (
    validate_id, validate_task_type, validate_email, validate_url
)

# Validate various inputs
try:
    task_id = validate_id("task-123", "Task ID")
    task_type = validate_task_type("data_processing")
    email = validate_email("user@example.com")
    url = validate_url("https://api.example.com/v1")
    
    print("All validations passed!")
    
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Configuration Validation

```python
from agent_orchestra.config_validator import (
    ConfigurationLoader, ConfigurationValidator
)

# Load configuration from environment
config = ConfigurationLoader.from_environment()

# Validate configuration
validator = ConfigurationValidator()
warnings = validator.validate(config)

if warnings:
    for warning in warnings:
        print(f"Warning: {warning}")
else:
    print("Configuration is valid!")
```

## Security and Authentication

### Using Security Context

```python
from agent_orchestra.security_utils import (
    TokenManager, RoleManager, SecurityContextBuilder,
    Role, Permission
)

# Setup security components
token_manager = TokenManager("your-secret-key")
role_manager = RoleManager()
context_builder = SecurityContextBuilder(role_manager, token_manager)

# Generate token for user
token = token_manager.generate_token(
    user_id="user123",
    roles=[Role.OPERATOR.value],
    permissions=[Permission.CREATE_TASKS.value]
)

# Create security context from token
context = context_builder.from_token(token)

# Check permissions
if context.has_permission(Permission.CREATE_TASKS):
    print("User can create tasks")
else:
    print("User cannot create tasks")
```

### Password Validation

```python
from agent_orchestra.security_utils import PasswordPolicy

# Create password policy
policy = PasswordPolicy(
    min_length=12,
    require_uppercase=True,
    require_lowercase=True,
    require_numbers=True,
    require_special=True
)

# Validate password
is_valid, violations = policy.validate_password("MySecureP@ssw0rd")

if is_valid:
    print("Password is valid")
else:
    for violation in violations:
        print(f"Password violation: {violation}")
```

## HTTP Client Usage

### Basic HTTP Operations

```python
import asyncio
from agent_orchestra.http_client import HttpClient, RetryPolicy, CircuitBreaker

async def http_examples():
    # Create HTTP client with retry and circuit breaker
    retry_policy = RetryPolicy(max_attempts=3, base_delay=1.0)
    circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    
    async with HttpClient(
        base_url="https://api.example.com",
        retry_policy=retry_policy,
        circuit_breaker=circuit_breaker
    ) as client:
        
        # GET request
        response = await client.get("/users")
        if response.is_success:
            users = response.json()
            print(f"Retrieved {len(users)} users")
        
        # POST request with JSON
        new_user = {"name": "John Doe", "email": "john@example.com"}
        response = await client.json_post("/users", new_user)
        if response.is_success:
            print(f"Created user with ID: {response.json()['id']}")
        
        # Get client statistics
        stats = client.get_stats()
        print(f"Success rate: {stats['success_rate']:.2%}")

# Run async example
asyncio.run(http_examples())
```

### HTTP Client Pool

```python
from agent_orchestra.http_client import create_http_client, get_http_client

# Create named HTTP clients
api_client = create_http_client(
    "main_api", 
    "https://api.example.com",
    default_timeout=30.0
)

auth_client = create_http_client(
    "auth_service",
    "https://auth.example.com",
    default_timeout=10.0
)

# Use clients by name
async def make_authenticated_request():
    auth_client = get_http_client("auth_service")
    api_client = get_http_client("main_api")
    
    # Get auth token
    token_response = await auth_client.post("/token", data={
        "username": "user", 
        "password": "pass"
    })
    
    if token_response.is_success:
        token = token_response.json()["access_token"]
        
        # Use token in API request
        headers = {"Authorization": f"Bearer {token}"}
        data_response = await api_client.get("/protected-data", headers=headers)
        
        return data_response.json()

# Run example
data = asyncio.run(make_authenticated_request())
```

## Database Operations

### Using Database Repository

```python
import asyncio
from agent_orchestra.database_utils import DatabaseManager, DatabaseRepository

async def database_examples():
    # Setup database
    from agent_orchestra.database_utils import DatabaseConfig
    
    config = DatabaseConfig(
        url="postgresql://user:pass@localhost/db",
        pool_size=10
    )
    
    db_manager = DatabaseManager(config)
    await db_manager.initialize()
    
    # Use repository pattern
    class UserRepository(DatabaseRepository):
        def __init__(self, db_manager):
            super().__init__(db_manager, "users")
        
        async def find_by_email(self, email: str):
            builder = self.query_builder()
            query, params = builder.where("email = :email", email=email).build_select()
            result = await self.db_manager.execute_query(query, params)
            return result.first
        
        async def find_active_users(self):
            builder = self.query_builder()
            query, params = builder.where("active = :active", active=True)\
                                  .order_by("created_at", "DESC")\
                                  .limit(100)\
                                  .build_select()
            result = await self.db_manager.execute_query(query, params)
            return result.rows
    
    # Use repository
    user_repo = UserRepository(db_manager)
    
    # Create user
    user_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "active": True
    }
    await user_repo.create(user_data)
    
    # Find user by email
    user = await user_repo.find_by_email("john@example.com")
    print(f"Found user: {user['name']}")
    
    # Get active users
    active_users = await user_repo.find_active_users()
    print(f"Active users: {len(active_users)}")
    
    await db_manager.close()

# Run example
asyncio.run(database_examples())
```

### Database Transactions

```python
async def transaction_example():
    async with db_manager.get_connection() as conn:
        async with conn.transaction():
            # Create user
            user_result = await conn.execute(
                "INSERT INTO users (name, email) VALUES (:name, :email) RETURNING id",
                {"name": "Jane Doe", "email": "jane@example.com"}
            )
            user_id = user_result.rows[0]["id"]
            
            # Create user profile
            await conn.execute(
                "INSERT INTO profiles (user_id, bio) VALUES (:user_id, :bio)",
                {"user_id": user_id, "bio": "Software developer"}
            )
            
            # Transaction is automatically committed
            print(f"Created user {user_id} with profile")

# Run transaction example
asyncio.run(transaction_example())
```

## Caching

### Basic Cache Usage

```python
from agent_orchestra.cache_manager import (
    create_cache, cached, EvictionPolicy, get_cache_stats
)

# Create a cache
cache = create_cache(
    name="user_cache",
    max_size=1000,
    max_memory_mb=50,
    default_ttl=300,  # 5 minutes
    eviction_policy=EvictionPolicy.LRU
)

# Basic cache operations
cache.set("user:123", {"name": "John", "email": "john@example.com"})
user = cache.get("user:123")
print(f"Cached user: {user['name']}")

# Use cache decorator
@cached(cache_name="user_cache", ttl=600)
def get_user_from_db(user_id: str):
    print(f"Loading user {user_id} from database...")  # Only called on cache miss
    return {"id": user_id, "name": f"User {user_id}"}

# First call hits database
user1 = get_user_from_db("456")

# Second call returns cached result
user2 = get_user_from_db("456")

# Get cache statistics
stats = get_cache_stats()
print(f"Cache hit rate: {stats['user_cache']['stats']['hit_rate']:.2%}")
```

### Cache with TTL

```python
import time

# Set item with custom TTL
cache.set("session:abc123", {"user_id": 123, "expires": "2024-12-31"}, ttl=3600)

# Check if item exists
if cache.exists("session:abc123"):
    session = cache.get("session:abc123")
    print(f"Session for user: {session['user_id']}")

# Wait for expiration (in real code, this would be natural timing)
time.sleep(2)  # Simulate time passing

# Item may be expired
session = cache.get("session:abc123", default="expired")
print(f"Session after wait: {session}")
```

## Performance Monitoring

### Basic Performance Tracking

```python
from agent_orchestra.performance_monitor import (
    performance_manager, profile, measure_operation
)

# Start performance monitoring
performance_manager.start_monitoring()

# Use decorator for profiling
@profile("user_processing")
def process_user_data(user_data):
    # Simulate processing
    time.sleep(0.1)
    return {"processed": True, "user_count": len(user_data)}

# Use context manager for measuring
async def process_orders():
    with measure_operation("order_processing", tags={"batch_size": 100}) as timer:
        # Simulate processing
        await asyncio.sleep(0.2)
        return {"orders_processed": 100}

# Execute operations
users = [{"id": i} for i in range(10)]
result = process_user_data(users)

orders = await process_orders()

# Get performance report
report = performance_manager.get_performance_report()
print(f"Current CPU usage: {report['current_resources']['cpu_percent']:.1f}%")
print(f"Average memory usage: {report['average_usage_5min']['memory_percent']:.1f}%")

# Stop monitoring
performance_manager.stop_monitoring()
```

### Custom Performance Metrics

```python
from agent_orchestra.performance_monitor import PerformanceMetric

# Record custom metrics
metrics_collector = performance_manager.metrics_collector

# Record a custom metric
metric = PerformanceMetric(
    name="queue_depth",
    value=150,
    unit="items",
    tags={"queue": "task_queue", "priority": "high"}
)

metrics_collector.record_metric(metric)

# Record operation duration
metrics_collector.record_duration(
    "database_query",
    2.5,  # seconds
    tags={"table": "users", "operation": "select"}
)

# Get operation statistics
stats = metrics_collector.get_operation_stats("database_query")
if stats:
    print(f"Database query stats:")
    print(f"  Average time: {stats['mean']:.3f}s")
    print(f"  95th percentile: {stats['p95']:.3f}s")
    print(f"  Total queries: {stats['count']}")
```

## Queue Management

### Different Queue Types

```python
from agent_orchestra.queue_manager import create_queue, QueueType

# Create different types of queues
fifo_queue = create_queue("tasks", QueueType.FIFO, max_size=1000)
priority_queue = create_queue("priority_tasks", QueueType.PRIORITY, max_size=500)
delay_queue = create_queue("scheduled_tasks", QueueType.DELAY, max_size=200)

# FIFO queue usage
fifo_queue.put("task1")
fifo_queue.put("task2")
task = fifo_queue.get()  # Returns "task1"

# Priority queue usage
priority_queue.put("low_priority_task", priority=1)
priority_queue.put("high_priority_task", priority=10)
task = priority_queue.get()  # Returns "high_priority_task"

# Delay queue usage
delay_queue.put("future_task", delay_seconds=300)  # Execute in 5 minutes
immediate_task = delay_queue.get()  # Returns None (no tasks ready)

# Check queue statistics
from agent_orchestra.queue_manager import get_queue_stats

stats = get_queue_stats()
for queue_name, queue_stats in stats.items():
    print(f"{queue_name}: {queue_stats['size']} items, "
          f"{queue_stats['throughput']:.2%} throughput")
```

### Async Queue Operations

```python
from agent_orchestra.queue_manager import AsyncQueue

# Wrap queue for async operations
async_queue = AsyncQueue(fifo_queue)

async def producer():
    for i in range(10):
        await async_queue.put(f"task_{i}")
        print(f"Produced task_{i}")
        await asyncio.sleep(0.1)

async def consumer():
    while True:
        task = await async_queue.get(timeout=5.0)
        if task is None:
            print("No more tasks, consumer stopping")
            break
        
        print(f"Processing {task}")
        await asyncio.sleep(0.2)  # Simulate processing

# Run producer and consumer
async def queue_example():
    await asyncio.gather(
        producer(),
        consumer()
    )

asyncio.run(queue_example())
```

## Data Transformation

### Format Conversion

```python
from agent_orchestra.data_transformers import transform_data, DataFormat

# Sample data
data = {
    "users": [
        {"id": 1, "name": "John", "active": True},
        {"id": 2, "name": "Jane", "active": False}
    ],
    "metadata": {
        "total": 2,
        "timestamp": "2024-01-15T10:00:00Z"
    }
}

# Convert to different formats
json_result = transform_data(data, DataFormat.PLAIN, DataFormat.JSON, pretty=True)
print("JSON conversion:")
print(json_result.data)
print(f"Size: {json_result.size_bytes} bytes")
print(f"Time: {json_result.transformation_time_ms:.2f}ms")

# Convert to YAML
yaml_result = transform_data(data, DataFormat.PLAIN, DataFormat.YAML, pretty=True)
print("\nYAML conversion:")
print(yaml_result.data)

# Convert list to CSV
user_list = data["users"]
csv_result = transform_data(user_list, DataFormat.PLAIN, DataFormat.CSV)
print("\nCSV conversion:")
print(csv_result.data)
```

### Data Processing

```python
from agent_orchestra.data_transformers import DataProcessor

# Sample dataset
sales_data = [
    {"region": "North", "product": "A", "sales": 100, "quarter": "Q1"},
    {"region": "North", "product": "B", "sales": 150, "quarter": "Q1"},
    {"region": "South", "product": "A", "sales": 200, "quarter": "Q1"},
    {"region": "South", "product": "B", "sales": 120, "quarter": "Q1"},
    {"region": "North", "product": "A", "sales": 110, "quarter": "Q2"},
    {"region": "South", "product": "A", "sales": 190, "quarter": "Q2"},
]

# Filter data
high_sales = DataProcessor.filter_data(
    sales_data, 
    lambda x: x["sales"] > 150
)
print(f"High sales records: {len(high_sales)}")

# Group by region
by_region = DataProcessor.group_by(sales_data, "region")
print(f"Regions: {list(by_region.keys())}")

# Aggregate data
aggregated = DataProcessor.aggregate_data(
    sales_data,
    "region",
    {
        "sales": sum,
        "sales_avg": lambda x: sum(x) / len(x)
    }
)

print("\nAggregated by region:")
for record in aggregated:
    print(f"  {record['region']}: total={record['sales_sum']}, "
          f"avg={record['sales_avg']:.1f}")

# Sort by sales
sorted_data = DataProcessor.sort_data(sales_data, "sales", reverse=True)
print(f"\nTop sales record: {sorted_data[0]}")
```

## Retry Mechanisms

### Basic Retry Usage

```python
from agent_orchestra.retry_mechanisms import (
    RetryConfig, RetryExecutor, RetryStrategy, with_retry
)
import random

# Configure retry behavior
retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    strategy=RetryStrategy.EXPONENTIAL,
    multiplier=2.0,
    jitter=True
)

# Use retry executor
retry_executor = RetryExecutor(retry_config)

def unreliable_operation():
    if random.random() < 0.7:  # 70% failure rate
        raise Exception("Simulated failure")
    return "Success!"

# Execute with retry
try:
    result = retry_executor.execute_sync(unreliable_operation)
    print(f"Operation succeeded: {result}")
except Exception as e:
    print(f"Operation failed after all retries: {e}")

# Use decorator
@with_retry(retry_config)
def another_unreliable_operation():
    if random.random() < 0.5:
        raise Exception("Random failure")
    return "Decorator success!"

result = another_unreliable_operation()
print(f"Decorator result: {result}")
```

### Circuit Breaker Pattern

```python
from agent_orchestra.retry_mechanisms import (
    CircuitBreakerConfig, with_circuit_breaker
)

# Configure circuit breaker
cb_config = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=10.0,
    success_threshold=2
)

@with_circuit_breaker("external_service", cb_config)
def call_external_service():
    # Simulate external service call
    if random.random() < 0.8:  # 80% failure rate initially
        raise Exception("Service unavailable")
    return "Service response"

# Multiple calls will eventually trigger circuit breaker
for i in range(10):
    try:
        result = call_external_service()
        print(f"Call {i+1}: {result}")
    except Exception as e:
        print(f"Call {i+1}: {e}")
    
    time.sleep(1)
```

This documentation provides comprehensive examples for using Agent Orchestra's APIs. Each example includes practical code that can be adapted to your specific use case.