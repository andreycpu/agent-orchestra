"""
HTTP client utilities for Agent Orchestra.

This module provides robust HTTP clients with retry logic, circuit breakers,
rate limiting, and comprehensive error handling for external service integration.
"""
import asyncio
import aiohttp
import time
import json
from typing import Dict, Any, Optional, Union, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from urllib.parse import urljoin, urlparse

from .exceptions import NetworkError, RetryableError, PermanentError, ThrottlingError
from .validation import validate_url


logger = logging.getLogger(__name__)


class HttpMethod(str, Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


@dataclass
class HttpResponse:
    """HTTP response wrapper."""
    status: int
    headers: Dict[str, str]
    body: Union[str, bytes, Dict[str, Any]]
    url: str
    method: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_success(self) -> bool:
        """Check if response indicates success."""
        return 200 <= self.status < 300
    
    @property
    def is_client_error(self) -> bool:
        """Check if response indicates client error."""
        return 400 <= self.status < 500
    
    @property
    def is_server_error(self) -> bool:
        """Check if response indicates server error."""
        return 500 <= self.status < 600
    
    def json(self) -> Dict[str, Any]:
        """Parse response body as JSON."""
        if isinstance(self.body, dict):
            return self.body
        
        if isinstance(self.body, (str, bytes)):
            try:
                return json.loads(self.body)
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"Response body is not valid JSON: {e}")
        
        raise ValueError("Response body cannot be parsed as JSON")
    
    def text(self) -> str:
        """Get response body as text."""
        if isinstance(self.body, str):
            return self.body
        elif isinstance(self.body, bytes):
            return self.body.decode('utf-8')
        elif isinstance(self.body, dict):
            return json.dumps(self.body)
        else:
            return str(self.body)


@dataclass
class RetryPolicy:
    """HTTP request retry policy."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_status_codes: List[int] = field(default_factory=lambda: [429, 502, 503, 504])
    
    def should_retry(self, attempt: int, status_code: int, exception: Optional[Exception] = None) -> bool:
        """Determine if request should be retried."""
        if attempt >= self.max_attempts:
            return False
        
        # Retry on network errors
        if exception and isinstance(exception, (NetworkError, RetryableError)):
            return True
        
        # Retry on specific status codes
        if status_code in self.retryable_status_codes:
            return True
        
        return False
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            jitter_amount = delay * 0.1 * (random.random() - 0.5)
            delay = max(0, delay + jitter_amount)
        
        return delay


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, block requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for external service calls."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 2  # Successes needed to close from half-open
    
    def __post_init__(self):
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED
    
    def can_proceed(self) -> bool:
        """Check if request can proceed through circuit breaker."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)):
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        
        # HALF_OPEN state - allow limited requests
        return True
    
    def record_success(self):
        """Record successful request."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0


class RateLimiter:
    """Token bucket rate limiter for HTTP requests."""
    
    def __init__(self, requests_per_second: float, burst_size: int = None):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size or int(requests_per_second)
        self.tokens = self.burst_size
        self.last_refill = time.time()
    
    async def acquire(self) -> bool:
        """Acquire a token for making a request."""
        now = time.time()
        time_passed = now - self.last_refill
        
        # Refill tokens
        self.tokens = min(
            self.burst_size,
            self.tokens + time_passed * self.requests_per_second
        )
        self.last_refill = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        
        return False
    
    async def wait_for_token(self):
        """Wait until a token is available."""
        while not await self.acquire():
            await asyncio.sleep(1.0 / self.requests_per_second)


class HttpClient:
    """Async HTTP client with advanced features."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        default_timeout: float = 30.0,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        rate_limiter: Optional[RateLimiter] = None,
        default_headers: Optional[Dict[str, str]] = None
    ):
        self.base_url = base_url
        self.default_timeout = default_timeout
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = circuit_breaker
        self.rate_limiter = rate_limiter
        self.default_headers = default_headers or {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Request statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.retry_count = 0
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
    
    async def start_session(self):
        """Start HTTP session."""
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            timeout = aiohttp.ClientTimeout(total=self.default_timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.default_headers
            )
    
    async def close_session(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _build_url(self, url: str) -> str:
        """Build full URL from base URL and path."""
        if self.base_url:
            return urljoin(self.base_url, url)
        return url
    
    def _should_retry_exception(self, exception: Exception) -> bool:
        """Determine if exception should trigger retry."""
        if isinstance(exception, asyncio.TimeoutError):
            return True
        if isinstance(exception, aiohttp.ClientError):
            return True
        if isinstance(exception, (NetworkError, RetryableError)):
            return True
        return False
    
    async def _make_request(
        self,
        method: HttpMethod,
        url: str,
        **kwargs
    ) -> HttpResponse:
        """Make HTTP request with full error handling."""
        if not self.session:
            await self.start_session()
        
        full_url = self._build_url(url)
        start_time = time.time()
        
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_proceed():
            raise NetworkError(f"Circuit breaker is open for {full_url}")
        
        # Rate limiting
        if self.rate_limiter:
            await self.rate_limiter.wait_for_token()
        
        try:
            async with self.session.request(method.value, full_url, **kwargs) as response:
                duration_ms = (time.time() - start_time) * 1000
                
                # Read response body
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    body = await response.json()
                elif 'text/' in content_type:
                    body = await response.text()
                else:
                    body = await response.read()
                
                http_response = HttpResponse(
                    status=response.status,
                    headers=dict(response.headers),
                    body=body,
                    url=full_url,
                    method=method.value,
                    duration_ms=duration_ms
                )
                
                # Update statistics and circuit breaker
                if http_response.is_success:
                    self.successful_requests += 1
                    if self.circuit_breaker:
                        self.circuit_breaker.record_success()
                else:
                    self.failed_requests += 1
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure()
                
                return http_response
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.failed_requests += 1
            
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            # Convert to appropriate exception type
            if isinstance(e, asyncio.TimeoutError):
                raise NetworkError(f"Request timeout for {full_url}") from e
            elif isinstance(e, aiohttp.ClientError):
                raise NetworkError(f"HTTP client error for {full_url}: {e}") from e
            else:
                raise
    
    async def request(
        self,
        method: HttpMethod,
        url: str,
        **kwargs
    ) -> HttpResponse:
        """Make HTTP request with retry logic."""
        self.total_requests += 1
        last_exception = None
        
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                response = await self._make_request(method, url, **kwargs)
                
                # Check if response indicates success
                if response.is_success:
                    return response
                
                # Check if we should retry on this status code
                if not self.retry_policy.should_retry(attempt, response.status):
                    return response
                
                # Log retry for non-success status
                logger.info(
                    f"Retrying request to {url} after status {response.status} "
                    f"(attempt {attempt}/{self.retry_policy.max_attempts})"
                )
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry on this exception
                if not self.retry_policy.should_retry(attempt, 0, e):
                    raise
                
                logger.info(
                    f"Retrying request to {url} after exception {type(e).__name__} "
                    f"(attempt {attempt}/{self.retry_policy.max_attempts})"
                )
            
            # Wait before retry (except on last attempt)
            if attempt < self.retry_policy.max_attempts:
                delay = self.retry_policy.get_delay(attempt)
                await asyncio.sleep(delay)
                self.retry_count += 1
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise NetworkError(f"All retries exhausted for {url}")
    
    # Convenience methods
    async def get(self, url: str, **kwargs) -> HttpResponse:
        """Make GET request."""
        return await self.request(HttpMethod.GET, url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> HttpResponse:
        """Make POST request."""
        return await self.request(HttpMethod.POST, url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> HttpResponse:
        """Make PUT request."""
        return await self.request(HttpMethod.PUT, url, **kwargs)
    
    async def patch(self, url: str, **kwargs) -> HttpResponse:
        """Make PATCH request."""
        return await self.request(HttpMethod.PATCH, url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> HttpResponse:
        """Make DELETE request."""
        return await self.request(HttpMethod.DELETE, url, **kwargs)
    
    async def json_post(self, url: str, data: Dict[str, Any], **kwargs) -> HttpResponse:
        """Make POST request with JSON data."""
        return await self.post(url, json=data, **kwargs)
    
    async def json_put(self, url: str, data: Dict[str, Any], **kwargs) -> HttpResponse:
        """Make PUT request with JSON data."""
        return await self.put(url, json=data, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        total = self.total_requests
        success_rate = (self.successful_requests / total) if total > 0 else 0
        
        return {
            'total_requests': total,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'retry_count': self.retry_count,
            'success_rate': success_rate,
            'circuit_breaker_state': self.circuit_breaker.state.value if self.circuit_breaker else None
        }


class HttpClientPool:
    """Pool of HTTP clients for different services."""
    
    def __init__(self):
        self.clients: Dict[str, HttpClient] = {}
    
    def add_client(self, name: str, client: HttpClient):
        """Add a named client to the pool."""
        self.clients[name] = client
    
    def get_client(self, name: str) -> Optional[HttpClient]:
        """Get a client by name."""
        return self.clients.get(name)
    
    def create_client(
        self,
        name: str,
        base_url: str,
        **client_kwargs
    ) -> HttpClient:
        """Create and add a new client."""
        client = HttpClient(base_url=base_url, **client_kwargs)
        self.clients[name] = client
        return client
    
    async def close_all(self):
        """Close all clients in the pool."""
        for client in self.clients.values():
            await client.close_session()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all clients."""
        return {
            name: client.get_stats()
            for name, client in self.clients.items()
        }


# Global HTTP client pool
http_client_pool = HttpClientPool()


def get_http_client(name: str) -> Optional[HttpClient]:
    """Get HTTP client from global pool."""
    return http_client_pool.get_client(name)


def create_http_client(
    name: str,
    base_url: str,
    **kwargs
) -> HttpClient:
    """Create and register HTTP client in global pool."""
    return http_client_pool.create_client(name, base_url, **kwargs)


async def close_all_http_clients():
    """Close all HTTP clients in global pool."""
    await http_client_pool.close_all()