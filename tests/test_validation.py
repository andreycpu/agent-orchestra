"""
Tests for validation utilities.
"""
import pytest
from datetime import datetime, timedelta

from agent_orchestra.validation import (
    validate_id, validate_task_type, validate_timeout, validate_retry_count,
    validate_max_retries, validate_url, validate_ip_address, validate_port,
    validate_email, validate_datetime, validate_json_serializable,
    validate_task_dependencies, validate_configuration, validate_resource_requirements,
    ValidationRules
)
from agent_orchestra.exceptions import ValidationError


class TestValidateId:
    """Test ID validation."""
    
    def test_valid_ids(self):
        """Test valid ID formats."""
        valid_ids = [
            "test_id",
            "agent-123",
            "task_abc_123",
            "a" * ValidationRules.MAX_TASK_ID_LENGTH
        ]
        
        for test_id in valid_ids:
            result = validate_id(test_id)
            assert result == test_id
    
    def test_invalid_ids(self):
        """Test invalid ID formats."""
        invalid_ids = [
            "",  # Empty
            " ",  # Whitespace only
            "id with spaces",  # Contains spaces
            "id@special",  # Special characters
            "a" * (ValidationRules.MAX_TASK_ID_LENGTH + 1),  # Too long
        ]
        
        for test_id in invalid_ids:
            with pytest.raises(ValidationError):
                validate_id(test_id)
    
    def test_non_string_id(self):
        """Test non-string ID input."""
        with pytest.raises(ValidationError):
            validate_id(123)


class TestValidateTaskType:
    """Test task type validation."""
    
    def test_valid_task_types(self):
        """Test valid task type formats."""
        valid_types = [
            "simple_task",
            "complex-task",
            "task123",
            "a" * ValidationRules.MAX_TASK_TYPE_LENGTH
        ]
        
        for task_type in valid_types:
            result = validate_task_type(task_type)
            assert result == task_type
    
    def test_invalid_task_types(self):
        """Test invalid task type formats."""
        invalid_types = [
            "",  # Empty
            "Task_With_Capitals",  # Uppercase
            "task with spaces",  # Spaces
            "task@special",  # Special characters
            "a" * (ValidationRules.MAX_TASK_TYPE_LENGTH + 1),  # Too long
        ]
        
        for task_type in invalid_types:
            with pytest.raises(ValidationError):
                validate_task_type(task_type)


class TestValidateTimeout:
    """Test timeout validation."""
    
    def test_valid_timeouts(self):
        """Test valid timeout values."""
        valid_timeouts = [
            None,  # No timeout
            1,  # Minimum
            3600,  # 1 hour
            ValidationRules.MAX_TIMEOUT_SECONDS
        ]
        
        for timeout in valid_timeouts:
            result = validate_timeout(timeout)
            assert result == timeout
    
    def test_invalid_timeouts(self):
        """Test invalid timeout values."""
        invalid_timeouts = [
            0,  # Zero
            -1,  # Negative
            ValidationRules.MAX_TIMEOUT_SECONDS + 1,  # Too large
            "60"  # String instead of int
        ]
        
        for timeout in invalid_timeouts:
            with pytest.raises(ValidationError):
                validate_timeout(timeout)


class TestValidateRetryCount:
    """Test retry count validation."""
    
    def test_valid_retry_counts(self):
        """Test valid retry count scenarios."""
        test_cases = [
            (0, 3),  # No retries yet
            (2, 5),  # Some retries
            (3, 3),  # At limit
        ]
        
        for retry_count, max_retries in test_cases:
            result = validate_retry_count(retry_count, max_retries)
            assert result == retry_count
    
    def test_invalid_retry_counts(self):
        """Test invalid retry count scenarios."""
        invalid_cases = [
            (-1, 3),  # Negative count
            (4, 3),  # Exceeds maximum
            ("2", 3),  # String instead of int
        ]
        
        for retry_count, max_retries in invalid_cases:
            with pytest.raises(ValidationError):
                validate_retry_count(retry_count, max_retries)


class TestValidateMaxRetries:
    """Test max retries validation."""
    
    def test_valid_max_retries(self):
        """Test valid max retries values."""
        valid_values = [0, 1, 5, ValidationRules.MAX_RETRY_COUNT]
        
        for max_retries in valid_values:
            result = validate_max_retries(max_retries)
            assert result == max_retries
    
    def test_invalid_max_retries(self):
        """Test invalid max retries values."""
        invalid_values = [
            -1,  # Negative
            ValidationRules.MAX_RETRY_COUNT + 1,  # Too high
            "3"  # String instead of int
        ]
        
        for max_retries in invalid_values:
            with pytest.raises(ValidationError):
                validate_max_retries(max_retries)


class TestValidateUrl:
    """Test URL validation."""
    
    def test_valid_urls(self):
        """Test valid URL formats."""
        valid_urls = [
            "http://example.com",
            "https://api.example.com/v1",
            "ws://localhost:8080",
            "wss://secure.example.com/ws"
        ]
        
        for url in valid_urls:
            result = validate_url(url)
            assert result == url
    
    def test_invalid_urls(self):
        """Test invalid URL formats."""
        invalid_urls = [
            "",  # Empty
            "not-a-url",  # No scheme
            "ftp://example.com",  # Invalid scheme
            "http://",  # No netloc
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValidationError):
                validate_url(url)


class TestValidateIpAddress:
    """Test IP address validation."""
    
    def test_valid_ip_addresses(self):
        """Test valid IP addresses."""
        valid_ips = [
            "192.168.1.1",
            "127.0.0.1",
            "10.0.0.1",
            "255.255.255.255",
            "2001:db8::1",  # IPv6
        ]
        
        for ip in valid_ips:
            result = validate_ip_address(ip)
            assert result == ip
    
    def test_invalid_ip_addresses(self):
        """Test invalid IP addresses."""
        invalid_ips = [
            "",  # Empty
            "256.1.1.1",  # Out of range
            "192.168.1",  # Incomplete
            "not-an-ip",  # Invalid format
        ]
        
        for ip in invalid_ips:
            with pytest.raises(ValidationError):
                validate_ip_address(ip)


class TestValidatePort:
    """Test port validation."""
    
    def test_valid_ports(self):
        """Test valid port numbers."""
        valid_ports = [1, 80, 443, 8080, 65535, "8080"]  # String should work too
        
        for port in valid_ports:
            result = validate_port(port)
            assert isinstance(result, int)
            assert 1 <= result <= 65535
    
    def test_invalid_ports(self):
        """Test invalid port numbers."""
        invalid_ports = [0, -1, 65536, "not-a-port", 1.5]
        
        for port in invalid_ports:
            with pytest.raises(ValidationError):
                validate_port(port)


class TestValidateEmail:
    """Test email validation."""
    
    def test_valid_emails(self):
        """Test valid email addresses."""
        valid_emails = [
            "user@example.com",
            "test.email@domain.org",
            "user+tag@example.co.uk"
        ]
        
        for email in valid_emails:
            result = validate_email(email)
            assert result == email.lower()
    
    def test_invalid_emails(self):
        """Test invalid email addresses."""
        invalid_emails = [
            "",  # Empty
            "not-an-email",  # No @
            "@domain.com",  # No local part
            "user@",  # No domain
            "user@domain",  # No TLD
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                validate_email(email)


class TestValidateDatetime:
    """Test datetime validation."""
    
    def test_valid_datetimes(self):
        """Test valid datetime objects."""
        valid_datetimes = [
            datetime.now(),
            datetime(2020, 1, 1),
            datetime(2099, 12, 31)
        ]
        
        for dt in valid_datetimes:
            result = validate_datetime(dt, allow_future=True)
            assert result == dt
    
    def test_future_datetime_not_allowed(self):
        """Test future datetime when not allowed."""
        future_dt = datetime.now() + timedelta(days=1)
        
        with pytest.raises(ValidationError):
            validate_datetime(future_dt, allow_future=False)
    
    def test_invalid_datetimes(self):
        """Test invalid datetime objects."""
        invalid_datetimes = [
            datetime(1950, 1, 1),  # Too far in past
            datetime(2150, 1, 1),  # Too far in future
            "2023-01-01",  # String instead of datetime
        ]
        
        for dt in invalid_datetimes:
            with pytest.raises(ValidationError):
                validate_datetime(dt) if isinstance(dt, datetime) else validate_datetime(dt)


class TestValidateJsonSerializable:
    """Test JSON serialization validation."""
    
    def test_valid_json_serializable(self):
        """Test valid JSON serializable objects."""
        valid_objects = [
            {"key": "value"},
            [1, 2, 3],
            "string",
            42,
            None,
            True
        ]
        
        for obj in valid_objects:
            result = validate_json_serializable(obj)
            assert result == obj
    
    def test_invalid_json_serializable(self):
        """Test invalid JSON serializable objects."""
        invalid_objects = [
            set([1, 2, 3]),  # Set is not JSON serializable
            lambda x: x,  # Function is not JSON serializable
        ]
        
        for obj in invalid_objects:
            with pytest.raises(ValidationError):
                validate_json_serializable(obj)


class TestValidateTaskDependencies:
    """Test task dependencies validation."""
    
    def test_valid_dependencies(self):
        """Test valid task dependencies."""
        task_id = "task_1"
        valid_deps = [
            [],  # No dependencies
            ["task_2"],  # Single dependency
            ["task_2", "task_3"]  # Multiple dependencies
        ]
        
        for deps in valid_deps:
            result = validate_task_dependencies(task_id, deps)
            assert result == deps
    
    def test_self_reference(self):
        """Test self-reference in dependencies."""
        task_id = "task_1"
        deps = ["task_1"]  # Self-reference
        
        with pytest.raises(ValidationError, match="cannot depend on itself"):
            validate_task_dependencies(task_id, deps)
    
    def test_duplicate_dependencies(self):
        """Test duplicate dependencies."""
        task_id = "task_1"
        deps = ["task_2", "task_2"]  # Duplicate
        
        with pytest.raises(ValidationError, match="Duplicate dependencies"):
            validate_task_dependencies(task_id, deps)


class TestValidateResourceRequirements:
    """Test resource requirements validation."""
    
    def test_valid_resource_requirements(self):
        """Test valid resource requirements."""
        valid_reqs = [
            {},  # Empty requirements
            {"cpu": 1.0},  # CPU only
            {"memory": "1GB"},  # Memory with unit
            {"cpu": 2.0, "memory": 1024**3, "disk": 100.0}  # All resources
        ]
        
        for reqs in valid_reqs:
            result = validate_resource_requirements(reqs)
            assert isinstance(result, dict)
    
    def test_invalid_resource_requirements(self):
        """Test invalid resource requirements."""
        invalid_reqs = [
            {"cpu": -1.0},  # Negative CPU
            {"memory": "invalid"},  # Invalid memory format
            {"disk": "not-a-number"}  # Invalid disk format
        ]
        
        for reqs in invalid_reqs:
            with pytest.raises(ValidationError):
                validate_resource_requirements(reqs)
    
    def test_memory_unit_parsing(self):
        """Test memory unit parsing."""
        memory_tests = [
            ("1GB", 1024**3),
            ("512MB", 512 * 1024**2),
            ("1024KB", 1024**2),
            ("2048B", 2048)
        ]
        
        for memory_str, expected_bytes in memory_tests:
            reqs = {"memory": memory_str}
            result = validate_resource_requirements(reqs)
            assert result["memory"] == expected_bytes