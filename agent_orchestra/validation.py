"""
Comprehensive validation utilities for Agent Orchestra.

This module provides robust validation functions for various data types,
configurations, and business logic rules used throughout the system.
"""
import re
import ipaddress
from typing import Any, Dict, List, Optional, Union, Callable
from urllib.parse import urlparse
from datetime import datetime, timedelta
import logging

from .types import TaskStatus, AgentStatus, TaskPriority
from .exceptions import ValidationError


logger = logging.getLogger(__name__)


class ValidationRules:
    """Container for validation rule constants."""
    
    MIN_TASK_ID_LENGTH = 1
    MAX_TASK_ID_LENGTH = 255
    MIN_AGENT_ID_LENGTH = 1
    MAX_AGENT_ID_LENGTH = 255
    MAX_TASK_TYPE_LENGTH = 100
    MAX_ERROR_MESSAGE_LENGTH = 2000
    MIN_TIMEOUT_SECONDS = 1
    MAX_TIMEOUT_SECONDS = 86400  # 24 hours
    MAX_RETRY_COUNT = 10


def validate_id(value: str, id_type: str = "ID") -> str:
    """
    Validate that an ID meets basic requirements.
    
    Args:
        value: The ID value to validate
        id_type: Type of ID for error messages
        
    Returns:
        The validated and normalized ID
        
    Raises:
        ValidationError: If ID is invalid
    """
    if not isinstance(value, str):
        raise ValidationError(f"{id_type} must be a string")
    
    value = value.strip()
    
    if not value:
        raise ValidationError(f"{id_type} cannot be empty")
    
    if len(value) < ValidationRules.MIN_TASK_ID_LENGTH:
        raise ValidationError(f"{id_type} too short (minimum {ValidationRules.MIN_TASK_ID_LENGTH} characters)")
    
    if len(value) > ValidationRules.MAX_TASK_ID_LENGTH:
        raise ValidationError(f"{id_type} too long (maximum {ValidationRules.MAX_TASK_ID_LENGTH} characters)")
    
    # Check for invalid characters
    if not re.match(r'^[a-zA-Z0-9_-]+$', value):
        raise ValidationError(f"{id_type} contains invalid characters (only alphanumeric, underscore, and hyphen allowed)")
    
    return value


def validate_task_type(task_type: str) -> str:
    """
    Validate task type string.
    
    Args:
        task_type: The task type to validate
        
    Returns:
        The validated and normalized task type
        
    Raises:
        ValidationError: If task type is invalid
    """
    if not isinstance(task_type, str):
        raise ValidationError("Task type must be a string")
    
    task_type = task_type.strip()
    
    if not task_type:
        raise ValidationError("Task type cannot be empty")
    
    if len(task_type) > ValidationRules.MAX_TASK_TYPE_LENGTH:
        raise ValidationError(f"Task type too long (maximum {ValidationRules.MAX_TASK_TYPE_LENGTH} characters)")
    
    # Task type should follow snake_case or kebab-case convention
    if not re.match(r'^[a-z0-9_-]+$', task_type):
        raise ValidationError("Task type must be lowercase alphanumeric with underscores or hyphens only")
    
    return task_type


def validate_timeout(timeout: Optional[int]) -> Optional[int]:
    """
    Validate timeout value.
    
    Args:
        timeout: Timeout in seconds
        
    Returns:
        The validated timeout
        
    Raises:
        ValidationError: If timeout is invalid
    """
    if timeout is None:
        return None
    
    if not isinstance(timeout, int):
        raise ValidationError("Timeout must be an integer")
    
    if timeout < ValidationRules.MIN_TIMEOUT_SECONDS:
        raise ValidationError(f"Timeout too small (minimum {ValidationRules.MIN_TIMEOUT_SECONDS} seconds)")
    
    if timeout > ValidationRules.MAX_TIMEOUT_SECONDS:
        raise ValidationError(f"Timeout too large (maximum {ValidationRules.MAX_TIMEOUT_SECONDS} seconds)")
    
    return timeout


def validate_retry_count(retry_count: int, max_retries: int) -> int:
    """
    Validate retry count against maximum retries.
    
    Args:
        retry_count: Current retry count
        max_retries: Maximum allowed retries
        
    Returns:
        The validated retry count
        
    Raises:
        ValidationError: If retry count is invalid
    """
    if not isinstance(retry_count, int):
        raise ValidationError("Retry count must be an integer")
    
    if retry_count < 0:
        raise ValidationError("Retry count cannot be negative")
    
    if retry_count > max_retries:
        raise ValidationError(f"Retry count ({retry_count}) exceeds maximum retries ({max_retries})")
    
    return retry_count


def validate_max_retries(max_retries: int) -> int:
    """
    Validate maximum retries setting.
    
    Args:
        max_retries: Maximum retry count
        
    Returns:
        The validated max retries
        
    Raises:
        ValidationError: If max retries is invalid
    """
    if not isinstance(max_retries, int):
        raise ValidationError("Max retries must be an integer")
    
    if max_retries < 0:
        raise ValidationError("Max retries cannot be negative")
    
    if max_retries > ValidationRules.MAX_RETRY_COUNT:
        raise ValidationError(f"Max retries too high (maximum {ValidationRules.MAX_RETRY_COUNT})")
    
    return max_retries


def validate_url(url: str) -> str:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        The validated URL
        
    Raises:
        ValidationError: If URL is invalid
    """
    if not isinstance(url, str):
        raise ValidationError("URL must be a string")
    
    url = url.strip()
    
    if not url:
        raise ValidationError("URL cannot be empty")
    
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValidationError("URL must have both scheme and netloc")
        
        if parsed.scheme not in ['http', 'https', 'ws', 'wss']:
            raise ValidationError("URL scheme must be http, https, ws, or wss")
    except Exception as e:
        raise ValidationError(f"Invalid URL format: {e}")
    
    return url


def validate_ip_address(ip: str) -> str:
    """
    Validate IP address format.
    
    Args:
        ip: IP address to validate
        
    Returns:
        The validated IP address
        
    Raises:
        ValidationError: If IP address is invalid
    """
    if not isinstance(ip, str):
        raise ValidationError("IP address must be a string")
    
    ip = ip.strip()
    
    if not ip:
        raise ValidationError("IP address cannot be empty")
    
    try:
        ipaddress.ip_address(ip)
    except ValueError as e:
        raise ValidationError(f"Invalid IP address: {e}")
    
    return ip


def validate_port(port: Union[int, str]) -> int:
    """
    Validate network port number.
    
    Args:
        port: Port number to validate
        
    Returns:
        The validated port number as integer
        
    Raises:
        ValidationError: If port is invalid
    """
    try:
        port_int = int(port)
    except (ValueError, TypeError):
        raise ValidationError("Port must be a valid integer")
    
    if port_int < 1 or port_int > 65535:
        raise ValidationError("Port must be between 1 and 65535")
    
    return port_int


def validate_email(email: str) -> str:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        The validated email address
        
    Raises:
        ValidationError: If email is invalid
    """
    if not isinstance(email, str):
        raise ValidationError("Email must be a string")
    
    email = email.strip().lower()
    
    if not email:
        raise ValidationError("Email cannot be empty")
    
    # Basic email regex validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise ValidationError("Invalid email format")
    
    return email


def validate_datetime(dt: datetime, allow_future: bool = True) -> datetime:
    """
    Validate datetime object.
    
    Args:
        dt: Datetime to validate
        allow_future: Whether future dates are allowed
        
    Returns:
        The validated datetime
        
    Raises:
        ValidationError: If datetime is invalid
    """
    if not isinstance(dt, datetime):
        raise ValidationError("Must be a datetime object")
    
    if not allow_future and dt > datetime.utcnow():
        raise ValidationError("Future datetime not allowed")
    
    # Check if datetime is reasonable (not too far in past or future)
    min_date = datetime(1970, 1, 1)
    max_date = datetime(2100, 1, 1)
    
    if dt < min_date:
        raise ValidationError(f"Datetime too far in past (before {min_date})")
    
    if dt > max_date:
        raise ValidationError(f"Datetime too far in future (after {max_date})")
    
    return dt


def validate_json_serializable(data: Any) -> Any:
    """
    Validate that data is JSON serializable.
    
    Args:
        data: Data to validate
        
    Returns:
        The validated data
        
    Raises:
        ValidationError: If data is not JSON serializable
    """
    import json
    
    try:
        json.dumps(data)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Data is not JSON serializable: {e}")
    
    return data


def validate_task_dependencies(task_id: str, dependencies: List[str]) -> List[str]:
    """
    Validate task dependencies for circular references and self-reference.
    
    Args:
        task_id: ID of the task
        dependencies: List of dependency task IDs
        
    Returns:
        The validated dependencies list
        
    Raises:
        ValidationError: If dependencies are invalid
    """
    if not isinstance(dependencies, list):
        raise ValidationError("Dependencies must be a list")
    
    # Check for self-reference
    if task_id in dependencies:
        raise ValidationError("Task cannot depend on itself")
    
    # Check for duplicates
    if len(dependencies) != len(set(dependencies)):
        raise ValidationError("Duplicate dependencies found")
    
    # Validate each dependency ID
    validated_deps = []
    for dep_id in dependencies:
        validated_deps.append(validate_id(dep_id, "Dependency ID"))
    
    return validated_deps


def validate_configuration(config: Dict[str, Any], schema: Dict[str, Callable]) -> Dict[str, Any]:
    """
    Validate configuration dictionary against schema.
    
    Args:
        config: Configuration to validate
        schema: Schema with field names as keys and validation functions as values
        
    Returns:
        The validated configuration
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Configuration must be a dictionary")
    
    validated_config = {}
    
    for field, validator_func in schema.items():
        if field in config:
            try:
                validated_config[field] = validator_func(config[field])
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(f"Validation error for field '{field}': {e}")
    
    # Check for unknown fields
    unknown_fields = set(config.keys()) - set(schema.keys())
    if unknown_fields:
        logger.warning(f"Unknown configuration fields ignored: {unknown_fields}")
    
    return validated_config


def validate_resource_requirements(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate resource requirements dictionary.
    
    Args:
        requirements: Resource requirements to validate
        
    Returns:
        The validated requirements
        
    Raises:
        ValidationError: If requirements are invalid
    """
    if not isinstance(requirements, dict):
        raise ValidationError("Resource requirements must be a dictionary")
    
    validated_reqs = {}
    
    # Validate CPU requirements
    if 'cpu' in requirements:
        cpu = requirements['cpu']
        if isinstance(cpu, (int, float)):
            if cpu <= 0:
                raise ValidationError("CPU requirement must be positive")
            validated_reqs['cpu'] = float(cpu)
        else:
            raise ValidationError("CPU requirement must be a number")
    
    # Validate memory requirements
    if 'memory' in requirements:
        memory = requirements['memory']
        if isinstance(memory, (int, float)):
            if memory <= 0:
                raise ValidationError("Memory requirement must be positive")
            validated_reqs['memory'] = float(memory)
        elif isinstance(memory, str):
            # Parse memory strings like "1GB", "512MB"
            memory_pattern = r'^(\d+(?:\.\d+)?)(GB|MB|KB|B)$'
            match = re.match(memory_pattern, memory.upper())
            if not match:
                raise ValidationError("Invalid memory format (use format like '1GB', '512MB')")
            
            value, unit = match.groups()
            value = float(value)
            
            # Convert to bytes
            multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
            validated_reqs['memory'] = value * multipliers[unit]
        else:
            raise ValidationError("Memory requirement must be a number or string with unit")
    
    # Validate disk requirements
    if 'disk' in requirements:
        disk = requirements['disk']
        if isinstance(disk, (int, float)):
            if disk <= 0:
                raise ValidationError("Disk requirement must be positive")
            validated_reqs['disk'] = float(disk)
        else:
            raise ValidationError("Disk requirement must be a number")
    
    return validated_reqs