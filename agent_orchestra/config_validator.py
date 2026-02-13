"""
Configuration validation and management utilities for Agent Orchestra.

This module provides comprehensive configuration validation, defaults,
and environment-based configuration loading.
"""
import os
import re
from typing import Dict, Any, Optional, List, Union, Type, get_type_hints
from pathlib import Path
from dataclasses import dataclass, fields, is_dataclass
from datetime import timedelta
import logging

from .validation import validate_timeout, validate_port, validate_url, validate_email
from .exceptions import ConfigurationError, ValidationError


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    pool_size: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.url:
            raise ConfigurationError("Database URL is required")
        
        if self.pool_size <= 0:
            raise ConfigurationError("Pool size must be positive")
        
        if self.pool_timeout <= 0:
            raise ConfigurationError("Pool timeout must be positive")


@dataclass 
class SecurityConfig:
    """Security configuration."""
    secret_key: str
    token_expiry_seconds: int = 3600
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    require_https: bool = True
    allowed_hosts: List[str] = None
    cors_origins: List[str] = None
    
    def __post_init__(self):
        """Validate security configuration."""
        if not self.secret_key:
            raise ConfigurationError("Secret key is required")
        
        if len(self.secret_key) < 32:
            raise ConfigurationError("Secret key must be at least 32 characters")
        
        if self.token_expiry_seconds <= 0:
            raise ConfigurationError("Token expiry must be positive")
        
        if self.max_failed_attempts <= 0:
            raise ConfigurationError("Max failed attempts must be positive")
        
        if self.lockout_duration_minutes <= 0:
            raise ConfigurationError("Lockout duration must be positive")
        
        # Set defaults for mutable fields
        if self.allowed_hosts is None:
            self.allowed_hosts = []
        
        if self.cors_origins is None:
            self.cors_origins = []


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    
    def __post_init__(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ConfigurationError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")
        
        valid_formats = ["json", "text"]
        if self.format.lower() not in valid_formats:
            raise ConfigurationError(f"Invalid log format: {self.format}. Must be one of {valid_formats}")
        
        if self.max_file_size <= 0:
            raise ConfigurationError("Max file size must be positive")
        
        if self.backup_count < 0:
            raise ConfigurationError("Backup count cannot be negative")


@dataclass
class TaskConfig:
    """Task execution configuration."""
    default_timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    retry_exponential_base: float = 2.0
    queue_size: int = 1000
    batch_size: int = 10
    
    def __post_init__(self):
        """Validate task configuration."""
        if self.default_timeout <= 0:
            raise ConfigurationError("Default timeout must be positive")
        
        if self.max_retries < 0:
            raise ConfigurationError("Max retries cannot be negative")
        
        if self.retry_delay <= 0:
            raise ConfigurationError("Retry delay must be positive")
        
        if self.max_retry_delay <= 0:
            raise ConfigurationError("Max retry delay must be positive")
        
        if self.retry_exponential_base <= 1:
            raise ConfigurationError("Retry exponential base must be greater than 1")
        
        if self.queue_size <= 0:
            raise ConfigurationError("Queue size must be positive")
        
        if self.batch_size <= 0:
            raise ConfigurationError("Batch size must be positive")


@dataclass
class AgentConfig:
    """Agent management configuration."""
    max_agents: int = 10
    heartbeat_interval: int = 30
    heartbeat_timeout: int = 60
    max_idle_time: int = 300
    registration_timeout: int = 120
    
    def __post_init__(self):
        """Validate agent configuration."""
        if self.max_agents <= 0:
            raise ConfigurationError("Max agents must be positive")
        
        if self.heartbeat_interval <= 0:
            raise ConfigurationError("Heartbeat interval must be positive")
        
        if self.heartbeat_timeout <= 0:
            raise ConfigurationError("Heartbeat timeout must be positive")
        
        if self.max_idle_time <= 0:
            raise ConfigurationError("Max idle time must be positive")
        
        if self.registration_timeout <= 0:
            raise ConfigurationError("Registration timeout must be positive")


@dataclass
class MetricsConfig:
    """Metrics and monitoring configuration."""
    enabled: bool = True
    endpoint: str = "/metrics"
    port: Optional[int] = None
    export_interval: int = 60
    retention_days: int = 30
    
    def __post_init__(self):
        """Validate metrics configuration."""
        if not self.endpoint.startswith("/"):
            raise ConfigurationError("Metrics endpoint must start with /")
        
        if self.port is not None:
            validate_port(self.port)
        
        if self.export_interval <= 0:
            raise ConfigurationError("Export interval must be positive")
        
        if self.retention_days <= 0:
            raise ConfigurationError("Retention days must be positive")


@dataclass
class OrchestraConfig:
    """Main orchestra configuration."""
    database: DatabaseConfig
    security: SecurityConfig
    logging: LoggingConfig = None
    tasks: TaskConfig = None
    agents: AgentConfig = None
    metrics: MetricsConfig = None
    
    def __post_init__(self):
        """Set defaults for optional configurations."""
        if self.logging is None:
            self.logging = LoggingConfig()
        
        if self.tasks is None:
            self.tasks = TaskConfig()
        
        if self.agents is None:
            self.agents = AgentConfig()
        
        if self.metrics is None:
            self.metrics = MetricsConfig()


class ConfigurationLoader:
    """Loads and validates configuration from various sources."""
    
    ENV_PREFIX = "ORCHESTRA_"
    
    @classmethod
    def from_environment(cls) -> OrchestraConfig:
        """Load configuration from environment variables.
        
        Environment variables should be prefixed with ORCHESTRA_ and use
        double underscores to separate nested configuration keys.
        
        Example:
            ORCHESTRA_DATABASE__URL=postgresql://...
            ORCHESTRA_SECURITY__SECRET_KEY=mysecretkey
            ORCHESTRA_LOGGING__LEVEL=DEBUG
        """
        env_config = {}
        
        # Collect all environment variables with our prefix
        for key, value in os.environ.items():
            if key.startswith(cls.ENV_PREFIX):
                # Remove prefix and convert to nested dict
                config_key = key[len(cls.ENV_PREFIX):].lower()
                cls._set_nested_value(env_config, config_key, value)
        
        return cls._build_config_from_dict(env_config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> OrchestraConfig:
        """Load configuration from dictionary."""
        return cls._build_config_from_dict(config_dict)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> OrchestraConfig:
        """Load configuration from JSON or YAML file."""
        import json
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    try:
                        import yaml
                        config_dict = yaml.safe_load(f)
                    except ImportError:
                        raise ConfigurationError("PyYAML is required to load YAML configuration files")
                else:
                    config_dict = json.load(f)
            
            return cls.from_dict(config_dict)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {file_path}: {e}")
    
    @classmethod
    def _set_nested_value(cls, target_dict: Dict[str, Any], key_path: str, value: str):
        """Set a nested dictionary value from a flattened key path."""
        keys = key_path.split('__')
        current = target_dict
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value with type conversion
        final_key = keys[-1]
        current[final_key] = cls._convert_env_value(value)
    
    @classmethod
    def _convert_env_value(cls, value: str) -> Any:
        """Convert environment variable string to appropriate Python type."""
        # Handle boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Handle None/null
        if value.lower() in ('none', 'null', ''):
            return None
        
        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Handle lists (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # Return as string
        return value
    
    @classmethod
    def _build_config_from_dict(cls, config_dict: Dict[str, Any]) -> OrchestraConfig:
        """Build OrchestraConfig from dictionary."""
        try:
            # Build sub-configurations
            database_config = None
            if 'database' in config_dict:
                database_config = DatabaseConfig(**config_dict['database'])
            else:
                # Database is required, check for minimal configuration
                db_url = config_dict.get('database_url') or os.environ.get('DATABASE_URL')
                if not db_url:
                    raise ConfigurationError("Database configuration is required")
                database_config = DatabaseConfig(url=db_url)
            
            security_config = None
            if 'security' in config_dict:
                security_config = SecurityConfig(**config_dict['security'])
            else:
                # Security is required
                secret_key = config_dict.get('secret_key') or os.environ.get('SECRET_KEY')
                if not secret_key:
                    raise ConfigurationError("Security configuration with secret key is required")
                security_config = SecurityConfig(secret_key=secret_key)
            
            # Build optional configurations
            logging_config = None
            if 'logging' in config_dict:
                logging_config = LoggingConfig(**config_dict['logging'])
            
            tasks_config = None
            if 'tasks' in config_dict:
                tasks_config = TaskConfig(**config_dict['tasks'])
            
            agents_config = None
            if 'agents' in config_dict:
                agents_config = AgentConfig(**config_dict['agents'])
            
            metrics_config = None
            if 'metrics' in config_dict:
                metrics_config = MetricsConfig(**config_dict['metrics'])
            
            return OrchestraConfig(
                database=database_config,
                security=security_config,
                logging=logging_config,
                tasks=tasks_config,
                agents=agents_config,
                metrics=metrics_config
            )
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to build configuration: {e}")


class ConfigurationValidator:
    """Validates configuration consistency and constraints."""
    
    @classmethod
    def validate(cls, config: OrchestraConfig) -> List[str]:
        """Validate configuration and return list of warnings.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Cross-component validation
        if config.agents.heartbeat_timeout <= config.agents.heartbeat_interval:
            warnings.append(
                "Agent heartbeat timeout should be greater than heartbeat interval"
            )
        
        if config.tasks.default_timeout < config.agents.heartbeat_interval * 2:
            warnings.append(
                "Task timeout should be at least 2x the agent heartbeat interval"
            )
        
        if config.tasks.queue_size < config.agents.max_agents * 10:
            warnings.append(
                "Task queue size might be too small for the number of agents"
            )
        
        # Security validation
        if not config.security.require_https:
            warnings.append("HTTPS is not required - consider enabling for production")
        
        if len(config.security.allowed_hosts) == 0:
            warnings.append("No allowed hosts specified - all hosts will be accepted")
        
        # Logging validation
        if config.logging.level == "DEBUG":
            warnings.append("Debug logging is enabled - may impact performance")
        
        if config.logging.file_path and not Path(config.logging.file_path).parent.exists():
            warnings.append(f"Log file directory does not exist: {config.logging.file_path}")
        
        # Metrics validation
        if not config.metrics.enabled:
            warnings.append("Metrics collection is disabled - monitoring may be limited")
        
        return warnings
    
    @classmethod
    def validate_environment(cls) -> List[str]:
        """Validate the runtime environment.
        
        Returns:
            List of environment issues
        """
        issues = []
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            issues.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        
        # Check required packages
        required_packages = [
            ('psutil', 'System monitoring'),
            ('structlog', 'Structured logging'),
            ('pydantic', 'Data validation'),
        ]
        
        for package, description in required_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(f"Missing package '{package}' required for {description}")
        
        # Check optional packages
        optional_packages = [
            ('yaml', 'YAML configuration files'),
            ('prometheus_client', 'Prometheus metrics'),
        ]
        
        missing_optional = []
        for package, description in optional_packages:
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(f"Optional package '{package}' not found ({description})")
        
        if missing_optional:
            issues.append("Optional packages missing: " + ", ".join(missing_optional))
        
        return issues


def create_default_config(
    database_url: str,
    secret_key: Optional[str] = None,
    **overrides
) -> OrchestraConfig:
    """Create a default configuration with minimal required settings.
    
    Args:
        database_url: Database connection URL
        secret_key: Secret key for security (generated if not provided)
        **overrides: Additional configuration overrides
        
    Returns:
        Default configuration
    """
    if not secret_key:
        import secrets
        secret_key = secrets.token_hex(32)
    
    config_dict = {
        'database': {'url': database_url},
        'security': {'secret_key': secret_key}
    }
    
    # Apply overrides
    for key, value in overrides.items():
        config_dict[key] = value
    
    return ConfigurationLoader.from_dict(config_dict)


def get_config_template() -> Dict[str, Any]:
    """Get a template configuration dictionary for documentation.
    
    Returns:
        Template configuration with all available options
    """
    return {
        'database': {
            'url': 'postgresql://user:password@host:port/database',
            'pool_size': 10,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'echo': False
        },
        'security': {
            'secret_key': 'your-secret-key-here',
            'token_expiry_seconds': 3600,
            'max_failed_attempts': 5,
            'lockout_duration_minutes': 15,
            'require_https': True,
            'allowed_hosts': ['localhost', '127.0.0.1'],
            'cors_origins': ['http://localhost:3000']
        },
        'logging': {
            'level': 'INFO',
            'format': 'json',
            'file_path': '/var/log/orchestra.log',
            'max_file_size': 10485760,  # 10MB
            'backup_count': 5,
            'enable_console': True
        },
        'tasks': {
            'default_timeout': 300,
            'max_retries': 3,
            'retry_delay': 1.0,
            'max_retry_delay': 60.0,
            'retry_exponential_base': 2.0,
            'queue_size': 1000,
            'batch_size': 10
        },
        'agents': {
            'max_agents': 10,
            'heartbeat_interval': 30,
            'heartbeat_timeout': 60,
            'max_idle_time': 300,
            'registration_timeout': 120
        },
        'metrics': {
            'enabled': True,
            'endpoint': '/metrics',
            'port': 9090,
            'export_interval': 60,
            'retention_days': 30
        }
    }