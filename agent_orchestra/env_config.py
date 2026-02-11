"""
Environment configuration management for Agent Orchestra
"""
import os
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass
from pathlib import Path
import structlog

from .exceptions import ConfigurationError, ValidationError

logger = structlog.get_logger(__name__)


def get_env_var(
    name: str,
    default: Any = None,
    required: bool = False,
    var_type: Type = str,
    choices: Optional[List[Any]] = None
) -> Any:
    """
    Get environment variable with type conversion and validation
    
    Args:
        name: Environment variable name
        default: Default value if not set
        required: Whether the variable is required
        var_type: Type to convert to (str, int, float, bool)
        choices: List of valid choices
        
    Returns:
        Converted environment variable value
        
    Raises:
        ConfigurationError: If required variable is missing or invalid
    """
    value = os.getenv(name)
    
    if value is None:
        if required:
            raise ConfigurationError(f"Required environment variable '{name}' is not set")
        return default
    
    # Type conversion
    try:
        if var_type == bool:
            # Handle boolean conversion
            value = value.lower() in ('true', '1', 'yes', 'on', 'enabled')
        elif var_type == list:
            # Handle list conversion (comma-separated)
            value = [item.strip() for item in value.split(',') if item.strip()]
        else:
            value = var_type(value)
    except (ValueError, TypeError) as e:
        raise ConfigurationError(f"Environment variable '{name}' has invalid type. Expected {var_type.__name__}, got: {value}")
    
    # Validate choices
    if choices is not None and value not in choices:
        raise ConfigurationError(f"Environment variable '{name}' has invalid value '{value}'. Valid choices: {choices}")
    
    return value


@dataclass
class DatabaseConfig:
    """Database configuration from environment variables"""
    url: str
    max_connections: int = 20
    timeout: float = 30.0
    ssl_required: bool = False
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        return cls(
            url=get_env_var("DATABASE_URL", required=True),
            max_connections=get_env_var("DATABASE_MAX_CONNECTIONS", default=20, var_type=int),
            timeout=get_env_var("DATABASE_TIMEOUT", default=30.0, var_type=float),
            ssl_required=get_env_var("DATABASE_SSL_REQUIRED", default=False, var_type=bool)
        )


@dataclass
class RedisConfig:
    """Redis configuration from environment variables"""
    url: str
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        return cls(
            url=get_env_var("REDIS_URL", default="redis://localhost:6379/0"),
            max_connections=get_env_var("REDIS_MAX_CONNECTIONS", default=50, var_type=int),
            socket_timeout=get_env_var("REDIS_SOCKET_TIMEOUT", default=5.0, var_type=float),
            socket_connect_timeout=get_env_var("REDIS_SOCKET_CONNECT_TIMEOUT", default=5.0, var_type=float),
            retry_on_timeout=get_env_var("REDIS_RETRY_ON_TIMEOUT", default=True, var_type=bool)
        )


@dataclass
class SecurityConfig:
    """Security configuration from environment variables"""
    jwt_secret: str
    jwt_expiry_hours: int = 24
    password_min_length: int = 8
    require_2fa: bool = False
    allowed_origins: List[str] = None
    rate_limit_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        jwt_secret = get_env_var("JWT_SECRET", required=True)
        if len(jwt_secret) < 32:
            raise ConfigurationError("JWT_SECRET must be at least 32 characters long")
        
        return cls(
            jwt_secret=jwt_secret,
            jwt_expiry_hours=get_env_var("JWT_EXPIRY_HOURS", default=24, var_type=int),
            password_min_length=get_env_var("PASSWORD_MIN_LENGTH", default=8, var_type=int),
            require_2fa=get_env_var("REQUIRE_2FA", default=False, var_type=bool),
            allowed_origins=get_env_var("ALLOWED_ORIGINS", default=["*"], var_type=list),
            rate_limit_enabled=get_env_var("RATE_LIMIT_ENABLED", default=True, var_type=bool)
        )


@dataclass
class MonitoringConfig:
    """Monitoring configuration from environment variables"""
    enabled: bool = True
    prometheus_port: int = 9090
    metrics_path: str = "/metrics"
    health_check_interval: int = 30
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'MonitoringConfig':
        log_level = get_env_var(
            "LOG_LEVEL", 
            default="INFO", 
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
        
        return cls(
            enabled=get_env_var("MONITORING_ENABLED", default=True, var_type=bool),
            prometheus_port=get_env_var("PROMETHEUS_PORT", default=9090, var_type=int),
            metrics_path=get_env_var("METRICS_PATH", default="/metrics"),
            health_check_interval=get_env_var("HEALTH_CHECK_INTERVAL", default=30, var_type=int),
            log_level=log_level
        )


@dataclass
class OrchestraConfig:
    """Complete orchestra configuration from environment variables"""
    environment: str
    debug: bool = False
    max_workers: int = 10
    task_timeout_default: int = 300
    heartbeat_interval: int = 30
    max_concurrent_tasks: int = 100
    enable_profiling: bool = False
    
    # Nested configurations
    database: DatabaseConfig
    redis: RedisConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    
    @classmethod
    def from_env(cls) -> 'OrchestraConfig':
        environment = get_env_var(
            "ENVIRONMENT", 
            default="production", 
            choices=["development", "testing", "staging", "production"]
        )
        
        debug = environment == "development" or get_env_var("DEBUG", default=False, var_type=bool)
        
        return cls(
            environment=environment,
            debug=debug,
            max_workers=get_env_var("MAX_WORKERS", default=10, var_type=int),
            task_timeout_default=get_env_var("TASK_TIMEOUT_DEFAULT", default=300, var_type=int),
            heartbeat_interval=get_env_var("HEARTBEAT_INTERVAL", default=30, var_type=int),
            max_concurrent_tasks=get_env_var("MAX_CONCURRENT_TASKS", default=100, var_type=int),
            enable_profiling=get_env_var("ENABLE_PROFILING", default=False, var_type=bool),
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            security=SecurityConfig.from_env(),
            monitoring=MonitoringConfig.from_env()
        )
    
    def validate(self) -> None:
        """Validate the complete configuration"""
        if self.max_workers <= 0:
            raise ValidationError("MAX_WORKERS must be positive")
        
        if self.task_timeout_default <= 0:
            raise ValidationError("TASK_TIMEOUT_DEFAULT must be positive")
        
        if self.heartbeat_interval <= 0:
            raise ValidationError("HEARTBEAT_INTERVAL must be positive")
        
        if self.max_concurrent_tasks <= 0:
            raise ValidationError("MAX_CONCURRENT_TASKS must be positive")
        
        # Validate nested configurations
        if self.security.password_min_length < 6:
            raise ValidationError("PASSWORD_MIN_LENGTH must be at least 6")
        
        if self.monitoring.health_check_interval <= 0:
            raise ValidationError("HEALTH_CHECK_INTERVAL must be positive")
        
        logger.info(
            "Configuration validated successfully",
            environment=self.environment,
            debug=self.debug,
            max_workers=self.max_workers
        )


def load_env_file(file_path: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from a .env file
    
    Args:
        file_path: Path to the .env file
        
    Returns:
        Dictionary of loaded environment variables
    """
    env_vars = {}
    env_file = Path(file_path)
    
    if not env_file.exists():
        logger.warning(f"Environment file {file_path} not found, using system environment only")
        return env_vars
    
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE format
                if '=' not in line:
                    logger.warning(f"Invalid line format in {file_path}:{line_num}: {line}")
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Set environment variable if not already set
                if key not in os.environ:
                    os.environ[key] = value
                    env_vars[key] = value
    
    except Exception as e:
        logger.error(f"Failed to load environment file {file_path}: {e}")
        raise ConfigurationError(f"Failed to load environment file: {e}")
    
    logger.info(f"Loaded {len(env_vars)} environment variables from {file_path}")
    return env_vars


def get_config() -> OrchestraConfig:
    """
    Get the complete orchestra configuration from environment
    
    This function should be called once at application startup
    """
    # Try to load .env file first
    load_env_file()
    
    # Load and validate configuration
    try:
        config = OrchestraConfig.from_env()
        config.validate()
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


# Export commonly used configuration getters
def get_redis_config() -> RedisConfig:
    """Get Redis configuration from environment"""
    return RedisConfig.from_env()


def get_security_config() -> SecurityConfig:
    """Get Security configuration from environment"""
    return SecurityConfig.from_env()


def get_monitoring_config() -> MonitoringConfig:
    """Get Monitoring configuration from environment"""
    return MonitoringConfig.from_env()


def is_development() -> bool:
    """Check if running in development environment"""
    return get_env_var("ENVIRONMENT", default="production") == "development"


def is_production() -> bool:
    """Check if running in production environment"""
    return get_env_var("ENVIRONMENT", default="production") == "production"