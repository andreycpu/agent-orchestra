"""
Environment management utilities for Agent Orchestra.

This module provides utilities for managing environment variables,
configuration discovery, and runtime environment detection.
"""
import os
import sys
import platform
from typing import Any, Dict, List, Optional, Union, Callable, Type
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

from .validation import validate_port, validate_url, validate_email
from .exceptions import ConfigurationError


logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class EnvironmentInfo:
    """Information about the runtime environment."""
    environment: Environment
    python_version: str
    platform: str
    architecture: str
    hostname: str
    working_directory: str
    executable_path: str
    package_version: Optional[str] = None
    debug_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'environment': self.environment.value,
            'python_version': self.python_version,
            'platform': self.platform,
            'architecture': self.architecture,
            'hostname': self.hostname,
            'working_directory': self.working_directory,
            'executable_path': self.executable_path,
            'package_version': self.package_version,
            'debug_mode': self.debug_mode
        }


class EnvironmentVariableManager:
    """Manages environment variables with type conversion and validation."""
    
    def __init__(self, prefix: str = "ORCHESTRA_"):
        self.prefix = prefix
        self.cached_values: Dict[str, Any] = {}
        
    def get_str(self, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """Get string environment variable."""
        full_key = f"{self.prefix}{key}"
        value = os.environ.get(full_key, default)
        
        if required and value is None:
            raise ConfigurationError(f"Required environment variable {full_key} is not set")
        
        return value
    
    def get_int(self, key: str, default: Optional[int] = None, required: bool = False) -> Optional[int]:
        """Get integer environment variable."""
        str_value = self.get_str(key, None, required)
        
        if str_value is None:
            return default
        
        try:
            return int(str_value)
        except ValueError:
            raise ConfigurationError(f"Environment variable {self.prefix}{key} must be an integer, got: {str_value}")
    
    def get_float(self, key: str, default: Optional[float] = None, required: bool = False) -> Optional[float]:
        """Get float environment variable."""
        str_value = self.get_str(key, None, required)
        
        if str_value is None:
            return default
        
        try:
            return float(str_value)
        except ValueError:
            raise ConfigurationError(f"Environment variable {self.prefix}{key} must be a float, got: {str_value}")
    
    def get_bool(self, key: str, default: Optional[bool] = None, required: bool = False) -> Optional[bool]:
        """Get boolean environment variable."""
        str_value = self.get_str(key, None, required)
        
        if str_value is None:
            return default
        
        str_value = str_value.lower()
        
        if str_value in ('true', '1', 'yes', 'on', 'enabled'):
            return True
        elif str_value in ('false', '0', 'no', 'off', 'disabled'):
            return False
        else:
            raise ConfigurationError(f"Environment variable {self.prefix}{key} must be a boolean value, got: {str_value}")
    
    def get_list(self, key: str, default: Optional[List[str]] = None, separator: str = ",", required: bool = False) -> Optional[List[str]]:
        """Get list environment variable."""
        str_value = self.get_str(key, None, required)
        
        if str_value is None:
            return default or []
        
        if not str_value.strip():
            return []
        
        return [item.strip() for item in str_value.split(separator) if item.strip()]
    
    def get_url(self, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """Get URL environment variable with validation."""
        str_value = self.get_str(key, default, required)
        
        if str_value is None:
            return None
        
        try:
            validate_url(str_value)
            return str_value
        except Exception as e:
            raise ConfigurationError(f"Environment variable {self.prefix}{key} must be a valid URL: {e}")
    
    def get_email(self, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """Get email environment variable with validation."""
        str_value = self.get_str(key, default, required)
        
        if str_value is None:
            return None
        
        try:
            return validate_email(str_value)
        except Exception as e:
            raise ConfigurationError(f"Environment variable {self.prefix}{key} must be a valid email: {e}")
    
    def get_port(self, key: str, default: Optional[int] = None, required: bool = False) -> Optional[int]:
        """Get port environment variable with validation."""
        int_value = self.get_int(key, default, required)
        
        if int_value is None:
            return None
        
        try:
            return validate_port(int_value)
        except Exception as e:
            raise ConfigurationError(f"Environment variable {self.prefix}{key} must be a valid port: {e}")
    
    def get_path(self, key: str, default: Optional[str] = None, required: bool = False, must_exist: bool = False) -> Optional[Path]:
        """Get path environment variable."""
        str_value = self.get_str(key, default, required)
        
        if str_value is None:
            return None
        
        path = Path(str_value)
        
        if must_exist and not path.exists():
            raise ConfigurationError(f"Path specified in {self.prefix}{key} does not exist: {path}")
        
        return path
    
    def get_enum(self, key: str, enum_class: Type[Enum], default: Optional[Enum] = None, required: bool = False) -> Optional[Enum]:
        """Get enum environment variable."""
        str_value = self.get_str(key, None, required)
        
        if str_value is None:
            return default
        
        try:
            return enum_class(str_value.lower())
        except ValueError:
            valid_values = [e.value for e in enum_class]
            raise ConfigurationError(
                f"Environment variable {self.prefix}{key} must be one of {valid_values}, got: {str_value}"
            )
    
    def set_cache(self, key: str, value: Any):
        """Cache a computed value."""
        self.cached_values[key] = value
    
    def get_cache(self, key: str) -> Any:
        """Get cached value."""
        return self.cached_values.get(key)
    
    def clear_cache(self):
        """Clear all cached values."""
        self.cached_values.clear()
    
    def get_all_with_prefix(self) -> Dict[str, str]:
        """Get all environment variables with the configured prefix."""
        result = {}
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                result[key[len(self.prefix):]] = value
        return result


class ConfigurationDiscovery:
    """Discovers configuration from multiple sources with precedence."""
    
    def __init__(self, app_name: str = "orchestra"):
        self.app_name = app_name
        self.search_paths = [
            Path.cwd() / f".{app_name}.env",
            Path.cwd() / f"{app_name}.conf", 
            Path.home() / f".{app_name}",
            Path(f"/etc/{app_name}"),
            Path.cwd() / "config" / f"{app_name}.conf"
        ]
    
    def find_config_files(self) -> List[Path]:
        """Find all existing configuration files."""
        found_files = []
        for path in self.search_paths:
            if path.exists() and path.is_file():
                found_files.append(path)
        return found_files
    
    def load_env_file(self, env_file: Path) -> Dict[str, str]:
        """Load environment variables from .env file."""
        env_vars = {}
        
        if not env_file.exists():
            return env_vars
        
        try:
            with open(env_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        env_vars[key] = value
                    else:
                        logger.warning(f"Invalid line {line_num} in {env_file}: {line}")
        
        except Exception as e:
            logger.error(f"Error loading env file {env_file}: {e}")
        
        return env_vars
    
    def discover_database_url(self, env_manager: EnvironmentVariableManager) -> Optional[str]:
        """Discover database URL from various sources."""
        # Try direct URL first
        db_url = env_manager.get_url("DATABASE_URL")
        if db_url:
            return db_url
        
        # Try component-based construction
        db_type = env_manager.get_str("DB_TYPE", "postgresql")
        db_host = env_manager.get_str("DB_HOST", "localhost")
        db_port = env_manager.get_port("DB_PORT", 5432)
        db_name = env_manager.get_str("DB_NAME")
        db_user = env_manager.get_str("DB_USER")
        db_pass = env_manager.get_str("DB_PASSWORD")
        
        if all([db_name, db_user]):
            if db_pass:
                return f"{db_type}://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            else:
                return f"{db_type}://{db_user}@{db_host}:{db_port}/{db_name}"
        
        return None
    
    def discover_redis_url(self, env_manager: EnvironmentVariableManager) -> Optional[str]:
        """Discover Redis URL from various sources."""
        # Try direct URL first
        redis_url = env_manager.get_url("REDIS_URL")
        if redis_url:
            return redis_url
        
        # Try component-based construction
        redis_host = env_manager.get_str("REDIS_HOST", "localhost")
        redis_port = env_manager.get_port("REDIS_PORT", 6379)
        redis_db = env_manager.get_int("REDIS_DB", 0)
        redis_pass = env_manager.get_str("REDIS_PASSWORD")
        
        if redis_pass:
            return f"redis://:{redis_pass}@{redis_host}:{redis_port}/{redis_db}"
        else:
            return f"redis://{redis_host}:{redis_port}/{redis_db}"


class EnvironmentDetector:
    """Detects and analyzes the runtime environment."""
    
    @staticmethod
    def detect_environment() -> Environment:
        """Detect current environment."""
        # Check explicit environment variable
        env_var = os.environ.get('ENVIRONMENT', '').lower()
        if env_var:
            try:
                return Environment(env_var)
            except ValueError:
                pass
        
        # Detect based on common patterns
        if any(key in os.environ for key in ['CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS']):
            return Environment.TESTING
        
        if 'HEROKU' in os.environ or 'DYNO' in os.environ:
            return Environment.PRODUCTION
        
        if 'KUBERNETES_SERVICE_HOST' in os.environ:
            return Environment.PRODUCTION
        
        if sys.argv[0].endswith('pytest') or 'pytest' in sys.modules:
            return Environment.TESTING
        
        # Default to development
        return Environment.DEVELOPMENT
    
    @staticmethod
    def get_environment_info() -> EnvironmentInfo:
        """Get comprehensive environment information."""
        try:
            from agent_orchestra import __version__
            package_version = __version__
        except ImportError:
            package_version = None
        
        return EnvironmentInfo(
            environment=EnvironmentDetector.detect_environment(),
            python_version=platform.python_version(),
            platform=platform.platform(),
            architecture=platform.architecture()[0],
            hostname=platform.node(),
            working_directory=str(Path.cwd()),
            executable_path=sys.executable,
            package_version=package_version,
            debug_mode=bool(os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes'))
        )
    
    @staticmethod
    def is_docker() -> bool:
        """Check if running inside Docker container."""
        return (
            Path('/.dockerenv').exists() or
            os.environ.get('DOCKER_CONTAINER') == 'true' or
            'docker' in Path('/proc/self/cgroup').read_text(errors='ignore')
        )
    
    @staticmethod
    def is_kubernetes() -> bool:
        """Check if running inside Kubernetes."""
        return 'KUBERNETES_SERVICE_HOST' in os.environ
    
    @staticmethod
    def is_cloud() -> bool:
        """Check if running in a cloud environment."""
        cloud_indicators = [
            'HEROKU',
            'AWS_EXECUTION_ENV',
            'GOOGLE_CLOUD_PROJECT',
            'AZURE_CLIENT_ID'
        ]
        return any(indicator in os.environ for indicator in cloud_indicators)
    
    @staticmethod
    def get_available_memory_mb() -> Optional[int]:
        """Get available system memory in MB."""
        try:
            import psutil
            return int(psutil.virtual_memory().available / 1024 / 1024)
        except ImportError:
            return None
    
    @staticmethod
    def get_cpu_count() -> int:
        """Get number of CPU cores."""
        return os.cpu_count() or 1


class DevelopmentHelper:
    """Helper utilities for development environments."""
    
    @staticmethod
    def setup_development_logging():
        """Setup logging for development environment."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        # Reduce noise from some libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
    
    @staticmethod
    def print_environment_summary(env_info: EnvironmentInfo):
        """Print environment summary for development."""
        print("=" * 50)
        print("Agent Orchestra Environment Summary")
        print("=" * 50)
        print(f"Environment: {env_info.environment.value}")
        print(f"Python Version: {env_info.python_version}")
        print(f"Platform: {env_info.platform}")
        print(f"Architecture: {env_info.architecture}")
        print(f"Hostname: {env_info.hostname}")
        print(f"Working Directory: {env_info.working_directory}")
        print(f"Package Version: {env_info.package_version or 'Unknown'}")
        print(f"Debug Mode: {env_info.debug_mode}")
        print(f"Docker: {EnvironmentDetector.is_docker()}")
        print(f"Kubernetes: {EnvironmentDetector.is_kubernetes()}")
        print(f"Cloud: {EnvironmentDetector.is_cloud()}")
        
        memory_mb = EnvironmentDetector.get_available_memory_mb()
        if memory_mb:
            print(f"Available Memory: {memory_mb} MB")
        print(f"CPU Cores: {EnvironmentDetector.get_cpu_count()}")
        print("=" * 50)


# Global instances
env_manager = EnvironmentVariableManager()
config_discovery = ConfigurationDiscovery()
env_detector = EnvironmentDetector()


def get_environment_info() -> EnvironmentInfo:
    """Get current environment information."""
    return env_detector.get_environment_info()


def is_development() -> bool:
    """Check if running in development environment."""
    return env_detector.detect_environment() == Environment.DEVELOPMENT


def is_production() -> bool:
    """Check if running in production environment."""
    return env_detector.detect_environment() == Environment.PRODUCTION


def setup_development_environment():
    """Setup development environment with logging and debugging."""
    if is_development():
        DevelopmentHelper.setup_development_logging()
        env_info = get_environment_info()
        DevelopmentHelper.print_environment_summary(env_info)