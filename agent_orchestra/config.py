"""
Configuration management for Agent Orchestra
"""
import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import structlog

from .exceptions import ConfigurationError, ValidationError

logger = structlog.get_logger(__name__)


@dataclass
class OrchestraConfig:
    """Main configuration for Orchestra"""
    max_concurrent_tasks: int = 100
    task_timeout_default: int = 300
    heartbeat_interval: int = 30
    redis_url: Optional[str] = None
    log_level: str = "INFO"
    metrics_enabled: bool = True
    monitoring_enabled: bool = True
    failure_retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 300
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.max_concurrent_tasks <= 0:
            raise ValidationError("max_concurrent_tasks must be positive")
            
        if self.task_timeout_default <= 0:
            raise ValidationError("task_timeout_default must be positive")
            
        if self.heartbeat_interval <= 0:
            raise ValidationError("heartbeat_interval must be positive")
            
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValidationError(f"Invalid log_level: {self.log_level}")
            
        if self.failure_retry_attempts < 0:
            raise ValidationError("failure_retry_attempts cannot be negative")
            
        if self.circuit_breaker_threshold <= 0:
            raise ValidationError("circuit_breaker_threshold must be positive")
            
        if self.circuit_breaker_timeout <= 0:
            raise ValidationError("circuit_breaker_timeout must be positive")


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    id: str
    name: Optional[str] = None
    capabilities: list = None
    max_concurrent_tasks: int = 10
    resource_limits: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.name is None:
            self.name = self.id


@dataclass
class RedisConfig:
    """Redis configuration"""
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    retry_on_timeout: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    metrics_port: int = 8080
    dashboard_enabled: bool = True
    export_interval: int = 60
    history_retention_days: int = 7
    alert_thresholds: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "error_rate": 0.1,
                "cpu_usage": 90.0,
                "memory_usage": 90.0,
                "task_timeout_rate": 0.05
            }


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    file_path: Optional[str] = None
    max_file_size: str = "100MB"
    backup_count: int = 5
    structured: bool = True


class ConfigLoader:
    """Configuration loader with support for multiple formats"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config_cache: Dict[str, Any] = {}
    
    def load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        config_file = config_file or self.config_path
        
        # Start with default configuration
        config = self._get_default_config()
        
        # Load from file if specified
        if config_file:
            file_config = self._load_from_file(config_file)
            config = self._merge_configs(config, file_config)
        
        # Override with environment variables
        env_config = self._load_from_environment()
        config = self._merge_configs(config, env_config)
        
        # Cache the loaded configuration
        self._config_cache = config
        
        logger.info("Configuration loaded", 
                   config_file=config_file,
                   sources=["default", "file" if config_file else None, "env"])
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "orchestra": asdict(OrchestraConfig()),
            "redis": asdict(RedisConfig()),
            "monitoring": asdict(MonitoringConfig()),
            "logging": asdict(LoggingConfig()),
            "agents": []
        }
    
    def _load_from_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning("Config file not found", path=config_file)
            return {}
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    logger.error("Unsupported config file format", path=config_file)
                    return {}
            
            logger.info("Configuration loaded from file", path=config_file)
            return config or {}
            
        except Exception as e:
            logger.error("Failed to load config file", 
                        path=config_file, error=str(e))
            return {}
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}
        
        # Orchestra configuration
        if os.getenv("ORCHESTRA_MAX_CONCURRENT_TASKS"):
            config.setdefault("orchestra", {})["max_concurrent_tasks"] = int(
                os.getenv("ORCHESTRA_MAX_CONCURRENT_TASKS")
            )
        
        if os.getenv("ORCHESTRA_TASK_TIMEOUT"):
            config.setdefault("orchestra", {})["task_timeout_default"] = int(
                os.getenv("ORCHESTRA_TASK_TIMEOUT")
            )
        
        if os.getenv("ORCHESTRA_HEARTBEAT_INTERVAL"):
            config.setdefault("orchestra", {})["heartbeat_interval"] = int(
                os.getenv("ORCHESTRA_HEARTBEAT_INTERVAL")
            )
        
        # Redis configuration
        if os.getenv("REDIS_URL"):
            config.setdefault("redis", {})["url"] = os.getenv("REDIS_URL")
        
        if os.getenv("REDIS_HOST"):
            config.setdefault("redis", {})["host"] = os.getenv("REDIS_HOST")
        
        if os.getenv("REDIS_PORT"):
            config.setdefault("redis", {})["port"] = int(os.getenv("REDIS_PORT"))
        
        if os.getenv("REDIS_PASSWORD"):
            config.setdefault("redis", {})["password"] = os.getenv("REDIS_PASSWORD")
        
        # Monitoring configuration
        if os.getenv("MONITORING_ENABLED"):
            config.setdefault("monitoring", {})["enabled"] = (
                os.getenv("MONITORING_ENABLED").lower() == "true"
            )
        
        if os.getenv("METRICS_PORT"):
            config.setdefault("monitoring", {})["metrics_port"] = int(
                os.getenv("METRICS_PORT")
            )
        
        # Logging configuration
        if os.getenv("LOG_LEVEL"):
            config.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")
        
        if os.getenv("LOG_FILE"):
            config.setdefault("logging", {})["file_path"] = os.getenv("LOG_FILE")
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and values"""
        try:
            # Validate orchestra config
            orchestra_config = config.get("orchestra", {})
            OrchestraConfig(**orchestra_config)
            
            # Validate redis config
            redis_config = config.get("redis", {})
            RedisConfig(**redis_config)
            
            # Validate monitoring config
            monitoring_config = config.get("monitoring", {})
            MonitoringConfig(**monitoring_config)
            
            # Validate logging config
            logging_config = config.get("logging", {})
            LoggingConfig(**logging_config)
            
            # Validate agent configs
            agents_config = config.get("agents", [])
            for agent_config in agents_config:
                AgentConfig(**agent_config)
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error("Configuration validation failed", error=str(e))
            return False
    
    def save_config(self, config: Dict[str, Any], output_file: str):
        """Save configuration to file"""
        output_path = Path(output_file)
        
        try:
            with open(output_path, 'w') as f:
                if output_path.suffix.lower() in ['.yml', '.yaml']:
                    yaml.safe_dump(config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config, f, indent=2)
            
            logger.info("Configuration saved", path=output_file)
            
        except Exception as e:
            logger.error("Failed to save configuration", 
                        path=output_file, error=str(e))


class ConfigurationManager:
    """Main configuration manager for Agent Orchestra"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.loader = ConfigLoader(config_path)
        self._config: Optional[Dict[str, Any]] = None
        self._orchestra_config: Optional[OrchestraConfig] = None
        self._redis_config: Optional[RedisConfig] = None
        self._monitoring_config: Optional[MonitoringConfig] = None
        self._logging_config: Optional[LoggingConfig] = None
    
    def load(self, config_file: Optional[str] = None) -> bool:
        """Load and validate configuration"""
        self._config = self.loader.load_config(config_file)
        
        if not self.loader.validate_config(self._config):
            return False
        
        # Create typed configuration objects
        self._orchestra_config = OrchestraConfig(**self._config.get("orchestra", {}))
        self._redis_config = RedisConfig(**self._config.get("redis", {}))
        self._monitoring_config = MonitoringConfig(**self._config.get("monitoring", {}))
        self._logging_config = LoggingConfig(**self._config.get("logging", {}))
        
        return True
    
    @property
    def orchestra(self) -> OrchestraConfig:
        """Get orchestra configuration"""
        if self._orchestra_config is None:
            raise RuntimeError("Configuration not loaded")
        return self._orchestra_config
    
    @property
    def redis(self) -> RedisConfig:
        """Get Redis configuration"""
        if self._redis_config is None:
            raise RuntimeError("Configuration not loaded")
        return self._redis_config
    
    @property
    def monitoring(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        if self._monitoring_config is None:
            raise RuntimeError("Configuration not loaded")
        return self._monitoring_config
    
    @property
    def logging(self) -> LoggingConfig:
        """Get logging configuration"""
        if self._logging_config is None:
            raise RuntimeError("Configuration not loaded")
        return self._logging_config
    
    def get_agent_configs(self) -> list[AgentConfig]:
        """Get agent configurations"""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        
        agent_configs = []
        for agent_data in self._config.get("agents", []):
            agent_configs.append(AgentConfig(**agent_data))
        
        return agent_configs
    
    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL for connection"""
        redis_config = self.redis
        
        if redis_config.url:
            return redis_config.url
        
        # Construct URL from components
        url = f"redis://"
        if redis_config.password:
            url += f":{redis_config.password}@"
        url += f"{redis_config.host}:{redis_config.port}/{redis_config.db}"
        
        return url
    
    def export_config(self, output_file: str):
        """Export current configuration to file"""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        
        self.loader.save_config(self._config, output_file)