"""
Plugin system for Agent Orchestra - extensibility framework
"""
import asyncio
import inspect
import importlib
import pkgutil
from typing import Dict, List, Any, Optional, Type, Callable, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import structlog

from .types import Task, AgentInfo, ExecutionResult
from .events import EventBus, Event

logger = structlog.get_logger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = None
    config_schema: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config_schema is None:
            self.config_schema = {}


class PluginInterface(ABC):
    """Base interface for all plugins"""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        pass
    
    @abstractmethod
    async def initialize(self, context: 'PluginContext'):
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Clean shutdown of the plugin"""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration"""
        return True


class TaskPlugin(PluginInterface):
    """Plugin interface for task-related functionality"""
    
    async def before_task_execution(self, task: Task, agent_info: AgentInfo) -> Task:
        """Called before task execution"""
        return task
    
    async def after_task_execution(self, task: Task, result: ExecutionResult) -> ExecutionResult:
        """Called after task execution"""
        return result
    
    async def on_task_failure(self, task: Task, error: Exception) -> bool:
        """Called when task fails. Return True if handled."""
        return False


class AgentPlugin(PluginInterface):
    """Plugin interface for agent-related functionality"""
    
    async def on_agent_registered(self, agent_info: AgentInfo):
        """Called when agent is registered"""
        pass
    
    async def on_agent_unregistered(self, agent_id: str):
        """Called when agent is unregistered"""
        pass
    
    async def modify_agent_selection(self, task: Task, candidates: List[AgentInfo]) -> List[AgentInfo]:
        """Modify agent selection for task"""
        return candidates


class MonitoringPlugin(PluginInterface):
    """Plugin interface for monitoring functionality"""
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect custom metrics"""
        return {}
    
    async def on_metric_threshold_exceeded(self, metric_name: str, value: float, threshold: float):
        """Called when metric threshold is exceeded"""
        pass


class RoutingPlugin(PluginInterface):
    """Plugin interface for task routing"""
    
    async def route_task(self, task: Task, available_agents: List[AgentInfo]) -> Optional[str]:
        """Custom task routing logic. Return agent_id or None."""
        return None
    
    async def calculate_priority(self, task: Task) -> float:
        """Calculate custom task priority"""
        return 0.0


@dataclass
class PluginContext:
    """Context provided to plugins during initialization"""
    orchestra: Any  # Orchestra instance
    event_bus: EventBus
    config: Dict[str, Any]
    shared_data: Dict[str, Any]
    logger: Any


class PluginManager:
    """Manages plugin loading, initialization, and lifecycle"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._plugins: Dict[str, PluginInterface] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
        self._plugin_dependencies: Dict[str, Set[str]] = {}
        self._shared_data: Dict[str, Any] = {}
        self._hooks: Dict[str, List[PluginInterface]] = {}
        
        # Register plugin event handlers
        self._register_event_handlers()
    
    def register_plugin(self, plugin: PluginInterface, config: Dict[str, Any] = None):
        """Manually register a plugin instance"""
        metadata = plugin.metadata
        
        if metadata.name in self._plugins:
            raise ValueError(f"Plugin {metadata.name} already registered")
        
        # Validate configuration
        config = config or {}
        if not plugin.validate_config(config):
            raise ValueError(f"Invalid configuration for plugin {metadata.name}")
        
        self._plugins[metadata.name] = plugin
        self._plugin_configs[metadata.name] = config
        self._plugin_dependencies[metadata.name] = set(metadata.dependencies)
        
        # Register plugin for hooks based on its interfaces
        self._register_plugin_hooks(plugin)
        
        logger.info("Plugin registered", 
                   name=metadata.name, 
                   version=metadata.version)
    
    def _register_plugin_hooks(self, plugin: PluginInterface):
        """Register plugin for appropriate hooks"""
        plugin_name = plugin.metadata.name
        
        if isinstance(plugin, TaskPlugin):
            for hook in ["before_task_execution", "after_task_execution", "on_task_failure"]:
                if hook not in self._hooks:
                    self._hooks[hook] = []
                self._hooks[hook].append(plugin)
        
        if isinstance(plugin, AgentPlugin):
            for hook in ["on_agent_registered", "on_agent_unregistered", "modify_agent_selection"]:
                if hook not in self._hooks:
                    self._hooks[hook] = []
                self._hooks[hook].append(plugin)
        
        if isinstance(plugin, MonitoringPlugin):
            for hook in ["collect_metrics", "on_metric_threshold_exceeded"]:
                if hook not in self._hooks:
                    self._hooks[hook] = []
                self._hooks[hook].append(plugin)
        
        if isinstance(plugin, RoutingPlugin):
            for hook in ["route_task", "calculate_priority"]:
                if hook not in self._hooks:
                    self._hooks[hook] = []
                self._hooks[hook].append(plugin)
    
    async def load_plugin_from_module(self, module_name: str, plugin_class: str, config: Dict[str, Any] = None):
        """Load plugin from module"""
        try:
            module = importlib.import_module(module_name)
            plugin_cls = getattr(module, plugin_class)
            
            if not issubclass(plugin_cls, PluginInterface):
                raise ValueError(f"{plugin_class} is not a valid plugin")
            
            plugin_instance = plugin_cls()
            self.register_plugin(plugin_instance, config)
            
            logger.info("Plugin loaded from module",
                       module=module_name,
                       plugin_class=plugin_class)
            
        except Exception as e:
            logger.error("Failed to load plugin from module",
                        module=module_name,
                        plugin_class=plugin_class,
                        error=str(e))
            raise
    
    async def discover_plugins(self, package_name: str):
        """Auto-discover plugins in a package"""
        try:
            package = importlib.import_module(package_name)
            
            for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
                if not ispkg:
                    full_module_name = f"{package_name}.{modname}"
                    try:
                        module = importlib.import_module(full_module_name)
                        
                        # Look for plugin classes
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(obj, PluginInterface) and 
                                obj is not PluginInterface and
                                hasattr(obj, 'metadata')):
                                
                                plugin_instance = obj()
                                self.register_plugin(plugin_instance)
                                
                                logger.info("Plugin auto-discovered",
                                           module=full_module_name,
                                           plugin_class=name)
                    
                    except Exception as e:
                        logger.warning("Failed to load plugin module",
                                     module=full_module_name,
                                     error=str(e))
        
        except Exception as e:
            logger.error("Plugin discovery failed",
                        package=package_name,
                        error=str(e))
    
    async def initialize_plugins(self, orchestra):
        """Initialize all registered plugins"""
        # Sort plugins by dependencies
        initialization_order = self._resolve_dependencies()
        
        for plugin_name in initialization_order:
            plugin = self._plugins[plugin_name]
            config = self._plugin_configs[plugin_name]
            
            context = PluginContext(
                orchestra=orchestra,
                event_bus=self.event_bus,
                config=config,
                shared_data=self._shared_data,
                logger=logger.bind(plugin=plugin_name)
            )
            
            try:
                await plugin.initialize(context)
                logger.info("Plugin initialized", name=plugin_name)
                
                # Emit plugin initialized event
                await self.event_bus.emit(
                    "plugin.initialized",
                    "plugin_manager",
                    {"plugin_name": plugin_name, "version": plugin.metadata.version}
                )
                
            except Exception as e:
                logger.error("Plugin initialization failed",
                           name=plugin_name,
                           error=str(e))
                raise
    
    async def shutdown_plugins(self):
        """Shutdown all plugins in reverse order"""
        initialization_order = self._resolve_dependencies()
        
        for plugin_name in reversed(initialization_order):
            plugin = self._plugins[plugin_name]
            
            try:
                await plugin.shutdown()
                logger.info("Plugin shutdown", name=plugin_name)
                
            except Exception as e:
                logger.error("Plugin shutdown failed",
                           name=plugin_name,
                           error=str(e))
    
    def _resolve_dependencies(self) -> List[str]:
        """Resolve plugin dependency order using topological sort"""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(plugin_name: str):
            if plugin_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {plugin_name}")
            
            if plugin_name not in visited:
                temp_visited.add(plugin_name)
                
                dependencies = self._plugin_dependencies.get(plugin_name, set())
                for dep in dependencies:
                    if dep in self._plugins:
                        visit(dep)
                    else:
                        logger.warning("Plugin dependency not found",
                                     plugin=plugin_name,
                                     dependency=dep)
                
                temp_visited.remove(plugin_name)
                visited.add(plugin_name)
                result.append(plugin_name)
        
        for plugin_name in self._plugins.keys():
            if plugin_name not in visited:
                visit(plugin_name)
        
        return result
    
    async def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Call all plugins registered for a specific hook"""
        results = []
        
        if hook_name in self._hooks:
            for plugin in self._hooks[hook_name]:
                try:
                    method = getattr(plugin, hook_name, None)
                    if method and callable(method):
                        if asyncio.iscoroutinefunction(method):
                            result = await method(*args, **kwargs)
                        else:
                            result = method(*args, **kwargs)
                        results.append(result)
                
                except Exception as e:
                    logger.error("Plugin hook failed",
                               plugin=plugin.metadata.name,
                               hook=hook_name,
                               error=str(e))
        
        return results
    
    def get_plugins_by_type(self, plugin_type: Type[PluginInterface]) -> List[PluginInterface]:
        """Get all plugins of a specific type"""
        return [
            plugin for plugin in self._plugins.values()
            if isinstance(plugin, plugin_type)
        ]
    
    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get plugin by name"""
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins"""
        return [
            {
                "name": plugin.metadata.name,
                "version": plugin.metadata.version,
                "description": plugin.metadata.description,
                "author": plugin.metadata.author,
                "dependencies": plugin.metadata.dependencies,
                "type": type(plugin).__name__
            }
            for plugin in self._plugins.values()
        ]
    
    def _register_event_handlers(self):
        """Register event handlers for plugin management"""
        self.event_bus.subscribe("plugin.reload_requested", self._handle_plugin_reload)
        self.event_bus.subscribe("plugin.config_changed", self._handle_config_change)
    
    async def _handle_plugin_reload(self, event: Event):
        """Handle plugin reload request"""
        plugin_name = event.data.get("plugin_name")
        
        if plugin_name and plugin_name in self._plugins:
            logger.info("Reloading plugin", name=plugin_name)
            
            # Shutdown plugin
            plugin = self._plugins[plugin_name]
            await plugin.shutdown()
            
            # Remove from registry
            del self._plugins[plugin_name]
            del self._plugin_configs[plugin_name]
            del self._plugin_dependencies[plugin_name]
            
            # Remove from hooks
            for hook_list in self._hooks.values():
                if plugin in hook_list:
                    hook_list.remove(plugin)
            
            # TODO: Reload plugin from module
            logger.info("Plugin reloaded", name=plugin_name)
    
    async def _handle_config_change(self, event: Event):
        """Handle plugin configuration change"""
        plugin_name = event.data.get("plugin_name")
        new_config = event.data.get("config", {})
        
        if plugin_name and plugin_name in self._plugins:
            plugin = self._plugins[plugin_name]
            
            if plugin.validate_config(new_config):
                self._plugin_configs[plugin_name] = new_config
                logger.info("Plugin configuration updated", name=plugin_name)
            else:
                logger.error("Invalid plugin configuration", name=plugin_name)


# Example plugins

class LoggingPlugin(TaskPlugin):
    """Example plugin that logs task execution"""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="logging_plugin",
            version="1.0.0",
            description="Logs task execution details",
            author="Agent Orchestra Team"
        )
    
    async def initialize(self, context: PluginContext):
        self.logger = context.logger
        self.config = context.config
        
    async def shutdown(self):
        pass
    
    async def before_task_execution(self, task: Task, agent_info: AgentInfo) -> Task:
        self.logger.info("Task starting",
                        task_id=task.id,
                        task_type=task.type,
                        agent_id=agent_info.id)
        return task
    
    async def after_task_execution(self, task: Task, result: ExecutionResult) -> ExecutionResult:
        self.logger.info("Task completed",
                        task_id=task.id,
                        success=result.success,
                        execution_time=result.execution_time)
        return result


class ResourceMonitoringPlugin(MonitoringPlugin):
    """Example plugin that monitors resource usage"""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="resource_monitoring",
            version="1.0.0", 
            description="Monitors system resource usage",
            author="Agent Orchestra Team"
        )
    
    async def initialize(self, context: PluginContext):
        self.context = context
        
    async def shutdown(self):
        pass
    
    async def collect_metrics(self) -> Dict[str, Any]:
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
    
    async def on_metric_threshold_exceeded(self, metric_name: str, value: float, threshold: float):
        self.context.logger.warning("Resource threshold exceeded",
                                   metric=metric_name,
                                   value=value,
                                   threshold=threshold)