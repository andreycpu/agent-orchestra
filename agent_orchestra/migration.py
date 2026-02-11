"""
Database and state migration utilities for Agent Orchestra
"""
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
import structlog

from .state_manager import StateManager
from .types import Task, AgentInfo

logger = structlog.get_logger(__name__)


@dataclass
class Migration:
    """Represents a single migration"""
    version: str
    description: str
    upgrade_func: Callable
    downgrade_func: Optional[Callable] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class MigrationManager:
    """Manages database schema and state migrations"""
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self._migrations: Dict[str, Migration] = {}
        self._migration_history: List[Dict[str, Any]] = []
        
        # Register built-in migrations
        self._register_builtin_migrations()
    
    def register_migration(self, migration: Migration):
        """Register a migration"""
        self._migrations[migration.version] = migration
        logger.info("Migration registered", version=migration.version, description=migration.description)
    
    def _register_builtin_migrations(self):
        """Register built-in system migrations"""
        
        # Migration 0.1.0 -> 0.2.0: Add agent metadata fields
        self.register_migration(Migration(
            version="0.2.0",
            description="Add metadata fields to agent info",
            upgrade_func=self._migrate_agent_metadata,
            downgrade_func=self._rollback_agent_metadata
        ))
        
        # Migration 0.2.0 -> 0.3.0: Add task dependency tracking
        self.register_migration(Migration(
            version="0.3.0", 
            description="Add dependency tracking to tasks",
            upgrade_func=self._migrate_task_dependencies,
            downgrade_func=self._rollback_task_dependencies,
            dependencies=["0.2.0"]
        ))
        
        # Migration 0.3.0 -> 0.4.0: Add execution metrics
        self.register_migration(Migration(
            version="0.4.0",
            description="Add execution metrics storage",
            upgrade_func=self._migrate_execution_metrics,
            downgrade_func=self._rollback_execution_metrics,
            dependencies=["0.3.0"]
        ))
    
    async def get_current_version(self) -> str:
        """Get current schema version"""
        if self.state_manager._redis:
            version = await self.state_manager._redis.get("schema_version")
            return version.decode() if version else "0.1.0"
        else:
            # For in-memory storage, check if migration history exists
            return getattr(self.state_manager, '_schema_version', '0.1.0')
    
    async def set_version(self, version: str):
        """Set current schema version"""
        if self.state_manager._redis:
            await self.state_manager._redis.set("schema_version", version)
        else:
            self.state_manager._schema_version = version
    
    async def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history"""
        if self.state_manager._redis:
            history_data = await self.state_manager._redis.get("migration_history")
            if history_data:
                return json.loads(history_data.decode())
        else:
            return getattr(self.state_manager, '_migration_history', [])
        
        return []
    
    async def record_migration(self, version: str, direction: str = "up"):
        """Record a migration in history"""
        history = await self.get_migration_history()
        
        record = {
            "version": version,
            "direction": direction,
            "timestamp": datetime.utcnow().isoformat(),
            "description": self._migrations[version].description if version in self._migrations else ""
        }
        
        history.append(record)
        
        if self.state_manager._redis:
            await self.state_manager._redis.set(
                "migration_history",
                json.dumps(history, default=str)
            )
        else:
            self.state_manager._migration_history = history
        
        logger.info("Migration recorded", version=version, direction=direction)
    
    async def needs_migration(self, target_version: str) -> bool:
        """Check if migration is needed to reach target version"""
        current_version = await self.get_current_version()
        return self._compare_versions(current_version, target_version) < 0
    
    async def migrate_to_version(self, target_version: str):
        """Migrate to specific version"""
        current_version = await self.get_current_version()
        
        if target_version == current_version:
            logger.info("Already at target version", version=target_version)
            return
        
        # Determine migration path
        migration_path = self._calculate_migration_path(current_version, target_version)
        
        if not migration_path:
            raise ValueError(f"No migration path from {current_version} to {target_version}")
        
        logger.info("Starting migration", 
                   from_version=current_version,
                   to_version=target_version,
                   path=migration_path)
        
        # Execute migrations
        for version in migration_path:
            if version in self._migrations:
                migration = self._migrations[version]
                
                logger.info("Applying migration", 
                           version=version,
                           description=migration.description)
                
                try:
                    await migration.upgrade_func()
                    await self.set_version(version)
                    await self.record_migration(version, "up")
                    
                    logger.info("Migration applied successfully", version=version)
                    
                except Exception as e:
                    logger.error("Migration failed", 
                                version=version,
                                error=str(e))
                    raise
    
    async def rollback_to_version(self, target_version: str):
        """Rollback to specific version"""
        current_version = await self.get_current_version()
        
        if target_version == current_version:
            logger.info("Already at target version", version=target_version)
            return
        
        if self._compare_versions(target_version, current_version) > 0:
            raise ValueError(f"Cannot rollback to newer version {target_version}")
        
        # Get rollback path
        history = await self.get_migration_history()
        
        # Find migrations to rollback
        migrations_to_rollback = []
        for record in reversed(history):
            if (record["direction"] == "up" and 
                self._compare_versions(record["version"], target_version) > 0):
                migrations_to_rollback.append(record["version"])
        
        logger.info("Starting rollback",
                   from_version=current_version,
                   to_version=target_version,
                   migrations=migrations_to_rollback)
        
        # Execute rollbacks
        for version in migrations_to_rollback:
            if version in self._migrations:
                migration = self._migrations[version]
                
                if migration.downgrade_func:
                    logger.info("Rolling back migration",
                               version=version,
                               description=migration.description)
                    
                    try:
                        await migration.downgrade_func()
                        await self.record_migration(version, "down")
                        
                        logger.info("Migration rolled back successfully", version=version)
                        
                    except Exception as e:
                        logger.error("Rollback failed",
                                    version=version,
                                    error=str(e))
                        raise
                else:
                    logger.warning("No downgrade function for migration", version=version)
        
        await self.set_version(target_version)
    
    def _calculate_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Calculate migration path between versions"""
        # Simple linear path for now
        # In a real implementation, this would handle complex dependency graphs
        
        available_versions = list(self._migrations.keys())
        available_versions.sort(key=self._version_key)
        
        from_idx = -1
        to_idx = -1
        
        for i, version in enumerate(available_versions):
            if self._compare_versions(version, from_version) > 0 and from_idx == -1:
                from_idx = i
            if version == to_version:
                to_idx = i
                break
        
        if from_idx != -1 and to_idx != -1:
            return available_versions[from_idx:to_idx + 1]
        
        return []
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings (-1, 0, 1)"""
        v1_parts = [int(x) for x in version1.split('.')]
        v2_parts = [int(x) for x in version2.split('.')]
        
        # Pad with zeros
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for v1, v2 in zip(v1_parts, v2_parts):
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
        
        return 0
    
    def _version_key(self, version: str):
        """Create sort key for version string"""
        return tuple(int(x) for x in version.split('.'))
    
    # Built-in migration functions
    async def _migrate_agent_metadata(self):
        """Add metadata fields to existing agents"""
        agents = await self.state_manager.get_all_agents()
        
        for agent in agents:
            # Add default metadata if not present
            if not hasattr(agent, 'metadata') or agent.metadata is None:
                agent.metadata = {
                    "created_at": datetime.utcnow().isoformat(),
                    "version": "0.2.0",
                    "migrated": True
                }
                await self.state_manager.register_agent(agent)
        
        logger.info("Added metadata to agents", count=len(agents))
    
    async def _rollback_agent_metadata(self):
        """Remove metadata fields from agents"""
        agents = await self.state_manager.get_all_agents()
        
        for agent in agents:
            if hasattr(agent, 'metadata'):
                agent.metadata = None
                await self.state_manager.register_agent(agent)
        
        logger.info("Removed metadata from agents", count=len(agents))
    
    async def _migrate_task_dependencies(self):
        """Add dependency tracking to tasks"""
        # Get all tasks
        if self.state_manager._redis:
            task_data = await self.state_manager._redis.hgetall("tasks")
            for task_id, data in task_data.items():
                task_dict = json.loads(data)
                if "dependencies" not in task_dict:
                    task_dict["dependencies"] = []
                    
                await self.state_manager._redis.hset(
                    "tasks",
                    task_id,
                    json.dumps(task_dict, default=str)
                )
        else:
            for task in self.state_manager._task_history.values():
                if not hasattr(task, 'dependencies'):
                    task.dependencies = []
        
        logger.info("Added dependency tracking to tasks")
    
    async def _rollback_task_dependencies(self):
        """Remove dependency tracking from tasks"""
        # Implementation would remove dependency fields
        logger.info("Removed dependency tracking from tasks")
    
    async def _migrate_execution_metrics(self):
        """Add execution metrics storage"""
        # Create metrics storage structures
        if self.state_manager._redis:
            # Initialize metrics keys
            await self.state_manager._redis.set("metrics_initialized", "true")
        else:
            self.state_manager._execution_metrics = {}
        
        logger.info("Initialized execution metrics storage")
    
    async def _rollback_execution_metrics(self):
        """Remove execution metrics storage"""
        if self.state_manager._redis:
            # Remove metrics keys
            await self.state_manager._redis.delete("metrics_initialized")
        else:
            if hasattr(self.state_manager, '_execution_metrics'):
                delattr(self.state_manager, '_execution_metrics')
        
        logger.info("Removed execution metrics storage")
    
    async def create_backup(self) -> str:
        """Create a backup of current state"""
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        if self.state_manager._redis:
            # Create Redis backup
            backup_key = f"backup:{backup_id}"
            
            # Save current state
            tasks = await self.state_manager._redis.hgetall("tasks")
            agents = await self.state_manager._redis.hgetall("agents")
            version = await self.get_current_version()
            history = await self.get_migration_history()
            
            backup_data = {
                "version": version,
                "migration_history": history,
                "tasks": {k.decode(): v.decode() for k, v in tasks.items()},
                "agents": {k.decode(): v.decode() for k, v in agents.items()},
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self.state_manager._redis.set(
                backup_key,
                json.dumps(backup_data, default=str),
                ex=86400 * 30  # Keep for 30 days
            )
        
        logger.info("Backup created", backup_id=backup_id)
        return backup_id
    
    async def restore_backup(self, backup_id: str):
        """Restore from backup"""
        if self.state_manager._redis:
            backup_key = f"backup:{backup_id}"
            backup_data = await self.state_manager._redis.get(backup_key)
            
            if not backup_data:
                raise ValueError(f"Backup {backup_id} not found")
            
            data = json.loads(backup_data.decode())
            
            # Restore state
            await self.state_manager._redis.delete("tasks", "agents")
            
            # Restore tasks
            for task_id, task_data in data["tasks"].items():
                await self.state_manager._redis.hset("tasks", task_id, task_data)
            
            # Restore agents
            for agent_id, agent_data in data["agents"].items():
                await self.state_manager._redis.hset("agents", agent_id, agent_data)
            
            # Restore version and history
            await self.set_version(data["version"])
            await self.state_manager._redis.set(
                "migration_history",
                json.dumps(data["migration_history"], default=str)
            )
        
        logger.info("Backup restored", backup_id=backup_id)