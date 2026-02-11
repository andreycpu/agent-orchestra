"""
Database migration utilities for Agent Orchestra
"""
import os
import json
import hashlib
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import structlog

from .exceptions import ConfigurationError, ValidationError

logger = structlog.get_logger(__name__)


@dataclass
class MigrationInfo:
    """Information about a database migration"""
    version: str
    name: str
    description: str
    checksum: str
    applied_at: Optional[datetime] = None
    execution_time: Optional[float] = None


class Migration(ABC):
    """Abstract base class for database migrations"""
    
    def __init__(self, version: str, name: str, description: str):
        self.version = version
        self.name = name
        self.description = description
    
    @abstractmethod
    async def up(self, connection) -> None:
        """Apply the migration"""
        pass
    
    @abstractmethod
    async def down(self, connection) -> None:
        """Rollback the migration"""
        pass
    
    def get_checksum(self) -> str:
        """Calculate checksum of migration content"""
        content = f"{self.version}{self.name}{self.description}"
        return hashlib.sha256(content.encode()).hexdigest()


class SchemaMigration(Migration):
    """Migration for schema changes"""
    
    def __init__(self, version: str, name: str, description: str, up_sql: str, down_sql: str):
        super().__init__(version, name, description)
        self.up_sql = up_sql
        self.down_sql = down_sql
    
    async def up(self, connection) -> None:
        """Apply schema changes"""
        if hasattr(connection, 'execute'):
            await connection.execute(self.up_sql)
        else:
            # For Redis or other non-SQL backends
            logger.info(f"Schema migration {self.version} applied (no SQL backend)")
    
    async def down(self, connection) -> None:
        """Rollback schema changes"""
        if hasattr(connection, 'execute'):
            await connection.execute(self.down_sql)
        else:
            logger.info(f"Schema migration {self.version} rolled back (no SQL backend)")
    
    def get_checksum(self) -> str:
        """Calculate checksum including SQL content"""
        content = f"{self.version}{self.name}{self.description}{self.up_sql}{self.down_sql}"
        return hashlib.sha256(content.encode()).hexdigest()


class DataMigration(Migration):
    """Migration for data transformations"""
    
    def __init__(self, version: str, name: str, description: str, 
                 up_func: Callable, down_func: Callable):
        super().__init__(version, name, description)
        self.up_func = up_func
        self.down_func = down_func
    
    async def up(self, connection) -> None:
        """Apply data transformation"""
        await self.up_func(connection)
    
    async def down(self, connection) -> None:
        """Rollback data transformation"""
        await self.down_func(connection)


class MigrationManager:
    """Manages database migrations for Agent Orchestra"""
    
    def __init__(self, connection_factory: Callable, migrations_dir: str = "migrations"):
        self.connection_factory = connection_factory
        self.migrations_dir = Path(migrations_dir)
        self._migrations: Dict[str, Migration] = {}
        self._applied_migrations: Dict[str, MigrationInfo] = {}
        
    def register_migration(self, migration: Migration):
        """Register a migration"""
        if migration.version in self._migrations:
            raise ValidationError(f"Migration version {migration.version} already registered")
        
        self._migrations[migration.version] = migration
        
        logger.debug(
            "Migration registered",
            version=migration.version,
            name=migration.name
        )
    
    def load_migrations_from_directory(self):
        """Load migrations from the migrations directory"""
        if not self.migrations_dir.exists():
            logger.warning(f"Migrations directory {self.migrations_dir} does not exist")
            return
        
        # Load SQL migrations
        for sql_file in self.migrations_dir.glob("*.sql"):
            self._load_sql_migration(sql_file)
        
        # Load Python migrations
        for py_file in self.migrations_dir.glob("*.py"):
            if py_file.name != "__init__.py":
                self._load_python_migration(py_file)
        
        logger.info(f"Loaded {len(self._migrations)} migrations from {self.migrations_dir}")
    
    def _load_sql_migration(self, file_path: Path):
        """Load a SQL migration file"""
        try:
            content = file_path.read_text()
            
            # Parse migration metadata from comments
            version = None
            name = None
            description = ""
            up_sql = ""
            down_sql = ""
            
            current_section = None
            
            for line in content.split('\n'):
                line = line.strip()
                
                if line.startswith('-- Version:'):
                    version = line.split(':', 1)[1].strip()
                elif line.startswith('-- Name:'):
                    name = line.split(':', 1)[1].strip()
                elif line.startswith('-- Description:'):
                    description = line.split(':', 1)[1].strip()
                elif line.startswith('-- UP'):
                    current_section = 'up'
                elif line.startswith('-- DOWN'):
                    current_section = 'down'
                elif not line.startswith('--') and line:
                    if current_section == 'up':
                        up_sql += line + '\n'
                    elif current_section == 'down':
                        down_sql += line + '\n'
            
            if version and name:
                migration = SchemaMigration(version, name, description, up_sql.strip(), down_sql.strip())
                self.register_migration(migration)
            else:
                logger.warning(f"Invalid migration file format: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to load SQL migration {file_path}: {e}")
    
    def _load_python_migration(self, file_path: Path):
        """Load a Python migration file"""
        try:
            # This would require dynamic import, which is complex
            # For now, just log that Python migrations are not yet supported
            logger.warning(f"Python migration loading not implemented: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load Python migration {file_path}: {e}")
    
    async def initialize_migration_tracking(self, connection):
        """Initialize migration tracking in the database"""
        try:
            if hasattr(connection, 'execute'):
                # SQL database - create migrations table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        checksum VARCHAR(64) NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        execution_time_ms INTEGER
                    )
                """)
            else:
                # Redis or other key-value store
                # Store migration info in a hash
                existing = await connection.hgetall("schema_migrations")
                if not existing:
                    await connection.hset("schema_migrations", "__initialized", str(datetime.utcnow()))
            
            logger.info("Migration tracking initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize migration tracking: {e}")
            raise ConfigurationError(f"Migration initialization failed: {e}")
    
    async def load_applied_migrations(self, connection):
        """Load list of applied migrations from the database"""
        try:
            self._applied_migrations.clear()
            
            if hasattr(connection, 'fetchall'):
                # SQL database
                rows = await connection.fetchall("SELECT version, name, description, checksum, applied_at, execution_time_ms FROM schema_migrations")
                for row in rows:
                    info = MigrationInfo(
                        version=row['version'],
                        name=row['name'],
                        description=row['description'],
                        checksum=row['checksum'],
                        applied_at=row['applied_at'],
                        execution_time=row['execution_time_ms']
                    )
                    self._applied_migrations[info.version] = info
            else:
                # Redis or other key-value store
                migration_data = await connection.hgetall("schema_migrations")
                for key, value in migration_data.items():
                    if key != "__initialized":
                        info_dict = json.loads(value)
                        info = MigrationInfo(**info_dict)
                        self._applied_migrations[info.version] = info
            
            logger.info(f"Loaded {len(self._applied_migrations)} applied migrations")
            
        except Exception as e:
            logger.error(f"Failed to load applied migrations: {e}")
            raise ConfigurationError(f"Failed to load migration history: {e}")
    
    async def get_pending_migrations(self) -> List[Migration]:
        """Get list of migrations that haven't been applied"""
        pending = []
        
        for version in sorted(self._migrations.keys()):
            migration = self._migrations[version]
            
            if version not in self._applied_migrations:
                pending.append(migration)
            else:
                # Check if migration has changed (checksum mismatch)
                applied_info = self._applied_migrations[version]
                if migration.get_checksum() != applied_info.checksum:
                    logger.warning(
                        "Migration checksum mismatch",
                        version=version,
                        expected=migration.get_checksum(),
                        actual=applied_info.checksum
                    )
        
        return pending
    
    async def apply_migration(self, migration: Migration, connection):
        """Apply a single migration"""
        start_time = datetime.utcnow()
        
        try:
            logger.info(
                "Applying migration",
                version=migration.version,
                name=migration.name
            )
            
            await migration.up(connection)
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            # Record migration as applied
            await self._record_migration_applied(migration, connection, start_time, execution_time)
            
            logger.info(
                "Migration applied successfully",
                version=migration.version,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(
                "Migration failed",
                version=migration.version,
                error=str(e)
            )
            raise
    
    async def _record_migration_applied(self, migration: Migration, connection, 
                                       applied_at: datetime, execution_time: float):
        """Record a migration as applied"""
        try:
            if hasattr(connection, 'execute'):
                # SQL database
                await connection.execute(
                    "INSERT INTO schema_migrations (version, name, description, checksum, applied_at, execution_time_ms) VALUES (?, ?, ?, ?, ?, ?)",
                    migration.version, migration.name, migration.description, 
                    migration.get_checksum(), applied_at, int(execution_time)
                )
            else:
                # Redis or other key-value store
                info = MigrationInfo(
                    version=migration.version,
                    name=migration.name,
                    description=migration.description,
                    checksum=migration.get_checksum(),
                    applied_at=applied_at,
                    execution_time=execution_time
                )
                await connection.hset("schema_migrations", migration.version, json.dumps(info.__dict__, default=str))
            
            # Update local cache
            self._applied_migrations[migration.version] = MigrationInfo(
                version=migration.version,
                name=migration.name,
                description=migration.description,
                checksum=migration.get_checksum(),
                applied_at=applied_at,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Failed to record migration: {e}")
            raise
    
    async def migrate(self, target_version: Optional[str] = None):
        """Apply all pending migrations up to target version"""
        async with self.connection_factory() as connection:
            await self.initialize_migration_tracking(connection)
            await self.load_applied_migrations(connection)
            
            pending = await self.get_pending_migrations()
            
            if not pending:
                logger.info("No pending migrations")
                return
            
            # Filter by target version if specified
            if target_version:
                pending = [m for m in pending if m.version <= target_version]
            
            logger.info(f"Applying {len(pending)} migrations")
            
            for migration in pending:
                await self.apply_migration(migration, connection)
            
            logger.info("All migrations applied successfully")
    
    async def rollback(self, target_version: str):
        """Rollback migrations to target version"""
        async with self.connection_factory() as connection:
            await self.load_applied_migrations(connection)
            
            # Find migrations to rollback (in reverse order)
            to_rollback = []
            for version in sorted(self._applied_migrations.keys(), reverse=True):
                if version > target_version:
                    if version in self._migrations:
                        to_rollback.append(self._migrations[version])
                    else:
                        logger.warning(f"Cannot rollback migration {version}: not found")
            
            logger.info(f"Rolling back {len(to_rollback)} migrations")
            
            for migration in to_rollback:
                logger.info(f"Rolling back migration {migration.version}")
                await migration.down(connection)
                
                # Remove from applied migrations table
                if hasattr(connection, 'execute'):
                    await connection.execute("DELETE FROM schema_migrations WHERE version = ?", migration.version)
                else:
                    await connection.hdel("schema_migrations", migration.version)
                
                # Remove from local cache
                del self._applied_migrations[migration.version]
            
            logger.info("Rollback completed successfully")
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        total_migrations = len(self._migrations)
        applied_migrations = len(self._applied_migrations)
        
        return {
            "total_migrations": total_migrations,
            "applied_migrations": applied_migrations,
            "pending_migrations": total_migrations - applied_migrations,
            "migration_versions": sorted(self._migrations.keys()),
            "applied_versions": sorted(self._applied_migrations.keys())
        }