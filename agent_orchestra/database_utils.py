"""
Database utilities and connection management for Agent Orchestra.

This module provides database abstraction, connection pooling,
migration utilities, and query helpers.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, AsyncContextManager
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import json

from .exceptions import DatabaseError, RetryableError, PermanentError
from .validation import validate_timeout


logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REDIS = "redis"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    connect_timeout: int = 10
    command_timeout: int = 60
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.url:
            raise ValueError("Database URL is required")
        
        validate_timeout(self.pool_timeout)
        validate_timeout(self.connect_timeout)
        validate_timeout(self.command_timeout)


@dataclass
class QueryResult:
    """Database query result."""
    rows: List[Dict[str, Any]]
    rowcount: int
    query: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def first(self) -> Optional[Dict[str, Any]]:
        """Get first row or None."""
        return self.rows[0] if self.rows else None
    
    @property
    def is_empty(self) -> bool:
        """Check if result is empty."""
        return len(self.rows) == 0


@dataclass
class DatabaseMetrics:
    """Database connection and query metrics."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_connections: int = 0
    active_connections: int = 0
    pool_size: int = 0
    avg_query_time_ms: float = 0.0
    slow_query_count: int = 0
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_queries': self.total_queries,
            'successful_queries': self.successful_queries,
            'failed_queries': self.failed_queries,
            'query_success_rate': self.successful_queries / max(self.total_queries, 1),
            'total_connections': self.total_connections,
            'active_connections': self.active_connections,
            'pool_size': self.pool_size,
            'avg_query_time_ms': self.avg_query_time_ms,
            'slow_query_count': self.slow_query_count,
            'last_error': self.last_error,
            'uptime_seconds': self.uptime_seconds
        }


class DatabaseTransaction:
    """Database transaction context manager."""
    
    def __init__(self, connection, rollback_on_exception: bool = True):
        self.connection = connection
        self.rollback_on_exception = rollback_on_exception
        self.committed = False
        self.rolled_back = False
    
    async def __aenter__(self):
        """Start transaction."""
        await self.connection.execute("BEGIN")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End transaction."""
        if exc_type and self.rollback_on_exception and not self.rolled_back:
            await self.rollback()
        elif not self.committed and not self.rolled_back:
            await self.commit()
    
    async def commit(self):
        """Commit transaction."""
        if not self.committed and not self.rolled_back:
            await self.connection.execute("COMMIT")
            self.committed = True
    
    async def rollback(self):
        """Rollback transaction."""
        if not self.committed and not self.rolled_back:
            await self.connection.execute("ROLLBACK")
            self.rolled_back = True


class QueryBuilder:
    """SQL query builder utility."""
    
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.reset()
    
    def reset(self):
        """Reset builder state."""
        self._select_fields = []
        self._where_conditions = []
        self._join_clauses = []
        self._order_by = []
        self._group_by = []
        self._having_conditions = []
        self._limit_value = None
        self._offset_value = None
        self._params = {}
        return self
    
    def select(self, *fields):
        """Add SELECT fields."""
        self._select_fields.extend(fields)
        return self
    
    def where(self, condition: str, **params):
        """Add WHERE condition."""
        self._where_conditions.append(condition)
        self._params.update(params)
        return self
    
    def join(self, table: str, on_condition: str):
        """Add JOIN clause."""
        self._join_clauses.append(f"JOIN {table} ON {on_condition}")
        return self
    
    def left_join(self, table: str, on_condition: str):
        """Add LEFT JOIN clause."""
        self._join_clauses.append(f"LEFT JOIN {table} ON {on_condition}")
        return self
    
    def order_by(self, field: str, direction: str = "ASC"):
        """Add ORDER BY clause."""
        self._order_by.append(f"{field} {direction}")
        return self
    
    def group_by(self, *fields):
        """Add GROUP BY clause."""
        self._group_by.extend(fields)
        return self
    
    def having(self, condition: str, **params):
        """Add HAVING condition."""
        self._having_conditions.append(condition)
        self._params.update(params)
        return self
    
    def limit(self, limit: int):
        """Add LIMIT clause."""
        self._limit_value = limit
        return self
    
    def offset(self, offset: int):
        """Add OFFSET clause."""
        self._offset_value = offset
        return self
    
    def build_select(self) -> tuple[str, Dict[str, Any]]:
        """Build SELECT query."""
        fields = ", ".join(self._select_fields) if self._select_fields else "*"
        query = f"SELECT {fields} FROM {self.table_name}"
        
        if self._join_clauses:
            query += " " + " ".join(self._join_clauses)
        
        if self._where_conditions:
            query += " WHERE " + " AND ".join(self._where_conditions)
        
        if self._group_by:
            query += " GROUP BY " + ", ".join(self._group_by)
        
        if self._having_conditions:
            query += " HAVING " + " AND ".join(self._having_conditions)
        
        if self._order_by:
            query += " ORDER BY " + ", ".join(self._order_by)
        
        if self._limit_value is not None:
            query += f" LIMIT {self._limit_value}"
        
        if self._offset_value is not None:
            query += f" OFFSET {self._offset_value}"
        
        return query, self._params.copy()
    
    def build_insert(self, data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Build INSERT query."""
        fields = ", ".join(data.keys())
        placeholders = ", ".join(f":{key}" for key in data.keys())
        query = f"INSERT INTO {self.table_name} ({fields}) VALUES ({placeholders})"
        return query, data
    
    def build_update(self, data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Build UPDATE query."""
        set_clause = ", ".join(f"{key} = :{key}" for key in data.keys())
        query = f"UPDATE {self.table_name} SET {set_clause}"
        
        params = data.copy()
        
        if self._where_conditions:
            query += " WHERE " + " AND ".join(self._where_conditions)
            params.update(self._params)
        
        return query, params
    
    def build_delete(self) -> tuple[str, Dict[str, Any]]:
        """Build DELETE query."""
        query = f"DELETE FROM {self.table_name}"
        
        if self._where_conditions:
            query += " WHERE " + " AND ".join(self._where_conditions)
        
        return query, self._params.copy()


class DatabaseConnection:
    """Database connection wrapper with utilities."""
    
    def __init__(self, connection, db_type: DatabaseType):
        self.connection = connection
        self.db_type = db_type
        self.query_count = 0
        self.last_query_time: Optional[datetime] = None
    
    async def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a query and return results."""
        import time
        
        start_time = time.time()
        self.query_count += 1
        self.last_query_time = datetime.utcnow()
        
        try:
            if parameters:
                cursor = await self.connection.execute(query, parameters)
            else:
                cursor = await self.connection.execute(query)
            
            # Fetch results for SELECT queries
            if query.strip().upper().startswith('SELECT'):
                rows = await cursor.fetchall()
                # Convert to list of dicts
                if hasattr(cursor, 'description') and cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = [dict(zip(columns, row)) for row in rows]
                else:
                    rows = []
                rowcount = len(rows)
            else:
                rows = []
                rowcount = cursor.rowcount if hasattr(cursor, 'rowcount') else 0
            
            duration_ms = (time.time() - start_time) * 1000
            
            return QueryResult(
                rows=rows,
                rowcount=rowcount,
                query=query,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Database query failed after {duration_ms:.2f}ms: {e}")
            raise DatabaseError(f"Query execution failed: {e}") from e
    
    async def execute_many(self, query: str, parameters_list: List[Dict[str, Any]]) -> int:
        """Execute query multiple times with different parameters."""
        try:
            cursor = await self.connection.executemany(query, parameters_list)
            return cursor.rowcount if hasattr(cursor, 'rowcount') else len(parameters_list)
        except Exception as e:
            raise DatabaseError(f"Batch execution failed: {e}") from e
    
    def transaction(self, rollback_on_exception: bool = True) -> DatabaseTransaction:
        """Create a database transaction."""
        return DatabaseTransaction(self, rollback_on_exception)
    
    def query_builder(self, table_name: str) -> QueryBuilder:
        """Create a query builder for a table."""
        return QueryBuilder(table_name)


class DatabaseManager:
    """Database connection and pool manager."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = None
        self.metrics = DatabaseMetrics()
        self._start_time = datetime.utcnow()
        self._slow_query_threshold_ms = 1000.0
    
    async def initialize(self):
        """Initialize database connection pool."""
        try:
            # This would create an actual connection pool based on the database type
            # For now, we'll create a placeholder
            logger.info(f"Initializing database connection pool for {self.config.url}")
            self.metrics.pool_size = self.config.pool_size
            self.metrics.total_connections = 1  # Placeholder
            
        except Exception as e:
            self.metrics.last_error = str(e)
            raise DatabaseError(f"Failed to initialize database pool: {e}") from e
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            # Close actual pool here
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncContextManager[DatabaseConnection]:
        """Get database connection from pool."""
        if not self.pool:
            await self.initialize()
        
        # This would get an actual connection from the pool
        # For now, create a mock connection
        connection = None  # Would be actual connection
        db_connection = DatabaseConnection(connection, DatabaseType.POSTGRESQL)
        
        try:
            self.metrics.active_connections += 1
            yield db_connection
        finally:
            self.metrics.active_connections -= 1
            # Return connection to pool
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a single query."""
        async with self.get_connection() as conn:
            result = await conn.execute(query, parameters)
            
            # Update metrics
            self.metrics.total_queries += 1
            if result:
                self.metrics.successful_queries += 1
                
                # Track slow queries
                if result.duration_ms > self._slow_query_threshold_ms:
                    self.metrics.slow_query_count += 1
                    logger.warning(f"Slow query detected: {result.duration_ms:.2f}ms - {query[:100]}...")
                
                # Update average query time
                total_time = self.metrics.avg_query_time_ms * (self.metrics.successful_queries - 1)
                self.metrics.avg_query_time_ms = (total_time + result.duration_ms) / self.metrics.successful_queries
            else:
                self.metrics.failed_queries += 1
            
            return result
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            start_time = datetime.utcnow()
            result = await self.execute_query("SELECT 1")
            response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.metrics.uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time_ms,
                'uptime_seconds': self.metrics.uptime_seconds,
                'metrics': self.metrics.to_dict()
            }
            
        except Exception as e:
            self.metrics.last_error = str(e)
            return {
                'status': 'unhealthy',
                'error': str(e),
                'uptime_seconds': self.metrics.uptime_seconds,
                'metrics': self.metrics.to_dict()
            }
    
    def set_slow_query_threshold(self, threshold_ms: float):
        """Set threshold for slow query detection."""
        self._slow_query_threshold_ms = threshold_ms


class MigrationManager:
    """Database migration management."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.migrations_table = "schema_migrations"
    
    async def ensure_migrations_table(self):
        """Ensure migrations tracking table exists."""
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.migrations_table} (
            version VARCHAR(255) PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
        """
        await self.db_manager.execute_query(query)
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        await self.ensure_migrations_table()
        
        query = f"SELECT version FROM {self.migrations_table} ORDER BY version"
        result = await self.db_manager.execute_query(query)
        return [row['version'] for row in result.rows]
    
    async def apply_migration(self, version: str, description: str, up_sql: str):
        """Apply a migration."""
        applied_migrations = await self.get_applied_migrations()
        
        if version in applied_migrations:
            logger.info(f"Migration {version} already applied")
            return
        
        async with self.db_manager.get_connection() as conn:
            async with conn.transaction():
                # Execute migration SQL
                await conn.execute(up_sql)
                
                # Record migration
                await conn.execute(
                    f"INSERT INTO {self.migrations_table} (version, description) VALUES (:version, :description)",
                    {'version': version, 'description': description}
                )
        
        logger.info(f"Applied migration {version}: {description}")
    
    async def rollback_migration(self, version: str, down_sql: str):
        """Rollback a migration."""
        applied_migrations = await self.get_applied_migrations()
        
        if version not in applied_migrations:
            logger.warning(f"Migration {version} not found in applied migrations")
            return
        
        async with self.db_manager.get_connection() as conn:
            async with conn.transaction():
                # Execute rollback SQL
                await conn.execute(down_sql)
                
                # Remove migration record
                await conn.execute(
                    f"DELETE FROM {self.migrations_table} WHERE version = :version",
                    {'version': version}
                )
        
        logger.info(f"Rolled back migration {version}")


class DatabaseRepository:
    """Base repository class for database operations."""
    
    def __init__(self, db_manager: DatabaseManager, table_name: str):
        self.db_manager = db_manager
        self.table_name = table_name
    
    def query_builder(self) -> QueryBuilder:
        """Create query builder for this table."""
        return QueryBuilder(self.table_name)
    
    async def create(self, data: Dict[str, Any]) -> QueryResult:
        """Create a new record."""
        builder = self.query_builder()
        query, params = builder.build_insert(data)
        return await self.db_manager.execute_query(query, params)
    
    async def find_by_id(self, record_id: Any) -> Optional[Dict[str, Any]]:
        """Find record by ID."""
        builder = self.query_builder()
        query, params = builder.where("id = :id", id=record_id).build_select()
        result = await self.db_manager.execute_query(query, params)
        return result.first
    
    async def find_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find all records."""
        builder = self.query_builder()
        
        if limit:
            builder.limit(limit)
        if offset:
            builder.offset(offset)
        
        query, params = builder.build_select()
        result = await self.db_manager.execute_query(query, params)
        return result.rows
    
    async def update(self, record_id: Any, data: Dict[str, Any]) -> QueryResult:
        """Update a record."""
        builder = self.query_builder()
        query, params = builder.where("id = :record_id", record_id=record_id).build_update(data)
        return await self.db_manager.execute_query(query, params)
    
    async def delete(self, record_id: Any) -> QueryResult:
        """Delete a record."""
        builder = self.query_builder()
        query, params = builder.where("id = :record_id", record_id=record_id).build_delete()
        return await self.db_manager.execute_query(query, params)
    
    async def count(self, where_conditions: Optional[Dict[str, Any]] = None) -> int:
        """Count records."""
        builder = self.query_builder().select("COUNT(*) as count")
        
        if where_conditions:
            for condition, value in where_conditions.items():
                builder.where(f"{condition} = :{condition}", **{condition: value})
        
        query, params = builder.build_select()
        result = await self.db_manager.execute_query(query, params)
        return result.first['count'] if result.first else 0
    
    async def exists(self, record_id: Any) -> bool:
        """Check if record exists."""
        count = await self.count({'id': record_id})
        return count > 0


# Global database manager instance
_global_db_manager: Optional[DatabaseManager] = None


def initialize_database(config: DatabaseConfig) -> DatabaseManager:
    """Initialize global database manager."""
    global _global_db_manager
    _global_db_manager = DatabaseManager(config)
    return _global_db_manager


def get_database_manager() -> Optional[DatabaseManager]:
    """Get global database manager."""
    return _global_db_manager


async def execute_query(query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
    """Execute query using global database manager."""
    if not _global_db_manager:
        raise DatabaseError("Database not initialized. Call initialize_database() first.")
    
    return await _global_db_manager.execute_query(query, parameters)