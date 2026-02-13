"""
Tests for logging utilities.
"""
import pytest
import json
import logging
import time
from datetime import datetime
from unittest.mock import Mock, patch

from agent_orchestra.logging_utils import (
    StructuredFormatter, PerformanceLogger, AuditLogger,
    TaskLogger, AgentLogger, performance_logging, setup_logging,
    create_specialized_loggers
)
from agent_orchestra.types import TaskStatus, AgentStatus


class TestStructuredFormatter:
    """Test structured JSON formatter."""
    
    def test_basic_formatting(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data['level'] == 'INFO'
        assert log_data['logger'] == 'test_logger'
        assert log_data['message'] == 'Test message'
        assert log_data['line'] == 10
        assert 'timestamp' in log_data
    
    def test_exception_formatting(self):
        """Test exception information formatting."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert 'exception' in log_data
        assert log_data['exception']['type'] == 'ValueError'
        assert log_data['exception']['message'] == 'Test exception'
        assert 'traceback' in log_data['exception']
    
    def test_extra_fields(self):
        """Test extra fields in log record."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.user_id = "test_user"
        record.request_id = "req_123"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert 'extra' in log_data
        assert log_data['extra']['user_id'] == 'test_user'
        assert log_data['extra']['request_id'] == 'req_123'


class TestPerformanceLogger:
    """Test performance logging."""
    
    def test_timer_operations(self):
        """Test timer start/stop operations."""
        mock_logger = Mock()
        perf_logger = PerformanceLogger(mock_logger)
        
        timer_id = perf_logger.start_timer("test_operation")
        time.sleep(0.1)
        perf_logger.stop_timer(timer_id)
        
        # Should have logged the operation
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "test_operation" in call_args[0][0]
        assert call_args[1]['extra']['operation'] == 'test_operation'
        assert call_args[1]['extra']['duration_ms'] >= 100
    
    def test_context_manager_timing(self):
        """Test timing with context manager."""
        mock_logger = Mock()
        perf_logger = PerformanceLogger(mock_logger)
        
        with perf_logger.time_operation("context_test") as timer:
            time.sleep(0.05)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "context_test" in call_args[0][0]
    
    @patch('agent_orchestra.logging_utils.psutil')
    def test_memory_logging(self, mock_psutil):
        """Test memory usage logging."""
        # Setup mock
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 10  # 10MB
        mock_memory_info.vms = 1024 * 1024 * 20  # 20MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 5.0
        mock_psutil.Process.return_value = mock_process
        
        mock_logger = Mock()
        perf_logger = PerformanceLogger(mock_logger)
        
        perf_logger.log_memory_usage("test_operation")
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[1]['extra']['operation'] == 'test_operation'
        assert call_args[1]['extra']['memory_rss_mb'] == 10.0
        assert call_args[1]['extra']['memory_percent'] == 5.0


class TestAuditLogger:
    """Test audit logging."""
    
    def test_authentication_logging(self):
        """Test authentication event logging."""
        mock_logger = Mock()
        audit_logger = AuditLogger(mock_logger)
        
        # Successful authentication
        audit_logger.log_authentication("user123", True, "192.168.1.1")
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "successful" in call_args[0][0]
        assert call_args[1]['extra']['user_id'] == 'user123'
        assert call_args[1]['extra']['success'] is True
        assert call_args[1]['extra']['source_ip'] == '192.168.1.1'
        
        # Failed authentication
        mock_logger.reset_mock()
        audit_logger.log_authentication("user456", False, "10.0.0.1")
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "failed" in call_args[0][0]
        assert call_args[1]['extra']['success'] is False
    
    def test_authorization_logging(self):
        """Test authorization event logging."""
        mock_logger = Mock()
        audit_logger = AuditLogger(mock_logger)
        
        audit_logger.log_authorization("user123", "read", "tasks", True)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "granted" in call_args[0][0]
        assert call_args[1]['extra']['user_id'] == 'user123'
        assert call_args[1]['extra']['action'] == 'read'
        assert call_args[1]['extra']['resource'] == 'tasks'
        assert call_args[1]['extra']['granted'] is True
    
    def test_data_access_logging(self):
        """Test data access event logging."""
        mock_logger = Mock()
        audit_logger = AuditLogger(mock_logger)
        
        audit_logger.log_data_access("user123", "tasks", "SELECT", 10)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "SELECT" in call_args[0][0]
        assert call_args[1]['extra']['record_count'] == 10


class TestTaskLogger:
    """Test task lifecycle logging."""
    
    def test_task_lifecycle_logging(self):
        """Test complete task lifecycle logging."""
        mock_logger = Mock()
        task_logger = TaskLogger(mock_logger)
        
        # Task created
        task_logger.log_task_created("task_123", "test_task", "normal")
        mock_logger.info.assert_called()
        
        # Task assigned
        mock_logger.reset_mock()
        task_logger.log_task_assigned("task_123", "agent_456")
        mock_logger.info.assert_called()
        
        # Task started
        mock_logger.reset_mock()
        task_logger.log_task_started("task_123", "agent_456")
        mock_logger.info.assert_called()
        
        # Task completed successfully
        mock_logger.reset_mock()
        task_logger.log_task_completed("task_123", "agent_456", TaskStatus.COMPLETED, 5.5)
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args
        assert call_args[1]['extra']['status'] == TaskStatus.COMPLETED.value
        assert call_args[1]['extra']['duration_seconds'] == 5.5
        
        # Task failed
        mock_logger.reset_mock()
        task_logger.log_task_completed("task_123", "agent_456", TaskStatus.FAILED)
        mock_logger.warning.assert_called()
    
    def test_task_retry_logging(self):
        """Test task retry logging."""
        mock_logger = Mock()
        task_logger = TaskLogger(mock_logger)
        
        task_logger.log_task_retry("task_123", 2, 3, "Connection timeout")
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "retry" in call_args[0][0]
        assert call_args[1]['extra']['attempt'] == 2
        assert call_args[1]['extra']['max_retries'] == 3
        assert call_args[1]['extra']['error'] == "Connection timeout"


class TestAgentLogger:
    """Test agent lifecycle logging."""
    
    def test_agent_registration_logging(self):
        """Test agent registration logging."""
        mock_logger = Mock()
        agent_logger = AgentLogger(mock_logger)
        
        capabilities = ["task_1", "task_2"]
        agent_logger.log_agent_registered("agent_123", capabilities)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "registered" in call_args[0][0]
        assert call_args[1]['extra']['agent_id'] == 'agent_123'
        assert call_args[1]['extra']['capabilities'] == capabilities
    
    def test_agent_status_change_logging(self):
        """Test agent status change logging."""
        mock_logger = Mock()
        agent_logger = AgentLogger(mock_logger)
        
        # Normal status change
        agent_logger.log_agent_status_change(
            "agent_123", 
            AgentStatus.IDLE, 
            AgentStatus.BUSY
        )
        
        mock_logger.info.assert_called_once()
        
        # Error status change
        mock_logger.reset_mock()
        agent_logger.log_agent_status_change(
            "agent_123",
            AgentStatus.BUSY,
            AgentStatus.ERROR
        )
        
        mock_logger.warning.assert_called_once()
    
    def test_agent_heartbeat_logging(self):
        """Test agent heartbeat logging."""
        mock_logger = Mock()
        agent_logger = AgentLogger(mock_logger)
        
        # Healthy heartbeat
        metrics = {"cpu_usage": 25.5, "memory_usage": 60.0}
        agent_logger.log_agent_heartbeat("agent_123", True, metrics)
        
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args
        assert call_args[1]['extra']['healthy'] is True
        assert call_args[1]['extra']['metrics'] == metrics
        
        # Unhealthy heartbeat
        mock_logger.reset_mock()
        agent_logger.log_agent_heartbeat("agent_123", False)
        
        mock_logger.warning.assert_called_once()


class TestPerformanceLoggingDecorator:
    """Test performance logging decorator."""
    
    def test_function_timing(self):
        """Test function execution timing."""
        mock_logger = Mock()
        
        @performance_logging(logger=mock_logger, threshold_seconds=0.0)
        def test_function():
            time.sleep(0.05)
            return "result"
        
        result = test_function()
        
        assert result == "result"
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "test_function" in call_args[0][0]
        assert call_args[1]['extra']['success'] is True
        assert call_args[1]['extra']['duration_seconds'] >= 0.05
    
    def test_function_exception_logging(self):
        """Test function exception logging."""
        mock_logger = Mock()
        
        @performance_logging(logger=mock_logger)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "failed" in call_args[0][0]
        assert call_args[1]['extra']['success'] is False
        assert call_args[1]['extra']['error'] == "Test error"
    
    def test_threshold_filtering(self):
        """Test that fast operations below threshold aren't logged."""
        mock_logger = Mock()
        
        @performance_logging(logger=mock_logger, threshold_seconds=1.0)
        def fast_function():
            return "result"
        
        result = fast_function()
        
        assert result == "result"
        mock_logger.info.assert_not_called()


class TestLoggingSetup:
    """Test logging setup utilities."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        logger = setup_logging(level='DEBUG', enable_console=True)
        
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0
    
    def test_create_specialized_loggers(self):
        """Test specialized logger creation."""
        base_logger = logging.getLogger("test_base")
        loggers = create_specialized_loggers(base_logger)
        
        assert 'performance' in loggers
        assert 'audit' in loggers
        assert 'tasks' in loggers
        assert 'agents' in loggers
        
        assert isinstance(loggers['performance'], PerformanceLogger)
        assert isinstance(loggers['audit'], AuditLogger)
        assert isinstance(loggers['tasks'], TaskLogger)
        assert isinstance(loggers['agents'], AgentLogger)