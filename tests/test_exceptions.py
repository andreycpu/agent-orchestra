"""
Tests for exception handling
"""
import pytest

from agent_orchestra.exceptions import (
    AgentOrchestraException, AgentRegistrationError, AgentNotFoundError,
    AgentUnavailableError, TaskRoutingError, TaskExecutionError,
    TaskTimeoutError, StateManagementError, ConcurrencyError,
    CircularDependencyError, ResourceExhaustionError, ValidationError,
    ConfigurationError, SecurityError, NetworkError
)


class TestAgentOrchestraException:
    """Test cases for base exception"""
    
    def test_exception_with_message_only(self):
        """Test exception with just a message"""
        exc = AgentOrchestraException("Test error")
        
        assert str(exc) == "Test error"
        assert exc.message == "Test error"
        assert exc.details == {}
    
    def test_exception_with_details(self):
        """Test exception with additional details"""
        details = {"agent_id": "agent_1", "task_id": "task_1"}
        exc = AgentOrchestraException("Test error", details=details)
        
        assert exc.message == "Test error"
        assert exc.details == details
        assert "Details: {'agent_id': 'agent_1', 'task_id': 'task_1'}" in str(exc)
    
    def test_exception_inheritance(self):
        """Test that custom exceptions inherit from base"""
        exceptions_to_test = [
            AgentRegistrationError,
            AgentNotFoundError,
            AgentUnavailableError,
            TaskRoutingError,
            TaskExecutionError,
            TaskTimeoutError,
            StateManagementError,
            ConcurrencyError,
            CircularDependencyError,
            ResourceExhaustionError,
            ValidationError,
            ConfigurationError,
            SecurityError,
            NetworkError
        ]
        
        for exc_class in exceptions_to_test:
            exc = exc_class("Test message")
            assert isinstance(exc, AgentOrchestraException)
            assert hasattr(exc, 'message')
            assert hasattr(exc, 'details')


class TestSpecificExceptions:
    """Test specific exception types"""
    
    def test_task_execution_error(self):
        """Test TaskExecutionError with details"""
        details = {
            "task_id": "task_123",
            "agent_id": "agent_456",
            "error_code": "TIMEOUT"
        }
        exc = TaskExecutionError("Task failed to execute", details=details)
        
        assert exc.details["task_id"] == "task_123"
        assert exc.details["agent_id"] == "agent_456"
        assert exc.details["error_code"] == "TIMEOUT"
    
    def test_security_error(self):
        """Test SecurityError with security context"""
        details = {
            "user_id": "user_123",
            "action": "unauthorized_access",
            "resource": "/admin/panel"
        }
        exc = SecurityError("Unauthorized access attempt", details=details)
        
        assert "unauthorized_access" in str(exc)
        assert exc.details["user_id"] == "user_123"
    
    def test_configuration_error(self):
        """Test ConfigurationError with config details"""
        details = {
            "config_file": "config.yaml",
            "missing_key": "database.host",
            "section": "database"
        }
        exc = ConfigurationError("Missing configuration", details=details)
        
        assert exc.details["missing_key"] == "database.host"
        assert "config.yaml" in str(exc)