"""
Tests for security utilities.
"""
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from agent_orchestra.security_utils import (
    PasswordPolicy, TokenManager, HashManager, RoleManager,
    SecurityAuditor, InputSanitizer, SecurityContextBuilder,
    Permission, Role, SecurityContext, generate_secure_key,
    generate_api_key, validate_api_key_format, secure_compare
)
from agent_orchestra.exceptions import SecurityError, AuthenticationError, AuthorizationError


class TestPasswordPolicy:
    """Test password policy enforcement."""
    
    def test_default_policy(self):
        """Test default password policy."""
        policy = PasswordPolicy()
        
        # Valid password
        is_valid, violations = policy.validate_password("SecureP@ss123")
        assert is_valid
        assert len(violations) == 0
        
        # Too short
        is_valid, violations = policy.validate_password("Short1!")
        assert not is_valid
        assert any("at least 8" in v for v in violations)
        
        # No uppercase
        is_valid, violations = policy.validate_password("lowercase123!")
        assert not is_valid
        assert any("uppercase" in v for v in violations)
        
        # No special character
        is_valid, violations = policy.validate_password("NoSpecial123")
        assert not is_valid
        assert any("special character" in v for v in violations)
        
        # Contains forbidden pattern
        is_valid, violations = policy.validate_password("MyPassword123!")
        assert not is_valid
        assert any("password" in v for v in violations)
    
    def test_custom_policy(self):
        """Test custom password policy settings."""
        policy = PasswordPolicy(
            min_length=12,
            require_special=False,
            forbidden_patterns=["custom"]
        )
        
        # Should pass without special char but fail length
        is_valid, violations = policy.validate_password("ShortPass1")
        assert not is_valid
        assert any("at least 12" in v for v in violations)
        
        # Should pass with custom settings
        is_valid, violations = policy.validate_password("LongPassword123")
        assert is_valid
        
        # Should fail with custom forbidden pattern
        is_valid, violations = policy.validate_password("CustomPassword123")
        assert not is_valid
        assert any("custom" in v for v in violations)


class TestTokenManager:
    """Test JWT token management."""
    
    def setUp(self):
        self.token_manager = TokenManager("test_secret_key", default_expiry=3600)
    
    def test_generate_token(self):
        """Test token generation."""
        self.setUp()
        token = self.token_manager.generate_token(
            user_id="test_user",
            roles=["operator"],
            permissions=["read_tasks"]
        )
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_token(self):
        """Test token verification."""
        self.setUp()
        token = self.token_manager.generate_token(
            user_id="test_user",
            roles=["operator"],
            permissions=["read_tasks"]
        )
        
        payload = self.token_manager.verify_token(token)
        
        assert payload["user_id"] == "test_user"
        assert payload["roles"] == ["operator"]
        assert payload["permissions"] == ["read_tasks"]
        assert "iat" in payload
        assert "exp" in payload
        assert "jti" in payload
    
    def test_expired_token(self):
        """Test expired token verification."""
        self.setUp()
        # Create token with very short expiry
        token = self.token_manager.generate_token(
            user_id="test_user",
            roles=["operator"],
            permissions=["read_tasks"],
            expiry_seconds=1
        )
        
        # Wait for expiration
        time.sleep(2)
        
        with pytest.raises(AuthenticationError, match="expired"):
            self.token_manager.verify_token(token)
    
    def test_invalid_token(self):
        """Test invalid token verification."""
        self.setUp()
        
        with pytest.raises(AuthenticationError, match="Invalid token"):
            self.token_manager.verify_token("invalid_token")
    
    def test_refresh_token(self):
        """Test token refresh."""
        self.setUp()
        original_token = self.token_manager.generate_token(
            user_id="test_user",
            roles=["operator"],
            permissions=["read_tasks"]
        )
        
        refreshed_token = self.token_manager.refresh_token(original_token)
        
        # Should be different tokens
        assert refreshed_token != original_token
        
        # Should have same payload data
        original_payload = self.token_manager.verify_token(original_token)
        refreshed_payload = self.token_manager.verify_token(refreshed_token)
        
        assert original_payload["user_id"] == refreshed_payload["user_id"]
        assert original_payload["roles"] == refreshed_payload["roles"]
        assert original_payload["permissions"] == refreshed_payload["permissions"]


class TestHashManager:
    """Test password hashing utilities."""
    
    def test_hash_password(self):
        """Test password hashing."""
        password = "test_password"
        hashed, salt = HashManager.hash_password(password)
        
        assert isinstance(hashed, str)
        assert isinstance(salt, str)
        assert len(hashed) > 0
        assert len(salt) > 0
        assert hashed != password  # Should be different from original
    
    def test_verify_password(self):
        """Test password verification."""
        password = "test_password"
        hashed, salt = HashManager.hash_password(password)
        
        # Correct password should verify
        assert HashManager.verify_password(password, hashed, salt)
        
        # Wrong password should not verify
        assert not HashManager.verify_password("wrong_password", hashed, salt)
    
    def test_consistent_hashing(self):
        """Test that same password with same salt produces same hash."""
        password = "test_password"
        salt_bytes = b"test_salt" * 4  # 32 bytes
        
        # Convert to base64 as the function expects
        import base64
        salt = base64.b64encode(salt_bytes).decode()
        
        hashed1, _ = HashManager.hash_password(password, salt_bytes)
        hashed2, _ = HashManager.hash_password(password, salt_bytes)
        
        assert hashed1 == hashed2


class TestRoleManager:
    """Test role-based access control."""
    
    def test_default_permissions(self):
        """Test default role permissions."""
        role_manager = RoleManager()
        
        # Viewer should have read permissions only
        viewer_perms = role_manager.get_permissions_for_roles([Role.VIEWER])
        assert Permission.READ_TASKS in viewer_perms
        assert Permission.CREATE_TASKS not in viewer_perms
        
        # Admin should have more permissions
        admin_perms = role_manager.get_permissions_for_roles([Role.ADMIN])
        assert Permission.READ_TASKS in admin_perms
        assert Permission.CREATE_TASKS in admin_perms
        assert Permission.DELETE_TASKS in admin_perms
        
        # Super admin should have all permissions
        super_admin_perms = role_manager.get_permissions_for_roles([Role.SUPER_ADMIN])
        assert len(super_admin_perms) == len(Permission)
    
    def test_multiple_roles(self):
        """Test permissions from multiple roles."""
        role_manager = RoleManager()
        
        # Should get combined permissions
        perms = role_manager.get_permissions_for_roles([Role.VIEWER, Role.OPERATOR])
        assert Permission.READ_TASKS in perms
        assert Permission.CREATE_TASKS in perms
        assert Permission.DELETE_TASKS not in perms  # Only admin has this
    
    def test_has_permission(self):
        """Test permission checking."""
        role_manager = RoleManager()
        
        assert role_manager.has_permission([Role.ADMIN], Permission.CREATE_TASKS)
        assert not role_manager.has_permission([Role.VIEWER], Permission.CREATE_TASKS)
    
    def test_add_remove_permissions(self):
        """Test adding and removing role permissions."""
        role_manager = RoleManager()
        
        # Add permission
        role_manager.add_role_permission(Role.VIEWER, Permission.CREATE_TASKS)
        assert role_manager.has_permission([Role.VIEWER], Permission.CREATE_TASKS)
        
        # Remove permission
        role_manager.remove_role_permission(Role.VIEWER, Permission.CREATE_TASKS)
        assert not role_manager.has_permission([Role.VIEWER], Permission.CREATE_TASKS)


class TestSecurityAuditor:
    """Test security auditing."""
    
    def test_successful_authentication_logging(self):
        """Test logging successful authentication."""
        logger_mock = Mock()
        auditor = SecurityAuditor(logger_mock)
        
        auditor.log_authentication_attempt("user1", True, "192.168.1.1")
        
        logger_mock.info.assert_called_once()
        call_args = logger_mock.info.call_args
        assert "successful" in call_args[0][0]
        assert call_args[1]["extra"]["user_id"] == "user1"
        assert call_args[1]["extra"]["success"] is True
    
    def test_failed_authentication_tracking(self):
        """Test tracking failed authentication attempts."""
        logger_mock = Mock()
        auditor = SecurityAuditor(logger_mock)
        
        # Multiple failed attempts
        for _ in range(3):
            auditor.log_authentication_attempt("user1", False, "192.168.1.1")
        
        # Should not be locked yet (default is 5 attempts)
        assert not auditor.is_account_locked("user1")
        
        # Two more failed attempts should trigger lockout
        for _ in range(2):
            auditor.log_authentication_attempt("user1", False, "192.168.1.1")
        
        assert auditor.is_account_locked("user1")
    
    def test_account_lockout_reset(self):
        """Test account lockout reset on successful login."""
        logger_mock = Mock()
        auditor = SecurityAuditor(logger_mock)
        
        # Failed attempts
        for _ in range(4):
            auditor.log_authentication_attempt("user1", False, "192.168.1.1")
        
        # Successful login should reset
        auditor.log_authentication_attempt("user1", True, "192.168.1.1")
        
        assert not auditor.is_account_locked("user1")
    
    def test_authorization_logging(self):
        """Test authorization decision logging."""
        logger_mock = Mock()
        auditor = SecurityAuditor(logger_mock)
        
        auditor.log_authorization_decision(
            "user1", "tasks", "create", True, "Has permission"
        )
        
        logger_mock.log.assert_called_once()
        call_args = logger_mock.log.call_args
        assert call_args[1]["extra"]["granted"] is True


class TestInputSanitizer:
    """Test input sanitization."""
    
    def test_sanitize_string(self):
        """Test string sanitization."""
        # Basic sanitization
        result = InputSanitizer.sanitize_string("  test string  ")
        assert result == "test string"
        
        # HTML removal
        result = InputSanitizer.sanitize_string("<script>alert('xss')</script>hello")
        assert "script" not in result.lower()
        assert "hello" in result
        
        # JavaScript patterns
        result = InputSanitizer.sanitize_string("javascript:alert('xss')")
        assert "javascript:" not in result
    
    def test_sanitize_string_max_length(self):
        """Test string length validation."""
        long_string = "a" * 1001
        
        with pytest.raises(SecurityError, match="too long"):
            InputSanitizer.sanitize_string(long_string, max_length=1000)
    
    def test_validate_user_input(self):
        """Test user input validation."""
        input_data = {
            "name": "John Doe",
            "age": 30,
            "active": True,
            "tags": ["user", "admin"],
            "metadata": {"role": "operator"}
        }
        
        result = InputSanitizer.validate_user_input(input_data)
        
        assert result["name"] == "John Doe"
        assert result["age"] == 30
        assert result["active"] is True
        assert "user" in result["tags"]
        assert result["metadata"]["role"] == "operator"


class TestSecurityContext:
    """Test security context."""
    
    def test_security_context_creation(self):
        """Test security context creation."""
        permissions = {Permission.READ_TASKS, Permission.CREATE_TASKS}
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        context = SecurityContext(
            user_id="test_user",
            roles=[Role.OPERATOR],
            permissions=permissions,
            expires_at=expires_at
        )
        
        assert context.user_id == "test_user"
        assert Role.OPERATOR in context.roles
        assert context.has_permission(Permission.READ_TASKS)
        assert not context.has_role(Role.ADMIN)
        assert not context.is_expired()
    
    def test_security_context_expiration(self):
        """Test security context expiration."""
        expires_at = datetime.utcnow() - timedelta(minutes=1)
        
        context = SecurityContext(
            user_id="test_user",
            roles=[Role.OPERATOR],
            permissions=set(),
            expires_at=expires_at
        )
        
        assert context.is_expired()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_generate_secure_key(self):
        """Test secure key generation."""
        key1 = generate_secure_key()
        key2 = generate_secure_key()
        
        assert isinstance(key1, str)
        assert isinstance(key2, str)
        assert len(key1) == 64  # 32 bytes * 2 (hex)
        assert key1 != key2  # Should be different
    
    def test_generate_api_key(self):
        """Test API key generation."""
        key1 = generate_api_key()
        key2 = generate_api_key()
        
        assert key1.startswith("ao-")
        assert key2.startswith("ao-")
        assert key1 != key2
        assert validate_api_key_format(key1)
        assert validate_api_key_format(key2)
    
    def test_validate_api_key_format(self):
        """Test API key format validation."""
        # Valid format
        valid_key = "ao-" + "a" * 43
        assert validate_api_key_format(valid_key)
        
        # Invalid formats
        invalid_keys = [
            "invalid-key",  # Wrong prefix
            "ao-short",  # Too short
            "ao-" + "a" * 44,  # Too long
            "ao-invalid@chars",  # Invalid characters
        ]
        
        for key in invalid_keys:
            assert not validate_api_key_format(key)
    
    def test_secure_compare(self):
        """Test timing-safe string comparison."""
        assert secure_compare("same", "same")
        assert not secure_compare("different", "strings")
        assert not secure_compare("", "non-empty")


class TestSecurityContextBuilder:
    """Test security context builder."""
    
    def setUp(self):
        self.token_manager = TokenManager("test_secret", 3600)
        self.role_manager = RoleManager()
        self.builder = SecurityContextBuilder(self.role_manager, self.token_manager)
    
    def test_from_token(self):
        """Test creating context from token."""
        self.setUp()
        
        token = self.token_manager.generate_token(
            user_id="test_user",
            roles=["operator"],
            permissions=["read_tasks"]
        )
        
        context = self.builder.from_token(token, "192.168.1.1")
        
        assert context.user_id == "test_user"
        assert Role.OPERATOR in context.roles
        assert context.source_ip == "192.168.1.1"
        assert context.has_permission(Permission.READ_TASKS)
    
    def test_from_user_data(self):
        """Test creating context from user data."""
        self.setUp()
        
        context = self.builder.from_user_data(
            user_id="test_user",
            roles=[Role.ADMIN],
            source_ip="127.0.0.1"
        )
        
        assert context.user_id == "test_user"
        assert Role.ADMIN in context.roles
        assert context.source_ip == "127.0.0.1"
        assert context.has_permission(Permission.DELETE_TASKS)  # Admin permission