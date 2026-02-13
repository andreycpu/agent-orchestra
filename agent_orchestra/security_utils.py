"""
Security utilities for Agent Orchestra.

This module provides authentication, authorization, encryption,
and security monitoring utilities for the orchestration system.
"""
import hashlib
import hmac
import secrets
import base64
import jwt
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re
import logging
from pathlib import Path

from .exceptions import SecurityError, AuthenticationError, AuthorizationError
from .validation import validate_email


logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """System permissions."""
    READ_TASKS = "read_tasks"
    CREATE_TASKS = "create_tasks"
    UPDATE_TASKS = "update_tasks"
    DELETE_TASKS = "delete_tasks"
    READ_AGENTS = "read_agents"
    MANAGE_AGENTS = "manage_agents"
    READ_RESULTS = "read_results"
    ADMIN_SYSTEM = "admin_system"
    MODIFY_CONFIG = "modify_config"
    VIEW_METRICS = "view_metrics"


class Role(str, Enum):
    """User roles with predefined permission sets."""
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass
class SecurityContext:
    """Security context for request validation."""
    user_id: str
    roles: List[Role]
    permissions: Set[Permission]
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if context has a specific permission."""
        return permission in self.permissions
    
    def has_role(self, role: Role) -> bool:
        """Check if context has a specific role."""
        return role in self.roles
    
    def is_expired(self) -> bool:
        """Check if the security context has expired."""
        return self.expires_at and datetime.utcnow() > self.expires_at


class PasswordPolicy:
    """Password policy enforcement."""
    
    def __init__(
        self,
        min_length: int = 8,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_numbers: bool = True,
        require_special: bool = True,
        forbidden_patterns: Optional[List[str]] = None
    ):
        self.min_length = min_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_numbers = require_numbers
        self.require_special = require_special
        self.forbidden_patterns = forbidden_patterns or [
            "password", "123456", "qwerty", "admin", "user"
        ]
    
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against policy.
        
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        if len(password) < self.min_length:
            violations.append(f"Password must be at least {self.min_length} characters long")
        
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            violations.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not re.search(r'[a-z]', password):
            violations.append("Password must contain at least one lowercase letter")
        
        if self.require_numbers and not re.search(r'\d', password):
            violations.append("Password must contain at least one number")
        
        if self.require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            violations.append("Password must contain at least one special character")
        
        # Check forbidden patterns
        password_lower = password.lower()
        for pattern in self.forbidden_patterns:
            if pattern.lower() in password_lower:
                violations.append(f"Password cannot contain '{pattern}'")
        
        return len(violations) == 0, violations


class TokenManager:
    """JWT token management for authentication."""
    
    def __init__(self, secret_key: str, default_expiry: int = 3600):
        self.secret_key = secret_key
        self.default_expiry = default_expiry  # seconds
        self.algorithm = "HS256"
    
    def generate_token(
        self,
        user_id: str,
        roles: List[str],
        permissions: List[str],
        expiry_seconds: Optional[int] = None
    ) -> str:
        """Generate a JWT token for a user."""
        expiry = expiry_seconds or self.default_expiry
        expires_at = datetime.utcnow() + timedelta(seconds=expiry)
        
        payload = {
            "user_id": user_id,
            "roles": roles,
            "permissions": permissions,
            "iat": datetime.utcnow(),
            "exp": expires_at,
            "jti": secrets.token_hex(16)  # JWT ID for token revocation
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    def refresh_token(self, token: str, extend_seconds: Optional[int] = None) -> str:
        """Refresh a token with new expiry."""
        payload = self.verify_token(token)
        
        # Remove old timing claims
        payload.pop('iat', None)
        payload.pop('exp', None)
        payload.pop('jti', None)
        
        return self.generate_token(
            user_id=payload['user_id'],
            roles=payload['roles'],
            permissions=payload['permissions'],
            expiry_seconds=extend_seconds
        )


class HashManager:
    """Password hashing and verification utilities."""
    
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """
        Hash a password with salt.
        
        Returns:
            Tuple of (hashed_password, salt_base64)
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 with SHA256
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        
        return base64.b64encode(hash_obj).decode(), base64.b64encode(salt).decode()
    
    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt_bytes = base64.b64decode(salt)
            expected_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt_bytes, 100000)
            expected_hash_b64 = base64.b64encode(expected_hash).decode()
            
            return hmac.compare_digest(expected_hash_b64, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False


class RoleManager:
    """Role-based access control management."""
    
    def __init__(self):
        self.role_permissions = {
            Role.VIEWER: {
                Permission.READ_TASKS,
                Permission.READ_AGENTS,
                Permission.READ_RESULTS,
                Permission.VIEW_METRICS
            },
            Role.OPERATOR: {
                Permission.READ_TASKS,
                Permission.CREATE_TASKS,
                Permission.UPDATE_TASKS,
                Permission.READ_AGENTS,
                Permission.READ_RESULTS,
                Permission.VIEW_METRICS
            },
            Role.ADMIN: {
                Permission.READ_TASKS,
                Permission.CREATE_TASKS,
                Permission.UPDATE_TASKS,
                Permission.DELETE_TASKS,
                Permission.READ_AGENTS,
                Permission.MANAGE_AGENTS,
                Permission.READ_RESULTS,
                Permission.VIEW_METRICS,
                Permission.MODIFY_CONFIG
            },
            Role.SUPER_ADMIN: set(Permission)  # All permissions
        }
    
    def get_permissions_for_roles(self, roles: List[Role]) -> Set[Permission]:
        """Get all permissions for a list of roles."""
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))
        return permissions
    
    def has_permission(self, roles: List[Role], permission: Permission) -> bool:
        """Check if roles have a specific permission."""
        user_permissions = self.get_permissions_for_roles(roles)
        return permission in user_permissions
    
    def add_role_permission(self, role: Role, permission: Permission):
        """Add a permission to a role."""
        if role not in self.role_permissions:
            self.role_permissions[role] = set()
        self.role_permissions[role].add(permission)
    
    def remove_role_permission(self, role: Role, permission: Permission):
        """Remove a permission from a role."""
        if role in self.role_permissions:
            self.role_permissions[role].discard(permission)


class SecurityAuditor:
    """Security event auditing and monitoring."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.suspicious_ips: Set[str] = set()
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
    
    def log_authentication_attempt(
        self,
        user_id: str,
        success: bool,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log authentication attempts."""
        event_data = {
            "event": "authentication_attempt",
            "user_id": user_id,
            "success": success,
            "source_ip": source_ip,
            "user_agent": user_agent,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if success:
            self.logger.info(f"Authentication successful for {user_id}", extra=event_data)
            # Clear failed attempts on successful login
            self.failed_attempts.pop(user_id, None)
        else:
            self.logger.warning(f"Authentication failed for {user_id}", extra=event_data)
            self._track_failed_attempt(user_id, source_ip)
    
    def _track_failed_attempt(self, user_id: str, source_ip: Optional[str]):
        """Track failed authentication attempts."""
        now = datetime.utcnow()
        
        # Clean old attempts
        if user_id in self.failed_attempts:
            self.failed_attempts[user_id] = [
                attempt for attempt in self.failed_attempts[user_id]
                if now - attempt < self.lockout_duration
            ]
        else:
            self.failed_attempts[user_id] = []
        
        # Add current attempt
        self.failed_attempts[user_id].append(now)
        
        # Check for suspicious activity
        if len(self.failed_attempts[user_id]) >= self.max_failed_attempts:
            self.logger.error(
                f"Account lockout triggered for {user_id}",
                extra={"event": "account_lockout", "user_id": user_id, "source_ip": source_ip}
            )
            
            if source_ip:
                self.suspicious_ips.add(source_ip)
    
    def is_account_locked(self, user_id: str) -> bool:
        """Check if an account is locked due to failed attempts."""
        if user_id not in self.failed_attempts:
            return False
        
        now = datetime.utcnow()
        recent_failures = [
            attempt for attempt in self.failed_attempts[user_id]
            if now - attempt < self.lockout_duration
        ]
        
        return len(recent_failures) >= self.max_failed_attempts
    
    def log_authorization_decision(
        self,
        user_id: str,
        resource: str,
        action: str,
        granted: bool,
        reason: Optional[str] = None
    ):
        """Log authorization decisions."""
        event_data = {
            "event": "authorization_decision",
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "granted": granted,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        level = logging.INFO if granted else logging.WARNING
        message = f"Authorization {'granted' if granted else 'denied'} for {user_id} on {resource}:{action}"
        self.logger.log(level, message, extra=event_data)
    
    def log_security_violation(
        self,
        violation_type: str,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security violations."""
        event_data = {
            "event": "security_violation",
            "violation_type": violation_type,
            "user_id": user_id,
            "source_ip": source_ip,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.error(f"Security violation: {violation_type}", extra=event_data)


class InputSanitizer:
    """Input sanitization and validation for security."""
    
    @staticmethod
    def sanitize_string(
        value: str,
        max_length: int = 1000,
        allow_html: bool = False,
        strip_whitespace: bool = True
    ) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            raise SecurityError("Input must be a string")
        
        if strip_whitespace:
            value = value.strip()
        
        if len(value) > max_length:
            raise SecurityError(f"Input too long (max {max_length} characters)")
        
        if not allow_html:
            # Remove potentially dangerous HTML/script tags
            value = re.sub(r'<[^>]*>', '', value)
            
            # Remove common script injection patterns
            dangerous_patterns = [
                r'javascript:', r'vbscript:', r'onload=', r'onerror=',
                r'<script', r'</script>', r'eval\(', r'document\.cookie'
            ]
            
            for pattern in dangerous_patterns:
                value = re.sub(pattern, '', value, flags=re.IGNORECASE)
        
        return value
    
    @staticmethod
    def validate_user_input(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize user input dictionary."""
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            clean_key = InputSanitizer.sanitize_string(key, max_length=100)
            
            if isinstance(value, str):
                sanitized[clean_key] = InputSanitizer.sanitize_string(value)
            elif isinstance(value, (int, float, bool)):
                sanitized[clean_key] = value
            elif isinstance(value, dict):
                sanitized[clean_key] = InputSanitizer.validate_user_input(value)
            elif isinstance(value, list):
                sanitized[clean_key] = [
                    InputSanitizer.sanitize_string(item) if isinstance(item, str) else item
                    for item in value[:100]  # Limit list size
                ]
            else:
                # Skip unknown types
                logger.warning(f"Skipping unknown input type for key {clean_key}: {type(value)}")
        
        return sanitized


class SecurityContextBuilder:
    """Builder for creating security contexts."""
    
    def __init__(self, role_manager: RoleManager, token_manager: TokenManager):
        self.role_manager = role_manager
        self.token_manager = token_manager
    
    def from_token(self, token: str, source_ip: Optional[str] = None) -> SecurityContext:
        """Create security context from JWT token."""
        try:
            payload = self.token_manager.verify_token(token)
            
            roles = [Role(role) for role in payload.get('roles', [])]
            permissions = self.role_manager.get_permissions_for_roles(roles)
            
            # Add explicit permissions from token
            token_permissions = payload.get('permissions', [])
            for perm_name in token_permissions:
                try:
                    permissions.add(Permission(perm_name))
                except ValueError:
                    logger.warning(f"Unknown permission in token: {perm_name}")
            
            expires_at = datetime.fromtimestamp(payload['exp']) if 'exp' in payload else None
            
            return SecurityContext(
                user_id=payload['user_id'],
                roles=roles,
                permissions=permissions,
                source_ip=source_ip,
                session_id=payload.get('jti'),
                expires_at=expires_at
            )
            
        except Exception as e:
            raise AuthenticationError(f"Failed to create security context: {e}")
    
    def from_user_data(
        self,
        user_id: str,
        roles: List[Role],
        additional_permissions: Optional[List[Permission]] = None,
        **kwargs
    ) -> SecurityContext:
        """Create security context from user data."""
        permissions = self.role_manager.get_permissions_for_roles(roles)
        
        if additional_permissions:
            permissions.update(additional_permissions)
        
        return SecurityContext(
            user_id=user_id,
            roles=roles,
            permissions=permissions,
            **kwargs
        )


def generate_secure_key(length: int = 32) -> str:
    """Generate a cryptographically secure key."""
    return secrets.token_hex(length)


def generate_api_key() -> str:
    """Generate a secure API key."""
    return f"ao-{secrets.token_urlsafe(32)}"


def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format."""
    return bool(re.match(r'^ao-[A-Za-z0-9_-]{43}$', api_key))


def secure_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison."""
    return hmac.compare_digest(a, b)


class SecurityMiddleware:
    """Security middleware for request processing."""
    
    def __init__(
        self,
        context_builder: SecurityContextBuilder,
        auditor: SecurityAuditor,
        require_auth: bool = True
    ):
        self.context_builder = context_builder
        self.auditor = auditor
        self.require_auth = require_auth
    
    def process_request(
        self,
        headers: Dict[str, str],
        source_ip: Optional[str] = None
    ) -> Optional[SecurityContext]:
        """Process incoming request for security."""
        # Extract authorization header
        auth_header = headers.get('Authorization', '').strip()
        
        if not auth_header:
            if self.require_auth:
                raise AuthenticationError("Missing authorization header")
            return None
        
        # Parse Bearer token
        if not auth_header.startswith('Bearer '):
            raise AuthenticationError("Invalid authorization header format")
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        try:
            context = self.context_builder.from_token(token, source_ip)
            
            if context.is_expired():
                raise AuthenticationError("Token has expired")
            
            self.auditor.log_authentication_attempt(
                context.user_id, True, source_ip
            )
            
            return context
            
        except AuthenticationError as e:
            # Extract user_id from token for logging (if possible)
            user_id = "unknown"
            try:
                import jwt
                unverified = jwt.decode(token, options={"verify_signature": False})
                user_id = unverified.get('user_id', 'unknown')
            except:
                pass
            
            self.auditor.log_authentication_attempt(user_id, False, source_ip)
            raise
    
    def require_permission(
        self,
        context: Optional[SecurityContext],
        permission: Permission,
        resource: str = "unknown"
    ):
        """Require specific permission for operation."""
        if not context:
            raise AuthenticationError("Authentication required")
        
        if not context.has_permission(permission):
            self.auditor.log_authorization_decision(
                context.user_id, resource, permission.value, False,
                "Insufficient permissions"
            )
            raise AuthorizationError(f"Permission {permission.value} required")
        
        self.auditor.log_authorization_decision(
            context.user_id, resource, permission.value, True
        )