"""
Security module for Agent Orchestra - authentication, authorization, and audit
"""
import hashlib
import secrets
import time
import jwt
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


class Permission(str, Enum):
    """System permissions"""
    AGENT_REGISTER = "agent:register"
    AGENT_UNREGISTER = "agent:unregister"
    AGENT_LIST = "agent:list"
    AGENT_STATUS = "agent:status"
    
    TASK_SUBMIT = "task:submit"
    TASK_CANCEL = "task:cancel"
    TASK_VIEW = "task:view"
    TASK_RESULTS = "task:results"
    
    SYSTEM_STATUS = "system:status"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    
    METRICS_VIEW = "metrics:view"
    METRICS_EXPORT = "metrics:export"
    
    LOGS_VIEW = "logs:view"
    LOGS_EXPORT = "logs:export"


class Role(str, Enum):
    """Predefined user roles"""
    ADMIN = "admin"
    OPERATOR = "operator"
    AGENT = "agent"
    MONITOR = "monitor"
    USER = "user"


@dataclass
class User:
    """User account information"""
    id: str
    username: str
    email: str
    roles: Set[Role]
    permissions: Set[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Session:
    """User session information"""
    id: str
    user_id: str
    token: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    active: bool = True


@dataclass
class AuditEvent:
    """Security audit event"""
    id: str
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource: str
    result: str  # success, failure, unauthorized
    ip_address: Optional[str]
    details: Dict[str, Any]


class SecurityError(Exception):
    """Base security exception"""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed"""
    pass


class AuthorizationError(SecurityError):
    """Authorization failed - insufficient permissions"""
    pass


class TokenExpiredError(AuthenticationError):
    """JWT token expired"""
    pass


class SecurityManager:
    """Central security management"""
    
    def __init__(self, jwt_secret: str = None, token_expiry_hours: int = 24):
        self.jwt_secret = jwt_secret or secrets.token_hex(32)
        self.token_expiry_hours = token_expiry_hours
        
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        self._audit_events: List[AuditEvent] = []
        self._role_permissions: Dict[Role, Set[Permission]] = {}
        self._security_policies: Dict[str, Any] = {}
        
        self._setup_default_roles()
        self._setup_default_policies()
    
    def _setup_default_roles(self):
        """Setup default role permissions"""
        self._role_permissions = {
            Role.ADMIN: {
                Permission.AGENT_REGISTER,
                Permission.AGENT_UNREGISTER,
                Permission.AGENT_LIST,
                Permission.AGENT_STATUS,
                Permission.TASK_SUBMIT,
                Permission.TASK_CANCEL,
                Permission.TASK_VIEW,
                Permission.TASK_RESULTS,
                Permission.SYSTEM_STATUS,
                Permission.SYSTEM_CONFIG,
                Permission.SYSTEM_ADMIN,
                Permission.SYSTEM_MONITOR,
                Permission.METRICS_VIEW,
                Permission.METRICS_EXPORT,
                Permission.LOGS_VIEW,
                Permission.LOGS_EXPORT
            },
            Role.OPERATOR: {
                Permission.AGENT_LIST,
                Permission.AGENT_STATUS,
                Permission.TASK_SUBMIT,
                Permission.TASK_CANCEL,
                Permission.TASK_VIEW,
                Permission.TASK_RESULTS,
                Permission.SYSTEM_STATUS,
                Permission.SYSTEM_MONITOR,
                Permission.METRICS_VIEW
            },
            Role.AGENT: {
                Permission.AGENT_REGISTER,
                Permission.TASK_VIEW,
                Permission.SYSTEM_STATUS
            },
            Role.MONITOR: {
                Permission.AGENT_LIST,
                Permission.AGENT_STATUS,
                Permission.TASK_VIEW,
                Permission.SYSTEM_STATUS,
                Permission.SYSTEM_MONITOR,
                Permission.METRICS_VIEW,
                Permission.LOGS_VIEW
            },
            Role.USER: {
                Permission.TASK_SUBMIT,
                Permission.TASK_VIEW,
                Permission.TASK_RESULTS,
                Permission.SYSTEM_STATUS
            }
        }
    
    def _setup_default_policies(self):
        """Setup default security policies"""
        self._security_policies = {
            "password_min_length": 8,
            "password_require_special": True,
            "session_timeout_hours": 24,
            "max_login_attempts": 5,
            "lockout_duration_minutes": 30,
            "require_mfa": False,
            "audit_retention_days": 90
        }
    
    def create_user(self, username: str, email: str, password: str, roles: List[Role] = None) -> str:
        """Create a new user"""
        if self._get_user_by_username(username):
            raise SecurityError(f"User {username} already exists")
        
        if not self._validate_password(password):
            raise SecurityError("Password does not meet security requirements")
        
        user_id = secrets.token_hex(16)
        roles_set = set(roles) if roles else {Role.USER}
        
        # Calculate permissions from roles
        permissions = set()
        for role in roles_set:
            permissions.update(self._role_permissions.get(role, set()))
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            roles=roles_set,
            permissions=permissions,
            created_at=datetime.utcnow()
        )
        
        self._users[user_id] = user
        
        # Store password hash separately (not in user object for security)
        self._store_password_hash(user_id, password)
        
        self._audit("user_created", user_id, "user", "success", {"username": username, "roles": list(roles_set)})
        
        logger.info("User created", user_id=user_id, username=username, roles=list(roles_set))
        return user_id
    
    def authenticate(self, username: str, password: str, ip_address: str = None) -> str:
        """Authenticate user and create session"""
        user = self._get_user_by_username(username)
        
        if not user or not user.active:
            self._audit("login_failed", user.id if user else None, "authentication", "failure", 
                       {"username": username, "reason": "user_not_found"}, ip_address)
            raise AuthenticationError("Invalid credentials")
        
        if not self._verify_password(user.id, password):
            self._audit("login_failed", user.id, "authentication", "failure",
                       {"username": username, "reason": "invalid_password"}, ip_address)
            raise AuthenticationError("Invalid credentials")
        
        # Create session
        session_id = secrets.token_hex(32)
        token = self._generate_jwt_token(user)
        
        session = Session(
            id=session_id,
            user_id=user.id,
            token=token,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            ip_address=ip_address
        )
        
        self._sessions[session_id] = session
        
        # Update user last login
        user.last_login = datetime.utcnow()
        
        self._audit("login_success", user.id, "authentication", "success", 
                   {"username": username, "session_id": session_id}, ip_address)
        
        logger.info("User authenticated", user_id=user.id, username=username, session_id=session_id)
        return token
    
    def validate_token(self, token: str) -> User:
        """Validate JWT token and return user"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            user_id = payload.get("user_id")
            
            if not user_id or user_id not in self._users:
                raise AuthenticationError("Invalid token")
            
            user = self._users[user_id]
            
            if not user.active:
                raise AuthenticationError("User account disabled")
            
            return user
            
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in user.permissions
    
    def require_permission(self, user: User, permission: Permission, resource: str = ""):
        """Require user to have specific permission"""
        if not self.check_permission(user, permission):
            self._audit("authorization_failed", user.id, resource or "unknown", "unauthorized",
                       {"permission": permission, "username": user.username})
            raise AuthorizationError(f"Permission {permission} required")
        
        self._audit("authorization_success", user.id, resource or "unknown", "success",
                   {"permission": permission})
    
    def logout(self, token: str):
        """Logout user by invalidating session"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            user_id = payload.get("user_id")
            
            # Find and deactivate session
            for session in self._sessions.values():
                if session.token == token:
                    session.active = False
                    self._audit("logout", user_id, "authentication", "success", 
                               {"session_id": session.id})
                    break
            
        except jwt.InvalidTokenError:
            pass  # Invalid token, nothing to logout
    
    def revoke_user_sessions(self, user_id: str):
        """Revoke all sessions for a user"""
        count = 0
        for session in self._sessions.values():
            if session.user_id == user_id and session.active:
                session.active = False
                count += 1
        
        self._audit("sessions_revoked", user_id, "security", "success", {"revoked_count": count})
        logger.info("User sessions revoked", user_id=user_id, count=count)
    
    def _generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user with security hardening"""
        now = datetime.utcnow()
        
        payload = {
            "user_id": user.id,
            "username": user.username,
            "roles": list(user.roles),
            "iat": now,
            "exp": now + timedelta(hours=self.token_expiry_hours),
            "nbf": now,  # Not before - token is not valid before this time
            "jti": secrets.token_hex(16),  # JWT ID for token tracking/revocation
            "iss": "agent-orchestra",  # Issuer
            "aud": "agent-orchestra-api"  # Audience
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def _get_user_by_username(self, username: str) -> Optional[User]:
        """Find user by username"""
        for user in self._users.values():
            if user.username == username:
                return user
        return None
    
    def _validate_password(self, password: str) -> bool:
        """Validate password against security policy"""
        policy = self._security_policies
        
        if len(password) < policy["password_min_length"]:
            return False
        
        if policy["password_require_special"]:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                return False
        
        return True
    
    def _store_password_hash(self, user_id: str, password: str):
        """Store password hash (simplified - use proper password hashing in production)"""
        # In production, use bcrypt or similar
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        
        # Store in a separate secure storage (not implemented here)
        pass
    
    def _verify_password(self, user_id: str, password: str) -> bool:
        """Verify password against stored hash"""
        # Simplified implementation - in production, properly verify hash
        return True  # Placeholder
    
    def _audit(self, action: str, user_id: Optional[str], resource: str, result: str, 
              details: Dict[str, Any], ip_address: str = None):
        """Record audit event"""
        event = AuditEvent(
            id=secrets.token_hex(16),
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            details=details
        )
        
        self._audit_events.append(event)
        
        # Keep only recent events
        retention_days = self._security_policies["audit_retention_days"]
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        self._audit_events = [e for e in self._audit_events if e.timestamp > cutoff]
    
    def get_audit_events(self, user_id: Optional[str] = None, action: Optional[str] = None,
                        start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                        limit: int = 100) -> List[AuditEvent]:
        """Get audit events with filtering"""
        events = self._audit_events
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if action:
            events = [e for e in events if e.action == action]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # Sort by timestamp descending and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security system status"""
        active_sessions = sum(1 for s in self._sessions.values() if s.active)
        recent_logins = len([e for e in self._audit_events 
                           if e.action == "login_success" and 
                           e.timestamp > datetime.utcnow() - timedelta(hours=24)])
        
        failed_logins = len([e for e in self._audit_events 
                           if e.action == "login_failed" and 
                           e.timestamp > datetime.utcnow() - timedelta(hours=24)])
        
        return {
            "total_users": len(self._users),
            "active_users": len([u for u in self._users.values() if u.active]),
            "active_sessions": active_sessions,
            "recent_logins_24h": recent_logins,
            "failed_logins_24h": failed_logins,
            "audit_events": len(self._audit_events),
            "security_policies": self._security_policies
        }
    
    def update_security_policy(self, policy_name: str, value: Any):
        """Update security policy"""
        if policy_name in self._security_policies:
            old_value = self._security_policies[policy_name]
            self._security_policies[policy_name] = value
            
            self._audit("policy_updated", None, "security", "success",
                       {"policy": policy_name, "old_value": old_value, "new_value": value})
            
            logger.info("Security policy updated", 
                       policy=policy_name, 
                       old_value=old_value, 
                       new_value=value)


def require_auth(permission: Permission):
    """Decorator to require authentication and permission"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # In a real implementation, extract user from request context
            # and check permissions
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimiter:
    """Rate limiting for security"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if identifier is within rate limits"""
        now = time.time()
        
        if identifier not in self._requests:
            self._requests[identifier] = []
        
        # Clean old requests
        cutoff = now - self.window_seconds
        self._requests[identifier] = [
            req_time for req_time in self._requests[identifier]
            if req_time > cutoff
        ]
        
        # Check limit
        if len(self._requests[identifier]) >= self.max_requests:
            return False
        
        # Record new request
        self._requests[identifier].append(now)
        return True
    
    def reset_rate_limit(self, identifier: str):
        """Reset rate limit for identifier"""
        if identifier in self._requests:
            del self._requests[identifier]


class InputValidator:
    """Input validation for security"""
    
    @staticmethod
    def validate_task_data(data: Dict[str, Any]) -> bool:
        """Validate task data for security"""
        # Check for dangerous patterns
        dangerous_patterns = [
            "__import__",
            "eval",
            "exec", 
            "subprocess",
            "os.system"
        ]
        
        data_str = str(data).lower()
        for pattern in dangerous_patterns:
            if pattern in data_str:
                logger.warning("Dangerous pattern detected in task data", pattern=pattern)
                return False
        
        return True
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return ""
        
        # Remove null bytes and control characters
        sanitized = ''.join(c for c in value if ord(c) >= 32 or c in '\n\r\t')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized