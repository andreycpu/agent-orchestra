# Security Guide

This document outlines security considerations, best practices, and implementation details for Agent Orchestra.

## Security Architecture

Agent Orchestra implements defense-in-depth security with multiple layers:

1. **Authentication & Authorization**: JWT-based tokens with role-based access control
2. **Input Validation**: Comprehensive validation and sanitization
3. **Network Security**: HTTPS enforcement and allowed host restrictions
4. **Data Protection**: Secure hashing and encryption utilities
5. **Audit Logging**: Comprehensive security event tracking
6. **Error Handling**: Secure error responses without information leakage

## Authentication

### JWT Token Authentication

Agent Orchestra uses JWT tokens for authentication with the following features:

- **Secure token generation** with configurable expiry
- **Role-based permissions** embedded in tokens
- **Token refresh** mechanisms for long-running sessions
- **Secure secret key management**

```python
from agent_orchestra.security_utils import TokenManager, Role, Permission

# Initialize token manager
token_manager = TokenManager("your-secret-key", default_expiry=3600)

# Generate token
token = token_manager.generate_token(
    user_id="user123",
    roles=[Role.OPERATOR.value],
    permissions=[Permission.CREATE_TASKS.value]
)

# Verify token
try:
    payload = token_manager.verify_token(token)
    print(f"Authenticated user: {payload['user_id']}")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
```

### Password Security

Passwords are protected using industry-standard practices:

- **PBKDF2 hashing** with SHA-256 and 100,000 iterations
- **Salt-based hashing** to prevent rainbow table attacks
- **Password policy enforcement** with configurable requirements
- **Timing-safe comparison** to prevent timing attacks

```python
from agent_orchestra.security_utils import PasswordPolicy, HashManager

# Password policy
policy = PasswordPolicy(
    min_length=12,
    require_uppercase=True,
    require_lowercase=True,
    require_numbers=True,
    require_special=True,
    forbidden_patterns=["password", "company_name"]
)

# Validate password
is_valid, violations = policy.validate_password("MySecureP@ssw0rd123")

# Hash password
hashed, salt = HashManager.hash_password("MySecureP@ssw0rd123")

# Verify password
is_correct = HashManager.verify_password("MySecureP@ssw0rd123", hashed, salt)
```

## Authorization

### Role-Based Access Control (RBAC)

Agent Orchestra implements RBAC with predefined roles and permissions:

#### Roles

- **VIEWER**: Read-only access to tasks, agents, and results
- **OPERATOR**: Can create and update tasks, read agents and results
- **ADMIN**: Full task and agent management, configuration access
- **SUPER_ADMIN**: Complete system access

#### Permissions

- `READ_TASKS`: View task information
- `CREATE_TASKS`: Create new tasks
- `UPDATE_TASKS`: Modify existing tasks
- `DELETE_TASKS`: Remove tasks
- `READ_AGENTS`: View agent information
- `MANAGE_AGENTS`: Register and manage agents
- `READ_RESULTS`: Access task execution results
- `ADMIN_SYSTEM`: System administration functions
- `MODIFY_CONFIG`: Change system configuration
- `VIEW_METRICS`: Access monitoring and metrics

```python
from agent_orchestra.security_utils import RoleManager, Role, Permission

role_manager = RoleManager()

# Check permissions
has_permission = role_manager.has_permission(
    [Role.OPERATOR], 
    Permission.CREATE_TASKS
)

# Get all permissions for roles
permissions = role_manager.get_permissions_for_roles([Role.ADMIN])
```

### Security Context

Security context provides request-level security information:

```python
from agent_orchestra.security_utils import SecurityContext, SecurityContextBuilder

# Create from token
context_builder = SecurityContextBuilder(role_manager, token_manager)
context = context_builder.from_token(jwt_token, source_ip="192.168.1.1")

# Check permissions
if context.has_permission(Permission.CREATE_TASKS):
    # User can create tasks
    pass

# Check expiration
if context.is_expired():
    # Token has expired
    pass
```

## Input Validation and Sanitization

### XSS and Injection Prevention

All user inputs are validated and sanitized:

```python
from agent_orchestra.security_utils import InputSanitizer

# Sanitize user input
clean_data = InputSanitizer.validate_user_input({
    "name": "John Doe",
    "description": "<script>alert('xss')</script>Safe content",
    "tags": ["tag1", "tag2"]
})

# Sanitize strings
clean_string = InputSanitizer.sanitize_string(
    "User input with <script>dangerous</script> content",
    max_length=200,
    allow_html=False
)
```

### Data Validation

Comprehensive validation for all data types:

```python
from agent_orchestra.validation import (
    validate_id, validate_task_type, validate_email, validate_url
)

try:
    # Validate various inputs
    task_id = validate_id("task-123")
    email = validate_email("user@example.com")
    url = validate_url("https://api.example.com")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Network Security

### HTTPS Enforcement

Configure HTTPS enforcement in production:

```python
from agent_orchestra.config_validator import SecurityConfig

security_config = SecurityConfig(
    secret_key="your-secret-key",
    require_https=True,
    allowed_hosts=["yourdomain.com", "api.yourdomain.com"],
    cors_origins=["https://frontend.yourdomain.com"]
)
```

### Request Security Middleware

Use security middleware for request processing:

```python
from agent_orchestra.security_utils import SecurityMiddleware

middleware = SecurityMiddleware(context_builder, auditor)

# Process request
try:
    context = middleware.process_request(
        headers={"Authorization": "Bearer " + token},
        source_ip="192.168.1.1"
    )
    
    # Require specific permission
    middleware.require_permission(
        context, 
        Permission.CREATE_TASKS, 
        resource="tasks"
    )
    
except AuthenticationError:
    # Handle authentication failure
    pass
except AuthorizationError:
    # Handle authorization failure
    pass
```

## Audit Logging

### Security Event Logging

Agent Orchestra provides comprehensive security audit logging:

```python
from agent_orchestra.security_utils import SecurityAuditor

auditor = SecurityAuditor(logger)

# Log authentication attempts
auditor.log_authentication_attempt(
    user_id="user123",
    success=True,
    source_ip="192.168.1.1"
)

# Log authorization decisions
auditor.log_authorization_decision(
    user_id="user123",
    resource="tasks",
    action="create",
    granted=True
)

# Log data access
auditor.log_data_access(
    user_id="user123",
    data_type="tasks",
    operation="SELECT",
    record_count=50
)

# Log configuration changes
auditor.log_configuration_change(
    user_id="admin",
    component="security",
    changes={"max_failed_attempts": {"old": 3, "new": 5}}
)
```

### Account Lockout

Automatic account lockout after failed attempts:

```python
# Check if account is locked
if auditor.is_account_locked("user123"):
    raise AuthenticationError("Account is locked due to multiple failed attempts")

# The auditor automatically tracks failed attempts and triggers lockout
```

## Data Protection

### Encryption Utilities

```python
from agent_orchestra.security_utils import generate_secure_key

# Generate cryptographically secure keys
secret_key = generate_secure_key(32)  # 32 bytes = 256 bits
api_key = generate_api_key()  # Formatted API key

# Validate API key format
if validate_api_key_format(api_key):
    print("Valid API key format")
```

### Secure Comparisons

Use timing-safe comparisons for sensitive data:

```python
from agent_orchestra.security_utils import secure_compare

# Safe string comparison
if secure_compare(provided_token, expected_token):
    # Tokens match
    pass
```

## Security Best Practices

### Development

1. **Never hardcode secrets** - Use environment variables or secure vaults
2. **Validate all inputs** - Use provided validation utilities
3. **Use secure defaults** - Enable security features by default
4. **Log security events** - Monitor authentication and authorization
5. **Regular security reviews** - Audit code and configurations

### Deployment

1. **Use HTTPS everywhere** - Encrypt all network traffic
2. **Restrict network access** - Use firewalls and security groups
3. **Regular updates** - Keep dependencies updated
4. **Monitor logs** - Set up alerting for security events
5. **Backup security logs** - Ensure audit trail preservation

### Configuration

```python
# Secure configuration example
from agent_orchestra.config_validator import ConfigurationLoader

config = ConfigurationLoader.from_dict({
    "security": {
        "secret_key": os.environ["ORCHESTRA_SECRET_KEY"],
        "token_expiry_seconds": 3600,
        "max_failed_attempts": 3,
        "lockout_duration_minutes": 15,
        "require_https": True,
        "allowed_hosts": ["yourdomain.com"],
        "cors_origins": ["https://frontend.yourdomain.com"]
    },
    "logging": {
        "level": "INFO",
        "format": "json"  # Structured logging for security events
    }
})
```

## Threat Mitigation

### Common Threats and Mitigations

| Threat | Mitigation |
|--------|------------|
| **SQL Injection** | Parameterized queries, input validation |
| **XSS** | Input sanitization, output encoding |
| **CSRF** | CSRF tokens, SameSite cookies |
| **Brute Force** | Account lockout, rate limiting |
| **Token Theft** | Short token expiry, secure storage |
| **MITM** | HTTPS enforcement, certificate pinning |
| **Data Breach** | Encryption at rest, minimal data retention |

### Rate Limiting

```python
from agent_orchestra.retry_mechanisms import RateLimiter

# API rate limiting
api_limiter = RateLimiter(requests_per_second=10, burst_size=20)

async def protected_endpoint():
    if not await api_limiter.acquire():
        raise ThrottlingError("Rate limit exceeded")
    
    # Process request
    pass
```

### Circuit Breaker for External Services

```python
from agent_orchestra.retry_mechanisms import CircuitBreaker, CircuitBreakerConfig

# Protect against failing external services
cb_config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=NetworkError
)

@with_circuit_breaker("external_auth_service", cb_config)
def verify_with_external_service(token):
    # Call external authentication service
    pass
```

## Security Monitoring

### Metrics and Alerts

Monitor security-related metrics:

```python
from agent_orchestra.performance_monitor import PerformanceMetric

# Custom security metrics
security_metric = PerformanceMetric(
    name="failed_auth_attempts",
    value=failed_count,
    unit="count",
    tags={"source_ip": client_ip, "user_id": user_id}
)

performance_manager.metrics_collector.record_metric(security_metric)
```

### Health Checks

Include security in health checks:

```python
def security_health_check():
    health_data = {
        "authentication_service": "healthy",
        "certificate_expiry": "30_days",
        "security_logs": "active"
    }
    
    return health_data
```

## Compliance

### Audit Requirements

Agent Orchestra supports compliance with:

- **SOC 2**: Comprehensive audit logging
- **PCI DSS**: Secure data handling practices
- **GDPR**: Data protection and privacy controls
- **HIPAA**: Healthcare data security (when configured)

### Data Retention

Configure appropriate data retention:

```python
# Security log retention
security_config = {
    "audit_log_retention_days": 90,
    "authentication_log_retention_days": 30,
    "access_log_retention_days": 7
}
```

## Incident Response

### Security Event Response

1. **Detection**: Monitor logs for security events
2. **Assessment**: Evaluate threat level and impact
3. **Containment**: Isolate affected systems
4. **Investigation**: Analyze logs and evidence
5. **Recovery**: Restore normal operations
6. **Lessons Learned**: Update security measures

### Emergency Procedures

```python
# Emergency account lockout
def emergency_lockout(user_id: str, reason: str):
    auditor.log_security_violation(
        violation_type="emergency_lockout",
        user_id=user_id,
        details={"reason": reason}
    )
    
    # Revoke all tokens for user
    # Lock account
    # Notify administrators
```

This security guide provides a comprehensive overview of Agent Orchestra's security features and best practices. Always follow the principle of least privilege and defense in depth when deploying in production environments.