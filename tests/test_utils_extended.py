"""
Tests for extended utility functions in Agent Orchestra
"""
import pytest
import time
from agent_orchestra.utils import (
    safe_dict_get, validate_json_schema, sanitize_data,
    calculate_memory_usage, RateLimiter, deep_merge_dicts,
    truncate_string
)


class TestSafeDictGet:
    """Test cases for safe_dict_get function"""
    
    def test_simple_key_access(self):
        """Test simple key access"""
        data = {"name": "John", "age": 30}
        
        assert safe_dict_get(data, "name") == "John"
        assert safe_dict_get(data, "age") == 30
        assert safe_dict_get(data, "missing") is None
        assert safe_dict_get(data, "missing", "default") == "default"
    
    def test_nested_key_access(self):
        """Test nested key access with dot notation"""
        data = {
            "user": {
                "profile": {
                    "name": "John",
                    "settings": {
                        "theme": "dark",
                        "notifications": True
                    }
                }
            }
        }
        
        assert safe_dict_get(data, "user.profile.name") == "John"
        assert safe_dict_get(data, "user.profile.settings.theme") == "dark"
        assert safe_dict_get(data, "user.profile.settings.notifications") is True
        assert safe_dict_get(data, "user.profile.age") is None
        assert safe_dict_get(data, "user.profile.age", 25) == 25
    
    def test_custom_separator(self):
        """Test custom separator"""
        data = {"user": {"profile": {"name": "John"}}}
        
        assert safe_dict_get(data, "user/profile/name", separator="/") == "John"
        assert safe_dict_get(data, "user->profile->name", separator="->") == "John"
    
    def test_non_dict_input(self):
        """Test with non-dictionary input"""
        assert safe_dict_get("not a dict", "key") is None
        assert safe_dict_get(["list"], "key") is None
        assert safe_dict_get(123, "key") is None
        assert safe_dict_get(None, "key") is None
        assert safe_dict_get(None, "key", "default") == "default"
    
    def test_partial_path_exists(self):
        """Test when partial path exists but not full path"""
        data = {"user": {"profile": None}}
        
        assert safe_dict_get(data, "user.profile.name") is None
        assert safe_dict_get(data, "user.profile.name", "default") == "default"


class TestValidateJsonSchema:
    """Test cases for validate_json_schema function"""
    
    def test_valid_schema(self):
        """Test validation with valid data"""
        schema = {"name": str, "age": int, "active": bool}
        data = {"name": "John", "age": 30, "active": True}
        
        is_valid, errors = validate_json_schema(data, schema)
        
        assert is_valid is True
        assert errors == []
    
    def test_missing_fields(self):
        """Test validation with missing fields"""
        schema = {"name": str, "age": int, "active": bool}
        data = {"name": "John", "age": 30}
        
        is_valid, errors = validate_json_schema(data, schema)
        
        assert is_valid is False
        assert len(errors) == 1
        assert "Missing required field: active" in errors[0]
    
    def test_wrong_types(self):
        """Test validation with wrong types"""
        schema = {"name": str, "age": int, "active": bool}
        data = {"name": "John", "age": "thirty", "active": "true"}
        
        is_valid, errors = validate_json_schema(data, schema)
        
        assert is_valid is False
        assert len(errors) == 2
        assert any("age" in error and "int" in error for error in errors)
        assert any("active" in error and "bool" in error for error in errors)
    
    def test_non_dict_input(self):
        """Test validation with non-dictionary input"""
        schema = {"name": str}
        
        is_valid, errors = validate_json_schema("not a dict", schema)
        
        assert is_valid is False
        assert "Data must be a dictionary" in errors[0]
    
    def test_empty_schema(self):
        """Test validation with empty schema"""
        schema = {}
        data = {"any": "data"}
        
        is_valid, errors = validate_json_schema(data, schema)
        
        assert is_valid is True
        assert errors == []


class TestSanitizeData:
    """Test cases for sanitize_data function"""
    
    def test_default_sensitive_keys(self):
        """Test sanitization with default sensitive keys"""
        data = {
            "name": "John",
            "password": "secret123",
            "api_key": "abc123",
            "token": "xyz789"
        }
        
        sanitized = sanitize_data(data)
        
        assert sanitized["name"] == "John"
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["token"] == "[REDACTED]"
    
    def test_custom_sensitive_keys(self):
        """Test sanitization with custom sensitive keys"""
        data = {
            "name": "John",
            "social_security": "123-45-6789",
            "credit_card": "4111-1111-1111-1111"
        }
        
        sensitive_keys = ["social_security", "credit_card"]
        sanitized = sanitize_data(data, sensitive_keys)
        
        assert sanitized["name"] == "John"
        assert sanitized["social_security"] == "[REDACTED]"
        assert sanitized["credit_card"] == "[REDACTED]"
    
    def test_nested_sanitization(self):
        """Test sanitization of nested dictionaries"""
        data = {
            "user": {
                "name": "John",
                "credentials": {
                    "password": "secret",
                    "api_key": "abc123"
                }
            }
        }
        
        sanitized = sanitize_data(data)
        
        assert sanitized["user"]["name"] == "John"
        assert sanitized["user"]["credentials"]["password"] == "[REDACTED]"
        assert sanitized["user"]["credentials"]["api_key"] == "[REDACTED]"
    
    def test_list_sanitization(self):
        """Test sanitization of lists containing dictionaries"""
        data = {
            "users": [
                {"name": "John", "password": "secret1"},
                {"name": "Jane", "password": "secret2"}
            ]
        }
        
        sanitized = sanitize_data(data)
        
        assert sanitized["users"][0]["name"] == "John"
        assert sanitized["users"][0]["password"] == "[REDACTED]"
        assert sanitized["users"][1]["name"] == "Jane"
        assert sanitized["users"][1]["password"] == "[REDACTED]"
    
    def test_case_insensitive_keys(self):
        """Test case-insensitive key matching"""
        data = {
            "Password": "secret",
            "API_KEY": "abc123",
            "Token": "xyz789"
        }
        
        sanitized = sanitize_data(data)
        
        assert sanitized["Password"] == "[REDACTED]"
        assert sanitized["API_KEY"] == "[REDACTED]"
        assert sanitized["Token"] == "[REDACTED]"


class TestCalculateMemoryUsage:
    """Test cases for calculate_memory_usage function"""
    
    def test_memory_usage_returns_dict(self):
        """Test that memory usage returns a dictionary"""
        usage = calculate_memory_usage()
        
        assert isinstance(usage, dict)
        # Should have some expected keys
        expected_keys = {"rss", "vms", "percent", "available", "current", "peak"}
        assert any(key in usage for key in expected_keys)
    
    def test_memory_usage_values_are_numeric(self):
        """Test that memory usage values are numeric"""
        usage = calculate_memory_usage()
        
        for key, value in usage.items():
            assert isinstance(value, (int, float))
            assert value >= 0  # Memory usage should be non-negative


class TestRateLimiter:
    """Test cases for RateLimiter class"""
    
    def test_rate_limiter_creation(self):
        """Test rate limiter creation"""
        limiter = RateLimiter(max_tokens=10, refill_rate=2.0)
        
        assert limiter.max_tokens == 10
        assert limiter.refill_rate == 2.0
        assert limiter.tokens == 10
    
    def test_token_consumption(self):
        """Test token consumption"""
        limiter = RateLimiter(max_tokens=5, refill_rate=1.0)
        
        # Should be able to consume tokens
        assert limiter.consume(3) is True
        assert limiter.consume(2) is True
        
        # Should not be able to consume more tokens
        assert limiter.consume(1) is False
    
    def test_token_refill(self):
        """Test token refilling over time"""
        limiter = RateLimiter(max_tokens=10, refill_rate=10.0)  # 10 tokens/second
        
        # Consume all tokens
        assert limiter.consume(10) is True
        assert limiter.consume(1) is False
        
        # Wait and verify tokens are refilled
        time.sleep(0.2)  # Should refill ~2 tokens
        assert limiter.consume(1) is True
    
    def test_time_until_available(self):
        """Test calculation of time until tokens are available"""
        limiter = RateLimiter(max_tokens=10, refill_rate=5.0)  # 5 tokens/second
        
        # Consume all tokens
        limiter.consume(10)
        
        # Should need time for tokens to become available
        time_needed = limiter.time_until_available(1)
        assert time_needed > 0
        
        # More tokens should need more time
        time_needed_5 = limiter.time_until_available(5)
        assert time_needed_5 > time_needed
    
    def test_max_tokens_limit(self):
        """Test that tokens don't exceed maximum"""
        limiter = RateLimiter(max_tokens=5, refill_rate=10.0)
        
        # Wait enough time for many tokens to be generated
        time.sleep(0.2)
        limiter._refill()
        
        # Should still be limited to max_tokens
        assert limiter.tokens <= 5


class TestDeepMergeDicts:
    """Test cases for deep_merge_dicts function"""
    
    def test_simple_merge(self):
        """Test simple dictionary merge"""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        
        result = deep_merge_dicts(dict1, dict2)
        
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}
    
    def test_override_values(self):
        """Test overriding values"""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 10, "c": 3}
        
        result = deep_merge_dicts(dict1, dict2)
        
        assert result == {"a": 10, "b": 2, "c": 3}
    
    def test_nested_merge(self):
        """Test nested dictionary merge"""
        dict1 = {
            "user": {"name": "John", "age": 30},
            "settings": {"theme": "light"}
        }
        dict2 = {
            "user": {"age": 31, "city": "NYC"},
            "settings": {"notifications": True}
        }
        
        result = deep_merge_dicts(dict1, dict2)
        
        expected = {
            "user": {"name": "John", "age": 31, "city": "NYC"},
            "settings": {"theme": "light", "notifications": True}
        }
        
        assert result == expected
    
    def test_non_dict_override(self):
        """Test overriding dict with non-dict value"""
        dict1 = {"a": {"nested": "value"}}
        dict2 = {"a": "simple_value"}
        
        result = deep_merge_dicts(dict1, dict2)
        
        assert result == {"a": "simple_value"}
    
    def test_original_dicts_unchanged(self):
        """Test that original dictionaries are not modified"""
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        
        result = deep_merge_dicts(dict1, dict2)
        
        assert dict1 == {"a": 1}  # Unchanged
        assert dict2 == {"b": 2}  # Unchanged
        assert result == {"a": 1, "b": 2}


class TestTruncateString:
    """Test cases for truncate_string function"""
    
    def test_no_truncation_needed(self):
        """Test string that doesn't need truncation"""
        text = "Short text"
        result = truncate_string(text, 20)
        
        assert result == text
    
    def test_basic_truncation(self):
        """Test basic string truncation"""
        text = "This is a very long text that needs to be truncated"
        result = truncate_string(text, 20)
        
        assert len(result) == 20
        assert result.endswith("...")
        assert result == "This is a very l..."
    
    def test_custom_suffix(self):
        """Test truncation with custom suffix"""
        text = "Long text here"
        result = truncate_string(text, 10, suffix="[...]")
        
        assert len(result) == 10
        assert result.endswith("[...]")
    
    def test_suffix_longer_than_max_length(self):
        """Test when suffix is longer than max length"""
        text = "Some text"
        result = truncate_string(text, 5, suffix="very long suffix")
        
        assert result == "very "  # Suffix truncated to max_length
        assert len(result) == 5
    
    def test_non_string_input(self):
        """Test with non-string input"""
        result = truncate_string(12345, 3)
        assert result == "1..."
        
        result = truncate_string(["list"], 8)
        assert result == "['list']"
    
    def test_empty_string(self):
        """Test with empty string"""
        result = truncate_string("", 10)
        assert result == ""
    
    def test_exact_length_match(self):
        """Test when string length exactly matches max_length"""
        text = "Exact"
        result = truncate_string(text, 5)
        
        assert result == "Exact"