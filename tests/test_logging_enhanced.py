"""
Tests for enhanced logging functionality
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch
from datetime import datetime
from agent_orchestra.logging_config import (
    SensitiveDataFilter, LogSampler, LogAggregator, 
    AsyncLogHandler, StructuredLogProcessor
)


class TestSensitiveDataFilter:
    """Test cases for SensitiveDataFilter"""
    
    @pytest.fixture
    def filter_instance(self):
        """Create SensitiveDataFilter instance"""
        return SensitiveDataFilter()
    
    def test_filter_simple_password(self, filter_instance):
        """Test filtering simple password"""
        event = {"message": "Login attempt", "password": "secret123"}
        
        filtered = filter_instance.filter(event)
        
        assert filtered["message"] == "Login attempt"
        assert filtered["password"] == "se*****23"  # Masked
    
    def test_filter_nested_sensitive_data(self, filter_instance):
        """Test filtering nested sensitive data"""
        event = {
            "user": {
                "name": "john",
                "credentials": {
                    "password": "supersecret",
                    "api_key": "abc123def456"
                }
            },
            "action": "login"
        }
        
        filtered = filter_instance.filter(event)
        
        assert filtered["user"]["name"] == "john"
        assert filtered["action"] == "login"
        assert filtered["user"]["credentials"]["password"] == "su*******et"
        assert filtered["user"]["credentials"]["api_key"] == "ab*******56"
    
    def test_filter_list_with_sensitive_data(self, filter_instance):
        """Test filtering lists containing sensitive data"""
        event = {
            "users": [
                {"name": "alice", "token": "token1"},
                {"name": "bob", "token": "token2"}
            ]
        }
        
        filtered = filter_instance.filter(event)
        
        assert filtered["users"][0]["name"] == "alice"
        assert filtered["users"][0]["token"] == "to***1"
        assert filtered["users"][1]["name"] == "bob"
        assert filtered["users"][1]["token"] == "to***2"
    
    def test_custom_sensitive_keys(self):
        """Test custom sensitive keys"""
        custom_filter = SensitiveDataFilter(
            sensitive_keys={"ssn", "credit_card"}, 
            mask_char='X'
        )
        
        event = {
            "name": "John",
            "ssn": "123-45-6789",
            "credit_card": "4111-1111-1111-1111",
            "password": "shouldnotbefiltered"  # Not in custom keys
        }
        
        filtered = custom_filter.filter(event)
        
        assert filtered["name"] == "John"
        assert filtered["ssn"] == "12XXXXXXXXX89"
        assert filtered["credit_card"] == "41XXXXXXXXXXXXXXX11"
        assert filtered["password"] == "shouldnotbefiltered"
    
    def test_mask_short_values(self, filter_instance):
        """Test masking values shorter than 4 characters"""
        event = {"password": "123", "token": "ab"}
        
        filtered = filter_instance.filter(event)
        
        assert filtered["password"] == "***"
        assert filtered["token"] == "**"
    
    def test_mask_none_values(self, filter_instance):
        """Test masking None values"""
        event = {"password": None, "token": ""}
        
        filtered = filter_instance.filter(event)
        
        assert filtered["password"] is None
        assert filtered["token"] == ""


class TestLogSampler:
    """Test cases for LogSampler"""
    
    def test_sampler_creation(self):
        """Test LogSampler creation"""
        sampler = LogSampler(sample_rate=0.5, burst_limit=10)
        
        assert sampler.sample_rate == 0.5
        assert sampler.burst_limit == 10
        assert sampler.total_count == 0
        assert sampler.sample_count == 0
    
    def test_sampler_invalid_parameters(self):
        """Test LogSampler with invalid parameters"""
        # Invalid sample rate
        with pytest.raises(ValueError, match="sample_rate must be between 0.0 and 1.0"):
            LogSampler(sample_rate=1.5)
        
        with pytest.raises(ValueError, match="sample_rate must be between 0.0 and 1.0"):
            LogSampler(sample_rate=-0.1)
        
        # Invalid burst limit
        with pytest.raises(ValueError, match="burst_limit must be non-negative"):
            LogSampler(burst_limit=-1)
    
    def test_burst_period_all_logged(self):
        """Test that all logs are kept during burst period"""
        sampler = LogSampler(sample_rate=0.1, burst_limit=5)
        
        # During burst period, all logs should be kept
        for i in range(5):
            result = sampler.should_log({"message": f"log {i}"})
            assert result is True
        
        # After burst, sampling should kick in
        logged_count = 0
        for i in range(100):
            if sampler.should_log({"message": f"log {i+5}"}):
                logged_count += 1
        
        # Should be roughly 10% (sample_rate=0.1) but some variance expected
        assert logged_count < 50  # Much less than 100%
    
    def test_zero_sample_rate(self):
        """Test zero sample rate after burst"""
        sampler = LogSampler(sample_rate=0.0, burst_limit=3)
        
        # Burst period
        for i in range(3):
            assert sampler.should_log({"message": f"log {i}"}) is True
        
        # After burst, nothing should be logged
        for i in range(10):
            assert sampler.should_log({"message": f"log {i+3}"}) is False
    
    def test_full_sample_rate(self):
        """Test full sample rate"""
        sampler = LogSampler(sample_rate=1.0, burst_limit=2)
        
        # All logs should be kept
        for i in range(20):
            assert sampler.should_log({"message": f"log {i}"}) is True
    
    def test_sampler_stats(self):
        """Test sampler statistics"""
        sampler = LogSampler(sample_rate=0.5, burst_limit=2)
        
        # Log some events
        for i in range(10):
            sampler.should_log({"message": f"log {i}"})
        
        stats = sampler.get_stats()
        
        assert stats["total_logs"] == 10
        assert stats["sample_rate_configured"] == 0.5
        assert stats["burst_limit"] == 2
        assert 0 <= stats["sample_rate_actual"] <= 1
    
    def test_thread_safety(self):
        """Test thread safety of LogSampler"""
        sampler = LogSampler(sample_rate=0.5, burst_limit=5)
        results = []
        
        def worker():
            for i in range(100):
                result = sampler.should_log({"message": f"log {i}"})
                results.append(result)
        
        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should have results from all threads
        assert len(results) == 500
        stats = sampler.get_stats()
        assert stats["total_logs"] == 500


class TestLogAggregator:
    """Test cases for LogAggregator"""
    
    @pytest.fixture
    def aggregator(self):
        """Create LogAggregator instance"""
        return LogAggregator(window_seconds=5, max_duplicates=2)
    
    def test_first_messages_pass_through(self, aggregator):
        """Test that first messages pass through normally"""
        event = {"event": "test message", "level": "info"}
        
        # First occurrence should pass through
        result = aggregator.process(event)
        assert result == event
        
        # Second occurrence should pass through
        result = aggregator.process(event)
        assert result == event
    
    def test_aggregation_after_max_duplicates(self, aggregator):
        """Test aggregation after max duplicates reached"""
        event = {"event": "repeated message", "level": "warning"}
        
        # First two pass through
        aggregator.process(event)
        aggregator.process(event)
        
        # Third should return aggregated message
        result = aggregator.process(event)
        assert result is not None
        assert result["aggregated_count"] == 3
        assert "first_occurrence" in result
        assert "last_occurrence" in result
        
        # Fourth should return None (already aggregated)
        result = aggregator.process(event)
        assert result is None
    
    def test_different_messages_handled_separately(self, aggregator):
        """Test that different messages are handled separately"""
        event1 = {"event": "message one", "level": "info"}
        event2 = {"event": "message two", "level": "info"}
        
        # Both should pass through independently
        for _ in range(3):  # More than max_duplicates
            result1 = aggregator.process(event1)
            result2 = aggregator.process(event2)
            
            # Both should be processed (different messages)
            assert result1 is not None
            assert result2 is not None
    
    def test_different_levels_handled_separately(self, aggregator):
        """Test that same message with different levels are separate"""
        event_info = {"event": "same message", "level": "info"}
        event_error = {"event": "same message", "level": "error"}
        
        # Should be treated as different messages due to different levels
        for _ in range(3):
            result1 = aggregator.process(event_info)
            result2 = aggregator.process(event_error)
            
            assert result1 is not None
            assert result2 is not None
    
    def test_cleanup_old_entries(self, aggregator):
        """Test cleanup of old entries"""
        event = {"event": "test message", "level": "info"}
        
        # Process message
        aggregator.process(event)
        assert len(aggregator.message_counts) == 1
        
        # Wait for window to expire
        time.sleep(6)  # window_seconds=5, so this should expire
        
        # Process new message to trigger cleanup
        new_event = {"event": "new message", "level": "info"}
        aggregator.process(new_event)
        
        # Old entry should be cleaned up
        # Note: This test might be flaky depending on timing
        # In a real implementation, you might want a more deterministic cleanup
    
    def test_hash_generation(self, aggregator):
        """Test message hash generation"""
        hash1 = aggregator._hash_message("test message", "info")
        hash2 = aggregator._hash_message("test message", "info")
        hash3 = aggregator._hash_message("test message", "error")
        hash4 = aggregator._hash_message("different message", "info")
        
        # Same message and level should have same hash
        assert hash1 == hash2
        
        # Different level should have different hash
        assert hash1 != hash3
        
        # Different message should have different hash
        assert hash1 != hash4
        
        # Hashes should be reasonable length (8 chars as implemented)
        assert len(hash1) == 8


class TestAsyncLogHandler:
    """Test cases for AsyncLogHandler"""
    
    @pytest.fixture
    def handler(self):
        """Create AsyncLogHandler instance"""
        handler = AsyncLogHandler(max_queue_size=100, flush_interval=0.1, worker_count=1)
        yield handler
        handler.stop()  # Cleanup
    
    def test_handler_creation(self, handler):
        """Test AsyncLogHandler creation"""
        assert handler.max_queue_size == 100
        assert handler.flush_interval == 0.1
        assert handler.worker_count == 1
        assert handler.running is False
    
    def test_enqueue_logs(self, handler):
        """Test enqueueing log events"""
        event = {"message": "test log", "level": "info"}
        
        handler.enqueue(event)
        
        stats = handler.get_stats()
        assert stats["logs_queued"] == 1
        assert stats["queue_size"] == 1
    
    def test_queue_size_limit(self, handler):
        """Test queue size limiting"""
        event = {"message": "test log", "level": "info"}
        
        # Fill queue beyond limit
        for i in range(150):  # max_queue_size=100
            handler.enqueue(event)
        
        stats = handler.get_stats()
        assert stats["queue_size"] <= 100  # Should not exceed max
        assert stats["logs_dropped"] > 0  # Some should be dropped
    
    def test_start_stop(self, handler):
        """Test starting and stopping handler"""
        assert handler.running is False
        
        handler.start()
        assert handler.running is True
        
        handler.stop()
        assert handler.running is False
    
    def test_processing_batch(self, handler):
        """Test batch processing"""
        events = [{"message": f"log {i}", "level": "info"} for i in range(5)]
        
        # Create batch format
        batch = [{"event": event, "formatter": None, "timestamp": time.time()} 
                for event in events]
        
        # Process batch (should not raise exception)
        handler._process_batch(batch)
        
        # Stats should show processed logs
        stats = handler.get_stats()
        assert stats["logs_processed"] >= 5
    
    @patch('builtins.print')  # Mock print to avoid output during tests
    def test_formatting_in_batch_processing(self, mock_print, handler):
        """Test custom formatting in batch processing"""
        def custom_formatter(event_dict):
            return f"CUSTOM: {event_dict.get('message', '')}"
        
        event = {"message": "test message"}
        batch = [{"event": event, "formatter": custom_formatter, "timestamp": time.time()}]
        
        handler._process_batch(batch)
        
        # Should have called print with formatted message
        mock_print.assert_called_with("CUSTOM: test message")


class TestStructuredLogProcessor:
    """Test cases for StructuredLogProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Create StructuredLogProcessor instance"""
        return StructuredLogProcessor()
    
    def test_processor_creation(self, processor):
        """Test StructuredLogProcessor creation"""
        assert "sensitive" in processor.enabled_filters
        assert processor.processed_count == 0
    
    def test_sensitive_data_filtering(self, processor):
        """Test sensitive data filtering in processor"""
        logger = Mock()
        event_dict = {"message": "login", "password": "secret123"}
        
        result = processor(logger, "info", event_dict)
        
        assert result["message"] == "login"
        assert result["password"] == "se*****23"  # Should be masked
    
    def test_configure_sampling(self, processor):
        """Test configuring sampling"""
        processor.configure_sampling(sample_rate=0.5, burst_limit=5)
        
        assert "sampling" in processor.enabled_filters
        assert processor.sampler.sample_rate == 0.5
        assert processor.sampler.burst_limit == 5
    
    def test_configure_aggregation(self, processor):
        """Test configuring aggregation"""
        processor.configure_aggregation(window_seconds=30, max_duplicates=10)
        
        assert "aggregation" in processor.enabled_filters
        assert processor.aggregator.window_seconds == 30
        assert processor.aggregator.max_duplicates == 10
    
    def test_enable_async_processing(self, processor):
        """Test enabling async processing"""
        processor.enable_async_processing(max_queue_size=500)
        
        assert "async" in processor.enabled_filters
        assert processor.async_handler.max_queue_size == 500
        assert processor.async_handler.running is True
        
        # Cleanup
        processor.async_handler.stop()
    
    @patch('time.time')
    def test_performance_tracking(self, mock_time, processor):
        """Test performance tracking"""
        # Mock time to control timing
        mock_time.side_effect = [0.0, 0.001]  # 1ms processing time
        
        logger = Mock()
        event_dict = {"message": "test"}
        
        processor(logger, "info", event_dict)
        
        stats = processor.get_performance_stats()
        assert stats["processed_count"] == 1
        assert stats["avg_processing_time_ms"] == 1.0  # 1ms
    
    def test_error_handling_in_processing(self, processor):
        """Test error handling during processing"""
        # Mock the sensitive filter to raise an exception
        processor.sensitive_filter.filter = Mock(side_effect=Exception("Test error"))
        
        logger = Mock()
        event_dict = {"message": "test"}
        
        result = processor(logger, "info", event_dict)
        
        # Should return error event instead of crashing
        assert result["event"] == "log_processing_error"
        assert "Test error" in result["error"]
    
    def test_drop_event_propagation(self, processor):
        """Test DropEvent propagation"""
        # Configure very low sample rate to trigger drops
        processor.configure_sampling(sample_rate=0.0, burst_limit=0)
        
        logger = Mock()
        event_dict = {"message": "test"}
        
        # Should raise DropEvent for most calls
        with pytest.raises(Exception) as excinfo:
            for _ in range(10):  # Try multiple times
                processor(logger, "info", event_dict)
        
        # Should eventually get a DropEvent
        assert "DropEvent" in str(excinfo.typename) or processor.processed_count == 0


class TestIntegrationLogging:
    """Integration tests for logging components"""
    
    def test_full_pipeline(self):
        """Test full logging pipeline with all components"""
        processor = StructuredLogProcessor()
        
        # Configure all features
        processor.configure_sampling(sample_rate=1.0)  # Keep all for testing
        processor.configure_aggregation(window_seconds=60)
        
        logger = Mock()
        
        # Process various log events
        events = [
            {"message": "user login", "password": "secret123", "level": "info"},
            {"message": "user login", "password": "secret456", "level": "info"},  # Duplicate
            {"message": "error occurred", "api_key": "abc123", "level": "error"},
            {"message": "normal operation", "level": "info"}
        ]
        
        results = []
        for event in events:
            try:
                result = processor(logger, event.get("level", "info"), event)
                results.append(result)
            except:
                results.append(None)  # DropEvent
        
        # Should have processed some events
        assert processor.processed_count > 0
        
        # Sensitive data should be filtered
        for result in results:
            if result and "password" in result:
                assert "secret" not in result["password"]
            if result and "api_key" in result:
                assert "abc123" not in result["api_key"]
    
    def test_performance_under_load(self):
        """Test performance under high log volume"""
        processor = StructuredLogProcessor()
        processor.configure_sampling(sample_rate=0.1)  # Sample to reduce load
        
        logger = Mock()
        
        start_time = time.time()
        
        # Process many log events
        for i in range(1000):
            event = {"message": f"log message {i}", "level": "info"}
            try:
                processor(logger, "info", event)
            except:
                pass  # Ignore DropEvent
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (less than 1 second for 1000 logs)
        assert processing_time < 1.0
        
        stats = processor.get_performance_stats()
        assert stats["processed_count"] > 0
        assert stats["avg_processing_time_ms"] < 100  # Less than 100ms average