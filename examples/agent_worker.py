"""
Standalone agent worker that connects to Orchestra
"""
import asyncio
import argparse
import signal
import sys
from typing import Dict, Any
import structlog

from agent_orchestra import Agent, Orchestra
from agent_orchestra.config import ConfigurationManager

logger = structlog.get_logger(__name__)


class WorkerAgent:
    """Standalone worker agent"""
    
    def __init__(self, agent_id: str, capabilities: list, redis_url: str = None):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.redis_url = redis_url
        self.orchestra = None
        self.agent = None
        self.running = False
    
    async def start(self):
        """Start the worker agent"""
        logger.info("Starting worker agent", 
                   agent_id=self.agent_id, 
                   capabilities=self.capabilities)
        
        # Create orchestra connection
        self.orchestra = Orchestra(redis_url=self.redis_url)
        
        # Create agent
        self.agent = Agent(
            agent_id=self.agent_id,
            name=f"Worker Agent {self.agent_id}",
            capabilities=self.capabilities
        )
        
        # Register task handlers based on capabilities
        self._register_task_handlers()
        
        # Register with orchestra
        self.orchestra.register_agent(self.agent)
        
        # Start orchestra (in worker mode, it just connects)
        await self.orchestra.start()
        
        self.running = True
        logger.info("Worker agent started successfully")
        
        # Keep running until stopped
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the worker agent"""
        logger.info("Stopping worker agent")
        self.running = False
        
        if self.orchestra:
            if self.agent:
                self.orchestra.unregister_agent(self.agent_id)
            await self.orchestra.stop()
        
        logger.info("Worker agent stopped")
    
    def _register_task_handlers(self):
        """Register task handlers based on capabilities"""
        for capability in self.capabilities:
            if capability == "text_processing":
                self.agent.register_task_handler("text_processing", self._handle_text_processing)
            elif capability == "data_processing":
                self.agent.register_task_handler("data_processing", self._handle_data_processing)
            elif capability == "math_processing":
                self.agent.register_task_handler("math_processing", self._handle_math_processing)
            elif capability == "image_processing":
                self.agent.register_task_handler("image_processing", self._handle_image_processing)
            elif capability == "content_analysis":
                self.agent.register_task_handler("content_analysis", self._handle_content_analysis)
            elif capability == "data_validation":
                self.agent.register_task_handler("data_validation", self._handle_data_validation)
            elif capability == "report_generation":
                self.agent.register_task_handler("report_generation", self._handle_report_generation)
            else:
                logger.warning("Unknown capability", capability=capability)
    
    async def _handle_text_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text processing tasks"""
        text = data.get("text", "")
        operation = data.get("operation", "analyze")
        
        logger.info("Processing text", 
                   agent_id=self.agent_id,
                   operation=operation,
                   text_length=len(text))
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        if operation == "upper":
            result = text.upper()
        elif operation == "lower":
            result = text.lower()
        elif operation == "reverse":
            result = text[::-1]
        elif operation == "word_count":
            result = len(text.split())
        elif operation == "analyze":
            words = text.split()
            result = {
                "word_count": len(words),
                "char_count": len(text),
                "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
                "uppercase_words": sum(1 for w in words if w.isupper()),
                "lowercase_words": sum(1 for w in words if w.islower())
            }
        else:
            result = f"Unknown operation: {operation}"
        
        return {
            "result": result,
            "original_text": text,
            "operation": operation,
            "processed_by": self.agent_id
        }
    
    async def _handle_data_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data processing tasks"""
        input_data = data.get("data", [])
        operation = data.get("operation", "analyze")
        
        logger.info("Processing data",
                   agent_id=self.agent_id,
                   operation=operation,
                   data_size=len(input_data) if isinstance(input_data, (list, dict)) else 0)
        
        # Simulate processing time
        await asyncio.sleep(0.2)
        
        if operation == "filter":
            condition = data.get("condition", lambda x: True)
            if isinstance(input_data, list):
                result = [item for item in input_data if condition(item)]
            else:
                result = input_data
        
        elif operation == "transform":
            transform_func = data.get("transform", lambda x: x)
            if isinstance(input_data, list):
                result = [transform_func(item) for item in input_data]
            else:
                result = transform_func(input_data)
        
        elif operation == "aggregate":
            if isinstance(input_data, list) and all(isinstance(x, (int, float)) for x in input_data):
                result = {
                    "sum": sum(input_data),
                    "average": sum(input_data) / len(input_data) if input_data else 0,
                    "min": min(input_data) if input_data else None,
                    "max": max(input_data) if input_data else None,
                    "count": len(input_data)
                }
            else:
                result = {"count": len(input_data) if hasattr(input_data, '__len__') else 1}
        
        else:
            result = {"error": f"Unknown operation: {operation}"}
        
        return {
            "result": result,
            "operation": operation,
            "input_size": len(input_data) if hasattr(input_data, '__len__') else 1,
            "processed_by": self.agent_id
        }
    
    async def _handle_math_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mathematical processing tasks"""
        numbers = data.get("numbers", [])
        operation = data.get("operation", "sum")
        
        logger.info("Processing math",
                   agent_id=self.agent_id,
                   operation=operation,
                   number_count=len(numbers))
        
        # Simulate processing time
        await asyncio.sleep(0.05)
        
        if not numbers:
            return {"error": "No numbers provided", "processed_by": self.agent_id}
        
        if operation == "sum":
            result = sum(numbers)
        elif operation == "product":
            result = 1
            for n in numbers:
                result *= n
        elif operation == "average":
            result = sum(numbers) / len(numbers)
        elif operation == "statistics":
            sorted_nums = sorted(numbers)
            n = len(sorted_nums)
            result = {
                "sum": sum(numbers),
                "average": sum(numbers) / n,
                "median": sorted_nums[n // 2] if n % 2 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2,
                "min": min(numbers),
                "max": max(numbers),
                "range": max(numbers) - min(numbers),
                "count": n
            }
        else:
            result = {"error": f"Unknown operation: {operation}"}
        
        return {
            "result": result,
            "operation": operation,
            "input_numbers": numbers,
            "processed_by": self.agent_id
        }
    
    async def _handle_image_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle image processing tasks (simulated)"""
        image_path = data.get("image_path", "")
        operation = data.get("operation", "analyze")
        
        logger.info("Processing image",
                   agent_id=self.agent_id,
                   operation=operation,
                   image_path=image_path)
        
        # Simulate image processing time
        await asyncio.sleep(0.5)
        
        # Simulated image processing results
        if operation == "analyze":
            result = {
                "width": 1920,
                "height": 1080,
                "format": "JPEG",
                "size_bytes": 245760,
                "dominant_colors": ["#FF5733", "#33FF57", "#3357FF"],
                "objects_detected": ["person", "car", "tree"],
                "confidence_scores": [0.95, 0.87, 0.76]
            }
        elif operation == "resize":
            target_width = data.get("width", 800)
            target_height = data.get("height", 600)
            result = {
                "original_size": [1920, 1080],
                "new_size": [target_width, target_height],
                "operation": "resize_complete"
            }
        elif operation == "filter":
            filter_type = data.get("filter", "blur")
            result = {
                "filter_applied": filter_type,
                "operation": "filter_complete"
            }
        else:
            result = {"error": f"Unknown operation: {operation}"}
        
        return {
            "result": result,
            "operation": operation,
            "image_path": image_path,
            "processed_by": self.agent_id
        }
    
    async def _handle_content_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content analysis tasks"""
        content = data.get("content", "")
        analysis_type = data.get("analysis_type", "sentiment")
        
        logger.info("Analyzing content",
                   agent_id=self.agent_id,
                   analysis_type=analysis_type,
                   content_length=len(content))
        
        # Simulate analysis time
        await asyncio.sleep(0.3)
        
        if analysis_type == "sentiment":
            # Simulated sentiment analysis
            positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
            negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
            
            content_lower = content.lower()
            positive_count = sum(word in content_lower for word in positive_words)
            negative_count = sum(word in content_lower for word in negative_words)
            
            if positive_count > negative_count:
                sentiment = "positive"
                score = 0.7 + (positive_count - negative_count) * 0.1
            elif negative_count > positive_count:
                sentiment = "negative"
                score = 0.3 - (negative_count - positive_count) * 0.1
            else:
                sentiment = "neutral"
                score = 0.5
            
            result = {
                "sentiment": sentiment,
                "confidence": max(0.0, min(1.0, score)),
                "positive_indicators": positive_count,
                "negative_indicators": negative_count
            }
        
        elif analysis_type == "keywords":
            words = content.split()
            word_freq = {}
            for word in words:
                clean_word = word.lower().strip('.,!?";')
                if len(clean_word) > 3:  # Skip short words
                    word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
            
            # Get top keywords
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            result = {
                "top_keywords": [{"word": word, "frequency": freq} for word, freq in top_keywords],
                "total_words": len(words),
                "unique_words": len(word_freq)
            }
        
        else:
            result = {"error": f"Unknown analysis type: {analysis_type}"}
        
        return {
            "result": result,
            "analysis_type": analysis_type,
            "content_length": len(content),
            "processed_by": self.agent_id
        }
    
    async def _handle_data_validation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data validation tasks"""
        input_data = data.get("data", {})
        rules = data.get("rules", [])
        
        logger.info("Validating data",
                   agent_id=self.agent_id,
                   rule_count=len(rules),
                   data_size=len(str(input_data)))
        
        # Simulate validation time
        await asyncio.sleep(0.15)
        
        validation_results = []
        errors = []
        warnings = []
        
        for rule in rules:
            rule_name = rule.get("name", "unknown")
            rule_type = rule.get("type", "required")
            field = rule.get("field", "")
            
            if rule_type == "required":
                if field not in input_data or input_data[field] is None:
                    errors.append(f"Required field '{field}' is missing")
                else:
                    validation_results.append(f"Required field '{field}' is present")
            
            elif rule_type == "type_check":
                expected_type = rule.get("expected_type", str)
                if field in input_data:
                    actual_type = type(input_data[field])
                    if actual_type != expected_type:
                        errors.append(f"Field '{field}' should be {expected_type.__name__}, got {actual_type.__name__}")
                    else:
                        validation_results.append(f"Field '{field}' has correct type")
            
            elif rule_type == "range_check":
                if field in input_data:
                    value = input_data[field]
                    min_val = rule.get("min")
                    max_val = rule.get("max")
                    
                    if min_val is not None and value < min_val:
                        errors.append(f"Field '{field}' value {value} is below minimum {min_val}")
                    elif max_val is not None and value > max_val:
                        errors.append(f"Field '{field}' value {value} is above maximum {max_val}")
                    else:
                        validation_results.append(f"Field '{field}' is within valid range")
        
        # Overall validation result
        is_valid = len(errors) == 0
        
        result = {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "validation_results": validation_results,
            "rules_checked": len(rules)
        }
        
        return {
            "result": result,
            "data_validated": True,
            "processed_by": self.agent_id
        }
    
    async def _handle_report_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle report generation tasks"""
        report_type = data.get("report_type", "summary")
        input_data = data.get("data", [])
        
        logger.info("Generating report",
                   agent_id=self.agent_id,
                   report_type=report_type,
                   data_items=len(input_data) if isinstance(input_data, list) else 1)
        
        # Simulate report generation time
        await asyncio.sleep(0.4)
        
        if report_type == "summary":
            if isinstance(input_data, list):
                total_items = len(input_data)
                processed_items = sum(1 for item in input_data if isinstance(item, dict) and item.get("processed"))
                error_items = sum(1 for item in input_data if isinstance(item, dict) and item.get("error"))
                
                result = {
                    "report_title": "Data Processing Summary",
                    "total_items": total_items,
                    "processed_successfully": processed_items,
                    "errors": error_items,
                    "success_rate": processed_items / total_items if total_items > 0 else 0,
                    "generated_at": asyncio.get_event_loop().time()
                }
            else:
                result = {
                    "report_title": "Simple Summary",
                    "data_type": type(input_data).__name__,
                    "data_length": len(str(input_data))
                }
        
        elif report_type == "detailed":
            result = {
                "report_title": "Detailed Analysis Report",
                "sections": [
                    {
                        "title": "Data Overview",
                        "content": f"Analyzed {len(input_data) if hasattr(input_data, '__len__') else 1} data items"
                    },
                    {
                        "title": "Processing Details",
                        "content": "All items processed according to configured rules and parameters"
                    },
                    {
                        "title": "Recommendations",
                        "content": "Continue monitoring data quality and processing performance"
                    }
                ],
                "metadata": {
                    "generated_by": self.agent_id,
                    "generation_time": 0.4,
                    "format_version": "1.0"
                }
            }
        
        else:
            result = {"error": f"Unknown report type: {report_type}"}
        
        return {
            "result": result,
            "report_type": report_type,
            "generated_by": self.agent_id
        }


async def main():
    """Main worker agent entry point"""
    parser = argparse.ArgumentParser(description="Agent Orchestra Worker")
    
    parser.add_argument("--agent-id", required=True, help="Unique agent identifier")
    parser.add_argument("--capabilities", nargs="+", required=True, 
                       help="Agent capabilities")
    parser.add_argument("--redis-url", help="Redis connection URL")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Load configuration if provided
    redis_url = args.redis_url
    if args.config:
        config_manager = ConfigurationManager()
        if config_manager.load(args.config):
            redis_url = redis_url or config_manager.get_redis_url()
    
    # Create and start worker
    worker = WorkerAgent(
        agent_id=args.agent_id,
        capabilities=args.capabilities,
        redis_url=redis_url
    )
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        worker.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await worker.start()
    except Exception as e:
        logger.error("Worker agent failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())