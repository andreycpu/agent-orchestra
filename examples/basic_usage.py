"""
Basic usage example for Agent Orchestra
"""
import asyncio
import structlog
from agent_orchestra import Orchestra, Agent

# Configure structured logging
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

logger = structlog.get_logger(__name__)


async def text_processor(data):
    """Example task handler for text processing"""
    text = data.get("text", "")
    operation = data.get("operation", "upper")
    
    logger.info("Processing text", text=text, operation=operation)
    
    # Simulate some processing time
    await asyncio.sleep(0.5)
    
    if operation == "upper":
        result = text.upper()
    elif operation == "lower":
        result = text.lower()
    elif operation == "reverse":
        result = text[::-1]
    else:
        result = text
    
    return {"processed_text": result, "original": text, "operation": operation}


async def math_processor(data):
    """Example task handler for math operations"""
    numbers = data.get("numbers", [])
    operation = data.get("operation", "sum")
    
    logger.info("Processing math", numbers=numbers, operation=operation)
    
    # Simulate processing
    await asyncio.sleep(0.2)
    
    if operation == "sum":
        result = sum(numbers)
    elif operation == "product":
        result = 1
        for n in numbers:
            result *= n
    elif operation == "average":
        result = sum(numbers) / len(numbers) if numbers else 0
    else:
        result = 0
    
    return {"result": result, "numbers": numbers, "operation": operation}


async def main():
    """Main example function"""
    logger.info("Starting Agent Orchestra example")
    
    # Create orchestra
    orchestra = Orchestra(
        max_concurrent_tasks=10,
        task_timeout_default=60
    )
    
    # Create agents with different capabilities
    text_agent = Agent("text-worker-1", capabilities=["text_processing"])
    text_agent.register_task_handler("text_processing", text_processor)
    
    math_agent = Agent("math-worker-1", capabilities=["math_processing"])  
    math_agent.register_task_handler("math_processing", math_processor)
    
    # Register agents
    orchestra.register_agent(text_agent)
    orchestra.register_agent(math_agent)
    
    # Start orchestra
    await orchestra.start()
    
    try:
        logger.info("Orchestra started, submitting tasks")
        
        # Submit various tasks
        tasks = []
        
        # Text processing tasks
        for i in range(5):
            task_id = await orchestra.submit_task({
                "type": "text_processing",
                "data": {
                    "text": f"Hello World {i}",
                    "operation": "upper" if i % 2 == 0 else "reverse"
                },
                "priority": "normal"
            })
            tasks.append(task_id)
            logger.info("Submitted text task", task_id=task_id, index=i)
        
        # Math processing tasks  
        for i in range(3):
            task_id = await orchestra.submit_task({
                "type": "math_processing", 
                "data": {
                    "numbers": [i, i+1, i+2],
                    "operation": "sum"
                },
                "priority": "high"
            })
            tasks.append(task_id)
            logger.info("Submitted math task", task_id=task_id, index=i)
        
        # Wait for all tasks to complete
        logger.info("Waiting for tasks to complete", total_tasks=len(tasks))
        
        results = []
        for task_id in tasks:
            try:
                result = await orchestra.wait_for_task(task_id, timeout=30)
                results.append(result)
                logger.info("Task completed", 
                           task_id=task_id, 
                           success=result.success,
                           execution_time=result.execution_time)
                
                if result.success:
                    logger.info("Task result", result=result.result)
                else:
                    logger.error("Task failed", error=result.error)
                    
            except asyncio.TimeoutError:
                logger.error("Task timed out", task_id=task_id)
        
        # Get orchestra status
        status = await orchestra.get_status()
        logger.info("Orchestra status", status=status)
        
        # Print summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_time = sum(r.execution_time for r in results)
        
        logger.info("Execution summary",
                   total_tasks=len(results),
                   successful=successful,
                   failed=failed,
                   total_execution_time=total_time,
                   average_time=total_time / len(results) if results else 0)
    
    finally:
        # Stop orchestra
        await orchestra.stop()
        logger.info("Orchestra stopped")


if __name__ == "__main__":
    asyncio.run(main())