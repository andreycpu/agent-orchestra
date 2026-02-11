"""
Advanced example demonstrating parallel execution and task dependencies
"""
import asyncio
import random
import time
from typing import Dict, Any
import structlog
from agent_orchestra import Orchestra, Agent
from agent_orchestra.types import TaskPriority

logger = structlog.get_logger(__name__)


async def data_fetcher(data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate data fetching from external sources"""
    source = data.get("source", "unknown")
    delay = random.uniform(0.5, 2.0)  # Random delay to simulate network
    
    logger.info("Fetching data", source=source, delay=delay)
    await asyncio.sleep(delay)
    
    # Simulate different data sources
    if source == "database":
        result = {"records": [{"id": i, "value": f"record_{i}"} for i in range(10)]}
    elif source == "api":
        result = {"data": {"timestamp": time.time(), "status": "active"}}
    elif source == "file":
        result = {"content": f"File content from {source}", "size": 1024}
    else:
        result = {"message": f"Data from {source}"}
    
    return {"source": source, "data": result, "fetch_time": delay}


async def data_processor(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process fetched data"""
    input_data = data.get("input_data", {})
    processing_type = data.get("processing_type", "transform")
    
    logger.info("Processing data", 
                processing_type=processing_type,
                data_size=len(str(input_data)))
    
    # Simulate CPU-intensive processing
    await asyncio.sleep(random.uniform(0.3, 1.0))
    
    if processing_type == "transform":
        result = {
            "transformed": True,
            "original_size": len(str(input_data)),
            "processed_at": time.time()
        }
    elif processing_type == "aggregate":
        result = {
            "summary": "Aggregated data",
            "count": len(str(input_data)),
            "hash": hash(str(input_data))
        }
    else:
        result = {"processed": input_data, "type": processing_type}
    
    return result


async def data_validator(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate processed data"""
    processed_data = data.get("processed_data", {})
    validation_rules = data.get("rules", ["not_empty", "valid_structure"])
    
    logger.info("Validating data", rules=validation_rules)
    
    # Simulate validation time
    await asyncio.sleep(random.uniform(0.1, 0.5))
    
    errors = []
    warnings = []
    
    # Simple validation logic
    if "not_empty" in validation_rules and not processed_data:
        errors.append("Data is empty")
    
    if "valid_structure" in validation_rules:
        if not isinstance(processed_data, dict):
            errors.append("Data is not a dictionary")
    
    # Random validation issues for demonstration
    if random.random() < 0.1:  # 10% chance of validation error
        errors.append("Random validation failure for demonstration")
    
    if random.random() < 0.2:  # 20% chance of warning
        warnings.append("Data quality warning")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "validated_data": processed_data
    }


async def report_generator(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate final report from validated data"""
    validation_results = data.get("validation_results", [])
    
    logger.info("Generating report", validation_count=len(validation_results))
    
    # Simulate report generation
    await asyncio.sleep(random.uniform(0.5, 1.5))
    
    total_records = len(validation_results)
    valid_records = sum(1 for r in validation_results if r.get("valid", False))
    error_count = sum(len(r.get("errors", [])) for r in validation_results)
    warning_count = sum(len(r.get("warnings", [])) for r in validation_results)
    
    report = {
        "timestamp": time.time(),
        "total_records": total_records,
        "valid_records": valid_records,
        "error_count": error_count,
        "warning_count": warning_count,
        "success_rate": valid_records / total_records if total_records > 0 else 0,
        "summary": f"Processed {total_records} records with {valid_records} valid"
    }
    
    return report


async def run_parallel_pipeline():
    """Run a complex parallel data processing pipeline"""
    logger.info("Starting parallel pipeline example")
    
    # Create orchestra with higher concurrency
    orchestra = Orchestra(
        max_concurrent_tasks=50,
        task_timeout_default=120
    )
    
    # Create specialized agents
    agents = [
        Agent("fetcher-1", capabilities=["data_fetching"]),
        Agent("fetcher-2", capabilities=["data_fetching"]),
        Agent("processor-1", capabilities=["data_processing"]),
        Agent("processor-2", capabilities=["data_processing"]),
        Agent("processor-3", capabilities=["data_processing"]),
        Agent("validator-1", capabilities=["data_validation"]),
        Agent("validator-2", capabilities=["data_validation"]),
        Agent("reporter-1", capabilities=["report_generation"])
    ]
    
    # Register task handlers
    for agent in agents:
        if "data_fetching" in [cap.name for cap in agent.capabilities]:
            agent.register_task_handler("data_fetching", data_fetcher)
        if "data_processing" in [cap.name for cap in agent.capabilities]:
            agent.register_task_handler("data_processing", data_processor)
        if "data_validation" in [cap.name for cap in agent.capabilities]:
            agent.register_task_handler("data_validation", data_validator)
        if "report_generation" in [cap.name for cap in agent.capabilities]:
            agent.register_task_handler("report_generation", report_generator)
    
    # Register all agents
    for agent in agents:
        orchestra.register_agent(agent)
    
    # Start orchestra
    await orchestra.start()
    
    try:
        # Phase 1: Parallel data fetching
        logger.info("Phase 1: Starting parallel data fetching")
        
        data_sources = ["database", "api", "file", "cache", "external_service"]
        fetch_tasks = []
        
        for i, source in enumerate(data_sources * 2):  # Fetch from each source twice
            task_id = await orchestra.submit_task({
                "type": "data_fetching",
                "data": {"source": f"{source}_{i}"},
                "priority": TaskPriority.HIGH if i < 5 else TaskPriority.NORMAL
            })
            fetch_tasks.append(task_id)
        
        # Wait for all fetching tasks
        logger.info("Waiting for fetch tasks to complete", count=len(fetch_tasks))
        fetch_results = []
        
        for task_id in fetch_tasks:
            result = await orchestra.wait_for_task(task_id, timeout=30)
            if result.success:
                fetch_results.append(result.result)
                logger.info("Data fetched successfully", task_id=task_id)
            else:
                logger.error("Data fetch failed", task_id=task_id, error=result.error)
        
        # Phase 2: Parallel data processing
        logger.info("Phase 2: Starting parallel data processing")
        
        processing_tasks = []
        for i, fetch_result in enumerate(fetch_results):
            processing_type = "transform" if i % 2 == 0 else "aggregate"
            
            task_id = await orchestra.submit_task({
                "type": "data_processing", 
                "data": {
                    "input_data": fetch_result,
                    "processing_type": processing_type
                },
                "priority": TaskPriority.NORMAL
            })
            processing_tasks.append(task_id)
        
        # Wait for processing
        logger.info("Waiting for processing tasks", count=len(processing_tasks))
        processing_results = []
        
        for task_id in processing_tasks:
            result = await orchestra.wait_for_task(task_id, timeout=30)
            if result.success:
                processing_results.append(result.result)
                logger.info("Data processed successfully", task_id=task_id)
            else:
                logger.error("Data processing failed", task_id=task_id, error=result.error)
        
        # Phase 3: Parallel validation
        logger.info("Phase 3: Starting parallel data validation")
        
        validation_tasks = []
        for i, processed_result in enumerate(processing_results):
            rules = ["not_empty", "valid_structure"]
            if i % 3 == 0:
                rules.append("strict_validation")
            
            task_id = await orchestra.submit_task({
                "type": "data_validation",
                "data": {
                    "processed_data": processed_result,
                    "rules": rules
                },
                "priority": TaskPriority.NORMAL
            })
            validation_tasks.append(task_id)
        
        # Wait for validation
        logger.info("Waiting for validation tasks", count=len(validation_tasks))
        validation_results = []
        
        for task_id in validation_tasks:
            result = await orchestra.wait_for_task(task_id, timeout=30)
            if result.success:
                validation_results.append(result.result)
                logger.info("Data validated successfully", task_id=task_id)
            else:
                logger.error("Data validation failed", task_id=task_id, error=result.error)
        
        # Phase 4: Generate final report
        logger.info("Phase 4: Generating final report")
        
        report_task_id = await orchestra.submit_task({
            "type": "report_generation",
            "data": {"validation_results": validation_results},
            "priority": TaskPriority.HIGH
        })
        
        report_result = await orchestra.wait_for_task(report_task_id, timeout=30)
        
        if report_result.success:
            logger.info("Pipeline completed successfully", 
                       report=report_result.result)
            print("\n=== PIPELINE RESULTS ===")
            print(f"Final Report: {report_result.result}")
        else:
            logger.error("Report generation failed", error=report_result.error)
        
        # Get final status
        status = await orchestra.get_status()
        logger.info("Final orchestra status", status=status)
        
    finally:
        await orchestra.stop()


if __name__ == "__main__":
    # Configure logging
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
    
    asyncio.run(run_parallel_pipeline())