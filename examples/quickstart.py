"""
Quick start example for Agent Orchestra
"""
import asyncio
from typing import Dict, Any

from agent_orchestra import Orchestra, Agent
from agent_orchestra.types import Task, TaskPriority


async def simple_text_processor(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simple text processing function"""
    text = task_data.get("text", "")
    
    # Simulate some processing
    await asyncio.sleep(0.1)
    
    return {
        "original_text": text,
        "word_count": len(text.split()),
        "char_count": len(text),
        "uppercase": text.upper(),
        "status": "processed"
    }


async def simple_calculator(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simple calculator function"""
    operation = task_data.get("operation", "add")
    numbers = task_data.get("numbers", [])
    
    if not numbers:
        raise ValueError("No numbers provided")
    
    # Simulate some processing
    await asyncio.sleep(0.05)
    
    if operation == "add":
        result = sum(numbers)
    elif operation == "multiply":
        result = 1
        for num in numbers:
            result *= num
    elif operation == "average":
        result = sum(numbers) / len(numbers)
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return {
        "operation": operation,
        "numbers": numbers,
        "result": result,
        "status": "calculated"
    }


async def main():
    """Main function demonstrating Agent Orchestra usage"""
    print("🎭 Agent Orchestra Quickstart Example")
    print("=" * 50)
    
    # Create and start the orchestra
    async with Orchestra() as orchestra:
        print("✅ Orchestra started")
        
        # Create agents
        text_agent = Agent(
            agent_id="text_agent",
            name="Text Processing Agent",
            capabilities=["text_processing"]
        )
        
        calc_agent = Agent(
            agent_id="calc_agent", 
            name="Calculator Agent",
            capabilities=["math_operations"]
        )
        
        # Register task handlers
        text_agent.register_handler("text_processing", simple_text_processor)
        calc_agent.register_handler("math_operations", simple_calculator)
        
        # Register agents with orchestra
        orchestra.register_agent(text_agent)
        orchestra.register_agent(calc_agent)
        
        print(f"✅ Registered {len([text_agent, calc_agent])} agents")
        
        # Submit some tasks
        tasks = []
        
        # Text processing tasks
        for i, text in enumerate([
            "Hello, world!",
            "Agent Orchestra is a powerful framework",
            "This is a longer text that contains more words for analysis"
        ]):
            task = Task(
                type="text_processing",
                data={"text": text},
                priority=TaskPriority.NORMAL
            )
            task_id = await orchestra.submit_task(task.__dict__)
            tasks.append(("text", task_id, text))
            print(f"📝 Submitted text task {i+1}: '{text[:30]}...'")
        
        # Math operation tasks
        for i, (operation, numbers) in enumerate([
            ("add", [1, 2, 3, 4, 5]),
            ("multiply", [2, 3, 4]),
            ("average", [10, 20, 30, 40, 50])
        ]):
            task = Task(
                type="math_operations",
                data={"operation": operation, "numbers": numbers},
                priority=TaskPriority.HIGH
            )
            task_id = await orchestra.submit_task(task.__dict__)
            tasks.append(("math", task_id, f"{operation}({numbers})"))
            print(f"🔢 Submitted math task {i+1}: {operation}({numbers})")
        
        print(f"\n⏳ Processing {len(tasks)} tasks...")
        
        # Wait for and display results
        success_count = 0
        for task_type, task_id, description in tasks:
            try:
                result = await orchestra.wait_for_task(task_id, timeout=10.0)
                
                if result.success:
                    success_count += 1
                    print(f"\n✅ {task_type.upper()} TASK COMPLETED")
                    print(f"   Input: {description}")
                    
                    if task_type == "text":
                        data = result.result
                        print(f"   Word count: {data.get('word_count')}")
                        print(f"   Character count: {data.get('char_count')}")
                        print(f"   Uppercase: {data.get('uppercase', '')[:50]}...")
                    
                    elif task_type == "math":
                        data = result.result
                        print(f"   Operation: {data.get('operation')}")
                        print(f"   Numbers: {data.get('numbers')}")
                        print(f"   Result: {data.get('result')}")
                    
                    print(f"   Execution time: {result.execution_time:.3f}s")
                else:
                    print(f"\n❌ {task_type.upper()} TASK FAILED")
                    print(f"   Input: {description}")
                    print(f"   Error: {result.error}")
                    
            except Exception as e:
                print(f"\n💥 TASK ERROR")
                print(f"   Task: {description}")
                print(f"   Error: {str(e)}")
        
        print(f"\n🎯 SUMMARY")
        print(f"   Total tasks: {len(tasks)}")
        print(f"   Successful: {success_count}")
        print(f"   Failed: {len(tasks) - success_count}")
        print(f"   Success rate: {success_count/len(tasks)*100:.1f}%")
        
        if success_count == len(tasks):
            print("🎉 All tasks completed successfully!")
        else:
            print("⚠️  Some tasks failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())