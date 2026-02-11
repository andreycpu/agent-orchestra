"""
Command line interface for Agent Orchestra
"""
import asyncio
import sys
import argparse
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
import structlog

from .orchestra import Orchestra
from .agent import Agent
from .config import ConfigurationManager
from .monitoring import OrchestrationMonitor

logger = structlog.get_logger(__name__)


class OrchestrationCLI:
    """Command line interface for managing Agent Orchestra"""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.orchestra: Optional[Orchestra] = None
        self.monitor: Optional[OrchestrationMonitor] = None
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command line argument parser"""
        parser = argparse.ArgumentParser(
            description="Agent Orchestra - Multi-agent orchestration framework",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        parser.add_argument(
            "--config", "-c",
            help="Configuration file path (YAML or JSON)",
            default=None
        )
        
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="Log level"
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Start command
        start_parser = subparsers.add_parser("start", help="Start the orchestra")
        start_parser.add_argument(
            "--daemon", "-d",
            action="store_true",
            help="Run as daemon"
        )
        start_parser.add_argument(
            "--monitor",
            action="store_true",
            help="Enable monitoring dashboard"
        )
        
        # Stop command
        subparsers.add_parser("stop", help="Stop the orchestra")
        
        # Status command
        subparsers.add_parser("status", help="Show orchestra status")
        
        # Submit task command
        task_parser = subparsers.add_parser("task", help="Submit a task")
        task_parser.add_argument("type", help="Task type")
        task_parser.add_argument(
            "--data", "-d",
            help="Task data as JSON string",
            default="{}"
        )
        task_parser.add_argument(
            "--priority", "-p",
            choices=["low", "normal", "high", "urgent"],
            default="normal",
            help="Task priority"
        )
        task_parser.add_argument(
            "--wait", "-w",
            action="store_true",
            help="Wait for task completion"
        )
        task_parser.add_argument(
            "--timeout", "-t",
            type=int,
            default=300,
            help="Task timeout in seconds"
        )
        
        # Agent management
        agent_parser = subparsers.add_parser("agent", help="Agent management")
        agent_subparsers = agent_parser.add_subparsers(dest="agent_action")
        
        # List agents
        agent_subparsers.add_parser("list", help="List registered agents")
        
        # Add agent
        add_agent_parser = agent_subparsers.add_parser("add", help="Add an agent")
        add_agent_parser.add_argument("agent_id", help="Agent ID")
        add_agent_parser.add_argument(
            "--name", "-n",
            help="Agent name"
        )
        add_agent_parser.add_argument(
            "--capabilities", "-c",
            nargs="+",
            help="Agent capabilities",
            default=[]
        )
        
        # Remove agent
        remove_agent_parser = agent_subparsers.add_parser("remove", help="Remove an agent")
        remove_agent_parser.add_argument("agent_id", help="Agent ID")
        
        # Monitor command
        monitor_parser = subparsers.add_parser("monitor", help="Show monitoring dashboard")
        monitor_parser.add_argument(
            "--refresh", "-r",
            type=int,
            default=5,
            help="Refresh interval in seconds"
        )
        monitor_parser.add_argument(
            "--continuous", "-c",
            action="store_true",
            help="Continuous monitoring"
        )
        
        # Config commands
        config_parser = subparsers.add_parser("config", help="Configuration management")
        config_subparsers = config_parser.add_subparsers(dest="config_action")
        
        # Validate config
        validate_parser = config_subparsers.add_parser("validate", help="Validate configuration")
        validate_parser.add_argument("config_file", help="Configuration file to validate")
        
        # Generate config
        generate_parser = config_subparsers.add_parser("generate", help="Generate sample configuration")
        generate_parser.add_argument("output_file", help="Output configuration file")
        generate_parser.add_argument(
            "--format",
            choices=["yaml", "json"],
            default="yaml",
            help="Configuration format"
        )
        
        # Show config
        config_subparsers.add_parser("show", help="Show current configuration")
        
        return parser
    
    async def run(self, args: argparse.Namespace):
        """Run the CLI command"""
        # Load configuration
        if not self.config_manager.load(args.config):
            logger.error("Failed to load configuration")
            sys.exit(1)
        
        # Setup logging
        self._setup_logging(args.log_level)
        
        try:
            if args.command == "start":
                await self._start_orchestra(args)
            elif args.command == "stop":
                await self._stop_orchestra()
            elif args.command == "status":
                await self._show_status()
            elif args.command == "task":
                await self._submit_task(args)
            elif args.command == "agent":
                await self._handle_agent_command(args)
            elif args.command == "monitor":
                await self._show_monitor(args)
            elif args.command == "config":
                await self._handle_config_command(args)
            else:
                logger.error("Unknown command", command=args.command)
                sys.exit(1)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            if self.orchestra:
                await self.orchestra.stop()
        except Exception as e:
            logger.error("Command failed", error=str(e))
            sys.exit(1)
    
    def _setup_logging(self, log_level: str):
        """Setup structured logging"""
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
    
    async def _start_orchestra(self, args: argparse.Namespace):
        """Start the orchestra"""
        logger.info("Starting Agent Orchestra")
        
        # Create orchestra with configuration
        config = self.config_manager.orchestra
        redis_url = self.config_manager.get_redis_url()
        
        self.orchestra = Orchestra(
            redis_url=redis_url,
            max_concurrent_tasks=config.max_concurrent_tasks,
            task_timeout_default=config.task_timeout_default,
            heartbeat_interval=config.heartbeat_interval
        )
        
        # Setup monitoring if enabled
        if args.monitor or self.config_manager.monitoring.enabled:
            self.monitor = OrchestrationMonitor()
            await self.monitor.start_monitoring()
        
        # Register configured agents
        agent_configs = self.config_manager.get_agent_configs()
        for agent_config in agent_configs:
            agent = Agent(
                agent_id=agent_config.id,
                name=agent_config.name,
                capabilities=agent_config.capabilities,
                metadata=agent_config.metadata
            )
            self.orchestra.register_agent(agent)
            logger.info("Registered configured agent", agent_id=agent.id)
        
        # Start orchestra
        await self.orchestra.start()
        
        logger.info("Orchestra started successfully")
        
        if not args.daemon:
            # Run interactively
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
        
        # Cleanup
        if self.monitor:
            await self.monitor.stop_monitoring()
        await self.orchestra.stop()
    
    async def _stop_orchestra(self):
        """Stop the orchestra"""
        if self.orchestra:
            logger.info("Stopping orchestra")
            await self.orchestra.stop()
            logger.info("Orchestra stopped")
        else:
            logger.warning("Orchestra is not running")
    
    async def _show_status(self):
        """Show orchestra status"""
        if not self.orchestra:
            print("Orchestra is not running")
            return
        
        status = await self.orchestra.get_status()
        
        print("\n=== ORCHESTRA STATUS ===")
        print(f"Running: {status['is_running']}")
        print(f"Running Tasks: {status['running_tasks']}")
        print(f"Max Concurrent Tasks: {status['max_concurrent_tasks']}")
        print(f"Registered Agents: {status['registered_agents']}")
        
        if 'global_state' in status:
            global_state = status['global_state']
            print(f"\n=== GLOBAL STATE ===")
            print(f"Total Agents: {global_state['agents']['total']}")
            print(f"Idle Agents: {global_state['agents']['idle']}")
            print(f"Busy Agents: {global_state['agents']['busy']}")
            print(f"Pending Tasks: {global_state['tasks']['pending']}")
            print(f"Running Tasks: {global_state['tasks']['running']}")
        
        if 'failure_statistics' in status:
            failure_stats = status['failure_statistics']
            print(f"\n=== FAILURE STATISTICS ===")
            print(f"Total Failures: {failure_stats['total_failures']}")
            if 'failure_types' in failure_stats:
                for failure_type, count in failure_stats['failure_types'].items():
                    print(f"  {failure_type}: {count}")
    
    async def _submit_task(self, args: argparse.Namespace):
        """Submit a task"""
        if not self.orchestra:
            print("Orchestra is not running. Please start it first.")
            return
        
        try:
            task_data = json.loads(args.data)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in task data", error=str(e))
            return
        
        task_request = {
            "type": args.type,
            "data": task_data,
            "priority": args.priority,
            "timeout": args.timeout
        }
        
        logger.info("Submitting task", task_request=task_request)
        
        task_id = await self.orchestra.submit_task(task_request)
        print(f"Task submitted: {task_id}")
        
        if args.wait:
            print("Waiting for task completion...")
            try:
                result = await self.orchestra.wait_for_task(task_id, timeout=args.timeout)
                
                print(f"\n=== TASK RESULT ===")
                print(f"Success: {result.success}")
                print(f"Execution Time: {result.execution_time:.2f}s")
                
                if result.success:
                    print(f"Result: {json.dumps(result.result, indent=2)}")
                else:
                    print(f"Error: {result.error}")
                    
            except asyncio.TimeoutError:
                print("Task timed out")
    
    async def _handle_agent_command(self, args: argparse.Namespace):
        """Handle agent management commands"""
        if args.agent_action == "list":
            await self._list_agents()
        elif args.agent_action == "add":
            await self._add_agent(args)
        elif args.agent_action == "remove":
            await self._remove_agent(args)
    
    async def _list_agents(self):
        """List registered agents"""
        if not self.orchestra:
            print("Orchestra is not running")
            return
        
        agents = self.orchestra.get_agents()
        
        print(f"\n=== REGISTERED AGENTS ({len(agents)}) ===")
        for agent in agents:
            capabilities = [cap.name for cap in agent.capabilities]
            print(f"ID: {agent.id}")
            print(f"  Name: {agent.name}")
            print(f"  Status: {agent.status}")
            print(f"  Capabilities: {', '.join(capabilities)}")
            print(f"  Last Heartbeat: {agent.last_heartbeat}")
            print()
    
    async def _add_agent(self, args: argparse.Namespace):
        """Add an agent"""
        if not self.orchestra:
            print("Orchestra is not running")
            return
        
        agent = Agent(
            agent_id=args.agent_id,
            name=args.name,
            capabilities=args.capabilities
        )
        
        self.orchestra.register_agent(agent)
        print(f"Agent {args.agent_id} added successfully")
    
    async def _remove_agent(self, args: argparse.Namespace):
        """Remove an agent"""
        if not self.orchestra:
            print("Orchestra is not running")
            return
        
        try:
            self.orchestra.unregister_agent(args.agent_id)
            print(f"Agent {args.agent_id} removed successfully")
        except Exception as e:
            print(f"Failed to remove agent: {e}")
    
    async def _show_monitor(self, args: argparse.Namespace):
        """Show monitoring dashboard"""
        if not self.monitor:
            print("Monitoring is not enabled")
            return
        
        if args.continuous:
            try:
                while True:
                    dashboard_data = self.monitor.get_dashboard_data()
                    
                    # Clear screen
                    print("\033[2J\033[H")
                    
                    print("=== AGENT ORCHESTRA DASHBOARD ===")
                    print(f"Timestamp: {dashboard_data['timestamp']}")
                    
                    # Performance metrics
                    perf = dashboard_data['performance']
                    print(f"\nPerformance:")
                    print(f"  Total Tasks: {perf['total_tasks']}")
                    print(f"  Completed: {perf['completed_tasks']}")
                    print(f"  Failed: {perf['failed_tasks']}")
                    print(f"  Error Rate: {perf['error_rate']:.2%}")
                    print(f"  Tasks/sec: {perf['tasks_per_second']:.2f}")
                    
                    # System health
                    health = dashboard_data['system_health']
                    print(f"\nSystem Health:")
                    print(f"  Active Agents: {health['active_agents']}")
                    print(f"  Healthy Agents: {health['healthy_agents']}")
                    print(f"  Unhealthy Agents: {health['unhealthy_agents']}")
                    print(f"  Queued Tasks: {health['queued_tasks']}")
                    
                    await asyncio.sleep(args.refresh)
                    
            except KeyboardInterrupt:
                print("\nMonitoring stopped")
        else:
            dashboard_data = self.monitor.get_dashboard_data()
            print(json.dumps(dashboard_data, indent=2))
    
    async def _handle_config_command(self, args: argparse.Namespace):
        """Handle configuration commands"""
        if args.config_action == "validate":
            await self._validate_config(args.config_file)
        elif args.config_action == "generate":
            await self._generate_config(args.output_file, args.format)
        elif args.config_action == "show":
            await self._show_config()
    
    async def _validate_config(self, config_file: str):
        """Validate configuration file"""
        config_manager = ConfigurationManager()
        if config_manager.load(config_file):
            print("Configuration is valid")
        else:
            print("Configuration validation failed")
    
    async def _generate_config(self, output_file: str, format: str):
        """Generate sample configuration"""
        config_manager = ConfigurationManager()
        config_manager.load()  # Load defaults
        config_manager.export_config(output_file)
        print(f"Sample configuration generated: {output_file}")
    
    async def _show_config(self):
        """Show current configuration"""
        config = self.config_manager._config
        if config:
            print(json.dumps(config, indent=2))
        else:
            print("No configuration loaded")


def main():
    """Main CLI entry point"""
    cli = OrchestrationCLI()
    parser = cli.create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Run the async command
    asyncio.run(cli.run(args))


if __name__ == "__main__":
    main()