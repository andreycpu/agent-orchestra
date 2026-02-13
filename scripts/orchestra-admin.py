#!/usr/bin/env python3
"""
Agent Orchestra Administration Script

This script provides command-line utilities for managing
Agent Orchestra instances, including health checks, statistics,
and basic maintenance operations.
"""
import sys
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from agent_orchestra import NEW_FEATURES_AVAILABLE
    if NEW_FEATURES_AVAILABLE:
        from agent_orchestra import (
            performance_manager, get_performance_report,
            get_cache_stats, get_queue_stats, get_error_statistics,
            ConfigurationLoader, ConfigurationValidator
        )
    from agent_orchestra.orchestra import Orchestra
    from agent_orchestra.health_checks import HealthCheckManager
except ImportError as e:
    print(f"Error importing Agent Orchestra: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)


def format_output(data, format_type="json"):
    """Format output data."""
    if format_type == "json":
        return json.dumps(data, indent=2, default=str)
    elif format_type == "table":
        # Simple table formatting for basic data
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                lines.append(f"{key:<20}: {value}")
            return "\n".join(lines)
        else:
            return str(data)
    else:
        return str(data)


async def health_check_command(args):
    """Perform system health check."""
    print("Agent Orchestra Health Check")
    print("=" * 40)
    
    health_results = {}
    
    # Basic system health
    try:
        if NEW_FEATURES_AVAILABLE:
            # Performance monitoring
            perf_report = get_performance_report()
            health_results["performance"] = {
                "cpu_percent": perf_report.get("current_resources", {}).get("cpu_percent", "N/A"),
                "memory_percent": perf_report.get("current_resources", {}).get("memory_percent", "N/A"),
                "uptime": "N/A"
            }
            
            # Cache health
            cache_stats = get_cache_stats()
            health_results["caches"] = {
                name: {
                    "hit_rate": stats.get("stats", {}).get("hit_rate", 0),
                    "size": stats.get("stats", {}).get("entry_count", 0)
                }
                for name, stats in cache_stats.items()
            }
            
            # Queue health
            queue_stats = get_queue_stats()
            health_results["queues"] = {
                name: {
                    "size": stats.get("size", 0),
                    "throughput": stats.get("throughput", 0)
                }
                for name, stats in queue_stats.items()
            }
            
            # Error statistics
            error_stats = get_error_statistics()
            health_results["errors"] = {
                "total_errors": error_stats.get("total_errors", 0),
                "recent_errors": error_stats.get("recent_error_count", 0),
                "most_common": error_stats.get("most_common_errors", [])[:3]
            }
        
        health_results["status"] = "healthy"
        
    except Exception as e:
        health_results["status"] = "error"
        health_results["error"] = str(e)
    
    print(format_output(health_results, args.format))


def stats_command(args):
    """Display system statistics."""
    print("Agent Orchestra Statistics")
    print("=" * 40)
    
    stats = {}
    
    try:
        if NEW_FEATURES_AVAILABLE:
            # Performance stats
            if args.component in ['all', 'performance']:
                perf_report = get_performance_report()
                stats["performance"] = perf_report
            
            # Cache stats
            if args.component in ['all', 'cache']:
                stats["cache"] = get_cache_stats()
            
            # Queue stats
            if args.component in ['all', 'queue']:
                stats["queues"] = get_queue_stats()
            
            # Error stats
            if args.component in ['all', 'errors']:
                stats["errors"] = get_error_statistics()
        
    except Exception as e:
        stats["error"] = str(e)
    
    print(format_output(stats, args.format))


def config_command(args):
    """Configuration operations."""
    print("Agent Orchestra Configuration")
    print("=" * 40)
    
    if not NEW_FEATURES_AVAILABLE:
        print("Configuration utilities not available")
        return
    
    try:
        if args.action == "validate":
            # Load and validate configuration
            config = ConfigurationLoader.from_environment()
            validator = ConfigurationValidator()
            warnings = validator.validate(config)
            
            result = {
                "status": "valid" if not warnings else "warnings",
                "warnings": warnings,
                "config_summary": {
                    "database_configured": bool(config.database.url),
                    "security_configured": bool(config.security.secret_key),
                    "logging_level": config.logging.level,
                    "max_agents": config.agents.max_agents
                }
            }
            
            print(format_output(result, args.format))
        
        elif args.action == "template":
            from agent_orchestra.config_validator import get_config_template
            template = get_config_template()
            print("Configuration Template:")
            print(format_output(template, "json"))
        
        elif args.action == "env-check":
            env_issues = ConfigurationValidator.validate_environment()
            result = {
                "environment_ok": len(env_issues) == 0,
                "issues": env_issues
            }
            print(format_output(result, args.format))
    
    except Exception as e:
        print(f"Configuration error: {e}")


def monitor_command(args):
    """Monitor system in real-time."""
    print("Agent Orchestra Monitor")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    
    try:
        import time
        while True:
            # Clear screen (simple version)
            print("\033[2J\033[H")
            
            print(f"Agent Orchestra Monitor - {datetime.now()}")
            print("=" * 50)
            
            if NEW_FEATURES_AVAILABLE:
                # Quick stats
                perf_report = get_performance_report()
                current_resources = perf_report.get("current_resources", {})
                
                print(f"CPU Usage:    {current_resources.get('cpu_percent', 'N/A'):.1f}%")
                print(f"Memory Usage: {current_resources.get('memory_percent', 'N/A'):.1f}%")
                
                # Cache stats
                cache_stats = get_cache_stats()
                if cache_stats:
                    print(f"Cache Count:  {len(cache_stats)}")
                    total_hit_rate = sum(
                        stats.get("stats", {}).get("hit_rate", 0) 
                        for stats in cache_stats.values()
                    ) / max(len(cache_stats), 1)
                    print(f"Avg Hit Rate: {total_hit_rate:.2%}")
                
                # Queue stats
                queue_stats = get_queue_stats()
                if queue_stats:
                    total_queue_size = sum(stats.get("size", 0) for stats in queue_stats.values())
                    print(f"Queue Items:  {total_queue_size}")
                
                # Error stats
                error_stats = get_error_statistics()
                print(f"Total Errors: {error_stats.get('total_errors', 0)}")
                print(f"Recent Errs:  {error_stats.get('recent_error_count', 0)}")
            
            print("\nPress Ctrl+C to stop monitoring...")
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def cleanup_command(args):
    """Cleanup operations."""
    print("Agent Orchestra Cleanup")
    print("=" * 40)
    
    if not NEW_FEATURES_AVAILABLE:
        print("Cleanup utilities not available")
        return
    
    try:
        from agent_orchestra.cache_manager import clear_all_caches
        
        if args.component in ['all', 'cache']:
            print("Clearing all caches...")
            clear_all_caches()
            print("Caches cleared.")
        
        # Add more cleanup operations as needed
        print("Cleanup completed.")
        
    except Exception as e:
        print(f"Cleanup error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agent Orchestra Administration Tool")
    parser.add_argument("--format", choices=["json", "table"], default="json",
                       help="Output format")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Perform health check")
    
    # Statistics command
    stats_parser = subparsers.add_parser("stats", help="Display statistics")
    stats_parser.add_argument("--component", 
                             choices=["all", "performance", "cache", "queue", "errors"],
                             default="all",
                             help="Component to show stats for")
    
    # Configuration command
    config_parser = subparsers.add_parser("config", help="Configuration operations")
    config_parser.add_argument("action",
                              choices=["validate", "template", "env-check"],
                              help="Configuration action")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Real-time monitoring")
    monitor_parser.add_argument("--interval", type=int, default=5,
                               help="Update interval in seconds")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup operations")
    cleanup_parser.add_argument("--component",
                               choices=["all", "cache"],
                               default="all",
                               help="Component to clean up")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "health":
            asyncio.run(health_check_command(args))
        elif args.command == "stats":
            stats_command(args)
        elif args.command == "config":
            config_command(args)
        elif args.command == "monitor":
            monitor_command(args)
        elif args.command == "cleanup":
            cleanup_command(args)
    except Exception as e:
        print(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()