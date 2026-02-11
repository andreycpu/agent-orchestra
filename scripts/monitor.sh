#!/bin/bash
# Monitoring script for Agent Orchestra

set -e

echo "🎭 Agent Orchestra Monitoring Script"
echo "===================================="

# Default values
DURATION=60
INTERVAL=5
FORMAT="text"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -f|--format)
            FORMAT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --duration SECONDS   Monitoring duration (default: 60)"
            echo "  -i, --interval SECONDS   Check interval (default: 5)"
            echo "  -f, --format FORMAT      Output format: text, json, csv (default: text)"
            echo "  -h, --help               Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                       # Monitor for 60 seconds"
            echo "  $0 -d 300 -i 10         # Monitor for 5 minutes, check every 10 seconds"
            echo "  $0 -f json              # Output in JSON format"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "Monitoring for ${DURATION} seconds (checking every ${INTERVAL}s)..."
echo ""

# Check if orchestra is running
if ! pgrep -f "agent_orchestra" > /dev/null; then
    echo "⚠️ Warning: No Agent Orchestra processes found"
    echo ""
fi

# Start monitoring
START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION))
ITERATION=1

if [ "$FORMAT" = "csv" ]; then
    echo "timestamp,cpu_percent,memory_mb,agent_count,task_queue_size,redis_memory_mb"
fi

while [ $(date +%s) -lt $END_TIME ]; do
    CURRENT_TIME=$(date +%s)
    TIMESTAMP=$(date -Iseconds)
    
    # Get system metrics
    CPU_PERCENT=$(python3 -c "import psutil; print(f'{psutil.cpu_percent(interval=1):.1f}')" 2>/dev/null || echo "N/A")
    MEMORY_MB=$(python3 -c "import psutil; print(int(psutil.virtual_memory().used / 1024 / 1024))" 2>/dev/null || echo "N/A")
    
    # Get Agent Orchestra metrics (if available)
    AGENT_COUNT="N/A"
    TASK_QUEUE_SIZE="N/A"
    
    # Try to get Redis metrics
    REDIS_MEMORY_MB="N/A"
    if command -v redis-cli &> /dev/null; then
        REDIS_INFO=$(redis-cli info memory 2>/dev/null || true)
        if [ ! -z "$REDIS_INFO" ]; then
            REDIS_MEMORY_MB=$(echo "$REDIS_INFO" | grep "used_memory:" | cut -d: -f2 | tr -d '\r' | xargs -I {} python3 -c "print(int({}/1024/1024))" 2>/dev/null || echo "N/A")
        fi
    fi
    
    # Output based on format
    case $FORMAT in
        "json")
            cat << EOF
{
  "timestamp": "$TIMESTAMP",
  "iteration": $ITERATION,
  "system": {
    "cpu_percent": "$CPU_PERCENT",
    "memory_mb": "$MEMORY_MB"
  },
  "orchestra": {
    "agent_count": "$AGENT_COUNT",
    "task_queue_size": "$TASK_QUEUE_SIZE"
  },
  "redis": {
    "memory_mb": "$REDIS_MEMORY_MB"
  }
}
EOF
            ;;
        "csv")
            echo "$TIMESTAMP,$CPU_PERCENT,$MEMORY_MB,$AGENT_COUNT,$TASK_QUEUE_SIZE,$REDIS_MEMORY_MB"
            ;;
        *)
            echo "📊 Iteration $ITERATION - $(date '+%H:%M:%S')"
            echo "   System: CPU ${CPU_PERCENT}%, Memory ${MEMORY_MB}MB"
            echo "   Redis: ${REDIS_MEMORY_MB}MB"
            echo "   Orchestra: ${AGENT_COUNT} agents, ${TASK_QUEUE_SIZE} queued"
            echo ""
            ;;
    esac
    
    ITERATION=$((ITERATION + 1))
    sleep $INTERVAL
done

echo ""
echo "📋 Monitoring Summary"
echo "===================="
echo "Duration: ${DURATION} seconds"
echo "Iterations: $((ITERATION - 1))"
echo "Interval: ${INTERVAL} seconds"

# Final system check
FINAL_CPU=$(python3 -c "import psutil; print(f'{psutil.cpu_percent(interval=1):.1f}')" 2>/dev/null || echo "N/A")
FINAL_MEMORY=$(python3 -c "import psutil; print(int(psutil.virtual_memory().used / 1024 / 1024))" 2>/dev/null || echo "N/A")

echo "Final CPU: ${FINAL_CPU}%"
echo "Final Memory: ${FINAL_MEMORY}MB"

# Check for high resource usage
if [ "$FINAL_CPU" != "N/A" ] && [ "${FINAL_CPU%.*}" -gt 80 ]; then
    echo "⚠️ Warning: High CPU usage detected (${FINAL_CPU}%)"
fi

if [ "$FINAL_MEMORY" != "N/A" ] && [ "$FINAL_MEMORY" -gt 1000 ]; then
    echo "⚠️ Warning: High memory usage detected (${FINAL_MEMORY}MB)"
fi

echo ""
echo "💡 For continuous monitoring, consider using:"
echo "   - python -m agent_orchestra.cli monitor --continuous"
echo "   - Prometheus + Grafana for production monitoring"
echo "   - System monitoring tools like htop, iotop"