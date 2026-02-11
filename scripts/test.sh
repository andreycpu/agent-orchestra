#!/bin/bash
# Test runner script for Agent Orchestra

set -e

echo "🎭 Agent Orchestra Test Runner"
echo "=============================="

# Default values
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE      Test type: unit, integration, all (default: all)"
            echo "  -c, --coverage       Generate coverage report"
            echo "  -v, --verbose        Verbose output"
            echo "  -h, --help          Show help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest is not installed. Install with:"
    echo "pip install pytest pytest-asyncio pytest-cov"
    exit 1
fi

# Build pytest command
PYTEST_ARGS=""

if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -v"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=agent_orchestra --cov-report=html --cov-report=term"
fi

# Run tests based on type
case $TEST_TYPE in
    "unit")
        echo "🧪 Running unit tests..."
        pytest tests/unit/ tests/test_*.py $PYTEST_ARGS
        ;;
    "integration")
        echo "🔗 Running integration tests..."
        pytest tests/integration/ $PYTEST_ARGS
        ;;
    "all")
        echo "🧪 Running all tests..."
        pytest tests/ $PYTEST_ARGS
        ;;
    *)
        echo "❌ Unknown test type: $TEST_TYPE"
        exit 1
        ;;
esac

echo ""
echo "✅ Tests completed successfully!"

if [ "$COVERAGE" = true ]; then
    echo ""
    echo "📊 Coverage report generated:"
    echo "  - HTML: htmlcov/index.html"
    echo "  - Terminal output above"
fi