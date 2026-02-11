#!/bin/bash
# Setup script for Agent Orchestra development environment

set -e

echo "🎭 Agent Orchestra Setup Script"
echo "==============================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📋 Installing dependencies..."
pip install -e .

if [ -f "requirements-dev.txt" ]; then
    echo "🛠️ Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Setup pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "🪝 Setting up pre-commit hooks..."
    pre-commit install
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs data config/local

# Generate sample config if not exists
if [ ! -f "config/local/development.yaml" ]; then
    echo "⚙️ Generating development configuration..."
    python -m agent_orchestra.cli config generate config/local/development.yaml --format yaml
fi

# Check Redis availability
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis is running and accessible"
    else
        echo "⚠️ Redis is not running. You can start it with: redis-server"
        echo "   Or use Docker: docker run -d -p 6379:6379 redis:7-alpine"
    fi
else
    echo "⚠️ Redis CLI not found. Install Redis or use Docker:"
    echo "   brew install redis  # macOS"
    echo "   apt-get install redis-server  # Ubuntu/Debian"
    echo "   docker run -d -p 6379:6379 redis:7-alpine  # Docker"
fi

# Run tests to verify installation
echo "🧪 Running quick verification tests..."
python -c "
import agent_orchestra
from agent_orchestra import Orchestra, Agent
print('✅ Import successful')

# Quick functionality test
import asyncio
async def test():
    orchestra = Orchestra()
    agent = Agent('test-agent', capabilities=['test'])
    orchestra.register_agent(agent)
    print('✅ Basic functionality working')

asyncio.run(test())
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Start Redis if not running: redis-server"
echo "  3. Run examples: python examples/basic_usage.py"
echo "  4. Start the orchestra: python -m agent_orchestra.cli start"
echo ""
echo "Development commands:"
echo "  - Run tests: pytest"
echo "  - Format code: black agent_orchestra tests examples"
echo "  - Type check: mypy agent_orchestra"
echo "  - Lint code: flake8 agent_orchestra"
echo ""
echo "Happy orchestrating! 🎭✨"