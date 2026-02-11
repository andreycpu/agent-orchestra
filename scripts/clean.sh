#!/bin/bash
# Clean up development artifacts

echo "🧹 Cleaning Agent Orchestra development artifacts..."

# Remove Python artifacts
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove build artifacts
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/

# Remove test artifacts
rm -rf .pytest_cache/
rm -rf htmlcov/
rm -f .coverage
rm -f coverage.xml

# Remove IDE artifacts
rm -rf .idea/
rm -f *.swp *.swo *~

# Remove logs
rm -rf logs/
rm -f *.log

# Remove temporary files
rm -f *.tmp *.temp *.bak *.pid

echo "✅ Cleanup completed!"