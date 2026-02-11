#!/bin/bash
# Deployment script for Agent Orchestra

set -e

echo "🎭 Agent Orchestra Deployment Script"
echo "===================================="

# Configuration
ENVIRONMENT=${1:-"staging"}
VERSION=${2:-"latest"}
DEPLOY_DIR="/opt/agent-orchestra"
SERVICE_NAME="agent-orchestra"

echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo ""

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    echo "❌ Invalid environment. Use: development, staging, or production"
    exit 1
fi

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create deployment directory
sudo mkdir -p $DEPLOY_DIR
cd $DEPLOY_DIR

# Download or update deployment files
echo "📥 Updating deployment files..."

if [ ! -d "agent-orchestra" ]; then
    git clone https://github.com/andreycpu/agent-orchestra.git
else
    cd agent-orchestra
    git pull origin main
    cd ..
fi

# Copy configuration for environment
echo "⚙️ Setting up configuration..."
sudo cp agent-orchestra/config/${ENVIRONMENT}.yaml config.yaml

# Set environment variables
export ENVIRONMENT
export VERSION
export COMPOSE_PROJECT_NAME="agent-orchestra-${ENVIRONMENT}"

# Pull latest images
echo "📦 Pulling Docker images..."
docker-compose -f agent-orchestra/docker-compose.yml pull

# Stop existing services
echo "🛑 Stopping existing services..."
docker-compose -f agent-orchestra/docker-compose.yml down || true

# Start services
echo "🚀 Starting services..."
docker-compose -f agent-orchestra/docker-compose.yml up -d

# Wait for services to be healthy
echo "🔍 Waiting for services to be healthy..."
sleep 30

# Check service health
echo "❤️ Checking service health..."
if docker-compose -f agent-orchestra/docker-compose.yml ps | grep -q "Up"; then
    echo "✅ Services are running"
else
    echo "❌ Some services failed to start"
    docker-compose -f agent-orchestra/docker-compose.yml logs
    exit 1
fi

# Run health check
echo "🩺 Running health checks..."
if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "⚠️ Health check failed - service may still be starting"
fi

# Show status
echo ""
echo "📊 Deployment Status"
echo "==================="
docker-compose -f agent-orchestra/docker-compose.yml ps

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "📋 Next steps:"
echo "  - Monitor logs: docker-compose -f agent-orchestra/docker-compose.yml logs -f"
echo "  - Check status: docker-compose -f agent-orchestra/docker-compose.yml ps"
echo "  - Access dashboard: http://localhost:3000 (if Grafana is enabled)"
echo "  - View metrics: http://localhost:8080/metrics"