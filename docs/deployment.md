# Deployment Guide

This guide covers deploying Agent Orchestra in different environments.

## Docker Deployment

### Single Container

```bash
docker run -d \
  --name agent-orchestra \
  -p 8080:8080 \
  -e REDIS_URL=redis://redis:6379 \
  agent-orchestra:latest
```

### Docker Compose

```bash
docker-compose up -d
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestra
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-orchestra
  template:
    metadata:
      labels:
        app: agent-orchestra
    spec:
      containers:
      - name: agent-orchestra
        image: agent-orchestra:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

## Environment Variables

- `REDIS_URL`: Redis connection URL
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MAX_CONCURRENT_TASKS`: Maximum concurrent tasks
- `METRICS_ENABLED`: Enable metrics collection

## Monitoring

### Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'agent-orchestra'
    static_configs:
      - targets: ['localhost:8080']
```

### Health Checks

```bash
curl http://localhost:8080/health
```

## Security

- Configure JWT secrets
- Set up proper authentication
- Use TLS for production
- Implement proper network policies