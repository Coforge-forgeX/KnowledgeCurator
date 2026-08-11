# Multi-stage build for Indexer Service
#
# Build context must be the repo root (services/), not indexer-service/,
# since `shared/` is vendored from there - same package main.py imports at
# runtime (shared.windows_encoding) and that the deploy workflow copies in
# for the non-Docker (Oryx) deployment path:
#   docker build -f indexer-service/Dockerfile -t indexer-service .
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY indexer-service/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy Python dependencies from builder, owned by appuser so it can read them
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code and the vendored shared package
COPY --chown=appuser:appuser indexer-service/ .
COPY --chown=appuser:appuser shared/ ./shared/

USER appuser

# Make sure scripts are executable
ENV PATH=/home/appuser/.local/bin:$PATH

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CLOUD_PROVIDER=azure
ENV STORAGE_PROVIDER=azure
ENV QUEUE_PROVIDER=azure
ENV PORT=8081
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE 8081

# Run the application
CMD ["python", "main.py"]
