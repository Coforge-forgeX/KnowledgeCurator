# ============================================================================
# Multi-stage Docker build for KB REST Service
# Target: <300MB final image size
# Optimized for: Azure Container Apps, AWS App Runner/Lambda, GCP Cloud Run
# ============================================================================

# ============================================================================
# Stage 1: Builder - Install dependencies and compile packages
# ============================================================================
FROM python:3.11-slim AS builder

# Build argument for optional cloud provider dependencies
# Options: "" (azure only), "aws", "gcp", "all-clouds"
ARG INSTALL_EXTRAS=""

# Install build dependencies (needed for compiling Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment (isolates dependencies)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements files first (leverage Docker layer caching)
# If requirements don't change, this layer is cached
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies from requirements.txt
# --no-cache-dir: Don't cache pip packages (saves space)
# --upgrade pip: Ensure latest pip for better dependency resolution
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim

# Install only runtime dependencies (no build tools)
# libpq5: PostgreSQL client library (needed for psycopg2)
# ca-certificates: SSL/TLS certificates for HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
# This includes all compiled packages WITHOUT build tools
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    STORAGE_PROVIDER=azure

# Create non-root user for security
# Running as root in container is a security risk
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application code (owned by appuser)
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port (configurable via PORT env var)
EXPOSE 8080

# Health check (used by container orchestrators)
# Checks /health endpoint every 30 seconds
# Marks container unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Start uvicorn server
# --host 0.0.0.0: Listen on all network interfaces
# --port ${PORT}: Use PORT environment variable
# --workers 1: Single worker (serverless platforms handle scaling)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
