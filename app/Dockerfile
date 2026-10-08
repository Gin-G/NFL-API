# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --prefer-offline
COPY frontend/ .
ENV VITE_API_BASE=""
RUN npm run build

# ── Stage 2: Python API ────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The FanDuel lineup optimizer (api/optimizer.py) lives in nfl_projections.
# --no-deps on purpose: the package declares TensorFlow for its training code,
# which this image must not carry, and the optimizer is pandas and a greedy
# loop. nfl_projections defers its model imports so this works (see that
# package's __init__). The projections CRONJOB installs the package properly,
# with TensorFlow, at runtime.
RUN pip install --no-cache-dir --no-deps \
    "nfl-projections @ https://github.com/Gin-G/nfl-data-py/archive/refs/heads/main.tar.gz"

# Copy the application source
COPY app/ .

# Copy built frontend assets
COPY --from=frontend-build /frontend/dist/ ./static/dashboard/

# Create __init__.py files for proper Python module structure
RUN touch __init__.py && \
    touch functions/__init__.py && \
    touch functions/players/__init__.py && \
    touch functions/coaching/__init__.py && \
    touch api/__init__.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
