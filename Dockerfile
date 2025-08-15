# NSRPO Docker Image
# CPU-only optimized version for easy deployment

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements-cpu.txt requirements-dev.txt ./

# Install Python dependencies (CPU-only)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-cpu.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy project files
COPY . .

# Create necessary directories
RUN python -c "from utils.paths import initialize_directories; initialize_directories()"

# Verify installation
RUN python verify_installation.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# Default command - run smoke tests
CMD ["python", "-m", "pytest", "tests/", "-v", "-m", "smoke"]