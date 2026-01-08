# Dockerfile for Finance RAG Pipeline

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    python3-dev \
    curl \
    git \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for production
RUN pip install --no-cache-dir \
    anthropic \
    uvicorn[standard] \
    gunicorn \
    psycopg2-binary \
    redis \
    prometheus_client

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Create non-root user for security
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Development override (use docker-compose override)
# CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
