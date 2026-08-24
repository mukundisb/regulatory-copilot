# Use Python 3.12 slim image for a balanced footprint with standard glibc
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Step 1: Pre-install CPU-only PyTorch to satisfy downstream dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies in a distinct layer for caching
COPY requirements.txt .

# Install dependencies (CPU-only PyTorch optimization can be used if specified)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and necessary assets
COPY . .

# Expose standard FastAPI application port
EXPOSE 8000

# Start Uvicorn bound to 0.0.0.0 (required for container ingress)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]