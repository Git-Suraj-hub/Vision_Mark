# Production Dockerfile for Vision Mark
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies for OpenCV and InsightFace models
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download buffalo_l model at build time to prevent download delays & memory spikes at container boot
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition']); app.prepare(ctx_id=0, det_size=(640, 640))" && \
    rm -f /root/.insightface/models/buffalo_l/1k3d68.onnx \
          /root/.insightface/models/buffalo_l/2d106det.onnx \
          /root/.insightface/models/buffalo_l/genderage.onnx

# Copy application files
COPY . /app/

# Create runtime directories
RUN mkdir -p /app/pythonProject4/storage/captures /app/pythonProject4/Dataset

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["python", "main.py"]
