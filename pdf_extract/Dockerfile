# Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables to avoid .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies for FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \ 
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /home/russellb/developments/DeepLearningNLP/sanicllmdocker/pdf_extract

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8003

# Default command
CMD ["python", "run.py"]

