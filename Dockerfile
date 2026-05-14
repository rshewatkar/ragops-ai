FROM python:3.10-slim

# Environment settings
ENV PYTHONUNBUFFERED=1
ENV GIT_PYTHON_REFRESH=quiet
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first
COPY requirements-api.txt .

# Install critical packages separately
RUN pip install numpy==1.26.4

RUN pip install --default-timeout=100 \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install -r requirements-api.txt

# Copy source code
COPY . .

# Create persistent directories
RUN mkdir -p /app/db /app/data

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]