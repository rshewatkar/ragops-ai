FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Uninstall numpy first to ensure clean install
RUN pip uninstall -y numpy || true

COPY requirements-api.txt .

# Install numpy first, then torch with CPU wheels, then rest
RUN pip install --no-cache-dir numpy==1.26.4 && \
    pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
