# Stage 1: Build & Download stage
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies for compiling llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy dependencies needed by download_model.py
COPY download_model.py .
COPY utils/ utils/

# Download the GGUF model weights during build time to package them inside the image
RUN python download_model.py

# Stage 2: Clean Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Copy compiled python site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy codebase
COPY main.py .
COPY router/ router/
COPY local_engine/ local_engine/
COPY api_engine/ api_engine/
COPY utils/ utils/
COPY AGents.md .

# Copy pre-downloaded model weights
COPY --from=builder /app/models/local_model.gguf /app/models/local_model.gguf

# Ensure input and output directories exist
RUN mkdir -p /input /output

# CMD specifies the entrypoint for the harness
CMD ["python", "main.py"]
