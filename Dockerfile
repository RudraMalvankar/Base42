# Stage 1: Build environment
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.11-slim

WORKDIR /app

# Copy built wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install the pre-built wheels
RUN pip install --no-cache /wheels/*

# Download model at build time to cache it in the image layer
COPY download_model.py .
RUN python download_model.py

# Copy the rest of the application
COPY core/ core/
COPY engine/ engine/
COPY models/ models/
COPY pipeline/ pipeline/
COPY main.py .

# Ensure input/output directories exist for local testing
RUN mkdir -p /input /output

# Command to run the application
CMD ["python", "main.py"]
