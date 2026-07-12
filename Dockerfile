# ==========================================
# STAGE 1: Builder
# ==========================================
FROM python:3.11-slim AS builder

# Install C++ build dependencies for llama-cpp-python compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy python dependencies and model download script
COPY requirements.txt .
COPY download_model.py .

# Install dependencies
# Install dependencies using pre-compiled universal CPU wheels to prevent hardware mismatch
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Pre-download the LLM weights into the image to achieve 0ms runtime network latency
RUN python download_model.py


# ==========================================
# STAGE 2: Runner
# ==========================================
FROM python:3.11-slim AS runner

# Set strict environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy the compiled virtual environment (Leaves GCC/build-essential behind)
COPY --from=builder /opt/venv /opt/venv

# Copy the pre-downloaded GGUF model
COPY --from=builder /model /model

# Create the required AMD Hackathon I/O directories
RUN mkdir -p /input /output

# Install libgomp1 required for llama-cpp-python universal CPU wheels
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set the working directory and copy the Base42 AI OS source code
WORKDIR /app
COPY . /app

# Execute the orchestrator
CMD ["python", "main.py"]
