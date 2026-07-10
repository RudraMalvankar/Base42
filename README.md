# AMD Developer Hackathon (Act II) - Track 1: General-Purpose AI Agent

This repository contains the implementation of a highly optimized, cost-efficient, and accurate General-Purpose AI Agent designed to solve Track 1 of the AMD Developer Hackathon.

## 🚀 Key Features

* **Hybrid Routing Architecture:** Classifies incoming tasks into 8 capability categories at zero cost using regex and heuristics. Simple tasks are routed to a local CPU-based model (Qwen 2.5 1.5B Instruct GGUF), and complex tasks (math, code generation, debugging, logic) are routed to Fireworks API.
* **Zero-Token Local Execution:** Saves Fireworks API tokens by resolving sentiment classification, named entity recognition (NER), factual queries, and text summarization locally.
* **Dynamic API Model Selection:** Reads `ALLOWED_MODELS` from the environment and chooses the optimal model size (e.g., lightweight models for API fallbacks and premium 70B/405B models for complex reasoning).
* **Robust Timeout & Error Recovery:** Wraps local CPU execution in a 20-second timeout block. If a task fails or times out, it seamlessly falls back to the Fireworks API, ensuring accuracy and pipeline resilience.
* **High Concurrency & Limit Throttling:** Processes tasks concurrently using Python's `asyncio` and throttles requests using a semaphore (limit: 5) to protect CPU memory and Fireworks API rate limits.
* **Production-Grade Containerization:** A multi-stage Docker build that compiles CPU-bound libraries (`llama-cpp-python`) and packages pre-downloaded GGUF weights. The final runtime container is stripped of build tools for a minimal pulled image size.

---

## 📁 Directory Structure

```text
amd/
├── router/
│   ├── __init__.py
│   └── classifier.py          # Heuristic/regex task classification
├── local_engine/
│   ├── __init__.py
│   └── llama_client.py        # Local GGUF CPU model controller (llama-cpp-python)
├── api_engine/
│   ├── __init__.py
│   └── fireworks_client.py    # Async Fireworks API client with retries and model mapping
├── utils/
│   ├── __init__.py
│   └── logger.py              # Central logging utility
├── input/
│   └── tasks.json             # Practice/evaluation tasks input
├── output/
│   └── results.json           # Evaluation outputs (generated on run completion)
├── .gitignore                 # Files excluded from version control
├── AGents.md                  # Comprehensive architectural plan and edge case mitigations
├── Dockerfile                 # Multi-stage Docker build targeting linux/amd64
├── download_model.py          # Pre-caches the GGUF model during Docker build
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
└── main.py                    # Entrypoint and execution coordinator
```

---

## ⚙️ Installation & Local Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Install Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```
*(Note: If compiling `llama-cpp-python` from source on your OS fails, ensure you have C/C++ compiler tools installed).*

### 3. Download the GGUF Model
Run the download script to retrieve and cache the 1.2 GB local model weights:
```bash
python download_model.py
```

### 4. Configure Environment Variables
Set up environment variables for the Fireworks API:
* **Linux/macOS:**
  ```bash
  export FIREWORKS_API_KEY="your-fireworks-api-key"
  export FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
  export ALLOWED_MODELS="accounts/fireworks/models/llama-v3p1-8b-instruct,accounts/fireworks/models/llama-v3p1-70b-instruct"
  ```
* **Windows (PowerShell):**
  ```powershell
  $env:FIREWORKS_API_KEY="your-fireworks-api-key"
  $env:FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
  $env:ALLOWED_MODELS="accounts/fireworks/models/llama-v3p1-8b-instruct,accounts/fireworks/models/llama-v3p1-70b-instruct"
  ```

---

## 🏃 Running the Agent

### Local Run
Prepare your input tasks inside `input/tasks.json` (or use the provided practice set) and run:
```bash
python main.py
```
This will read the tasks, run the hybrid routing logic, and write the output list containing answers to `output/results.json`.

---

## 🐳 Docker Deployment

The judging VM runs `linux/amd64`. You must build and push a compatible Docker image.

### 1. Build the Docker Image
```bash
docker build -t amd-agent:latest .
```
*(Apple Silicon Users: Build targeting AMD64)*
```bash
docker buildx build --platform linux/amd64 -t amd-agent:latest .
```

### 2. Run the Container Locally for Testing
Mount your local input/output directories to verify:
```bash
docker run --rm \
  -e FIREWORKS_API_KEY="your-fireworks-api-key" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="accounts/fireworks/models/llama-v3p1-8b-instruct,accounts/fireworks/models/llama-v3p1-70b-instruct" \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  amd-agent:latest
```

---

## 🧠 Edge Case Analysis & Resilience

| Potential Issue | System Resilience / Mitigation |
| :--- | :--- |
| **Local CPU hangs** | Wrapped in a 20-second asyncio timeout; aborts execution and redirects request to Fireworks API. |
| **Out-Of-Memory (OOM)** | Restricts context size (`n_ctx=512`) and specifies `n_threads=2` to fit well within the 4 GB limit. |
| **API Rate-Limiting (429)** | Implements `tenacity` exponential retry backoff decorator on API completion methods. |
| **Schema Violations** | Fallback catch blocks produce standardized string error messages per task, preventing empty or missing JSON output records. |
| **Model Invalidation** | Reads allowed list at runtime and dynamically chooses models from `ALLOWED_MODELS` rather than hardcoding. |
