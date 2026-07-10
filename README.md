# Base42: Enterprise AI Orchestration Engine

Base42 is a highly optimized, deterministic AI Operating System built for the AMD Developer Hackathon (Track 1). It uses a hybrid orchestration pipeline to route tasks dynamically between a zero-token local Python environment, a 4-bit quantized local LLM, and the premium Fireworks API.

## 🚀 Quick Start: Docker Build & Run

To test Base42 locally in an environment identical to the grading platform, use Docker. The Dockerfile uses a multi-stage build to compile `llama-cpp-python` and caches the 1.5B Qwen model inside the image.

### 1. Build the Docker Image

Run the following command in the root directory. This process might take 5-10 minutes as it downloads the model weights and builds C++ bindings.

```bash
docker build -t base42-agent .
```

### 2. Run the Container (Local Testing)

To simulate the grading environment, you need to mount local `/input` and `/output` directories into the container and provide your Fireworks API key.

1. Ensure you have a `tasks.json` file inside a local `./input` folder.
2. Run the container:

```bash
docker run --rm \
  -v ${PWD}/input:/input \
  -v ${PWD}/output:/output \
  -e FIREWORKS_API_KEY="your_api_key_here" \
  -e ALLOWED_MODELS="accounts/fireworks/models/llama-v3p1-8b-instruct,accounts/fireworks/models/llama-v3p1-70b-instruct" \
  --memory="4g" \
  --cpus="2.0" \
  base42-agent
```

### 3. Submission / Pushing to Registry

When you are ready to submit to the AMD platform, you need to push the built image to the GitHub Container Registry (GHCR) or DockerHub as specified by the Hackathon rules.

```bash
# 1. Login to GitHub Container Registry (Use your GitHub PAT as password)
docker login ghcr.io -u RudraMalvankar

# 2. Tag the image
docker tag base42-agent ghcr.io/rudramalvankar/base42:latest

# 3. Push to GHCR
docker push ghcr.io/rudramalvankar/base42:latest
```

## 🧠 Architecture Highlights
- **Zero-Token Pre-Processing:** Uses Regex and heuristics to classify prompts.
- **Fail-Over Confidence Engine:** Evaluates local model outputs for hallucinations and automatically escalates to Fireworks if the confidence is low.
- **Async Execution:** Uses `asyncio.Semaphore` combined with ThreadPool isolation for synchronous C++ inference, preventing event loop deadlocks.
