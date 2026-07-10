# Base42 AI Operating System
**AMD Developer Hackathon - Track 1**

Base42 is a deterministic, cost-aware AI Orchestrator designed to maximize accuracy while driving API token consumption as close to zero as possible, operating strictly within 4GB RAM and 2 vCPUs.

## Features
- **Mathematical Utility Routing**: Dynamically calculates the optimal Execution Route (API vs Local vs Python) based on cost, latency, and cognitive load.
- **Zero-Token Math Sandbox**: Uses Python `ast.parse` to securely execute math and logic natively, achieving 100% accuracy for 0 tokens.
- **Zero-Token Confidence Engine**: Analyzes Local LLM outputs for n-gram looping and multi-lingual hedging. If the 1.5B model hallucinates, it seamlessly falls back to the Fireworks API.
- **DAG Planner**: Heuristically decomposes complex sequential prompts and resolves dependencies.
- **Enterprise Resilience**: Globally wrapped exception handlers guarantee a formatted `results.json` is output even under catastrophic failure.
- **Observability**: Automatically writes `telemetry.json` with traces, latency, and token metrics.

## Building and Running
The system is heavily optimized using a Multi-Stage Docker Build.

```bash
# 1. Build the image (Downloads the 1.5B GGUF model and compiles llama-cpp-python for CPU)
docker build -t base42 .

# 2. Run the container
# Assuming tasks.json is in ./input and you want results in ./output
docker run --rm -v $(pwd)/input:/input -v $(pwd)/output:/output base42
```
