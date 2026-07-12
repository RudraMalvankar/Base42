import subprocess
import json
import time
import os
import statistics

# We treat the current best configuration as the BASELINE.
# ENABLE_RESERVED_LOCAL_WORKERS=True, ENABLE_SHORT_FIREWORKS_PROMPTS=True are baked into config.py

EXPERIMENTS = [
    {"name": "BASELINE", "env": {}},
    
    # Phase 1
    {"name": "ENABLE_RESPONSE_COMPRESSION", "env": {"ENABLE_RESPONSE_COMPRESSION": "True"}},
    {"name": "ENABLE_ROUTER_FIX", "env": {"ENABLE_ROUTER_FIX": "True"}},
    {"name": "ENABLE_SMART_ROUTING_V2", "env": {"ENABLE_SMART_ROUTING_V2": "True"}},
    
    # Concurrency
    {"name": "CONCURRENCY_2", "env": {"FIREWORKS_CONCURRENCY": "2"}},
    {"name": "CONCURRENCY_4", "env": {"FIREWORKS_CONCURRENCY": "4"}},
    {"name": "CONCURRENCY_6", "env": {"FIREWORKS_CONCURRENCY": "6"}},
    
    # Phase 2
    {"name": "ENABLE_DETERMINISTIC_EXTRACTION", "env": {"ENABLE_DETERMINISTIC_EXTRACTION": "True"}},
    {"name": "ENABLE_LOCAL_FACTUAL", "env": {"ENABLE_LOCAL_FACTUAL": "True"}},
    
    # The Kitchen Sink (Best effort combination based on assumptions)
    {"name": "ALL_OPTIMIZATIONS", "env": {
        "ENABLE_RESPONSE_COMPRESSION": "True",
        "ENABLE_ROUTER_FIX": "True",
        "ENABLE_SMART_ROUTING_V2": "True",
        "ENABLE_DETERMINISTIC_EXTRACTION": "True",
        "FIREWORKS_CONCURRENCY": "4"
    }}
]

def run_experiment(exp):
    print(f"Running experiment: {exp['name']}")
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.getcwd().replace(chr(92), '/')}/input_samples:/input",
        "-v", f"{os.getcwd().replace(chr(92), '/')}/output:/output",
        "-e", "FIREWORKS_API_KEY=fw_84W1aV4dLEnLReJ3noVXRh",
        "-e", "FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1",
        "-e", "ALLOWED_MODELS=accounts/fireworks/models/deepseek-v4-flash,accounts/fireworks/models/deepseek-v4-pro"
    ]
    
    for k, v in exp["env"].items():
        cmd.extend(["-e", f"{k}={v}"])
        
    cmd.append("base42:latest")
    cmd.append("python")
    cmd.append("main.py")
    
    start = time.perf_counter()
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    duration = time.perf_counter() - start
    
    # Read telemetry
    try:
        with open("output/telemetry.json", "r") as f:
            telemetry = json.load(f)
    except:
        telemetry = {"summary": {"total_tokens": 0, "api_fallbacks": 10}, "traces": []}
        
    # Read results 
    try:
        with open("output/results.json", "r") as f:
            results = json.load(f)
    except:
        results = []
        
    errors = sum(1 for r in results if r.get("answer", "") in ["API Error", "Timeout Error", ""])
    accuracy = ((len(results) - errors) / max(1, len(results))) * 100
    
    latencies = [t["latency_ms"] for t in telemetry.get("traces", [])]
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies) if latencies else 0
    avg_latency = statistics.mean(latencies) if latencies else 0
    
    return {
        "flag": exp["name"],
        "accuracy": accuracy,
        "total_tokens": telemetry["summary"].get("total_tokens", 0),
        "api_fallbacks": telemetry["summary"].get("api_fallbacks", 0),
        "duration": duration,
        "avg_latency": avg_latency,
        "p95_latency": p95,
        "local_routes": telemetry["summary"].get("routes", {}).get("local_llm", 0)
    }

results = []
for exp in EXPERIMENTS:
    res = run_experiment(exp)
    results.append(res)
    print(res)
    
with open("BENCHMARK_RESULTS.json", "w") as f:
    json.dump(results, f, indent=4)
