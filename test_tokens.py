import os
import re
import subprocess
import json
import time

file_path = "engine/executors/local_llm.py"
with open(file_path, "r") as f:
    content = f.read()

for mt in [48, 64, 80, 96]:
    print(f"\n==========================================")
    print(f"=== Testing max_tokens={mt} ===")
    print(f"==========================================")
    new_content = re.sub(r'max_tokens = min\(max_tokens, \d+\)', f'max_tokens = min(max_tokens, {mt})', content)
    with open(file_path, "w") as f:
        f.write(new_content)
    
    start = time.time()
    res = subprocess.run(["python", "run_experiments.py"], capture_output=True, text=True)
    duration = time.time() - start
    
    print(f"\nTotal Execution Time: {duration:.2f}s")
    
    # Parse telemetry for inference latency
    try:
        with open("output/telemetry.json", "r") as f:
            telemetry = json.load(f)
            
        print("\nTelemetry:")
        for t in telemetry['traces']:
            if t['task_id'] in ['T01', 'T01b', 'T01c']:
                print(f"{t['task_id']} Latency: {t['latency_ms']/1000:.2f}s | Route: {t['route']}")
    except Exception as e:
        print("Could not read telemetry:", e)

    print(f"\nOutputs:")
    try:
        with open("output/results.json", "r") as f:
            results = json.load(f)
        for r in results:
            if r['task_id'] in ['T01', 'T01b', 'T01c']:
                print(f"[{r['task_id']}]: {r['answer']}\n")
    except Exception as e:
        print("Could not read results:", e)

# Restore original
with open(file_path, "w") as f:
    f.write(content)
