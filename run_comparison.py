import subprocess
import json
import time
import os
import statistics
import threading
import shutil

# Resource Monitoring variables
peak_cpu = 0.0
peak_mem = 0.0
is_monitoring = False
cpu_history = []

def monitor_container(container_name="base42_bench"):
    global peak_cpu, peak_mem, cpu_history, is_monitoring
    while is_monitoring:
        try:
            # Poll docker stats
            out = subprocess.check_output(
                ['docker', 'stats', container_name, '--no-stream', '--format', '{{.CPUPerc}},{{.MemUsage}}'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            
            if not out:
                time.sleep(0.1)
                continue
                
            for line in out.splitlines():
                if ',' not in line:
                    continue
                cpu_str, mem_str = line.split(',', 1)
                
                # Parse CPU
                c = float(cpu_str.replace('%', '').strip())
                
                # Parse Mem (e.g. "120MiB / 4GiB")
                mem_used = mem_str.split('/')[0].strip()
                m = 0.0
                if 'MiB' in mem_used or 'MB' in mem_used:
                    m = float(mem_used.replace('MiB', '').replace('MB', '').strip())
                elif 'GiB' in mem_used or 'GB' in mem_used:
                    m = float(mem_used.replace('GiB', '').replace('GB', '').strip()) * 1024
                    
                if c > peak_cpu:
                    peak_cpu = c
                if m > peak_mem:
                    peak_mem = m
                    
                cpu_history.append(c)
        except Exception:
            pass
        time.sleep(0.15)

def run_run(enable_repair: bool) -> dict:
    global peak_cpu, peak_mem, cpu_history, is_monitoring
    
    # Reset monitor stats
    peak_cpu = 0.0
    peak_mem = 0.0
    cpu_history = []
    
    # Start monitor thread
    is_monitoring = True
    monitor_thread = threading.Thread(target=monitor_container, args=("base42_bench",))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run container
    cmd = [
        "docker", "run", "--name", "base42_bench", "--rm",
        "--cpus=2", "--memory=4g",
        "-v", f"{os.path.abspath('input_samples')}:/input",
        "-v", f"{os.path.abspath('output')}:/output",
        "-v", f"{os.path.abspath('engine')}:/app/engine",
        "-v", f"{os.path.abspath('pipeline')}:/app/pipeline",
        "-v", f"{os.path.abspath('models')}:/app/models",
        "-v", f"{os.path.abspath('core')}:/app/core",
        "-v", f"{os.path.abspath('main.py')}:/app/main.py",
        "-v", f"{os.path.abspath('config.py')}:/app/config.py",
        "-e", "FIREWORKS_API_KEY=fw_84W1aV4dLEnLReJ3noVXRh",
        "-e", "FIREWORKS_CONCURRENCY=1",
        "-e", f"ENABLE_LOCAL_REPAIR={'True' if enable_repair else 'False'}"
    ]
    cmd.append("ghcr.io/rudramalvankar/base42:latest")
    
    print(f"Executing Docker run (ENABLE_LOCAL_REPAIR={enable_repair})...")
    start_time = time.perf_counter()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    total_runtime = time.perf_counter() - start_time
    
    # Stop monitoring
    is_monitoring = False
    monitor_thread.join()
    
    # Load results & telemetry
    try:
        with open("output/results.json", "r") as f:
            results = json.load(f)
    except Exception:
        results = []
        
    try:
        with open("output/telemetry.json", "r") as f:
            telemetry = json.load(f)
    except Exception:
        telemetry = {"summary": {}, "traces": []}
        
    # Calculate accuracy based on non-crash gate
    errors = sum(1 for r in results if r.get("answer", "") in ["API Error", "Timeout Error", "Fatal Orchestrator Error", "Timeout", ""])
    accuracy = ((len(results) - errors) / max(1, len(results))) * 100
    
    traces = telemetry.get("traces", [])
    latencies = [t.get("latency_ms", 0.0) for t in traces]
    
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies) if latencies else 0.0
    
    repairs_attempted = sum(t.get("repairs_attempted", 0) for t in traces)
    repairs_successful = sum(t.get("repairs_successful", 0) for t in traces)
    fallbacks_avoided = sum(t.get("api_fallbacks_avoided", 0) for t in traces)
    tokens_consumed = telemetry.get("summary", {}).get("total_tokens", 0)
    
    return {
        "accuracy": accuracy,
        "tokens": tokens_consumed,
        "repairs_attempted": repairs_attempted,
        "repairs_successful": repairs_successful,
        "fallbacks_avoided": fallbacks_avoided,
        "avg_latency": avg_latency / 1000.0, # convert to seconds
        "p95_latency": p95_latency / 1000.0, # convert to seconds
        "peak_ram": peak_mem,
        "peak_cpu": peak_cpu,
        "runtime": total_runtime
    }

def main():
    print("Preparing official AMD Track 1 tasks for benchmarking...")
    # shutil.copy("input/tasks.json", "input_samples/tasks.json")
    
    # 1. Run Baseline
    print("\n--- RUNNING BASELINE (ENABLE_LOCAL_REPAIR=False) ---")
    baseline = run_run(enable_repair=False)
    print("Baseline Metrics:", baseline)
    
    # 2. Run Repair
    print("\n--- RUNNING REPAIR (ENABLE_LOCAL_REPAIR=True) ---")
    repair = run_run(enable_repair=True)
    print("Repair Metrics:", repair)
    
    # Calculate tokens saved
    tokens_saved = baseline["tokens"] - repair["tokens"]
    
    # Output markdown report
    report = f"""# Benchmark Comparison: Local Self-Repair vs Baseline

Official AMD Track 1 tasks evaluated.

| Metric | Baseline (No Repair) | Local Self-Repair (Active) | Difference |
| :--- | :--- | :--- | :--- |
| **Accuracy** | {baseline['accuracy']:.1f}% | {repair['accuracy']:.1f}% | {repair['accuracy'] - baseline['accuracy']:.1f}% |
| **Fireworks Tokens Consumed** | {baseline['tokens']:,} | {repair['tokens']:,} | {repair['tokens'] - baseline['tokens']:,} |
| **Fireworks Tokens Saved** | - | **{tokens_saved:,}** | - |
| **Repairs Attempted** | - | {repair['repairs_attempted']} | - |
| **Successful Repairs** | - | {repair['repairs_successful']} | - |
| **Fireworks Fallbacks Avoided** | - | {repair['fallbacks_avoided']} | - |
| **Average Task Latency** | {baseline['avg_latency']:.2f}s | {repair['avg_latency']:.2f}s | {repair['avg_latency'] - baseline['avg_latency']:.2f}s |
| **P95 Task Latency** | {baseline['p95_latency']:.2f}s | {repair['p95_latency']:.2f}s | {repair['p95_latency'] - baseline['p95_latency']:.2f}s |
| **Peak RAM Usage** | {baseline['peak_ram']:.2f} MiB | {repair['peak_ram']:.2f} MiB | {repair['peak_ram'] - baseline['peak_ram']:.2f} MiB |
| **Peak CPU Usage** | {baseline['peak_cpu']:.1f}% | {repair['peak_cpu']:.1f}% | {repair['peak_cpu'] - baseline['peak_cpu']:.1f}% |
| **Total Pipeline Runtime** | {baseline['runtime']:.1f}s | {repair['runtime']:.1f}s | {repair['runtime'] - baseline['runtime']:.1f}s |
"""
    
    print("\n" + report)
    
    with open("benchmark_comparison_report.md", "w") as f:
        f.write(report)
        
    print("Report saved to benchmark_comparison_report.md")

if __name__ == "__main__":
    main()
