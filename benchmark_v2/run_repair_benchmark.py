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

def main():
    print("Backing up baseline evaluation report...")
    if os.path.exists("evaluation_report.json"):
        shutil.copy("evaluation_report.json", "baseline_report.json")
    else:
        print("Error: evaluation_report.json not found. Make sure you are in benchmark_v2 directory.")
        return

    print("Copying 240 benchmark tasks to input_samples...")
    shutil.copy("tasks.json", "../input_samples/tasks.json")
    
    # Reset monitor stats
    global peak_cpu, peak_mem, cpu_history, is_monitoring
    peak_cpu = 0.0
    peak_mem = 0.0
    cpu_history = []
    
    # Start monitor thread
    is_monitoring = True
    monitor_thread = threading.Thread(target=monitor_container, args=("base42_bench",))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run container with ENABLE_LOCAL_REPAIR=True
    cmd = [
        "docker", "run", "--name", "base42_bench", "--rm",
        "--cpus=2", "--memory=4g",
        "-v", f"{os.path.abspath('../input_samples')}:/input",
        "-v", f"{os.path.abspath('../output')}:/output",
        "-v", f"{os.path.abspath('../engine')}:/app/engine",
        "-v", f"{os.path.abspath('../pipeline')}:/app/pipeline",
        "-v", f"{os.path.abspath('../models')}:/app/models",
        "-v", f"{os.path.abspath('../main.py')}:/app/main.py",
        "-v", f"{os.path.abspath('../config.py')}:/app/config.py",
        "-e", "FIREWORKS_API_KEY=fw_84W1aV4dLEnLReJ3noVXRh",
        "-e", "ENABLE_LOCAL_REPAIR=True"
    ]
    cmd.append("ghcr.io/rudramalvankar/base42:latest")
    
    print("\n--- RUNNING REPAIR BENCHMARK (ENABLE_LOCAL_REPAIR=True) ---")
    start_time = time.perf_counter()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    total_runtime = time.perf_counter() - start_time
    
    # Stop monitoring
    is_monitoring = False
    monitor_thread.join()
    
    print("Evaluating results...")
    # Run evaluator.py to produce the new evaluation_report.json
    subprocess.run(["python", "evaluator.py"])
    
    # Load both reports
    with open("baseline_report.json", "r") as f:
        baseline_report = json.load(f)
    with open("evaluation_report.json", "r") as f:
        repair_report = json.load(f)
        
    try:
        with open("../output/telemetry.json", "r") as f:
            telemetry = json.load(f)
    except Exception:
        telemetry = {"summary": {}, "traces": []}
        
    traces = telemetry.get("traces", [])
    repairs_attempted = sum(t.get("repairs_attempted", 0) for t in traces)
    repairs_successful = sum(t.get("repairs_successful", 0) for t in traces)
    fallbacks_avoided = sum(t.get("api_fallbacks_avoided", 0) for t in traces)
    
    baseline_summary = baseline_report["summary"]
    repair_summary = repair_report["summary"]
    
    # Collect latencies
    latencies = [t.get("latency_ms", 0.0) for t in traces]
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies) if latencies else 0.0
    
    # Load baseline latencies
    baseline_traces = baseline_report.get("details", [])
    baseline_latencies = [t.get("latency_ms", 0.0) for t in baseline_traces]
    baseline_avg_latency = statistics.mean(baseline_latencies) if baseline_latencies else 0.0
    baseline_p95_latency = statistics.quantiles(baseline_latencies, n=20)[18] if len(baseline_latencies) >= 20 else max(baseline_latencies) if baseline_latencies else 0.0
    
    tokens_saved = baseline_summary["total_tokens_consumed"] - repair_summary["total_tokens_consumed"]
    
    report = f"""# Benchmark Comparison: Local Self-Repair vs Baseline

Dataset: 240 custom benchmark tasks (8 categories, 3 difficulties).

| Metric | Baseline (No Repair) | Local Self-Repair (Active) | Difference |
| :--- | :--- | :--- | :--- |
| **Accuracy** | {baseline_summary['accuracy_percent']:.1f}% | {repair_summary['accuracy_percent']:.1f}% | {repair_summary['accuracy_percent'] - baseline_summary['accuracy_percent']:.1f}% |
| **Fireworks Tokens Consumed** | {baseline_summary['total_tokens_consumed']:,} | {repair_summary['total_tokens_consumed']:,} | {repair_summary['total_tokens_consumed'] - baseline_summary['total_tokens_consumed']:,} |
| **Fireworks Tokens Saved** | - | **{tokens_saved:,}** | - |
| **Repairs Attempted** | 0 | {repairs_attempted} | +{repairs_attempted} |
| **Successful Repairs** | 0 | {repairs_successful} | +{repairs_successful} |
| **Fireworks Fallbacks Avoided** | 0 | {fallbacks_avoided} | +{fallbacks_avoided} |
| **Average Task Latency** | {baseline_avg_latency / 1000.0:.2f}s | {avg_latency / 1000.0:.2f}s | {(avg_latency - baseline_avg_latency) / 1000.0:.2f}s |
| **P95 Task Latency** | {baseline_p95_latency / 1000.0:.2f}s | {p95_latency / 1000.0:.2f}s | {(p95_latency - baseline_p95_latency) / 1000.0:.2f}s |
| **Peak RAM Usage** | N/A | {peak_mem:.2f} MiB | - |
| **Peak CPU Usage** | N/A | {peak_cpu:.1f}% | - |
| **Total Pipeline Runtime** | ~236.0s | {total_runtime:.1f}s | {total_runtime - 236.0:.1f}s |
"""
    
    print("\n" + report)
    
    with open("../benchmark_comparison_report.md", "w") as f:
        f.write(report)
    print("Report written to ../benchmark_comparison_report.md")

if __name__ == "__main__":
    main()
