import subprocess
import time
import threading

peak_cpu = 0.0
peak_mem = 0.0
is_running = True

cpu_history = []

def monitor():
    global peak_cpu, peak_mem, cpu_history
    while is_running:
        try:
            out = subprocess.check_output(['docker', 'stats', '--no-stream', '--format', '{{.CPUPerc}},{{.MemUsage}}']).decode('utf-8').strip()
            if not out: 
                time.sleep(0.2)
                continue
            for line in out.splitlines():
                if ',' not in line: continue
                cpu_str, mem_str = line.split(',', 1)
                
                # Parse CPU
                c = float(cpu_str.replace('%', '').strip())
                
                # Parse Mem
                mem_used = mem_str.split('/')[0].strip()
                m = 0.0
                if 'MiB' in mem_used or 'MB' in mem_used:
                    m = float(mem_used.replace('MiB', '').replace('MB', '').strip())
                elif 'GiB' in mem_used or 'GB' in mem_used:
                    m = float(mem_used.replace('GiB', '').replace('GB', '').strip()) * 1024
                    
                if c > peak_cpu: peak_cpu = c
                if m > peak_mem: peak_mem = m
                
                cpu_history.append((time.time(), c))
        except Exception as e:
            pass
        time.sleep(0.2)

t = threading.Thread(target=monitor)
t.start()

print("Running pipeline and tracking peak resources...")
subprocess.run('docker run --name base42_bench --rm --cpus="2" --memory="4g" -e FIREWORKS_API_KEY=fw_84W1aV4dLEnLReJ3noVXRh -v "C:/Users/mypc/Desktop/amd:/app" -v "C:/Users/mypc/Desktop/amd/input_samples:/input" -v "C:/Users/mypc/Desktop/amd/output:/output" ghcr.io/rudramalvankar/base42:latest python main.py 2>&1', shell=True)

is_running = False
t.join()

print("\n--- PERFORMANCE METRICS ---")
print(f"Peak RAM Usage: {peak_mem:.2f} MiB")

if cpu_history:
    avg_cpu = sum(c for _, c in cpu_history) / len(cpu_history)
    print(f"Average CPU Usage: {avg_cpu:.2f}%")
    
    # 5-second rolling average
    max_5s = 0.0
    exceeded_2vcpus_for_5s = False
    
    for i in range(len(cpu_history)):
        t_start = cpu_history[i][0]
        window = [c for t, c in cpu_history if t_start <= t <= t_start + 5.0]
        if len(window) > 0:
            avg_5s = sum(window) / len(window)
            if avg_5s > max_5s:
                max_5s = avg_5s
            if avg_5s > 200.0:
                exceeded_2vcpus_for_5s = True
                
    print(f"Maximum sustained CPU (5s rolling): {max_5s:.2f}%")
    print(f"Exceeded 2-vCPU (200%) for 5 consecutive seconds: {exceeded_2vcpus_for_5s}")
else:
    print("No CPU data collected.")

