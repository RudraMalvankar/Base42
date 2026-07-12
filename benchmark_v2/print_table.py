import json
from collections import defaultdict

with open('evaluation_report.json') as f:
    d = json.load(f)

details = d['details']
agg = defaultdict(lambda: {'passed':0,'total':0,'tokens':0,'latency':0,'fallbacks':0})

for t in details:
    k = (t['category'], t['difficulty'])
    agg[k]['passed'] += int(t['pass'])
    agg[k]['total'] += 1
    agg[k]['tokens'] += t['tokens']
    agg[k]['latency'] += t['latency_ms']
    agg[k]['fallbacks'] += int(t['fallback_triggered'])

print('| Category | Difficulty | Accuracy (%) | Tokens Used | Avg Latency (s) | Fallbacks |')
print('| :--- | :--- | :--- | :--- | :--- | :--- |')
for k, v in sorted(agg.items()):
    acc = v['passed'] / v['total'] * 100
    lat = v['latency'] / v['total'] / 1000
    print(f"| {k[0]} | {k[1]} | {acc:.1f}% | {v['tokens']} | {lat:.2f}s | {v['fallbacks']} |")
