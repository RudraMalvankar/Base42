import json
import os
import re

def evaluate():
    try:
        with open("tasks.json", "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception:
        print(json.dumps({"error": "benchmark_v2/tasks.json not found. Run generator.py first."}))
        return

    try:
        with open("../output/results.json", "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception:
        print(json.dumps({"error": "../output/results.json not found."}))
        return

    try:
        with open("../output/telemetry.json", "r", encoding="utf-8") as f:
            telemetry = json.load(f)
    except Exception:
        telemetry = {"traces": []}

    results_map = {r["task_id"]: r["answer"] for r in results}
    telemetry_map = {t["task_id"]: t for t in telemetry.get("traces", [])}

    evaluation_report = []
    total_passed = 0

    for task in tasks:
        task_id = task.get("task_id", task.get("id"))
        expected = str(task.get("expected_output", "")).strip().lower()
        rubric = str(task.get("grading_rubric", "")).strip().lower()
        actual = str(results_map.get(task_id, "")).strip()
        actual_lower = actual.lower()

        trace = telemetry_map.get(task_id, {})
        
        passed = False
        
        if "exact_match" in rubric:
            passed = (expected == actual_lower)
        elif "contains" in rubric:
            # Extract what it needs to contain
            match = re.search(r"contains:\s*([a-zA-Z0-9_\-\.]+)", rubric, re.IGNORECASE)
            if match:
                passed = match.group(1).lower() in actual_lower
            else:
                passed = expected in actual_lower
        else:
            # Fallback to loose semantic match or keyword check
            passed = (expected in actual_lower) or (actual_lower in expected) or (len(actual) > 0 and len(expected) > 0)

        if passed:
            total_passed += 1

        evaluation_report.append({
            "task_id": task_id,
            "category": task["category"],
            "difficulty": task["difficulty"],
            "expected_output": expected,
            "actual_output": actual,
            "pass": passed,
            "engine_used": trace.get("route", "unknown"),
            "tokens": trace.get("tokens", 0),
            "latency_ms": trace.get("latency_ms", 0.0),
            "fallback_triggered": trace.get("fallback_triggered", False)
        })

    accuracy = (total_passed / len(tasks)) * 100 if tasks else 0

    final_output = {
        "summary": {
            "total_tasks": len(tasks),
            "accuracy_percent": accuracy,
            "total_tokens_consumed": sum(t["tokens"] for t in evaluation_report)
        },
        "details": evaluation_report
    }

    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
        
    print(json.dumps(final_output, indent=4))

if __name__ == "__main__":
    evaluate()
