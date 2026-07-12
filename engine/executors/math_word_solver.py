import json, re
from engine.executors.python import MathSandbox

EXTRACTION_PROMPT = """Convert this word problem into an ordered list of arithmetic 
expressions that together answer every question asked. Use only numbers and + - * / ().
Reference an earlier step's result as $1, $2, etc. (1-indexed).
Output strict JSON only, no explanation:
{{"steps": ["<expr>", "<expr>"], "labels": ["<what step 1 answers>", "<what step 2 answers>"]}}

Problem: {prompt}"""

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def solve_word_problem(prompt: str, local_llm_call):
    raw = local_llm_call(EXTRACTION_PROMPT.format(prompt=prompt))
    try:
        plan = json.loads(_strip_fences(raw))
        steps = plan["steps"]
        labels = plan.get("labels", [])
    except Exception:
        return None  # fall through to Fireworks

    results = []
    for expr in steps:
        # substitute $1, $2 with prior results before evaluating
        for i, r in enumerate(results, start=1):
            expr = expr.replace(f"${i}", str(r))
        
        # Ensure only safe characters are evaluated
        clean_expr = re.sub(r"[^0-9+\-*/(). ]", "", expr)
        try:
            val = float(MathSandbox.evaluate(clean_expr))
            results.append(val)
        except Exception:
            return None

    # Build a clean, human-readable answer instead of dumping raw numbers
    parts = []
    for i, val in enumerate(results):
        val_str = f"{val:.2f}".rstrip("0").rstrip(".") if isinstance(val, float) else str(val)
        parts.append(val_str)
    
    return ", ".join(parts) if len(parts) > 1 else (parts[0] if parts else None)
