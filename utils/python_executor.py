"""
python_executor.py

Safe evaluation of local python code and arithmetic. Used for:
1. math_reasoning: converting word problems to arithmetic formulas and executing them.
2. code_debug: smoke-testing bug fixes locally before trusting them.
"""

import ast
import operator as op
import re
from utils.logger import setup_logger

logger = setup_logger("python_executor")

# Safe arithmetic operators mapping
_ALLOWED_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
    ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
}

def safe_eval(expr: str):
    """
    Evaluates a pure mathematical expression safely using AST.
    Does not allow variable names, lookups, or function calls.
    """
    node = ast.parse(expr, mode="eval").body
    return _eval_node(node)

def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Non-numeric constant found")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Disallowed node type: {type(node)}")

def try_solve_arithmetic(expression: str):
    """
    Safely evaluates an arithmetic expression and returns a float/int,
    or None if the expression is invalid or unsafe.
    """
    # Keep only numbers, basic operators, and brackets
    cleaned = re.sub(r"[^0-9+\-*/().% ]", "", expression)
    if not cleaned.strip():
        return None
    try:
        return safe_eval(cleaned)
    except Exception as e:
        logger.debug(f"Arithmetic evaluation failed for '{cleaned}': {e}")
        return None

def extract_function_source(prompt: str) -> str:
    """
    Finds and extracts a Python function definition block from the prompt.
    """
    match = re.search(r"(def\s+\w+\(.*?\):.*)", prompt, re.DOTALL)
    return match.group(1).strip() if match else ""

def find_function_name(source: str) -> str:
    """
    Extracts the name of the function from its source code.
    """
    match = re.search(r"def\s+(\w+)\(", source)
    return match.group(1) if match else ""

def run_function(source: str, func_name: str, args_list: list) -> list:
    """
    Runs a Python function source in a highly restricted namespace for basic execution verification.
    Returns a list of tuples: (arguments, result_or_error_string).
    """
    restricted_globals = {
        "__builtins__": {
            "len": len, "range": range, "max": max, "min": min,
            "sum": sum, "sorted": sorted, "list": list, "set": set,
            "int": int, "float": float, "str": str, "bool": bool,
            "abs": abs, "enumerate": enumerate, "isinstance": isinstance,
        }
    }
    local_ns = {}
    try:
        # Execute the function definition block in the restricted namespace
        exec(source, restricted_globals, local_ns)
    except Exception as e:
        logger.warning(f"Failed to define function {func_name} in executor: {e}")
        return [((), f"ERROR: Definition failed: {e}")]

    func = local_ns.get(func_name)
    if func is None:
        return [((), f"ERROR: Function {func_name} not found after exec")]

    results = []
    for args in args_list:
        try:
            # Call the function with arguments
            res = func(*args)
            results.append((args, res))
        except Exception as e:
            results.append((args, f"ERROR: Runtime execution failed: {e}"))
    return results
