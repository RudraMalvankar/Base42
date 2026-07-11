from .base import BaseExecutor
from models.schemas import TaskContext, ExecutionResult
from models.enums import ExecutionRoute
from core.logger import setup_logger
import ast
import operator
import re

import math

logger = setup_logger("python_executor")

class MathSandbox:
    """Secure deterministic evaluator for math expressions."""
    
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv
    }

    MATH_FUNCTIONS = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'pow': math.pow,
        'abs': abs
    }
    
    @classmethod
    def extract_equation(cls, prompt: str) -> str:
        # Prevent parsing dates (e.g. 2024-05-12) as math unless explicit words exist
        if re.search(r'\b\d{4}-\d{2}-\d{2}\b', prompt) or re.search(r'\b\d{2}/\d{2}/\d{4}\b', prompt):
            if not re.search(r'(?i)(calculate|compute|solve|\+)', prompt):
                return ""

        clean = re.sub(r'(?i)(what is|calculate|compute|solve|find the value of|equals|answer to)', '', prompt)
        
        # Match function calls like math.sqrt(144) or pure equations
        match = re.search(r'((?:math\.)?[a-z]{3,5}\s*\([\d\s\+\-\*\/\.\%]+\)|[\d\s\+\-\*\/\(\)\.\%]{3,})', clean, re.IGNORECASE)
        if match:
            expr = match.group(1).strip()
            if expr and not re.fullmatch(r'[\d\.\s]+', expr):
                return expr
        return ""

    @classmethod
    def _eval_node(cls, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex)):
                return node.value
            raise TypeError("Only numeric constants allowed")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in cls.OPERATORS:
                raise TypeError(f"Unsupported operator: {type(node.op)}")
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            if type(node.op) in (ast.Div, ast.Mod, ast.FloorDiv) and right == 0:
                raise ZeroDivisionError("Division by zero")
            return cls.OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in cls.OPERATORS:
                raise TypeError(f"Unsupported unary operator: {type(node.op)}")
            return cls.OPERATORS[type(node.op)](cls._eval_node(node.operand))
        elif isinstance(node, ast.Call):
            # Allow whitelisted math functions like sqrt(), math.sin()
            if isinstance(node.func, ast.Name) and node.func.id in cls.MATH_FUNCTIONS:
                func = cls.MATH_FUNCTIONS[node.func.id]
                args = [cls._eval_node(arg) for arg in node.args]
                return func(*args)
            elif isinstance(node.func, ast.Attribute) and node.func.attr in cls.MATH_FUNCTIONS:
                func = cls.MATH_FUNCTIONS[node.func.attr]
                args = [cls._eval_node(arg) for arg in node.args]
                return func(*args)
            raise TypeError(f"Unsupported function call")
        else:
            raise TypeError(f"Unsupported node type: {type(node)}")

    @classmethod
    def evaluate(cls, equation: str) -> str:
        try:
            tree = ast.parse(equation, mode='eval').body
            result = cls._eval_node(tree)
            # Format nicely, drop .0 for integers
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            return str(result)
        except Exception as e:
            logger.warning(f"MathSandbox failed to evaluate '{equation}': {e}")
            raise ValueError(f"Math Error: {e}")

class PythonExecutor(BaseExecutor):
    async def execute(self, context: TaskContext) -> ExecutionResult:
        prompt = context.request.prompt
        
        equation = MathSandbox.extract_equation(prompt)
        
        if not equation:
            # Empty means it was likely a word problem we failed to parse
            logger.warning(f"Task {context.request.task_id}: Could not extract equation. Triggering API fallback.")
            return ExecutionResult(
                task_id=context.request.task_id,
                answer="",
                route_taken=ExecutionRoute.PYTHON,
                fallback_triggered=True
            )
            
        try:
            logger.info(f"Task {context.request.task_id}: Executing math in AST Sandbox -> {equation}")
            answer = MathSandbox.evaluate(equation)
        except Exception:
            answer = "" # Trigger API fallback
            
        return ExecutionResult(
            task_id=context.request.task_id,
            answer=answer,
            route_taken=ExecutionRoute.PYTHON,
            fallback_triggered=(answer == "")
        )
