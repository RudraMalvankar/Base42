from .base import BaseExecutor
from models.schemas import TaskContext, ExecutionResult
from models.enums import ExecutionRoute
from core.logger import setup_logger
import ast
import operator
import re

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
    
    @classmethod
    def extract_equation(cls, prompt: str) -> str:
        # Strip common natural language wrappers
        clean = re.sub(r'(?i)(what is|calculate|compute|solve|find the value of|equals|answer to)', '', prompt)
        
        # Look for the longest contiguous math expression
        # Matches numbers, operators, parens, modulo, and spaces
        match = re.search(r'([\d\s\+\-\*\/\(\)\.\%]{3,})', clean)
        if match:
            expr = match.group(1).strip()
            # If it's just a raw number with no operators, it's not a real math equation
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
                answer="", # Empty string triggers fallback in ConfidenceEngine
                route_taken=ExecutionRoute.PYTHON
            )
            
        try:
            logger.info(f"Task {context.request.task_id}: Executing math in AST Sandbox -> {equation}")
            answer = MathSandbox.evaluate(equation)
        except Exception:
            answer = "" # Trigger API fallback
            
        return ExecutionResult(
            task_id=context.request.task_id,
            answer=answer,
            route_taken=ExecutionRoute.PYTHON
        )
