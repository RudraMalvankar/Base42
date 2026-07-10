from .base import BaseExecutor
from models.schemas import TaskContext, ExecutionResult
from models.enums import ExecutionRoute
import ast
import operator
import re

class PythonExecutor(BaseExecutor):
    def __init__(self):
        # Safe math evaluation mapping
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg
        }
        
    def _eval_expr(self, node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return self.operators[type(node.op)](self._eval_expr(node.left), self._eval_expr(node.right))
        elif isinstance(node, ast.UnaryOp):
            return self.operators[type(node.op)](self._eval_expr(node.operand))
        else:
            raise TypeError(node)

    async def execute(self, context: TaskContext) -> ExecutionResult:
        prompt = context.request.prompt
        
        # Clean the prompt to extract just the math expression
        # E.g. "Calculate 5 + 3" -> "5+3"
        expr_match = re.search(r'([\d\s\+\-\*\/\(\)\.]+)', prompt)
        
        if not expr_match:
            return ExecutionResult(
                task_id=context.request.task_id,
                answer="Could not parse math expression.",
                route_taken=ExecutionRoute.PYTHON
            )
            
        cleaned_expr = expr_match.group(1).strip()
        
        try:
            # Parse the expression securely
            tree = ast.parse(cleaned_expr, mode='eval').body
            result = self._eval_expr(tree)
            answer = str(result)
        except Exception as e:
            answer = f"Math error: {str(e)}"
            
        return ExecutionResult(
            task_id=context.request.task_id,
            answer=answer,
            route_taken=ExecutionRoute.PYTHON
        )
