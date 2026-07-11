import pytest
import asyncio
from models.schemas import TaskContext, TaskRequest, ExecutionResult
from models.enums import TaskCategory, ExecutionRoute
from engine.executors.python import PythonExecutor

@pytest.mark.asyncio
async def test_python_executor_math():
    prompt = "Calculate 10 + 15"
    req = TaskRequest(task_id="3", prompt=prompt)
    ctx = TaskContext(request=req)
    
    executor = PythonExecutor()
    result = await executor.execute(ctx)
    
    assert result.answer == "25"
    assert result.route_taken == ExecutionRoute.PYTHON

@pytest.mark.asyncio
async def test_python_executor_invalid_math():
    prompt = "What is the meaning of life?"
    req = TaskRequest(task_id="4", prompt=prompt)
    ctx = TaskContext(request=req)
    
    executor = PythonExecutor()
    result = await executor.execute(ctx)
    
    assert result.answer == ""
    assert result.fallback_triggered is True
