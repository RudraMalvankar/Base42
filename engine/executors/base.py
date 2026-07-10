from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
from models.schemas import TaskContext, ExecutionResult

@runtime_checkable
class IExecutor(Protocol):
    async def execute(self, context: TaskContext) -> ExecutionResult:
        ...

class BaseExecutor(ABC, IExecutor):
    @abstractmethod
    async def execute(self, context: TaskContext) -> ExecutionResult:
        """
        Execute the task given the context.
        Must return an ExecutionResult containing the final answer.
        """
        pass
