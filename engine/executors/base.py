from abc import ABC, abstractmethod
from models.schemas import TaskContext, ExecutionResult

class BaseExecutor(ABC):
    @abstractmethod
    async def execute(self, context: TaskContext) -> ExecutionResult:
        """
        Execute the task given the context.
        Must return an ExecutionResult containing the final answer.
        """
        pass
