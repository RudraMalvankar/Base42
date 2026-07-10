from models.schemas import TaskContext
from models.enums import TaskCategory, ComplexityLevel, ExecutionRoute

class DecisionEngine:
    @staticmethod
    def route(context: TaskContext) -> ExecutionRoute:
        # High trust categories that can be handled locally if not hard
        local_categories = [
            TaskCategory.SENTIMENT, 
            TaskCategory.NER, 
            TaskCategory.SUMMARIZATION, 
            TaskCategory.FACTUAL
        ]
        
        if context.category in local_categories and context.complexity != ComplexityLevel.HARD:
            return ExecutionRoute.LOCAL_LLM
            
        # Fast path for basic math
        if context.category == TaskCategory.MATH and context.complexity == ComplexityLevel.EASY:
            return ExecutionRoute.PYTHON
            
        # Default fallback is Fireworks
        return ExecutionRoute.FIREWORKS
