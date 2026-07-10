from models.schemas import TaskContext
from models.enums import TaskCategory, ComplexityLevel
import re

class ComplexityEstimator:
    @staticmethod
    def estimate(context: TaskContext) -> ComplexityLevel:
        prompt = context.request.prompt
        word_count = context.metadata.word_count
        
        # Heuristic rules
        if word_count > 60 or context.category in [TaskCategory.LOGIC, TaskCategory.CODE_GEN]:
            return ComplexityLevel.HARD
            
        constraints = len(re.findall(r'(must|exactly|only|require|if|not|and)', prompt, re.IGNORECASE))
        if word_count > 25 or constraints > 1 or context.category in [TaskCategory.DEBUGGING]:
            return ComplexityLevel.MEDIUM
            
        return ComplexityLevel.EASY
