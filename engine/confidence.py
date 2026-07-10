from models.schemas import ExecutionResult, TaskContext
from models.enums import ExecutionRoute, TaskCategory

class ConfidenceEngine:
    @staticmethod
    def evaluate(result: ExecutionResult, context: TaskContext) -> bool:
        """
        Returns True if the result is confident and trustworthy.
        Returns False if hallucination or failure is detected (triggering an API escalation).
        """
        # If it already used Fireworks, we trust it
        if result.route_taken == ExecutionRoute.FIREWORKS:
            return True
            
        answer = result.answer.strip()
        
        # If the local LLM failed or threw an error
        if not answer or "error" in answer.lower():
            return False
            
        # Specific check for NER: Must look like JSON or a list
        if context.category == TaskCategory.NER:
            if not ("[" in answer and "]" in answer):
                return False
                
        # Heuristic: if the answer is ridiculously long for a factual query, it's probably hallucinating
        if context.category == TaskCategory.FACTUAL and len(answer.split()) > 100:
            return False
            
        return True
