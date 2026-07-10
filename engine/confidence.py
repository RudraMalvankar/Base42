from models.schemas import ExecutionResult, TaskContext
from models.enums import ExecutionRoute, TaskCategory
import ast
import json

class ConfidenceEngine:
    @staticmethod
    def calculate_confidence(result: ExecutionResult, context: TaskContext) -> float:
        """
        Calculates a mathematical confidence score between 0.0 and 1.0.
        Conf = V_R (Rule Validation) * V_D (Deterministic) * V_H (Heuristic)
        If Conf < 0.6, escalation is triggered.
        """
        # API fallbacks are trusted by definition
        if result.route_taken == ExecutionRoute.FIREWORKS:
            return 1.0
            
        answer = result.answer.strip()
        if not answer or "error" in answer.lower():
            return 0.0 # V_D fails completely
            
        v_r = 1.0 # Rule Validation
        v_d = 1.0 # Deterministic Validation
        v_h = 1.0 # Heuristic Validation
        
        if context.category == TaskCategory.MATH:
            try:
                ast.literal_eval(answer)
                v_d = 1.0
            except (ValueError, SyntaxError):
                v_d = 0.5 # Requires regex stripping, confidence reduced
                
        elif context.category == TaskCategory.NER:
            if "[" in answer and "]" in answer:
                try:
                    json.loads(answer[answer.find("["):answer.rfind("]")+1])
                    v_r = 1.0
                except json.JSONDecodeError:
                    v_r = 0.0
            else:
                v_r = 0.0
                
        elif context.category == TaskCategory.FACTUAL:
            if len(answer.split()) > 100:
                v_h = 0.2
                
        return v_r * v_d * v_h

    @staticmethod
    def evaluate(result: ExecutionResult, context: TaskContext) -> bool:
        confidence = ConfidenceEngine.calculate_confidence(result, context)
        return confidence >= 0.6
