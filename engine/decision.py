from models.schemas import TaskContext
from models.enums import TaskCategory, ExecutionRoute

class DecisionEngine:
    @staticmethod
    def route(context: TaskContext) -> ExecutionRoute:
        profile = context.profile
        
        # Fast deterministic check
        if context.category == TaskCategory.MATH and profile.complexity_score < 0.4:
            return ExecutionRoute.PYTHON
            
        # Mathematical Scoring Engine (Phase 3)
        scores = {
            ExecutionRoute.PYTHON: 0.0,
            ExecutionRoute.LOCAL_LLM: 0.0,
            ExecutionRoute.FIREWORKS: 0.0
        }
        
        # Score Local LLM (Accuracy)
        if context.category in [TaskCategory.SENTIMENT, TaskCategory.SUMMARIZATION, TaskCategory.NER, TaskCategory.FACTUAL]:
            scores[ExecutionRoute.LOCAL_LLM] += 0.85
        
        # Penalize Local LLM for complexity
        if profile.reasoning_depth > 2 or profile.complexity_score > 0.6:
            scores[ExecutionRoute.LOCAL_LLM] -= 0.6
            
        # Score Fireworks (High Accuracy, High Cost)
        scores[ExecutionRoute.FIREWORKS] += 0.99
        
        # Cost Penalty (Phase 4 Cost Estimation)
        # Assume max score is 1.0. A high token count reduces Fireworks score to prioritize Local.
        cost_penalty = min(0.6, (profile.estimated_output_tokens / 500.0) * 0.5)
        scores[ExecutionRoute.FIREWORKS] -= cost_penalty
        
        best_route = max(scores.items(), key=lambda x: x[1])[0]
        
        # Absolute Safeguard
        if best_route == ExecutionRoute.LOCAL_LLM and profile.requires_deterministic:
            return ExecutionRoute.FIREWORKS
            
        return best_route
