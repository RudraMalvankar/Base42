from abc import ABC, abstractmethod
from typing import List, Dict
from models.schemas import TaskContext, PromptProfile, ExecutionRoute
from models.enums import TaskCategory
from core.logger import setup_logger

logger = setup_logger("decision_engine")

class ExecutorScorer(ABC):
    @abstractmethod
    def calculate_utility(self, profile: PromptProfile) -> float:
        pass

class PythonScorer(ExecutorScorer):
    def calculate_utility(self, profile: PromptProfile) -> float:
        # Python is only useful for deterministic math/code without heavy reasoning
        if not profile.has_math:
            return -1.0
            
        if profile.reasoning_depth > 2:
            return 0.1 # Likely a word problem requiring NLP
            
        # Perfect utility: Free, 100% accurate for basic math, zero latency
        return 1.0

class LocalLLMScorer(ExecutorScorer):
    def __init__(self, max_context_tokens: int = 512):
        self.max_context = max_context_tokens
        self.safe_categories = {
            TaskCategory.SENTIMENT: 0.9,
            TaskCategory.NER: 0.85,
            TaskCategory.FACTUAL: 0.8,
            TaskCategory.SUMMARIZATION: 0.75
        }

    def calculate_utility(self, profile: PromptProfile) -> float:
        # Veto: Out of Memory / Context Window Exceeded
        if profile.estimated_input_tokens + profile.estimated_output_tokens > self.max_context:
            logger.warning(f"Local LLM vetoed: Context overflow ({profile.estimated_input_tokens} > {self.max_context})")
            return -1.0

        # Calculate base accuracy from the semantic profile distribution
        base_accuracy = 0.0
        for category_str, probability in profile.task_types:
            try:
                cat = TaskCategory(category_str)
                # Weighted sum of capabilities
                capability = self.safe_categories.get(cat, 0.3)
                base_accuracy += capability * probability
            except ValueError:
                continue

        # Penalize for cognitive complexity (1.5B models fail on complex reasoning)
        complexity_penalty = (profile.complexity_score * 0.4) + (profile.reasoning_depth * 0.1)
        
        # Utility = Accuracy - Penalty (Cost is 0)
        utility = base_accuracy - complexity_penalty
        
        return max(0.0, utility)

class FireworksScorer(ExecutorScorer):
    def calculate_utility(self, profile: PromptProfile) -> float:
        # Base accuracy is very high for 70B models
        base_accuracy = 0.95
        
        # Calculate token cost (Assume max budget per prompt is ~1000 tokens)
        total_estimated_tokens = profile.estimated_input_tokens + profile.estimated_output_tokens
        
        # Cost Penalty: We aggressively penalize high token consumption
        # If it costs 500 tokens, penalty is 0.5.
        cost_penalty = min(0.9, (total_estimated_tokens / 1000.0))
        
        # Utility = High Accuracy - Cost
        return base_accuracy - cost_penalty

class DecisionEngine:
    def __init__(self):
        # Initialize scorers via Strategy Pattern
        self.scorers: Dict[ExecutionRoute, ExecutorScorer] = {
            ExecutionRoute.PYTHON: PythonScorer(),
            ExecutionRoute.LOCAL_LLM: LocalLLMScorer(max_context_tokens=512),
            ExecutionRoute.FIREWORKS: FireworksScorer()
        }

    def route(self, context: TaskContext) -> ExecutionRoute:
        profile = context.profile
        if not profile:
            logger.error("DecisionEngine received TaskContext with no PromptProfile. Defaulting to Fireworks.")
            return ExecutionRoute.FIREWORKS

        best_route = ExecutionRoute.FIREWORKS
        highest_utility = -float('inf')
        utilities = {}

        for route, scorer in self.scorers.items():
            utility = scorer.calculate_utility(profile)
            utilities[route.value] = round(utility, 3)
            
            # strict greater than ensures we break ties in favor of the order of dictionary keys
            # (which favors Python -> Local -> Fireworks if we sort it, but here we just >)
            # Actually, to favor free tiers on exact ties, we check >=
            if utility >= highest_utility and utility >= 0.0:
                # If tied with Fireworks, favor free tier (which evaluates earlier)
                if utility == highest_utility and route == ExecutionRoute.FIREWORKS:
                    continue # Keep the free tier
                highest_utility = utility
                best_route = route

        logger.info(f"Decision Engine Utilities: {utilities} -> Selected: {best_route.value}")
        return best_route
