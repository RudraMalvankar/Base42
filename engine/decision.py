from abc import ABC, abstractmethod
from typing import List, Dict
from dataclasses import dataclass
from models.schemas import TaskContext, PromptProfile, ExecutionRoute
from models.enums import TaskCategory
from core.logger import setup_logger

logger = setup_logger("decision_engine")

@dataclass
class ScoringWeights:
    local_complexity_penalty: float = 0.4
    local_reasoning_penalty: float = 0.1
    fireworks_base_acc: float = 0.99
    fireworks_ambiguity_penalty: float = 0.2
    fireworks_cost_weight: float = 50.0

class ExecutorScorer(ABC):
    def __init__(self, weights: ScoringWeights):
        self.weights = weights
        
    @abstractmethod
    def calculate_utility(self, context: TaskContext) -> float:
        pass

class PythonScorer(ExecutorScorer):
    def calculate_utility(self, context: TaskContext) -> float:
        profile = context.profile
        if not profile.has_math:
            return -1.0
            
        if profile.reasoning_depth > 2:
            return 0.1
            
        if context.failed_attempts > 0 and context.route == ExecutionRoute.PYTHON:
            return -1.0
            
        return 1.0

class LocalLLMScorer(ExecutorScorer):
    def __init__(self, weights: ScoringWeights, max_context_tokens: int = 512):
        super().__init__(weights)
        self.max_context = max_context_tokens
        self.safe_categories = {
            TaskCategory.SENTIMENT: 0.9,
            TaskCategory.NER: 0.85,
            TaskCategory.FACTUAL: 0.8,
            TaskCategory.SUMMARIZATION: 0.75
        }

    def calculate_utility(self, context: TaskContext) -> float:
        profile = context.profile
        
        if context.failed_attempts > 0 and context.route == ExecutionRoute.LOCAL_LLM:
            return -1.0

        if profile.estimated_input_tokens + profile.estimated_output_tokens > self.max_context:
            logger.warning(f"Local LLM vetoed: Context overflow")
            return -1.0

        base_accuracy = 0.0
        for category_str, probability in profile.task_types:
            try:
                cat = TaskCategory(category_str)
                base_accuracy += self.safe_categories.get(cat, 0.3) * probability
            except ValueError:
                continue

        complexity_penalty = (profile.complexity_score * self.weights.local_complexity_penalty) + \
                             (profile.reasoning_depth * self.weights.local_reasoning_penalty)
        
        utility = base_accuracy - complexity_penalty
        return max(0.0, utility)

class FireworksScorer(ExecutorScorer):
    def calculate_utility(self, context: TaskContext) -> float:
        profile = context.profile
        
        base_accuracy = self.weights.fireworks_base_acc - (profile.complexity_score * self.weights.fireworks_ambiguity_penalty)
        
        total_tokens = profile.estimated_input_tokens + profile.estimated_output_tokens
        cost_penalty = min(0.9, (total_tokens / 1000.0) * self.weights.fireworks_cost_weight)
        
        if context.failed_attempts > 0:
            cost_penalty = 0.0
            
        return base_accuracy - cost_penalty

class DecisionEngine:
    def __init__(self, weights: ScoringWeights = None):
        self.weights = weights or ScoringWeights()
        self.scorers: Dict[ExecutionRoute, ExecutorScorer] = {
            ExecutionRoute.PYTHON: PythonScorer(self.weights),
            ExecutionRoute.LOCAL_LLM: LocalLLMScorer(self.weights, max_context_tokens=512),
            ExecutionRoute.FIREWORKS: FireworksScorer(self.weights)
        }

    def route(self, context: TaskContext) -> ExecutionRoute:
        if not context.profile:
            return ExecutionRoute.FIREWORKS

        best_route = ExecutionRoute.FIREWORKS
        highest_utility = -float('inf')
        utilities = {}

        for route, scorer in self.scorers.items():
            utility = scorer.calculate_utility(context)
            utilities[route.value] = round(utility, 3)
            
            if utility >= highest_utility and utility >= 0.0:
                if utility == highest_utility and route == ExecutionRoute.FIREWORKS:
                    continue
                highest_utility = utility
                best_route = route

        logger.info(f"Decision Engine Utilities: {utilities} -> Selected: {best_route.value}")
        return best_route
