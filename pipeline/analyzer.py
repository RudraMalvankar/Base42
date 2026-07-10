"""
Prompt Analyzer: Cascade Classifier Orchestrator.

Layer 1 (Structural) always runs — zero cost, zero latency.
Layer 2 (Semantic) only runs when structural confidence < CONFIDENCE_THRESHOLD.

Design decision: CONFIDENCE_THRESHOLD = 0.85
Tuned conservatively to ensure ~80% of prompts resolve via Layer 1.
"""
from typing import List, Tuple, Optional
from models.enums import TaskCategory
from models.schemas import PromptProfile
from pipeline.prompt_models import ClassificationMethod
from pipeline.structural_extractor import StructuralExtractor
from pipeline.semantic_classifier import SemanticClassifier
from core.logger import setup_logger

logger = setup_logger("prompt_analyzer")

CONFIDENCE_THRESHOLD = 0.85

# Maps structural signals to primary category
STRUCTURAL_CATEGORY_MAP = {
    "has_code_block": TaskCategory.DEBUGGING,   # Will be refined by imperative/what
    "has_math_expression": TaskCategory.MATH,
    "has_ner_signal": TaskCategory.NER,
}

# Output token estimates by category (tuned from real model outputs)
OUTPUT_TOKEN_ESTIMATES = {
    TaskCategory.MATH: 30,
    TaskCategory.SENTIMENT: 15,
    TaskCategory.SUMMARIZATION: 100,
    TaskCategory.NER: 120,
    TaskCategory.DEBUGGING: 350,
    TaskCategory.LOGIC: 200,
    TaskCategory.CODE_GEN: 400,
    TaskCategory.FACTUAL: 60,
}

class PromptAnalyzer:
    def __init__(self):
        self._structural = StructuralExtractor()
        self._semantic: Optional[SemanticClassifier] = SemanticClassifier()

    def analyze(self, prompt: str) -> PromptProfile:
        # --- Layer 1: Always Run ---
        features = self._structural.extract(prompt)
        
        task_types: List[Tuple[str, float]] = []
        primary_category: TaskCategory = TaskCategory.FACTUAL
        method = ClassificationMethod.STRUCTURAL

        if features.structural_confidence >= CONFIDENCE_THRESHOLD:
            # High confidence — build category from structural signals
            if features.has_code_block and features.question_type.value in ("imperative", "how"):
                primary_category = TaskCategory.CODE_GEN
            elif features.has_code_block:
                primary_category = TaskCategory.DEBUGGING
            elif features.has_math_expression:
                primary_category = TaskCategory.MATH
            elif features.has_ner_signal:
                primary_category = TaskCategory.NER
            task_types = [(primary_category.value, features.structural_confidence)]
            method = ClassificationMethod.STRUCTURAL
        else:
            # --- Layer 2: Semantic Fallback ---
            if self._semantic and self._semantic._available:
                semantic_results = self._semantic.classify(prompt)
                if semantic_results:
                    task_types = [(cat.value, prob) for cat, prob in semantic_results]
                    primary_category = semantic_results[0][0]
                    method = ClassificationMethod.SEMANTIC if features.structural_confidence < 0.4 else ClassificationMethod.HYBRID
            else:
                # Degraded mode: pure structural with low confidence
                primary_category = TaskCategory.FACTUAL
                task_types = [(primary_category.value, 0.5)]
                method = ClassificationMethod.STRUCTURAL

        # --- Compute Complexity Score ---
        complexity_score = min(1.0, (
            (features.word_count / 200.0) * 0.3 +
            (features.logical_operator_count / 5.0) * 0.4 +
            (0.3 if features.has_code_block else 0.0)
        ))

        # --- Compute Reasoning Depth (1-5) ---
        reasoning_depth = 1
        if features.logical_operator_count > 3:
            reasoning_depth = 4
        elif features.logical_operator_count > 1:
            reasoning_depth = 3
        elif primary_category in [TaskCategory.DEBUGGING, TaskCategory.CODE_GEN]:
            reasoning_depth = 3
        elif primary_category == TaskCategory.LOGIC:
            reasoning_depth = 5

        estimated_output = OUTPUT_TOKEN_ESTIMATES.get(primary_category, 60)

        logger.info(
            f"Prompt analyzed via {method.value} | "
            f"Category: {primary_category.value} | "
            f"Complexity: {complexity_score:.2f} | "
            f"Structural Confidence: {features.structural_confidence:.2f}"
        )

        return PromptProfile(
            task_types=task_types,
            primary_category=primary_category,
            complexity_score=complexity_score,
            reasoning_depth=reasoning_depth,
            requires_deterministic=features.has_math_expression or features.has_code_block,
            estimated_input_tokens=features.token_estimate,
            estimated_output_tokens=estimated_output,
            classification_method=method,
            # Legacy fields for compatibility with Classifier and Complexity modules
            has_code=features.has_code_block,
            has_math=features.has_math_expression,
            has_ner=features.has_ner_signal,
            word_count=features.word_count,
        )
