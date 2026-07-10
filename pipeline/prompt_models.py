from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from models.enums import TaskCategory

class QuestionType(str, Enum):
    WHAT = "what"
    HOW = "how"
    WHICH = "which"
    OPEN = "open"
    IMPERATIVE = "imperative"  # e.g., "Write a function..."

class ClassificationMethod(str, Enum):
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

class StructuralFeatures(BaseModel):
    word_count: int
    token_estimate: int
    has_code_block: bool
    has_math_expression: bool
    has_ner_signal: bool
    logical_operator_count: int
    question_type: QuestionType
    structural_confidence: float

class PromptProfile(BaseModel):
    # Multi-label probability distribution (sorted desc by confidence)
    task_types: List[Tuple[str, float]]
    primary_category: TaskCategory
    complexity_score: float          # 0.0 to 1.0
    reasoning_depth: int             # 1 (retrieve) to 5 (complex multi-hop)
    requires_deterministic: bool
    estimated_input_tokens: int
    estimated_output_tokens: int
    classification_method: ClassificationMethod
    # Legacy compatibility fields
    has_code: bool
    has_math: bool
    has_ner: bool
    word_count: int
