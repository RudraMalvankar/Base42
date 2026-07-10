import re
from enum import Enum

class TaskCategory(str, Enum):
    FACTUAL = "factual_knowledge"
    MATH = "math_reasoning"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    NER = "ner"
    DEBUGGING = "code_debug"
    LOGIC = "logic_puzzle"
    CODE_GEN = "code_generation"

# Category keywords to search for
CATEGORY_KEYWORDS = {
    TaskCategory.MATH: [
        r"\bhow many\b", r"\bpercent\b", r"%", r"\btotal\b", r"\bremain",
        r"\bcalculate\b", r"\bsum\b", r"\baverage\b", r"\bprofit\b",
        r"\bdiscount\b", r"\bratio\b",
    ],
    TaskCategory.SENTIMENT: [
        r"\bsentiment\b", r"\bclassify.*review\b", r"\bpositive or negative\b",
        r"\bopinion\b", r"\btone of this review\b", r"\bpositive, negative\b"
    ],
    TaskCategory.SUMMARIZATION: [
        r"\bsummar", r"\bcondense\b", r"\bin one sentence\b", r"\bin exactly one sentence\b",
        r"\btl;?dr\b", r"\bshorten\b",
    ],
    TaskCategory.NER: [
        r"\bnamed entit", r"\bextract.*entit", r"\bpeople, organizations\b",
        r"\bidentify.*(person|organization|location|date)\b",
        r"\bextract all entities\b"
    ],
    TaskCategory.DEBUGGING: [
        r"\bbug\b", r"\bfix\b.*\bfunction\b", r"\bfind and fix\b",
        r"\bwhat'?s wrong with this code\b", r"\bcorrect(ed)? implementation\b",
        r"def\s+\w+\(",
    ],
    TaskCategory.LOGIC: [
        r"\beach own[s]? a different\b", r"\bwho owns\b", r"\bconstraint",
        r"\ball conditions\b", r"\bdeduce\b", r"\bpuzzle\b",
    ],
    TaskCategory.CODE_GEN: [
        r"\bwrite a (python )?function\b", r"\bimplement a function\b",
        r"\bwrite code that\b", r"\bwrite a program\b",
    ],
}

# The order we check them (specific first, factual/default last)
CATEGORY_ORDER = [
    TaskCategory.MATH,
    TaskCategory.SENTIMENT,
    TaskCategory.SUMMARIZATION,
    TaskCategory.NER,
    TaskCategory.DEBUGGING,
    TaskCategory.LOGIC,
    TaskCategory.CODE_GEN
]

def classify_task(prompt: str) -> TaskCategory:
    """
    Heuristically classifies a task prompt into one of the 8 capability categories.
    """
    text = prompt.lower()
    for category in CATEGORY_ORDER:
        for pattern in CATEGORY_KEYWORDS[category]:
            if re.search(pattern, text):
                return category
    return TaskCategory.FACTUAL

def estimate_complexity(prompt: str, category: TaskCategory) -> str:
    """
    Estimates a task prompt's complexity level ('low', 'medium', or 'high')
    based on length and constraint count heuristics.
    """
    length = len(prompt.split())
    # Count basic logical separators as proxy for constraints
    constraint_count = len(re.findall(r"\band\b|\bbut\b|,", prompt.lower()))
    
    if category == TaskCategory.LOGIC:
        # Logic puzzles are highly complex for 3B-class models
        return "high"
    if category == TaskCategory.MATH and constraint_count >= 3:
        return "high"
    if length > 60:
        return "high"
    if length > 25 or constraint_count >= 2:
        return "medium"
    return "low"
