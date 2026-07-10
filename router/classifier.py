import re
from enum import Enum

class TaskCategory(str, Enum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    NER = "ner"
    DEBUGGING = "debugging"
    LOGIC = "logic"
    CODE_GEN = "code_gen"

def classify_task(prompt: str) -> TaskCategory:
    """
    Heuristically classifies a task prompt into one of the 8 capability categories
    based on keyword matching, regex patterns, and context.
    """
    p_lower = prompt.lower()
    
    # 1. Sentiment Classification
    sentiment_keywords = [
        "sentiment", "classify the sentiment", "classify sentiment",
        "review sentiment", "positive or negative", "positive, negative"
    ]
    if any(k in p_lower for k in sentiment_keywords):
        return TaskCategory.SENTIMENT
        
    # 2. Named Entity Recognition (NER)
    ner_keywords = [
        "named entities", "named entity", "extract all entities",
        "extract entities", "extract all named entities", "extract named entities"
    ]
    if any(k in p_lower for k in ner_keywords) or (
        "extract" in p_lower and ("person" in p_lower or "location" in p_lower or "organization" in p_lower or "org" in p_lower)
    ):
        return TaskCategory.NER
        
    # 3. Code Debugging
    debugging_keywords = [
        "has a bug", "bug:", "find and fix", "correct the implementation",
        "debugging", "debug this", "fix the bug", "correct this code"
    ]
    if any(k in p_lower for k in debugging_keywords) or (
        "bug" in p_lower and ("def " in prompt or "class " in prompt or "function" in p_lower)
    ):
        return TaskCategory.DEBUGGING

    # 4. Code Generation
    code_gen_keywords = [
        "write a python function", "write a function", "implement a function",
        "write code", "python function that", "write a script", "write an algorithm",
        "create a function", "write a program"
    ]
    if any(k in p_lower for k in code_gen_keywords) or (
        "function" in p_lower and ("python" in p_lower or "code" in p_lower) and ("return" in p_lower or "write" in p_lower)
    ):
        return TaskCategory.CODE_GEN

    # 5. Text Summarisation
    summarization_keywords = [
        "summarize", "summarise", "condense", "summary",
        "in one sentence", "in exactly one sentence", "in a single sentence"
    ]
    if any(k in p_lower for k in summarization_keywords):
        return TaskCategory.SUMMARIZATION

    # 6. Mathematical Reasoning
    math_keywords = [
        "how many", "calculate", "solve the math", "arithmetic", "percentage",
        "ratio", "multiplied by", "divided by", "math problem", "solve for"
    ]
    # Simple regex to detect multi-step math or numbers combined with math questions
    has_math_ops = bool(re.search(r'\d+\s*[%+\-*/]\s*\d+', prompt))
    has_math_words = any(k in p_lower for k in math_keywords)
    if (has_math_words or has_math_ops) and not any(w in p_lower for w in ["python", "code", "function"]):
        return TaskCategory.MATH

    # 7. Logical / Deductive Reasoning
    logic_keywords = [
        "puzzle", "logic", "deduce", "deductive", "who owns", "different pet",
        "friends", "constraint", "riddle", "truth teller", "liar"
    ]
    if any(k in p_lower for k in logic_keywords):
        return TaskCategory.LOGIC

    # 8. Factual Knowledge (Default)
    return TaskCategory.FACTUAL
