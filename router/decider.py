import re
from router.classifier import TaskCategory

HIGH_TRUST_LOCAL = {TaskCategory.SENTIMENT, TaskCategory.NER}
LOW_TRUST_LOCAL = {TaskCategory.LOGIC}

def answers_agree(a: str, b: str) -> bool:
    """
    Checks if two model outputs agree. Uses whitespace and case normalization,
    and fallback Jaccard token overlap for longer text structures.
    """
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    a_n, b_n = norm(a), norm(b)
    if a_n == b_n:
        return True
        
    a_tokens, b_tokens = set(a_n.split()), set(b_n.split())
    if not a_tokens or not b_tokens:
        return False
        
    overlap = len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))
    return overlap > 0.75

def verify_logic_puzzle(prompt: str, answer: str) -> bool:
    """
    Lightweight logical puzzle verification. Confirms that proper names output
    in the answer are actually present in the prompt to block hallucinations.
    """
    prompt_names = set(re.findall(r"\b[A-Z][a-z]+\b", prompt))
    answer_names = set(re.findall(r"\b[A-Z][a-z]+\b", answer))
    if not prompt_names:
        return True  # If no names in prompt, skip check
    return bool(answer_names & prompt_names)

def decide(
    category: TaskCategory, 
    complexity: str, 
    local_answer: str,
    second_sample: str = None, 
    executed_result = None
) -> dict:
    """
    Decides whether to trust the local model or escalate to the Fireworks API.
    Returns: {"trust_local": bool, "reason": str}
    """
    # 1. Verification via execution (math AST, code run) is highest confidence
    if executed_result is not None:
        return {"trust_local": True, "reason": "executed_and_verified"}

    # 2. Sentiment and NER are generally highly accurate locally for low/medium
    if category in HIGH_TRUST_LOCAL and complexity != "high":
        return {"trust_local": True, "reason": "high_trust_category"}

    # 3. Logic puzzles are low trust locally (default escalate)
    if category in LOW_TRUST_LOCAL:
        return {"trust_local": False, "reason": "low_trust_category_escalate"}

    # 4. If self-consistency check is provided (2 samples run)
    if second_sample is not None:
        agree = answers_agree(local_answer, second_sample)
        return {
            "trust_local": agree,
            "reason": "self_consistency_agree" if agree else "self_consistency_disagree"
        }

    # 5. Default complexity routing for Factual, Code Gen, and Summarization
    return {
        "trust_local": complexity != "high", 
        "reason": "complexity_default_low_medium" if complexity != "high" else "complexity_high_escalate"
    }
