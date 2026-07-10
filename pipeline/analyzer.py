import re
from models.schemas import PromptProfile

class PromptAnalyzer:
    @staticmethod
    def analyze(prompt: str) -> PromptProfile:
        word_count = len(prompt.split())
        
        has_code = bool(re.search(r'(def |class |import |```|function)', prompt, re.IGNORECASE))
        has_math = bool(re.search(r'(\d+\s*[\+\-\*\/]\s*\d+|calculate|compute|math)', prompt, re.IGNORECASE))
        has_reasoning = bool(re.search(r'(logic|puzzle|deduce|if and only if|therefore)', prompt, re.IGNORECASE))
        has_extraction = bool(re.search(r'(extract|find|list all|entities)', prompt, re.IGNORECASE))
        
        # Calculate heuristics
        requires_deterministic = has_math or has_code
        reasoning_depth = 3 if has_reasoning else (2 if has_code else 1)
        
        # Complexity Score: (0.0 to 1.0)
        complexity_score = min(1.0, (word_count / 200.0) + (reasoning_depth * 0.15))
        
        estimated_input_tokens = max(1, len(prompt) // 4)
        estimated_output_tokens = 50
        if has_code:
            estimated_output_tokens = 400
        elif has_extraction:
            estimated_output_tokens = 150
            
        return PromptProfile(
            word_count=word_count,
            complexity_score=complexity_score,
            reasoning_depth=reasoning_depth,
            requires_deterministic=requires_deterministic,
            has_code=has_code,
            has_math=has_math,
            has_ner=has_extraction,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens
        )
