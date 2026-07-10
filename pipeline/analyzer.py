import re
from models.schemas import AnalyzerMetadata

class PromptAnalyzer:
    @staticmethod
    def analyze(prompt: str) -> AnalyzerMetadata:
        word_count = len(prompt.split())
        
        # Regex heuristics for fast zero-token analysis
        has_code = bool(re.search(r'(def |class |import |```|function)', prompt, re.IGNORECASE))
        has_math = bool(re.search(r'(\d+\s*[\+\-\*\/]\s*\d+|calculate|compute|math)', prompt, re.IGNORECASE))
        has_reasoning = bool(re.search(r'(logic|puzzle|deduce|if and only if|therefore)', prompt, re.IGNORECASE))
        has_extraction = bool(re.search(r'(extract|find|list all|entities)', prompt, re.IGNORECASE))
        has_generation = bool(re.search(r'(write|generate|create|summarize)', prompt, re.IGNORECASE))
        
        return AnalyzerMetadata(
            word_count=word_count,
            has_code=has_code,
            has_math=has_math,
            has_reasoning=has_reasoning,
            has_extraction=has_extraction,
            has_generation=has_generation
        )
