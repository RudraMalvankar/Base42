from dataclasses import dataclass
from models.enums import TaskCategory
from pipeline.prompt_models import StructuralFeatures

@dataclass
class TokenPrediction:
    input_tokens: int
    predicted_output_tokens: int
    confidence_interval: float

class TokenPredictor:
    """Predicts token consumption mathematically based on task features."""
    
    # Hard boundaries for AMD Hackathon API budgets
    MAX_OUTPUT = 800
    MIN_OUTPUT = 20
    
    @classmethod
    def predict(cls, prompt: str, category: TaskCategory, features: StructuralFeatures) -> TokenPrediction:
        input_tokens = features.token_estimate
        output_tokens = 0
        confidence = 0.9
        
        if category == TaskCategory.MATH:
            output_tokens = 30 + (features.logical_operator_count * 15)
        elif category == TaskCategory.FACTUAL:
            output_tokens = 40 + (features.logical_operator_count * 10)
        elif category == TaskCategory.SENTIMENT:
            output_tokens = cls.MIN_OUTPUT
        elif category == TaskCategory.SUMMARIZATION:
            output_tokens = max(cls.MIN_OUTPUT, int(input_tokens * 0.25))
            confidence = 0.7 # Summarization length is highly variable
        elif category == TaskCategory.NER:
            output_tokens = min(int(input_tokens * 0.5), 300)
        elif category == TaskCategory.CODE_GEN:
            output_tokens = 150 + (features.logical_operator_count * 30)
            confidence = 0.6
        elif category == TaskCategory.DEBUGGING:
            output_tokens = 100 + (features.word_count) 
        elif category == TaskCategory.LOGIC:
            output_tokens = 150 + (features.logical_operator_count * 40)
            confidence = 0.5
        else:
            output_tokens = 100
            
        # Detect potential foreign language (crude heuristic: non-ascii characters)
        # Tokenizers encode non-Latin scripts inefficiently (up to 3x more tokens)
        if not prompt.isascii():
            output_tokens = int(output_tokens * 2.5)
            input_tokens = int(input_tokens * 2.5)
            confidence -= 0.2
            
        final_output = max(cls.MIN_OUTPUT, min(cls.MAX_OUTPUT, int(output_tokens)))
        
        return TokenPrediction(
            input_tokens=input_tokens,
            predicted_output_tokens=final_output,
            confidence_interval=max(0.1, confidence)
        )
