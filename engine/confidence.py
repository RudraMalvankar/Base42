import re
from models.schemas import TaskContext, ExecutionResult
from models.enums import ExecutionRoute, TaskCategory
from core.logger import setup_logger

logger = setup_logger("confidence_engine")

class ConfidenceEngine:
    """Zero-token heuristic hallucination and failure detector."""

    HEDGING_TERMS = re.compile(
        r'\b(not sure|maybe|could be|might be|possibly|as an AI|unclear|don\'t know|cannot answer|hard to say)\b', 
        re.IGNORECASE
    )

    @classmethod
    def _calculate_hedging_penalty(cls, text: str) -> float:
        matches = len(cls.HEDGING_TERMS.findall(text))
        return min(0.5, matches * 0.15) # Cap penalty at 0.5

    @classmethod
    def _calculate_repetition_penalty(cls, text: str) -> float:
        # Detect token looping (e.g. "the the the the the") which is a common failure mode for quantized small models
        words = text.split()
        if len(words) < 10:
            return 0.0
            
        loop_count = 0
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                loop_count += 1
                
        return min(0.6, loop_count * 0.2) # High penalty for loops

    @classmethod
    def evaluate(cls, result: ExecutionResult, context: TaskContext) -> bool:
        # We implicitly trust Fireworks 70B and Python for this hackathon's accuracy gate
        if result.route_taken in [ExecutionRoute.FIREWORKS, ExecutionRoute.PYTHON]:
            return True

        answer = result.answer.strip()
        
        # Hard failure
        if not answer or answer == "API Error":
            return False

        confidence_score = 1.0

        # 1. Structural Breakdown Penalty
        # If NER or Logic was expected, and the validator failed to produce a valid JSON array/object
        if context.category in [TaskCategory.NER, TaskCategory.LOGIC]:
            if not (answer.startswith("[") or answer.startswith("{")):
                logger.warning(f"Task {result.task_id}: Confidence drop. Local LLM failed structural JSON validation.")
                confidence_score -= 0.4

        # 2. Linguistic Hedging (Hallucination marker)
        hedging_penalty = cls._calculate_hedging_penalty(answer)
        confidence_score -= hedging_penalty

        # 3. Repetition/Looping (Model breakdown marker)
        rep_penalty = cls._calculate_repetition_penalty(answer)
        confidence_score -= rep_penalty

        logger.info(
            f"Task {result.task_id}: Local LLM Confidence = {confidence_score:.2f} "
            f"(Hedge: -{hedging_penalty:.2f}, Rep: -{rep_penalty:.2f})"
        )

        # 0.75 is a strict threshold. Any structural failure OR heavy hedging forces a fallback.
        return confidence_score >= 0.75
