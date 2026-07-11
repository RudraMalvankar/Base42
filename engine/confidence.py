import re
from models.schemas import TaskContext, ExecutionResult
from models.enums import ExecutionRoute, TaskCategory
from core.logger import setup_logger

logger = setup_logger("confidence_engine")

class ConfidenceEngine:
    """Zero-token heuristic hallucination and failure detector."""

    HEDGING_TERMS = re.compile(
        r'\b('
        r'not sure|maybe|could be|might be|possibly|as an AI|unclear|don\'t know|cannot answer|hard to say|' # English
        r'je ne suis pas sûr|peut-être|pourrait être|je ne sais pas|' # French
        r'no estoy seguro|tal vez|podría ser|no sé|' # Spanish
        r'ich bin mir nicht sicher|vielleicht|könnte sein|ich weiß nicht' # German
        r')\b', 
        re.IGNORECASE
    )

    @classmethod
    def _calculate_hedging_penalty(cls, text: str) -> float:
        matches = len(cls.HEDGING_TERMS.findall(text))
        return min(0.5, matches * 0.15) # Cap penalty at 0.5

    @classmethod
    def _calculate_repetition_penalty(cls, text: str) -> float:
        words = text.split()
        if len(words) < 10:
            return 0.0
            
        loop_count = 0
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                loop_count += 1
                
        return min(0.6, loop_count * 0.2)

    @classmethod
    def evaluate(cls, result: ExecutionResult, context: TaskContext) -> bool:
        # We implicitly trust Python for this hackathon's accuracy gate as it is deterministic
        if result.route_taken == ExecutionRoute.PYTHON:
            return bool(result.answer.strip())

        answer = result.answer.strip()
        
        # Hard failure
        if not answer or answer == "API Error":
            return False

        confidence_score = 1.0

        # 1. Structural Breakdown Penalty
        if context.category in [TaskCategory.NER, TaskCategory.LOGIC]:
            if not (answer.startswith("[") or answer.startswith("{")):
                logger.warning(f"Task {result.task_id}: Confidence drop. Failed structural JSON validation.")
                confidence_score -= 0.4

        # 2. Linguistic Hedging
        hedging_penalty = cls._calculate_hedging_penalty(answer)
        confidence_score -= hedging_penalty

        # 3. Repetition/Looping
        rep_penalty = cls._calculate_repetition_penalty(answer)
        confidence_score -= rep_penalty

        logger.info(
            f"Task {result.task_id}: Confidence = {confidence_score:.2f} "
            f"(Hedge: -{hedging_penalty:.2f}, Rep: -{rep_penalty:.2f})"
        )

        # Dynamic Threshold: Fireworks gets a slightly more lenient threshold to prevent infinite fallback loops
        threshold = 0.60 if result.route_taken == ExecutionRoute.FIREWORKS else 0.75
        return confidence_score >= threshold
