import re
from models.schemas import ExecutionResult

class ResultValidator:
    @staticmethod
    def sanitize(result: ExecutionResult) -> ExecutionResult:
        """
        Strips conversational filler and ensures the schema matches what the grader expects.
        """
        answer = result.answer
        
        # Strip common local model conversational filler
        fillers = [
            r"^here is the answer:?\s*",
            r"^here's the.*:?\s*",
            r"^sure,.*:?\s*",
            r"^the answer is:?\s*"
        ]
        
        for filler in fillers:
            answer = re.sub(filler, "", answer, flags=re.IGNORECASE).strip()
            
        # Ensure it's not entirely empty
        if not answer:
            answer = "Failed to generate an answer."
            
        result.answer = answer
        return result
