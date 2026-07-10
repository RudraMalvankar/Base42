from abc import ABC, abstractmethod
from typing import Dict, Type
import re
import ast
import json
from models.schemas import ExecutionResult
from models.enums import TaskCategory
from core.logger import setup_logger

logger = setup_logger("validator")

class BaseValidator(ABC):
    @abstractmethod
    def validate(self, result: ExecutionResult) -> ExecutionResult:
        pass
        
    def strip_fillers(self, text: str) -> str:
        fillers = [
            r"^here is the.*:?\s*",
            r"^sure,.*:?\s*",
            r"^the answer is:?\s*"
        ]
        for filler in fillers:
            text = re.sub(filler, "", text, flags=re.IGNORECASE).strip()
        return text

class MathValidator(BaseValidator):
    def validate(self, result: ExecutionResult) -> ExecutionResult:
        result.answer = self.strip_fillers(result.answer)
        # Check if the answer can be evaluated or is a pure number
        try:
            val = ast.literal_eval(result.answer)
            if isinstance(val, (int, float)):
                return result
        except (ValueError, SyntaxError):
            pass
            
        # Fallback regex to just extract the first floating point or int
        match = re.search(r'-?\d+(?:\.\d+)?', result.answer)
        if match:
            result.answer = match.group(0)
            
        return result

class NERValidator(BaseValidator):
    def validate(self, result: ExecutionResult) -> ExecutionResult:
        result.answer = self.strip_fillers(result.answer)
        # Attempt to parse it as JSON
        try:
            if "[" in result.answer and "]" in result.answer:
                start = result.answer.find("[")
                end = result.answer.rfind("]") + 1
                json_str = result.answer[start:end]
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    result.answer = json.dumps(parsed)
        except json.JSONDecodeError:
            logger.warning(f"Failed to validate JSON for NER task {result.task_id}")
        return result

class DefaultValidator(BaseValidator):
    def validate(self, result: ExecutionResult) -> ExecutionResult:
        result.answer = self.strip_fillers(result.answer)
        if not result.answer:
            result.answer = "Failed to generate an answer."
        return result

class ResultValidator:
    _validators: Dict[TaskCategory, Type[BaseValidator]] = {
        TaskCategory.MATH: MathValidator,
        TaskCategory.NER: NERValidator
    }

    @classmethod
    def sanitize(cls, result: ExecutionResult, category: TaskCategory) -> ExecutionResult:
        validator_class = cls._validators.get(category, DefaultValidator)
        validator = validator_class()
        return validator.validate(result)
