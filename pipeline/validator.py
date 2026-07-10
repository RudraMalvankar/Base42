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
        # Aggressively remove conversational filler
        fillers = [
            r"^(?i)here is the.*?:?\s*\n*",
            r"^(?i)sure,.*?:?\s*\n*",
            r"^(?i)the answer is:?\s*\n*",
            r"^(?i)certainly!.*?:?\s*\n*",
            r"^(?i)here is your.*?:?\s*\n*"
        ]
        for filler in fillers:
            text = re.sub(filler, "", text).strip()
        return text

    def strip_markdown_blocks(self, text: str) -> str:
        # Match ```json ... ``` or ```python ... ``` and extract content
        match = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

class JsonValidator(BaseValidator):
    def validate(self, result: ExecutionResult) -> ExecutionResult:
        original = result.answer
        text = self.strip_fillers(original)
        text = self.strip_markdown_blocks(text)
        
        # Aggressive JSON extraction (find first [ or { to last ] or })
        try:
            start_obj = text.find("{")
            start_arr = text.find("[")
            # Get the earliest valid start bracket
            start = start_obj if (start_arr == -1 or (start_obj != -1 and start_obj < start_arr)) else start_arr
            
            end_obj = text.rfind("}")
            end_arr = text.rfind("]")
            # Get the latest valid end bracket
            end = end_obj if (end_arr == -1 or (end_obj != -1 and end_obj > end_arr)) else end_arr
            
            if start != -1 and end != -1 and start < end:
                json_str = text[start:end+1]
                parsed = json.loads(json_str)
                # Dump minified json (saves tokens on final output size constraints if any)
                result.answer = json.dumps(parsed, separators=(',', ':'))
                return result
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON for task {result.task_id}. Returning raw.")
            
        result.answer = original # Fallback to raw if all parsing fails
        return result

class CodeValidator(BaseValidator):
    def validate(self, result: ExecutionResult) -> ExecutionResult:
        text = self.strip_fillers(result.answer)
        text = self.strip_markdown_blocks(text)
        result.answer = text
        return result

class MathValidator(BaseValidator):
    def validate(self, result: ExecutionResult) -> ExecutionResult:
        text = self.strip_fillers(result.answer)
        # Attempt to evaluate if it's purely a numeric response
        try:
            val = ast.literal_eval(text)
            if isinstance(val, (int, float)):
                result.answer = str(val)
                return result
        except (ValueError, SyntaxError):
            pass
            
        # Fallback to extract the last number mentioned (often the final answer)
        matches = re.findall(r'-?\d+(?:\.\d+)?', text)
        if matches:
            result.answer = matches[-1]
        else:
            result.answer = text
            
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
        TaskCategory.NER: JsonValidator,
        TaskCategory.LOGIC: JsonValidator, # Often expects strict schema
        TaskCategory.CODE_GEN: CodeValidator,
        TaskCategory.DEBUGGING: CodeValidator,
    }

    @classmethod
    def sanitize(cls, result: ExecutionResult, category: TaskCategory) -> ExecutionResult:
        validator_class = cls._validators.get(category, DefaultValidator)
        validator = validator_class()
        return validator.validate(result)
