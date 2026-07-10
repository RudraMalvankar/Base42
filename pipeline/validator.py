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

class SchemaCoercer:
    """Zero-token heuristic schema mapping for JSON."""
    
    COMMON_MAPPINGS = {
        # NER hallucination mappings
        "name": "entity",
        "value": "entity",
        "text": "entity",
        "category": "type",
        "class": "type",
        "label": "type",
        "entity_name": "entity",
        "entity_type": "type"
    }

    @classmethod
    def coerce(cls, data: dict | list, category: TaskCategory) -> dict | list:
        if category != TaskCategory.NER:
            return data
            
        def coerce_dict(d: dict):
            new_d = {}
            for k, v in d.items():
                lower_k = k.lower()
                if lower_k in cls.COMMON_MAPPINGS:
                    new_d[cls.COMMON_MAPPINGS[lower_k]] = v
                else:
                    new_d[k] = v
            return new_d
            
        if isinstance(data, list):
            return [coerce_dict(item) if isinstance(item, dict) else item for item in data]
        elif isinstance(data, dict):
            return coerce_dict(data)
        return data

class JsonValidator(BaseValidator):
    def __init__(self, category: TaskCategory):
        self.category = category

    def validate(self, result: ExecutionResult) -> ExecutionResult:
        original = result.answer
        text = self.strip_fillers(original)
        text = self.strip_markdown_blocks(text)
        
        try:
            start_obj = text.find("{")
            start_arr = text.find("[")
            start = start_obj if (start_arr == -1 or (start_obj != -1 and start_obj < start_arr)) else start_arr
            
            end_obj = text.rfind("}")
            end_arr = text.rfind("]")
            end = end_obj if (end_arr == -1 or (end_obj != -1 and end_obj > end_arr)) else end_arr
            
            if start != -1 and end != -1 and start < end:
                json_str = text[start:end+1]
                parsed = json.loads(json_str)
                
                # Apply structural schema coercion
                parsed = SchemaCoercer.coerce(parsed, self.category)
                
                result.answer = json.dumps(parsed, separators=(',', ':'))
                return result
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON for task {result.task_id}. Returning raw.")
            
        result.answer = original
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
        
        try:
            val = ast.literal_eval(text)
            if isinstance(val, (int, float)):
                result.answer = str(val)
                return result
        except (ValueError, SyntaxError):
            pass
            
        # Look for explicit linguistic markers denoting the final answer
        marker_regex = r"(?:answer is|equals|=|therefore,?|result is)\s*(-?\d+(?:\.\d+)?)"
        match = re.search(marker_regex, text, re.IGNORECASE)
        if match:
            result.answer = match.group(1)
            return result
            
        # Fallback to pure regex search
        matches = re.findall(r'-?\d+(?:\.\d+)?', text)
        if matches:
            if len(matches) == 1:
                result.answer = matches[0]
            else:
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
        # Pass category to validators that require it
        if validator_class == JsonValidator:
            validator = validator_class(category)
        else:
            validator = validator_class()
        return validator.validate(result)
