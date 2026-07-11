from models.schemas import TaskContext, PromptProfile
from models.enums import TaskCategory
import re

class TaskClassifier:
    @staticmethod
    def classify(context: TaskContext) -> TaskCategory:
        prompt = context.request.prompt.lower()
        meta = context.profile
        
        # Heuristic rules for classification
        if meta.has_math or re.search(r'(\+|-|\*|/|equation|calculate)', prompt):
            return TaskCategory.MATH
            
        if re.search(r'(sentiment|positive|negative|neutral)', prompt):
            return TaskCategory.SENTIMENT
            
        if re.search(r'(summarize|summary|tldr|briefly)', prompt):
            return TaskCategory.SUMMARIZATION
            
        if meta.has_extraction or re.search(r'(ner|named entities|person|organization|location)', prompt):
            return TaskCategory.NER
            
        if re.search(r'(bug|debug|fix|error|traceback)', prompt):
            return TaskCategory.DEBUGGING
            
        if meta.has_reasoning or re.search(r'(logic|puzzle|deduce)', prompt):
            return TaskCategory.LOGIC
            
        if meta.has_code or re.search(r'(write.*code|generate.*python|script)', prompt):
            return TaskCategory.CODE_GEN
            
        if re.search(r'(design.*system|architecture|distributed.*system|compare.*database|infrastructure|pseudocode)', prompt):
            return TaskCategory.ARCHITECTURE
            
        return TaskCategory.FACTUAL
