import re
from typing import List
from models.schemas import TaskContext, ExecutionResult
from models.enums import TaskCategory
from dataclasses import dataclass, field

@dataclass
class SubTask:
    id: str
    prompt: str
    category: TaskCategory
    dependencies: List[str] = field(default_factory=list)

class DependencyParser:
    """Zero-token heuristic NLP parser to detect sequential execution order."""
    SEQUENTIAL_MARKERS = re.compile(
        r'\b(then|after that|followed by|and finally|once that is done)\b', 
        re.IGNORECASE
    )

    @classmethod
    def parse_order(cls, prompt: str, num_tasks: int) -> List[List[int]]:
        """Returns an adjacency list of dependencies (idx -> [depends_on_idx])."""
        dependencies = [[] for _ in range(num_tasks)]
        
        # If we detect explicit sequential words, we assume sequential dependency (Task 1 depends on Task 0)
        if cls.SEQUENTIAL_MARKERS.search(prompt) and num_tasks > 1:
            for i in range(1, num_tasks):
                dependencies[i].append(i - 1)
                
        return dependencies

class PromptRewriter:
    """Rewrites prompts for sub-tasks to cleanly isolate focus and prevent LLM confusion."""
    @classmethod
    def rewrite(cls, original: str, category: TaskCategory, depends_on: List[str]) -> str:
        prefix = ""
        if depends_on:
            dep_str = ", ".join(depends_on)
            prefix = f"[DEPENDENCY DETECTED: Use the output from task(s) {dep_str} to complete this step]\n"
            
        instructions = {
            TaskCategory.SUMMARIZATION: "Extract and summarize the main points from the text provided.",
            TaskCategory.NER: "Extract all named entities (people, places, organizations) as a strict JSON list.",
            TaskCategory.SENTIMENT: "Analyze the sentiment (positive, negative, neutral) of the text.",
            TaskCategory.MATH: "Solve only the mathematical equations or logic presented.",
            TaskCategory.CODE_GEN: "Write or generate the requested code.",
            TaskCategory.DEBUGGING: "Identify bugs and provide fixed code."
        }
        
        instruction = instructions.get(category, f"Focus only on the {category.value} requirement.")
        
        # Structured layout prevents attention dilution
        return f"{prefix}INSTRUCTION: {instruction}\n\nORIGINAL REQUEST:\n{original}"

class TaskPlanner:
    def plan(self, context: TaskContext) -> List[SubTask]:
        profile = context.profile
        
        valid_categories = []
        for cat_str, prob in profile.task_types:
            if prob > 0.6: 
                valid_categories.append(TaskCategory(cat_str))
                
        valid_categories = valid_categories[:2]
        
        if len(valid_categories) <= 1:
            return [SubTask(id=f"{context.request.task_id}_0", prompt=context.request.prompt, category=context.category)]
            
        deps = DependencyParser.parse_order(context.request.prompt, len(valid_categories))
        
        sub_tasks = []
        for idx, cat in enumerate(valid_categories):
            task_id = f"{context.request.task_id}_{idx}"
            dependent_ids = [f"{context.request.task_id}_{d_idx}" for d_idx in deps[idx]]
            
            rewritten_prompt = PromptRewriter.rewrite(context.request.prompt, cat, dependent_ids)
            
            sub_tasks.append(SubTask(
                id=task_id, 
                prompt=rewritten_prompt, 
                category=cat,
                dependencies=dependent_ids
            ))
            
        return sub_tasks

class ResultAggregator:
    def aggregate(self, original_task_id: str, results: List[ExecutionResult]) -> ExecutionResult:
        if len(results) == 1:
            results[0].task_id = original_task_id
            return results[0]
            
        combined_answer = ""
        total_tokens = 0
        
        for idx, res in enumerate(results):
            combined_answer += f"[{res.route_taken.value.upper()} Output {idx+1}]:\n{res.answer}\n\n"
            total_tokens += res.fireworks_tokens
            
        final_route = results[0].route_taken 
        
        return ExecutionResult(
            task_id=original_task_id,
            answer=combined_answer.strip(),
            route_taken=final_route,
            fireworks_tokens=total_tokens
        )
