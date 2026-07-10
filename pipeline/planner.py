from typing import List
from models.schemas import TaskContext, ExecutionResult
from models.enums import TaskCategory
from dataclasses import dataclass

@dataclass
class SubTask:
    id: str
    prompt: str
    category: TaskCategory

class TaskPlanner:
    def plan(self, context: TaskContext) -> List[SubTask]:
        profile = context.profile
        
        # We only split if we have strong multi-label probabilities
        valid_categories = []
        for cat_str, prob in profile.task_types:
            if prob > 0.6:  # Threshold for independent task existence
                valid_categories.append(TaskCategory(cat_str))
                
        # Prevent over-splitting (Hard Cap)
        valid_categories = valid_categories[:2]
        
        # If single task, return as-is
        if len(valid_categories) <= 1:
            return [SubTask(id=f"{context.request.task_id}_0", prompt=context.request.prompt, category=context.category)]
            
        sub_tasks = []
        for idx, cat in enumerate(valid_categories):
            # Zero-token heuristic splitting via specialized focus prompting
            specialized_prompt = f"Focus ONLY on the {cat.value} aspect of the following request. Ignore other instructions. Request: {context.request.prompt}"
            sub_tasks.append(SubTask(id=f"{context.request.task_id}_{idx}", prompt=specialized_prompt, category=cat))
            
        return sub_tasks

class ResultAggregator:
    def aggregate(self, original_task_id: str, results: List[ExecutionResult]) -> ExecutionResult:
        if len(results) == 1:
            results[0].task_id = original_task_id
            return results[0]
            
        combined_answer = ""
        total_tokens = 0
        
        for idx, res in enumerate(results):
            # Cleanly format multi-part answers for the LLM Judge
            combined_answer += f"[{res.route_taken.value.upper()} Output {idx+1}]:\n{res.answer}\n\n"
            total_tokens += res.fireworks_tokens
            
        # The final reported route is the most expensive one taken in the DAG
        final_route = results[0].route_taken 
        
        return ExecutionResult(
            task_id=original_task_id,
            answer=combined_answer.strip(),
            route_taken=final_route,
            fireworks_tokens=total_tokens
        )
