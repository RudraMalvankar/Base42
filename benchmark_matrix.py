import json
import time
import asyncio
from engine.executors.local_llm import LocalLLMExecutor
from models.schemas import TaskRequest, TaskContext, ExecutionResult
from models.enums import TaskCategory, ExecutionRoute
import config

async def test_all_categories():
    print("\n" + "="*60)
    print("TINYLLAMA CAPABILITY MATRIX BENCHMARK")
    print("="*60 + "\n")

    with open("input_samples/tasks.json", "r") as f:
        tasks = json.load(f)
        
    local_executor = LocalLLMExecutor(model_path="/model/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
    
    # Map task IDs to categories based on the PDF definitions
    category_map = {
        'T01': TaskCategory.FACTUAL,
        'T01b': TaskCategory.FACTUAL,
        'T01c': TaskCategory.FACTUAL,
        'T02': TaskCategory.MATH,
        'T02b': TaskCategory.MATH,
        'T03': TaskCategory.SENTIMENT,
        'T03b': TaskCategory.SENTIMENT,
        'T04': TaskCategory.SUMMARIZATION,
        'T04b': TaskCategory.SUMMARIZATION,
        'T05': TaskCategory.NER
    }
    
    await asyncio.sleep(1) # Warmup
    
    for t in tasks:
        task_id = t['task_id']
        category = category_map.get(task_id, TaskCategory.FACTUAL)
        
        print(f"--- Task {task_id} [{category.name}] ---")
        req = TaskRequest(task_id=task_id, prompt=t['prompt'])
        ctx = TaskContext(request=req, category=category)
        
        start = time.time()
        res_local = await local_executor.execute(ctx)
        latency = time.time() - start
        
        print(f"Latency: {latency:.2f}s")
        print(f"Output:\n{res_local.answer}")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_all_categories())
