import asyncio
import json
import time
import os
from models.schemas import TaskRequest, TaskContext, PromptProfile
from engine.executors.local_llm import LocalLLMExecutor
from models.schemas import TaskRequest
from models.enums import TaskCategory

async def main():
    print("Loading TinyLlama...")
    executor = LocalLLMExecutor(model_path="/model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    
    with open("./input_samples/tasks.json", "r") as f:
        tasks = json.load(f)
        
    targets = {"T03": "sentiment", "T03b": "sentiment", "T05": "ner"}
    
    async def run_task(t_id, cat, prompt):
        req = TaskRequest(task_id=t_id, prompt=prompt)
        ctx = TaskContext(request=req, category=TaskCategory(cat), profile=PromptProfile(word_count=20, estimated_output_tokens=100, complexity_score=0.5))
        
        start = time.time()
        result = await executor.execute(ctx)
        latency = time.time() - start
        
        print(f"\nTask {t_id} ({cat})")
        print(f"Latency: {latency:.2f}s")
        print(f"Raw Output: {result.answer}")
        
    coroutines = []
    for t in tasks:
        if t["task_id"] in targets:
            coroutines.append(run_task(t["task_id"], targets[t["task_id"]], t["prompt"]))
            
    await asyncio.gather(*coroutines)

if __name__ == "__main__":
    asyncio.run(main())
