import json
import time
import asyncio
from engine.executors.local_llm import LocalLLMExecutor
from engine.executors.fireworks import FireworksExecutor
import config
from models.schemas import TaskRequest, TaskContext, ExecutionResult
from models.enums import TaskCategory, ExecutionRoute

async def test_sentiment():
    print("\n" + "="*50)
    print("SENTIMENT TASK BENCHMARK: LOCAL VS FIREWORKS")
    print("="*50 + "\n")

    # Load tasks
    with open("input_samples/tasks.json", "r") as f:
        tasks = json.load(f)
        
    sentiment_tasks = [t for t in tasks if t['task_id'] in ['T03', 'T03b']]
    
    local_executor = LocalLLMExecutor(model_path="/model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    fireworks = FireworksExecutor()
    
    # Let the model warm up
    await asyncio.sleep(1)
    
    for t in sentiment_tasks:
        print(f"--- Task {t['task_id']} ---")
        prompt = t['prompt']
        
        req = TaskRequest(task_id=t['task_id'], prompt=prompt)
        ctx = TaskContext(request=req, category=TaskCategory.SENTIMENT)
        
        # LOCAL TEST
        start = time.time()
        res_local = await local_executor.execute(ctx)
        time_local = time.time() - start
        
        print(f"[TinyLlama (Local)]")
        print(f"Latency: {time_local:.2f}s")
        print(f"Output: {res_local.answer}")
        
        # FIREWORKS TEST
        start = time.time()
        res_fw = await fireworks.execute(ctx)
        time_fw = time.time() - start
        
        print(f"\n[Fireworks API]")
        print(f"Latency: {time_fw:.2f}s")
        print(f"Output: {res_fw.answer}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    asyncio.run(test_sentiment())
