import asyncio
import json
from models.schemas import TaskRequest, TaskContext
from engine.executors.local_llm import LocalLLMExecutor
from core.logger import setup_logger

logger = setup_logger("test")

async def run():
    executor = LocalLLMExecutor(model_path='/model/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf')
    
    with open('input_samples/tasks.json') as f:
        tasks = json.load(f)
        
    t03 = next(t for t in tasks if t['task_id'] == 'T03')
    t03b = next(t for t in tasks if t['task_id'] == 'T03b')
    
    from pipeline.analyzer import TaskCategory
    ctx3 = TaskContext(request=TaskRequest(**t03), profile=None)
    ctx3.category = TaskCategory.SENTIMENT
    ctx3b = TaskContext(request=TaskRequest(**t03b), profile=None)
    ctx3b.category = TaskCategory.SENTIMENT
    
    print("Executing T03...")
    r3 = await executor.execute(ctx3)
    print('T03 Local Output:', r3.answer)
    
    print("Executing T03b...")
    r3b = await executor.execute(ctx3b)
    print('T03b Local Output:', r3b.answer)

if __name__ == "__main__":
    asyncio.run(run())
