import asyncio
import json
import os
from typing import List
from models.schemas import TaskRequest, TaskContext, ExecutionResult
from models.enums import ExecutionRoute
from pipeline.analyzer import PromptAnalyzer
from pipeline.classifier import TaskClassifier
from pipeline.complexity import ComplexityEstimator
from pipeline.validator import ResultValidator
from engine.decision import DecisionEngine
from engine.confidence import ConfidenceEngine
from engine.executors.python import PythonExecutor
from engine.executors.fireworks import FireworksExecutor
from engine.executors.local_llm import LocalLLMExecutor
from core.logger import setup_logger

logger = setup_logger("base42_main")

class Base42Orchestrator:
    def __init__(self):
        self.python_exec = PythonExecutor()
        self.local_exec = LocalLLMExecutor(model_path="./weights/model.gguf")
        self.api_exec = FireworksExecutor()
        
    async def process_task(self, request: TaskRequest) -> ExecutionResult:
        # 1. Analyze
        metadata = PromptAnalyzer.analyze(request.prompt)
        context = TaskContext(request=request, metadata=metadata)
        
        # 2. Classify & Estimate
        context.category = TaskClassifier.classify(context)
        context.complexity = ComplexityEstimator.estimate(context)
        
        # 3. Decide Route
        route = DecisionEngine.route(context)
        context.route = route
        
        logger.info(f"Task {request.task_id} -> Category: {context.category.value}, Route: {route.value}")
        
        # 4. Execute
        if route == ExecutionRoute.PYTHON:
            result = await self.python_exec.execute(context)
        elif route == ExecutionRoute.LOCAL_LLM:
            try:
                result = await self.local_exec.execute(context)
            except Exception:
                result = ExecutionResult(task_id=request.task_id, answer="", route_taken=ExecutionRoute.LOCAL_LLM)
        else:
            result = await self.api_exec.execute(context)
            
        # 5. Evaluate Confidence
        if not ConfidenceEngine.evaluate(result, context):
            logger.warning(f"Task {request.task_id}: Low confidence in {route.value}. Escalating to Fireworks.")
            result = await self.api_exec.execute(context)
            
        # 6. Validate & Sanitize
        final_result = ResultValidator.sanitize(result)
        return final_result

async def main():
    input_path = "/input/tasks.json"
    output_path = "/output/results.json"
    
    # Local fallback paths for testing outside docker
    if not os.path.exists("/input"):
        input_path = "./input/tasks.json"
        output_path = "./output/results.json"
        os.makedirs("./input", exist_ok=True)
        os.makedirs("./output", exist_ok=True)
        
        # Create dummy input if missing
        if not os.path.exists(input_path):
            with open(input_path, "w") as f:
                json.dump([{"task_id": "1", "prompt": "What is 2+2?"}], f)

    try:
        with open(input_path, "r") as f:
            tasks_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        tasks_data = []

    orchestrator = Base42Orchestrator()
    semaphore = asyncio.Semaphore(4) # Protect memory & API limits

    async def _bounded_process(task_dict):
        async with semaphore:
            request = TaskRequest(**task_dict)
            return await orchestrator.process_task(request)

    logger.info(f"Starting execution of {len(tasks_data)} tasks.")
    
    tasks = [_bounded_process(t) for t in tasks_data]
    results: List[ExecutionResult] = await asyncio.gather(*tasks)
    
    # Format for AMD Grader
    output_data = [{"task_id": r.task_id, "answer": r.answer} for r in results]
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
        
    logger.info(f"Finished processing. Results written to {output_path}")
    
    total_tokens = sum(r.fireworks_tokens for r in results)
    logger.info(f"Total Fireworks Tokens Used: {total_tokens}")

if __name__ == "__main__":
    asyncio.run(main())
