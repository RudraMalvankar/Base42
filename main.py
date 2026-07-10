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
from engine.dag import DAGEngine, DAGNode
from pipeline.planner import TaskPlanner, ResultAggregator
from engine.executors.python import PythonExecutor
from engine.executors.fireworks import FireworksExecutor
from engine.executors.local_llm import LocalLLMExecutor
from core.logger import setup_logger

logger = setup_logger("base42_main")

class Base42Orchestrator:
    def __init__(self):
        self.analyzer = PromptAnalyzer()  # Loads semantic model ONCE at startup
        self.decision_engine = DecisionEngine()
        self.python_exec = PythonExecutor()
        self.local_exec = LocalLLMExecutor(model_path="./weights/model.gguf")
        self.api_exec = FireworksExecutor()
        
    async def _execute_single_route(self, context: TaskContext) -> ExecutionResult:
        """Helper to execute a single atomic task."""
        route = context.route
        if route == ExecutionRoute.FIREWORKS:
            result = await self.api_exec.execute(context)
        elif route == ExecutionRoute.LOCAL_LLM:
            result = await self.local_exec.execute(context)
        else:
            result = await self.python_exec.execute(context)
            
        final_result = ResultValidator.sanitize(result, context.category)
        
        # Fallback Loop
        if not ConfidenceEngine.evaluate(final_result, context):
            logger.warning(f"Task {context.request.task_id}: Low confidence in {route.value}. Recalculating route.")
            context.failed_attempts += 1
            context.route = self.decision_engine.route(context)
            
            if context.route == ExecutionRoute.FIREWORKS:
                result = await self.api_exec.execute(context)
            elif context.route == ExecutionRoute.LOCAL_LLM:
                result = await self.local_exec.execute(context)
            else:
                result = await self.python_exec.execute(context)
                
            final_result = ResultValidator.sanitize(result, context.category)
            
        return final_result

    async def process_task(self, request: TaskRequest) -> ExecutionResult:
        # 1. Analyze
        profile = self.analyzer.analyze(request.prompt)
        context = TaskContext(request=request, profile=profile)
        context.category = profile.primary_category
        context.complexity = ComplexityEstimator.estimate(context)
        
        # 2. Plan (Heuristic Decomposition)
        planner = TaskPlanner()
        sub_tasks = planner.plan(context)
        
        # 3. Build DAG & Execute
        dag = DAGEngine()
        for st in sub_tasks:
            st_req = TaskRequest(task_id=st.id, prompt=st.prompt)
            st_ctx = TaskContext(request=st_req, profile=profile, category=st.category, complexity=context.complexity)
            
            async def _exec_node(ctx=st_ctx):
                ctx.route = self.decision_engine.route(ctx)
                return await self._execute_single_route(ctx)
                
            node = DAGNode(task_id=st.id, executable=_exec_node, context=st_ctx)
            dag.add_node(node)
            
        # Execute all sub-tasks concurrently
        sub_results = await dag.execute_graph()
        
        # 4. Aggregate
        aggregator = ResultAggregator()
        return aggregator.aggregate(request.task_id, sub_results)

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
        
    logger.info("Finished processing.", extra={"telemetry": {
        "total_tasks": len(results),
        "total_fireworks_tokens": total_tokens,
        "output_path": output_path
    }})

if __name__ == "__main__":
    asyncio.run(main())
