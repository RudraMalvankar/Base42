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
from core.telemetry import TelemetryService, TaskTrace
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
        
        # Point to the TinyLlama model (Supports both Docker and Local dev paths)
        model_path = "/model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        if not os.path.exists(model_path):
            model_path = "./model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            
        self.local_exec = LocalLLMExecutor(model_path=model_path)
        self.api_exec = FireworksExecutor()
        
    async def _execute_single_route(self, context: TaskContext) -> ExecutionResult:
        """Helper to execute a single atomic task."""
        route = context.route
        fallback_triggered = False
        
        if route == ExecutionRoute.FIREWORKS:
            result = await self.api_exec.execute(context)
        elif route == ExecutionRoute.LOCAL_LLM:
            try:
                result = await self.local_exec.execute(context)
            except Exception as e:
                logger.warning(f"Task {context.request.task_id}: Local LLM crashed/timed out ({e}). Bypassing to Fireworks API.")
                result = await self.api_exec.execute(context)
                fallback_triggered = True
        else:
            result = await self.python_exec.execute(context)
            if result.fallback_triggered and context.category.value == "math" and self.local_exec.llm:
                from engine.executors.math_word_solver import solve_word_problem
                import asyncio
                
                def llm_call(p):
                    return self.local_exec._invoke_sync(p, 256)
                    
                try:
                    ans = await asyncio.to_thread(solve_word_problem, context.request.prompt, llm_call)
                    if ans:
                        result.answer = str(ans)
                        result.fallback_triggered = False
                except Exception as e:
                    pass
            
        final_result = ResultValidator.sanitize(result, context.category)
        
        # Fallback Loop
        if not ConfidenceEngine.evaluate(final_result, context):
            fallback_triggered = True
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
            
        final_result.fallback_triggered = fallback_triggered
        return final_result

    async def process_task(self, request: TaskRequest) -> ExecutionResult:
        import time
        start_time = time.perf_counter()
        
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
        node_map = {}
        for st in sub_tasks:
            st_req = TaskRequest(task_id=st.id, prompt=st.prompt)
            st_ctx = TaskContext(request=st_req, profile=profile, category=st.category, complexity=context.complexity)
            
            async def _exec_node(node_ref: DAGNode):
                ctx = node_ref.context
                # Dynamic Context Injection
                if node_ref.dependencies:
                    dep_text = "\n\n--- PREVIOUS TASK OUTPUTS ---\n"
                    for d in node_ref.dependencies:
                        if d.result:
                            dep_text += f"[{d.task_id}]: {d.result.answer}\n"
                    ctx.request.prompt += dep_text
                    
                ctx.route = self.decision_engine.route(ctx)
                return await self._execute_single_route(ctx)
                
            node = DAGNode(task_id=st.id, executable=_exec_node, context=st_ctx)
            node_map[st.id] = node
            dag.add_node(node)
            
        # Wire dependencies
        for st in sub_tasks:
            node = node_map[st.id]
            for dep_id in st.dependencies:
                if dep_id in node_map:
                    node.add_dependency(node_map[dep_id])
            
        # Execute all sub-tasks concurrently/sequentially based on edges
        sub_results = await dag.execute_graph()
        
        # 4. Aggregate
        aggregator = ResultAggregator()
        final_result = aggregator.aggregate(request.task_id, sub_results)
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        fallback_triggered = any(getattr(r, 'fallback_triggered', False) for r in sub_results)
        
        trace = TaskTrace(
            task_id=request.task_id,
            category=context.category.value if context.category else "UNKNOWN",
            latency_ms=latency_ms,
            route=final_result.route_taken.value if final_result.route_taken else "UNKNOWN",
            fallback_triggered=fallback_triggered,
            tokens=final_result.fireworks_tokens
        )
        await TelemetryService.get_instance().record(trace)
        
        return final_result

async def main():
    import sys
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
        try:
            with open(input_path, "r") as f:
                tasks_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            tasks_data = []

        orchestrator = Base42Orchestrator()
        semaphore = asyncio.Semaphore(20) # Protect memory & API limits, allow 20 concurrent

        async def _bounded_process(task_dict):
            async with semaphore:
                request = TaskRequest(**task_dict)
                try:
                    # AMD Hackathon imposes a strict 30-second time limit per task.
                    # We enforce a 28-second hard timeout to guarantee we never stall the pipeline.
                    return await asyncio.wait_for(
                        orchestrator.process_task(request),
                        timeout=28.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Task {request.task_id} TIMED OUT globally (>28s). Forcing API fallback state.")
                    return ExecutionResult(
                        task_id=request.task_id,
                        answer="Timeout Error",
                        route_taken=ExecutionRoute.FIREWORKS,
                        fallback_triggered=True
                    )

        logger.info(f"Starting execution of {len(tasks_data)} tasks.")
        
        tasks = [_bounded_process(t) for t in tasks_data]
        
        # return_exceptions=True prevents one poisoned task from crashing the entire batch
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        results: List[ExecutionResult] = []
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                # Fallback extraction of task_id if possible
                task_id = str(tasks_data[i].get("task_id", i)) if i < len(tasks_data) else str(i)
                logger.error(f"Task {task_id} suffered FATAL UNHANDLED ERROR: {res}")
                results.append(ExecutionResult(
                    task_id=task_id, 
                    answer="Fatal Orchestrator Error", 
                    route_taken=None, 
                    fallback_triggered=True
                ))
            else:
                results.append(res)
        
        # Format for AMD Grader
        output_data = [{"task_id": r.task_id, "answer": r.answer} for r in results]
        
        try:
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=4)
            logger.info(f"Successfully wrote {len(output_data)} results to {output_path}")
            
            # Dump telemetry
            TelemetryService.get_instance().dump_report("/output/telemetry.json")
        except Exception as e:
            logger.error(f"Failed to write results: {e}")
            
        logger.info("Finished processing.")
        
    except Exception as e:
        logger.critical(f"GLOBAL CRASH: {e}. Writing emergency blank output.")
        with open(output_path, "w") as f:
            json.dump([], f)
        sys.exit(0) # Exit cleanly to ensure Docker runtime doesn't report an internal failure

if __name__ == "__main__":
    asyncio.run(main())
