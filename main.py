import os
import sys
import json
import asyncio
import time
from typing import Dict, Any, Optional
from utils.logger import setup_logger
from router.classifier import classify_task, estimate_complexity, TaskCategory
from router.decider import decide, verify_logic_puzzle
from api_engine.fireworks_client import FireworksClient
from local_engine.llama_client import LocalLlamaClient
from utils import python_executor as pyexec

logger = setup_logger("runner")

# Input/Output paths defined by the harness
DEFAULT_INPUT_PATH = "/input/tasks.json"
DEFAULT_OUTPUT_PATH = "/output/results.json"

# Local testing fallbacks
LOCAL_INPUT_PATH = "input/tasks.json"
LOCAL_OUTPUT_PATH = "output/results.json"

CONCURRENCY_LIMIT = 5

def _strip_code_fences(text: str) -> str:
    """Removes markdown code block fences if returned by the model."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()

async def handle_local_math(prompt: str, local_client: LocalLlamaClient) -> tuple:
    """
    Asks the local model to convert a math word problem into a clean arithmetic formula
    and evaluates it locally using AST. Returns (answer_str, float_result, raw_expr).
    """
    extraction_prompt = (
        "Convert this word problem into a single arithmetic expression "
        "using only numbers and + - * / ( ). Output ONLY the expression, "
        "nothing else.\n\n" + prompt
    )
    expr = await local_client.generate_completion_async(extraction_prompt, TaskCategory.MATH, temperature=0.0)
    result = pyexec.try_solve_arithmetic(expr)
    if result is not None:
        return str(result), result, expr
    return None, None, expr

async def handle_local_code_debug(prompt: str, local_client: LocalLlamaClient) -> tuple:
    """
    Extracts code block, asks local LLM to fix it, then runs smoke test checks.
    Returns (fixed_source, ran_clean).
    """
    source = pyexec.extract_function_source(prompt)
    if not source:
        return None, False
        
    func_name = pyexec.find_function_name(source)
    if not func_name:
        return None, False

    fix_prompt = (
        f"This function has a bug:\n\n{source}\n\n"
        "Output ONLY the corrected function definition, nothing else."
    )
    fixed_source = await local_client.generate_completion_async(fix_prompt, TaskCategory.DEBUGGING, temperature=0.0)
    fixed_source = _strip_code_fences(fixed_source)
    fixed_func_name = pyexec.find_function_name(fixed_source) or func_name

    # Smoke-test with generic arguments to verify it compiles and runs without exception
    test_args = [([1, 2, 3],), ([5],), ([-1, -5, -2],)]
    try:
        results = pyexec.run_function(fixed_source, fixed_func_name, test_args)
        # Verify no runtime ERROR prefix returned
        ran_clean = all(not str(r[1]).startswith("ERROR") for r in results)
    except Exception:
        ran_clean = False

    return fixed_source, ran_clean

async def process_task(
    task: Dict[str, Any], 
    local_client: LocalLlamaClient, 
    api_client: FireworksClient, 
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    task_id = task.get("task_id", "unknown")
    prompt = task.get("prompt", "")
    
    if not prompt:
        logger.warning(f"Task {task_id} has an empty prompt.")
        return {"task_id": task_id, "answer": ""}

    async with semaphore:
        # 1. Classify Task & Estimate Complexity
        category = classify_task(prompt)
        complexity = estimate_complexity(prompt, category)
        logger.info(f"Task {task_id} classified as category={category.value} complexity={complexity}")
        
        local_answer = None
        executed_result = None
        second_sample = None
        
        # 2. Local Execution Path (if local model loaded)
        if local_client.loaded:
            # Category-specific routing
            if category == TaskCategory.MATH:
                try:
                    # Try safe arithmetic formula execution first
                    answer_str, result, expr = await asyncio.wait_for(
                        handle_local_math(prompt, local_client),
                        timeout=20.0
                    )
                    if result is not None:
                        executed_result = result
                        local_answer = answer_str
                    else:
                        # Fall back to standard local prompt call
                        local_answer = await local_client.generate_completion_async(prompt, category, temperature=0.1)
                except Exception as e:
                    logger.warning(f"Local math handling failed/timed out for {task_id}: {e}")
                    
            elif category == TaskCategory.DEBUGGING:
                try:
                    # Try local code fix & smoke test validation
                    fixed_source, ran_clean = await asyncio.wait_for(
                        handle_local_code_debug(prompt, local_client),
                        timeout=20.0
                    )
                    if fixed_source and ran_clean:
                        executed_result = fixed_source
                        local_answer = fixed_source
                    else:
                        local_answer = fixed_source or await local_client.generate_completion_async(prompt, category, temperature=0.1)
                except Exception as e:
                    logger.warning(f"Local debug handling failed/timed out for {task_id}: {e}")
                    
            elif category == TaskCategory.LOGIC:
                # Logic puzzles always default escalate (3B models are too weak)
                pass
                
            else:
                # Factual, NER, Sentiment, Summarization, and Code Gen
                try:
                    # Run first sample
                    local_answer = await asyncio.wait_for(
                        local_client.generate_completion_async(prompt, category, temperature=0.1),
                        timeout=20.0
                    )
                    # For medium-risk categories, spend a second local query to check self-consistency
                    is_medium_risk = category in [TaskCategory.FACTUAL, TaskCategory.SUMMARIZATION, TaskCategory.CODE_GEN]
                    if is_medium_risk and complexity != "high":
                        second_sample = await asyncio.wait_for(
                            local_client.generate_completion_async(prompt, category, temperature=0.7),
                            timeout=20.0
                        )
                except Exception as e:
                    logger.warning(f"Local prompt completion failed/timed out for {task_id}: {e}")

        # 3. Confidence Decider
        verdict = decide(category, complexity, local_answer, second_sample, executed_result)
        
        # If we trust the local model, return its response (costs 0 tokens!)
        if verdict["trust_local"] and local_answer is not None:
            logger.info(f"[{task_id}] trust=LOCAL reason={verdict['reason']}")
            return {"task_id": task_id, "answer": local_answer}
            
        # 4. Escalation to Fireworks API
        logger.info(f"[{task_id}] trust=FIREWORKS reason={verdict['reason']}")
        try:
            api_answer = await api_client.generate_completion_async(prompt, category)
            
            # Logic check sanity safeguard
            if category == TaskCategory.LOGIC:
                if not verify_logic_puzzle(prompt, api_answer):
                    logger.warning(f"[{task_id}] logic answer failed name constraints check; reporting anyway.")
                    
            return {"task_id": task_id, "answer": api_answer}
        except Exception as e:
            logger.error(f"Fireworks API call failed for task {task_id}: {e}")
            # Dynamic recovery fallback
            fallback_ans = local_answer if local_answer is not None else "Error: Processing failure."
            return {"task_id": task_id, "answer": fallback_ans}

async def main_async():
    logger.info("Starting Hybrid-Confidence General-Purpose AI Agent...")
    
    # 1. Resolve Input/Output Paths
    input_path = DEFAULT_INPUT_PATH
    output_path = DEFAULT_OUTPUT_PATH
    
    if not os.path.exists(input_path):
        logger.info(f"Harness input path '{input_path}' not found. Falling back to local testing path.")
        input_path = LOCAL_INPUT_PATH
        output_path = LOCAL_OUTPUT_PATH
        
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at: {input_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([], f)
        sys.exit(0)
        
    # 2. Load Tasks
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        logger.info(f"Loaded {len(tasks)} tasks from {input_path}")
    except Exception as e:
        logger.error(f"Failed to load or parse input JSON: {e}")
        sys.exit(1)
        
    if not isinstance(tasks, list):
        logger.error("Input JSON is not a list of tasks.")
        sys.exit(1)

    # 3. Initialize Clients
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "local_model.gguf")
    local_client = LocalLlamaClient(model_path=model_path)
    
    try:
        api_client = FireworksClient()
    except Exception as e:
        logger.error(f"Failed to initialize Fireworks API client: {e}")
        sys.exit(1)
        
    # 4. Process Tasks concurrently
    start_time = time.time()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks_to_run = [process_task(t, local_client, api_client, semaphore) for t in tasks]
    
    results = await asyncio.gather(*tasks_to_run)
    
    # 5. Write Results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        temp_output_path = f"{output_path}.tmp"
        with open(temp_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        if os.path.exists(output_path):
            os.remove(output_path)
            
        os.rename(temp_output_path, output_path)
        logger.info(f"Successfully wrote {len(results)} results in {time.time() - start_time:.2f}s to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write results file: {e}")
        sys.exit(1)
        
    logger.info("Runner finished successfully.")
    sys.exit(0)

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        sys.exit(130)

if __name__ == "__main__":
    main()
