import os
import sys
import json
import asyncio
from typing import Dict, List, Any
from utils.logger import setup_logger
from router.classifier import classify_task, TaskCategory
from api_engine.fireworks_client import FireworksClient
from local_engine.llama_client import LocalLlamaClient

logger = setup_logger("runner")

# Input/Output paths defined by the harness
DEFAULT_INPUT_PATH = "/input/tasks.json"
DEFAULT_OUTPUT_PATH = "/output/results.json"

# Local testing fallbacks
LOCAL_INPUT_PATH = "input/tasks.json"
LOCAL_OUTPUT_PATH = "output/results.json"

CONCURRENCY_LIMIT = 5

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
        # Classify the task
        category = classify_task(prompt)
        logger.info(f"Task {task_id} classified as: {category}")
        
        answer = None
        
        # Determine routing logic:
        # Sentiment & NER -> Local if available
        # Summarisation & Factual -> Local-first, fallback to API
        # Math, Logic, Debugging, Code Gen -> API directly
        
        use_local = (
            local_client.loaded and 
            category in [TaskCategory.SENTIMENT, TaskCategory.NER, TaskCategory.SUMMARIZATION, TaskCategory.FACTUAL]
        )
        
        if use_local:
            try:
                logger.info(f"Routing Task {task_id} locally...")
                # Impose a 20-second timeout on local CPU inference
                answer = await asyncio.wait_for(
                    local_client.generate_completion_async(prompt, category),
                    timeout=20.0
                )
                logger.info(f"Task {task_id} processed locally.")
            except asyncio.TimeoutError:
                logger.warning(f"Local inference timed out (20s limit) for task {task_id}. Falling back to Fireworks API.")
            except Exception as e:
                logger.error(f"Local inference failed for task {task_id}: {e}. Falling back to Fireworks API.")
                
        # Call API if local was not selected, or if local run timed out / failed
        if answer is None:
            try:
                logger.info(f"Routing Task {task_id} to Fireworks API...")
                answer = await api_client.generate_completion_async(prompt, category)
                logger.info(f"Task {task_id} processed via Fireworks API.")
            except Exception as e:
                logger.error(f"Fireworks API call failed for task {task_id}: {e}")
                # Fallback placeholder to guarantee schema validity
                answer = "Error: Unable to generate response due to internal processing failure."
                
        return {"task_id": task_id, "answer": answer}

async def main_async():
    logger.info("Starting General-Purpose AI Agent Runner...")
    
    # 1. Resolve Input/Output Paths
    input_path = DEFAULT_INPUT_PATH
    output_path = DEFAULT_OUTPUT_PATH
    
    if not os.path.exists(input_path):
        logger.info(f"Harness input path '{input_path}' not found. Falling back to local testing path.")
        input_path = LOCAL_INPUT_PATH
        output_path = LOCAL_OUTPUT_PATH
        
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at: {input_path}")
        # Write empty array to output to satisfy schema, and exit cleanly
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
        # Exiting non-zero is preferred for malformed files
        sys.exit(1)
        
    if not isinstance(tasks, list):
        logger.error("Input JSON is not a list of tasks.")
        sys.exit(1)

    # 3. Initialize Clients
    # Look for model weights relative to the workspace directory
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "local_model.gguf")
    local_client = LocalLlamaClient(model_path=model_path)
    
    try:
        api_client = FireworksClient()
    except Exception as e:
        logger.error(f"Failed to initialize Fireworks API client: {e}")
        sys.exit(1)
        
    # 4. Process Tasks concurrently
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks_to_run = [process_task(t, local_client, api_client, semaphore) for t in tasks]
    
    results = await asyncio.gather(*tasks_to_run)
    
    # 5. Write Results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        # Write atomically using a temporary file
        temp_output_path = f"{output_path}.tmp"
        with open(temp_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        if os.path.exists(output_path):
            os.remove(output_path)
            
        os.rename(temp_output_path, output_path)
        logger.info(f"Successfully wrote {len(results)} results to {output_path}")
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
