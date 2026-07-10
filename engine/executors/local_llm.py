from .base import BaseExecutor
from models.schemas import TaskContext, ExecutionResult
from models.enums import ExecutionRoute
from core.logger import setup_logger
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

logger = setup_logger("local_llm_executor")

class LocalLLMExecutor(BaseExecutor):
    def __init__(self, model_path: str = "./weights/model.gguf"):
        self.model_path = model_path
        self.llm = None
        self.executor = ThreadPoolExecutor(max_workers=2) # Prevent asyncio deadlock
        
        if HAS_LLAMA:
            try:
                # Load with strict resource limits for 4GB/2CPU env
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=512,
                    n_threads=2,
                    verbose=False
                )
            except Exception as e:
                logger.error(f"Failed to load local model at {model_path}: {e}")
                self.llm = None
        else:
            logger.warning("llama-cpp-python is not installed. Local LLM will failover to Fireworks.")

    def _invoke_sync(self, prompt: str) -> str:
        if not self.llm:
            raise RuntimeError("Local LLM not loaded.")
            
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a concise AI. Answer directly with no filler text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.1
        )
        return response["choices"][0]["message"]["content"].strip()

    async def execute(self, context: TaskContext) -> ExecutionResult:
        if not self.llm:
            # Fallback will be handled by the router/engine if this raises or returns empty
            raise RuntimeError("Local inference unavailable.")
            
        try:
            logger.info(f"Task {context.request.task_id}: Routing to Local LLM")
            # Run C++ synchronous binding in a separate thread to prevent event loop blocking
            loop = asyncio.get_running_loop()
            answer = await asyncio.wait_for(
                loop.run_in_executor(self.executor, self._invoke_sync, context.request.prompt),
                timeout=20.0 # Strict timeout to ensure we don't blow the 30s limit
            )
        except asyncio.TimeoutError:
            logger.error(f"Task {context.request.task_id}: Local LLM timed out (>20s)")
            raise TimeoutError("Local LLM timeout")
        except Exception as e:
            logger.error(f"Task {context.request.task_id}: Local LLM error - {e}")
            raise e
            
        return ExecutionResult(
            task_id=context.request.task_id,
            answer=answer,
            route_taken=ExecutionRoute.LOCAL_LLM
        )
