from .base import BaseExecutor
from models.schemas import TaskContext, ExecutionResult
from models.enums import ExecutionRoute
from core.logger import setup_logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import os

try:
    from llama_cpp import Llama, LlamaGrammar
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

logger = setup_logger("local_llm_executor")

# Grammar-Constrained Decoding
SENTIMENT_GRAMMAR = r'''
root ::= "Positive" | "Negative" | "Neutral" | "Mixed"
'''

NER_GRAMMAR = r'''
root ::= "[" entity ("," entity)* "]"
entity ::= "{\"entity\": \"" [^"]+ "\", \"label\": \"" label "\"}"
label ::= "PERSON" | "ORGANIZATION" | "LOCATION" | "DATE"
'''

class LocalLLMWorker:
    def __init__(self, worker_id: int, model_path: str):
        self.worker_id = worker_id
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.llm = None
        load_start = time.time()
        try:
            from config import FeatureFlags
            n_threads = 2 if getattr(FeatureFlags, "ENABLE_FIXED_LLAMA_THREADS", False) else 1
            self.llm = Llama(
                model_path=model_path,
                n_ctx=512, # Qwen2.5 benefits from larger context for complex prompts
                n_batch=512,
                n_threads=n_threads, 
                verbose=False
            )
            load_time = time.time() - load_start
            logger.info(f"Worker {worker_id} successfully loaded model in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Worker {worker_id} failed to load model: {e}")

class LocalLLMExecutor(BaseExecutor):
    def __init__(self, model_path: str = "./weights/model.gguf"):
        self.model_path = model_path
        self.workers = []
        self.circuit_breaker_tripped = False
        if HAS_LLAMA:
            logger.info("Initializing SINGLETON Qwen2.5 Worker with n_threads=2 for max CPU utilization...")
            w1 = LocalLLMWorker(1, model_path)
            # Override n_threads to 2 for the singleton worker if flag is not set
            from config import FeatureFlags
            if w1.llm and not getattr(FeatureFlags, "ENABLE_FIXED_LLAMA_THREADS", False):
                w1.llm.n_threads = 2
            self.workers.append(w1)
            # Model warm-up
            try:
                logger.info("Warming up model with empty prompt...")
                w1.llm.create_chat_completion([{"role": "user", "content": "hi"}], max_tokens=2, temperature=0.0)
                logger.info("Model warm-up complete.")
            except:
                pass
        else:
            logger.warning("llama-cpp-python is not installed.")

    def is_available(self) -> bool:
        from config import FeatureFlags
        if getattr(FeatureFlags, "ENABLE_LONG_LOCAL_QUEUE", False) and self.circuit_breaker_tripped:
            return False
        return len(self.workers) > 0

    def _get_grammar(self, category: str):
        try:
            if category == "ner":
                return LlamaGrammar.from_string(NER_GRAMMAR)
        except Exception as e:
            logger.warning(f"Failed to compile grammar for {category}: {e}")
        return None

    def _get_prompt(self, category: str, prompt: str) -> list:
        # Few-shot prompting for strict adherence
        if category == "sentiment":
            return [
                {"role": "system", "content": "Classify sentiment as Positive, Negative, Neutral, or Mixed, and provide a one-sentence reason."},
                {"role": "user", "content": "I hate this late delivery, but the customer service was nice."},
                {"role": "assistant", "content": "Mixed - The delivery was late but the customer service was nice."},
                {"role": "user", "content": prompt}
            ]
        elif category == "ner":
            return [
                {"role": "system", "content": "Extract entities into JSON list format."},
                {"role": "user", "content": "John went to Apple in New York on Jan 1."},
                {"role": "assistant", "content": '[{"entity": "John", "label": "PERSON"}, {"entity": "Apple", "label": "ORGANIZATION"}, {"entity": "New York", "label": "LOCATION"}, {"entity": "Jan 1", "label": "DATE"}]'},
                {"role": "user", "content": prompt}
            ]
        else:
            return [
                {"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": prompt}
            ]

    def _invoke_sync(self, worker: LocalLLMWorker, prompt: str, max_tokens: int, category: str = "general") -> str:
        grammar = self._get_grammar(category)
        messages = self._get_prompt(category, prompt)
        
        inf_start = time.time()
        
        # Performance optimizations for Factual
        stop_seqs = ["User:", "user:", "<|im_end|>"]
        if category == "factual":
            max_tokens = min(max_tokens, 128) # Qwen2.5 generates concise factual answers
            
        response = worker.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0, # Qwen2.5 handles deterministic decoding well unlike TinyLlama
            grammar=grammar,
            stop=stop_seqs
        )
        inf_time = time.time() - inf_start
        logger.info(f"[DIAGNOSTIC] Actual local inference time: {inf_time:.2f}s")
        return response["choices"][0]["message"]["content"].strip()

    async def execute(self, context: TaskContext) -> ExecutionResult:
        if not self.is_available():
            logger.warning(f"Task {context.request.task_id}: Local LLM unavailable.")
            return self._fallback(context.request.task_id)

        # Admission Control
        idle_worker = None
        for w in self.workers:
            if not w.lock.locked():
                idle_worker = w
                break
                
        from config import FeatureFlags
        wait_time = 0.0
        experiment_mode = getattr(FeatureFlags, "ENABLE_LOCAL_QUEUE_EXPERIMENT", False)
        breaker_mode = getattr(FeatureFlags, "ENABLE_LONG_LOCAL_QUEUE", False)
        
        if not idle_worker and (FeatureFlags.ENABLE_SMART_ROUTING_V2 or experiment_mode or breaker_mode):
            wait_start = time.time()
            if experiment_mode:
                logger.info(f"Task {context.request.task_id}: Local workers busy. Entering INFINITE wait queue (experiment)...")
            elif breaker_mode:
                logger.info(f"Task {context.request.task_id}: Local workers busy. Entering wait queue (max 60s breaker)...")
            else:
                logger.info(f"Task {context.request.task_id}: Local workers busy. Entering wait queue (max 12s)...")
                
            try:
                while not idle_worker:
                    elapsed = time.time() - wait_start
                    if breaker_mode and elapsed >= 60.0:
                        break
                    elif not experiment_mode and not breaker_mode and elapsed >= 12.0:
                        break
                    sleep_time = 0.05 if getattr(FeatureFlags, "ENABLE_QUEUE_SLEEP", False) else 0.5
                    await asyncio.sleep(sleep_time)
                    for w in self.workers:
                        if not w.lock.locked():
                            idle_worker = w
                            break
                wait_time = time.time() - wait_start
                logger.info(f"[DIAGNOSTIC] Task {context.request.task_id} Queue wait time: {wait_time:.2f}s")
            except asyncio.TimeoutError:
                logger.error(f"[DIAGNOSTIC] Task {context.request.task_id} Timeout occurred while WAITING in queue")
                return self._fallback(context.request.task_id)

        if not idle_worker:
            if breaker_mode:
                logger.warning(f"Circuit Breaker tripped for {context.request.task_id} during queue!")
                self.circuit_breaker_tripped = True
            logger.warning(f"[DIAGNOSTIC] Task {context.request.task_id}: Timeout occurred while WAITING. Admission Control REJECT. Routing to Fireworks.")
            return self._fallback(context.request.task_id)

        # Formal async lock acquisition to provably guarantee mutually exclusive entry
        await idle_worker.lock.acquire()
        
        max_tokens = 256
        if context.profile and context.profile.estimated_output_tokens:
            max_tokens = int(context.profile.estimated_output_tokens * 1.2)
            max_tokens = min(max(max_tokens, 256), 512)

        start_time = time.time()
        try:
            logger.info(f"Task {context.request.task_id}: Routing to Local Worker {idle_worker.worker_id}")
            loop = asyncio.get_running_loop()
            
            inf_timeout = 60.0 if getattr(FeatureFlags, "ENABLE_LONG_LOCAL_QUEUE", False) else 18.0
            
            answer = await asyncio.wait_for(
                loop.run_in_executor(idle_worker.executor, self._invoke_sync, idle_worker, context.request.prompt, max_tokens, context.category.value),
                timeout=inf_timeout
            )
            
            latency = time.time() - start_time
            logger.info(f"Task {context.request.task_id}: Local Worker {idle_worker.worker_id} finished in {latency:.2f}s")
            return ExecutionResult(
                task_id=context.request.task_id,
                answer=answer,
                route_taken=ExecutionRoute.LOCAL_LLM,
                tokens=max_tokens,
                latency=latency
            )
        except asyncio.TimeoutError:
            if getattr(FeatureFlags, "ENABLE_LONG_LOCAL_QUEUE", False):
                self.circuit_breaker_tripped = True
                logger.warning(f"Circuit breaker TRIPPED! Task {context.request.task_id} inference timeout.")
            logger.error(f"Task {context.request.task_id}: Local Worker {idle_worker.worker_id} timed out")
            return self._fallback(context.request.task_id)
        except Exception as e:
            if getattr(FeatureFlags, "ENABLE_LONG_LOCAL_QUEUE", False):
                self.circuit_breaker_tripped = True
            logger.error(f"Task {context.request.task_id}: Local LLM error - {e}")
            return self._fallback(context.request.task_id)
        finally:
            idle_worker.lock.release()

    def _fallback(self, task_id: str) -> ExecutionResult:
        return ExecutionResult(
            task_id=task_id,
            answer="",
            route_taken=ExecutionRoute.LOCAL_LLM,
            fallback_triggered=True
        )

    async def math_extraction_fallback(self, prompt: str) -> str:
        from config import FeatureFlags
        if FeatureFlags.ENABLE_RESERVED_LOCAL_WORKERS:
            # Bypass math to Fireworks to reserve local workers for classification tasks
            return None
            
        if not self.is_available():
            return None
            
        idle_worker = None
        for w in self.workers:
            if not w.lock.locked():
                idle_worker = w
                break
                
        if not idle_worker:
            return None # Admission control reject
            
        await idle_worker.lock.acquire()
        try:
            loop = asyncio.get_running_loop()
            from engine.executors.math_word_solver import solve_word_problem
            
            def math_worker_sync():
                def llm_call(p):
                    return self._invoke_sync(idle_worker, p, 256, "general")
                return solve_word_problem(prompt, llm_call)
                
            ans = await asyncio.wait_for(
                loop.run_in_executor(idle_worker.executor, math_worker_sync),
                timeout=5.0
            )
            return ans
        except Exception:
            return None
        finally:
            idle_worker.lock.release()
