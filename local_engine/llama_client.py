import os
import asyncio
from typing import Optional
from utils.logger import setup_logger
from router.classifier import TaskCategory

logger = setup_logger("local_llama")

class LocalLlamaClient:
    def __init__(self, model_path: str = "models/local_model.gguf"):
        self.model_path = model_path
        self.llm = None
        self.loaded = False
        
        # We will attempt to load the model on initialization if it exists.
        # Otherwise, the runner will know it is unavailable and fall back to API.
        self.load_model()

    def load_model(self) -> bool:
        if not os.path.exists(self.model_path):
            logger.warning(f"Local model weights not found at '{self.model_path}'. Local inference will be disabled.")
            return False
            
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading local GGUF model from {self.model_path}...")
            # Configure llama-cpp carefully to fit in 4 GB RAM / 2 vCPU
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=512,         # Small context size to save RAM
                n_threads=2,       # Fit 2 vCPU budget
                n_batch=256,       # Batch size for prompt processing
                verbose=False      # Suppress verbose stderr output
            )
            self.loaded = True
            logger.info("Local model successfully loaded!")
            return True
        except ImportError:
            logger.error("llama-cpp-python package is not installed. Local inference will be disabled.")
            return False
        except Exception as e:
            logger.error(f"Failed to load local model: {e}. Local inference will be disabled.")
            return False

    def _get_system_prompt(self, category: TaskCategory) -> str:
        """
        Returns a concise system prompt optimized for the local model.
        """
        if category == TaskCategory.SENTIMENT:
            return "Classify sentiment as Positive, Negative, or Neutral. Output: classification and brief 1-sentence justification."
        elif category == TaskCategory.NER:
            return "Extract Named Entities (Person, Organization, Location, Date). Output JSON: [{\"entity\": \"...\", \"type\": \"...\"}]."
        elif category == TaskCategory.SUMMARIZATION:
            return "Summarize the text in exactly one sentence. Do not add intro/outro."
        else:
            return "Answer the question directly and concisely in 1-2 sentences."

    def _format_prompt(self, prompt: str, category: TaskCategory) -> str:
        """
        Formats prompt using Qwen ChatML standard template.
        """
        system_prompt = self._get_system_prompt(category)
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _run_inference_sync(self, formatted_prompt: str) -> str:
        """
        Synchronous wrapper run in a background thread executor.
        """
        if not self.llm:
            raise RuntimeError("Model is not initialized.")
            
        res = self.llm(
            formatted_prompt,
            max_tokens=150,     # Limit tokens to save CPU time
            temperature=0.1,    # Greedy decoding for factual/sentiment
            stop=["<|im_end|>", "<|im_start|>", "<|im_start|>assistant"]
        )
        return res["choices"][0]["text"].strip()

    async def generate_completion_async(self, prompt: str, category: TaskCategory) -> str:
        """
        Runs local model inference asynchronously via thread-pool executor.
        """
        if not self.loaded:
            raise RuntimeError("Local model is not loaded.")
            
        formatted_prompt = self._format_prompt(prompt, category)
        
        loop = asyncio.get_running_loop()
        # Execute llama-cpp CPU-bound inference in a separate thread to prevent blocking asyncio loop
        answer = await loop.run_in_executor(None, self._run_inference_sync, formatted_prompt)
        return answer
