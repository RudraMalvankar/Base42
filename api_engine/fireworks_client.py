import os
import httpx
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.logger import setup_logger
from router.classifier import TaskCategory

logger = setup_logger("fireworks_client")

class FireworksClient:
    def __init__(self):
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        
        # Strip trailing slash if present
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]
            
        allowed_models_str = os.environ.get("ALLOWED_MODELS", "")
        self.allowed_models = [m.strip() for m in allowed_models_str.split(",") if m.strip()]
        
        logger.info(f"Initialized FireworksClient with base URL: {self.base_url}")
        logger.info(f"Allowed models: {self.allowed_models}")

    def select_model(self, category: TaskCategory) -> str:
        """
        Dynamically selects the most suitable model from ALLOWED_MODELS based on task category complexity.
        """
        if not self.allowed_models:
            raise ValueError("ALLOWED_MODELS environment variable is empty or not set.")
            
        if len(self.allowed_models) == 1:
            return self.allowed_models[0]
            
        # Determine if the task needs a premium model
        needs_premium = category in [
            TaskCategory.MATH,
            TaskCategory.LOGIC,
            TaskCategory.DEBUGGING,
            TaskCategory.CODE_GEN
        ]
        
        if needs_premium:
            # Prefer largest model
            for keyword in ["405b", "70b", "72b", "22b", "8x22b"]:
                for model in self.allowed_models:
                    if keyword in model.lower():
                        logger.info(f"Selected premium model: {model} for category: {category}")
                        return model
            # Fallback to the first available model
            return self.allowed_models[0]
        else:
            # Prefer smaller model to save tokens
            for keyword in ["8b", "8x7b", "3b", "2b", "1.5b"]:
                for model in self.allowed_models:
                    if keyword in model.lower():
                        logger.info(f"Selected lightweight model: {model} for category: {category}")
                        return model
            # Try to avoid massive models if possible for simple tasks
            for model in self.allowed_models:
                if not any(k in model.lower() for k in ["405b", "70b", "72b"]):
                    return model
            return self.allowed_models[0]

    def _get_system_prompt(self, category: TaskCategory) -> str:
        """
        Returns a highly optimized, short system prompt designed to ensure high accuracy
        while minimizing input/output tokens.
        """
        if category == TaskCategory.SENTIMENT:
            return "Classify sentiment as Positive, Negative, or Neutral. Output: classification and brief 1-sentence justification."
        elif category == TaskCategory.NER:
            return "Extract Named Entities (Person, Organization, Location, Date). Output JSON: [{\"entity\": \"...\", \"type\": \"...\"}]."
        elif category == TaskCategory.SUMMARIZATION:
            return "Summarize the text in exactly one sentence. Do not add intro/outro."
        elif category == TaskCategory.MATH:
            return "Solve the math problem step-by-step. Provide the final numerical answer clearly at the end."
        elif category == TaskCategory.DEBUGGING:
            return "Identify the bug, explain it in 1 sentence, and output the corrected Python code block."
        elif category == TaskCategory.CODE_GEN:
            return "Write a correct, well-structured Python function matching the spec. Return only the Python code."
        elif category == TaskCategory.LOGIC:
            return "Solve the logic puzzle step-by-step and state the final answer clearly."
        else:
            return "Answer the question directly and concisely in 1-2 sentences."

    # Retry API calls with exponential backoff for rate limits or network issues
    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def generate_completion_async(self, prompt: str, category: TaskCategory, timeout: float = 25.0) -> str:
        """
        Calls Fireworks API asynchronously using httpx with retries and timeout.
        """
        model = self.select_model(category)
        system_prompt = self._get_system_prompt(category)
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Configure temperature based on need for deterministic outputs
        temp = 0.0 if category in [TaskCategory.MATH, TaskCategory.LOGIC, TaskCategory.DEBUGGING, TaskCategory.NER, TaskCategory.SENTIMENT] else 0.3
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temp,
            "max_tokens": 512  # Keep max response length constrained to optimize output token count
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            answer = result["choices"][0]["message"]["content"].strip()
            return answer
