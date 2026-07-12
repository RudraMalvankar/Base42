from .base import BaseExecutor
from models.schemas import TaskContext, ExecutionResult
from models.enums import ExecutionRoute
from core.config import settings
from core.logger import setup_logger
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = setup_logger("fireworks_executor")

class FireworksExecutor(BaseExecutor):
    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url=settings.fireworks_base_url,
            headers={
                "Authorization": f"Bearer {settings.fireworks_api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
        
    def _select_model(self, context: TaskContext) -> str:
        # Step 4: Retry with Pro if Flash failed (failed_attempts > 1)
        if getattr(context, "failed_attempts", 0) > 1:
            return settings.fireworks_reasoning_model
            
        # Step 3: Default to Fast model for first Fireworks attempt
        return settings.fireworks_fast_model
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _post_chat(self, payload: dict) -> httpx.Response:
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response

    async def execute(self, context: TaskContext) -> ExecutionResult:
        model = self._select_model(context)
        
        system_prompt = "You are a highly efficient AI. Output ONLY the final answer. Do not use conversational filler, do not say 'Here is...', and do not explain your steps unless explicitly requested."
        
        if getattr(context, "failed_attempts", 0) > 0:
            system_prompt += " Retry: Follow format strictly. No filler."
        
        # Static max_tokens allocation to prevent truncation retries and latency spikes
        max_tokens = 4096
            
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context.request.prompt}
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }
        
        try:
            logger.info(f"Task {context.request.task_id}: Escalating to Fireworks API ({model})")
            resp = await self._post_chat(payload)
            data = resp.json()
            choice = data["choices"][0]
            answer = choice["message"]["content"].strip()
            tokens = data.get("usage", {}).get("total_tokens", 0)
            
            # --- Output Token Validator (Truncation Recovery) ---
            if choice.get("finish_reason") == "length":
                logger.warning(f"Task {context.request.task_id}: Fireworks truncated output (length). Retrying with expanded limits.")
                payload["max_tokens"] = min(max_tokens * 2, 4096)
                resp2 = await self._post_chat(payload)
                data2 = resp2.json()
                answer = data2["choices"][0]["message"]["content"].strip()
                tokens += data2.get("usage", {}).get("total_tokens", 0)
                
        except Exception as e:
            logger.error(f"Task {context.request.task_id}: Fireworks API failed - {e}")
            answer = "API Error"
            tokens = 0
            
        return ExecutionResult(
            task_id=context.request.task_id,
            answer=answer,
            route_taken=ExecutionRoute.FIREWORKS,
            fireworks_tokens=tokens
        )
