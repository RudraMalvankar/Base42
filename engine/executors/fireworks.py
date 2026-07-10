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
        models = settings.allowed_models_list
        # Pick largest allowed model for hard problems, smallest for fallbacks
        if not models:
            raise ValueError("No ALLOWED_MODELS configured.")
            
        if "70b" in models[-1].lower() or "405b" in models[-1].lower():
            # Use largest model for logic/code
            return models[-1]
        return models[0]
        
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
        
        system_prompt = "You are a highly efficient AI. Answer the user's prompt directly, correctly, and completely without any conversational filler like 'Here is the answer'."
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context.request.prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }
        
        try:
            logger.info(f"Task {context.request.task_id}: Escalating to Fireworks API ({model})")
            resp = await self._post_chat(payload)
            data = resp.json()
            answer = data["choices"][0]["message"]["content"].strip()
            tokens = data.get("usage", {}).get("total_tokens", 0)
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
