import os
from typing import List
from pydantic import BaseModel

class Settings(BaseModel):
    fireworks_api_key: str = os.getenv("FIREWORKS_API_KEY", "")
    fireworks_base_url: str = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    allowed_models: str = os.getenv("ALLOWED_MODELS", "accounts/fireworks/models/llama-v3p1-8b-instruct,accounts/fireworks/models/llama-v3p1-70b-instruct")
    
    @property
    def allowed_models_list(self) -> List[str]:
        return [m.strip() for m in self.allowed_models.split(",") if m.strip()]

settings = Settings()
