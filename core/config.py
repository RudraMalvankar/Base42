import os
from typing import List
from pydantic import BaseModel

class Settings(BaseModel):
    fireworks_api_key: str = os.getenv("FIREWORKS_API_KEY", "")
    fireworks_base_url: str = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    allowed_models: str = os.getenv("ALLOWED_MODELS", "accounts/fireworks/models/deepseek-v4-flash,accounts/fireworks/models/deepseek-v4-pro")
    
    @property
    def fireworks_fast_model(self) -> str:
        return self.allowed_models_list[0]

    @property
    def fireworks_reasoning_model(self) -> str:
        return self.allowed_models_list[-1] if len(self.allowed_models_list) > 1 else self.allowed_models_list[0]
    @property
    def allowed_models_list(self) -> List[str]:
        return [m.strip() for m in self.allowed_models.split(",") if m.strip()]

settings = Settings()
