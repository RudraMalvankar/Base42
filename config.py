import os

def get_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).lower() in ('true', '1', 'yes')

class FeatureFlags:
    ENABLE_LOCAL_FACTUAL = False
    ENABLE_LOCAL_SUMMARIZATION = False
    ENABLE_RESERVED_LOCAL_WORKERS = True
    ENABLE_SHORT_FIREWORKS_PROMPTS = True
    ENABLE_DYNAMIC_MAX_TOKENS = False
    
    ENABLE_RESPONSE_COMPRESSION = True
    ENABLE_ROUTER_FIX = True
    ENABLE_SMART_ROUTING_V2 = True
    FIREWORKS_CONCURRENCY = 4
    
    ENABLE_DETERMINISTIC_EXTRACTION = True
    ENABLE_LOCAL_FACTUAL = False
