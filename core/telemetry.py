import json
import time
import asyncio
from typing import List, Dict
from dataclasses import dataclass, asdict
from core.logger import setup_logger

logger = setup_logger("telemetry")

@dataclass
class TaskTrace:
    task_id: str
    category: str
    latency_ms: float
    route: str
    fallback_triggered: bool
    tokens: int

class TelemetryService:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TelemetryService, cls).__new__(cls)
            cls._instance.traces = []
            cls._instance.counters = {
                "total_tasks": 0,
                "api_fallbacks": 0,
                "total_tokens": 0,
                "routes": {}
            }
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls()
        return cls._instance

    async def record(self, trace: TaskTrace):
        async with self._lock:
            self.traces.append(trace)
            self.counters["total_tasks"] += 1
            self.counters["total_tokens"] += trace.tokens
            
            if trace.fallback_triggered:
                self.counters["api_fallbacks"] += 1
                
            route = trace.route
            if route not in self.counters["routes"]:
                self.counters["routes"][route] = 0
            self.counters["routes"][route] += 1
            
            # Lightweight checkpointing every 10 tasks to prevent total loss on OOM kills
            if self.counters["total_tasks"] % 10 == 0:
                self._flush_checkpoint()

    def _flush_checkpoint(self):
        try:
            self.dump_report("/output/telemetry_checkpoint.json")
        except Exception:
            pass

    def dump_report(self, path: str):
        report = {
            "summary": self.counters,
            "traces": [asdict(t) for t in self.traces]
        }
        try:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Telemetry report dumped to {path}")
        except Exception as e:
            logger.error(f"Failed to write telemetry: {e}")
