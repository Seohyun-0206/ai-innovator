import time
import requests
from config import OLLAMA_BASE_URL
from src.models.base import ModelClient, ModelResponse


class OllamaModel(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._base_url = OLLAMA_BASE_URL

    def call(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            latency = time.perf_counter() - start
            return ModelResponse(
                text=data.get("response", ""),
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
                latency=latency,
            )
        except Exception as e:
            latency = time.perf_counter() - start
            return ModelResponse(text="", latency=latency, error=str(e))
