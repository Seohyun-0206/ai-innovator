import time
import openai
from config import OPENAI_API_KEY, OPENAI_BASE_URL
from src.models.base import ModelClient, ModelResponse

# o1/o3/o4 reasoning 모델은 temperature 미지원, max_completion_tokens 사용
_REASONING_PREFIXES = ("o1", "o3", "o4")


class OpenAIModel(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        self._is_reasoning = model_name.lower().startswith(_REASONING_PREFIXES)

    def call(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        try:
            params: dict = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self._is_reasoning:
                params["max_completion_tokens"] = 16
            else:
                params["max_tokens"] = 16
                params["temperature"] = 0

            response = self._client.chat.completions.create(**params)
            latency = time.perf_counter() - start
            usage = response.usage
            return ModelResponse(
                text=response.choices[0].message.content or "",
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                latency=latency,
            )
        except Exception as e:
            latency = time.perf_counter() - start
            return ModelResponse(text="", latency=latency, error=str(e))
