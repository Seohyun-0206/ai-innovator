import time
import openai
from config import OPENAI_API_KEY, OPENAI_BASE_URL
from src.models.base import ModelClient, ModelResponse


class OpenAIModel(ModelClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

    def call(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0,
            )
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
