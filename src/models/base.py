from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# Per 1M tokens in USD
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku": {"input": 0.8, "output": 4.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "o1": {"input": 15.0, "output": 60.0},
    "o1-mini": {"input": 3.0, "output": 12.0},
    "o3": {"input": 10.0, "output": 40.0},
    "o3-mini": {"input": 1.1, "output": 4.4},
    "_local": {"input": 0.0, "output": 0.0},
}


def get_pricing(model_name: str) -> dict[str, float]:
    name = model_name.lower()
    for key, pricing in MODEL_PRICING.items():
        if key in name:
            return pricing
    return MODEL_PRICING["_local"]


class ModelClient(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._pricing = get_pricing(model_name)

    @abstractmethod
    def call(self, prompt: str) -> ModelResponse:
        pass

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self._pricing["input"] / 1_000_000
            + output_tokens * self._pricing["output"] / 1_000_000
        )
