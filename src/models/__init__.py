from config import LOCAL_MODEL_PREFIXES
from src.models.base import ModelClient, ModelResponse
from src.models.openai_model import OpenAIModel
from src.models.ollama_model import OllamaModel


def get_model(model_name: str) -> ModelClient:
    name = model_name.lower()
    if any(name.startswith(p) for p in LOCAL_MODEL_PREFIXES):
        return OllamaModel(model_name)
    return OpenAIModel(model_name)
