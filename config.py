import os

# ── OpenAI-compatible API ─────────────────────────────────────────────────────
# 모든 API 모델은 OpenAI SDK를 통해 호출됩니다.
# base_url을 바꾸면 다른 OpenAI-compatible 엔드포인트(Azure, Together 등)도 사용 가능합니다.

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-api-key-here")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ── Ollama (로컬 모델) ────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── 로컬 모델로 처리할 모델명 접두사 ─────────────────────────────────────────
# 이 목록에 해당하면 Ollama로 라우팅됩니다.
LOCAL_MODEL_PREFIXES: list[str] = [
    "llama", "mistral", "gemma", "qwen", "phi", "deepseek",
    "codellama", "solar", "exaone", "vicuna", "orca",
]
