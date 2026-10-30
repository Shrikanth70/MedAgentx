"""Ollama & Foundation Model Interface for MedAgentX."""

import httpx
import structlog
from config.settings import settings

logger = structlog.get_logger()


class LLMClient:
    """Client for generating completions via Ollama (llama3.2) or fallback LLM providers."""

    def __init__(
        self,
        provider: str = settings.LLM_PROVIDER,
        model: str = settings.LLM_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text completion from Ollama llama3.2 with extended timeout."""
        if self.provider == "ollama":
            try:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = httpx.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": settings.TEMPERATURE,
                            "num_predict": 350,
                        },
                    },
                    timeout=120.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    res_text = data.get("response", "").strip()
                    logger.info("Ollama llama3.2 generation successful", model=self.model)
                    return res_text
            except Exception as e:
                logger.error("Ollama API call failed; falling back to internal synthesis", error=str(e))

        return ""
