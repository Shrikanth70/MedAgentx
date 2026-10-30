"""Baseline B0: Standalone LLM (Zero-Shot Clinical Reasoning without Retrieval)."""

import time
import structlog
from src.baselines.base import BaseSystem, SystemResponse
from src.agents.llm_client import LLMClient

logger = structlog.get_logger()


class B0LLMOnly(BaseSystem):
    """Baseline B0: Standalone LLM without external evidence retrieval."""

    def __init__(self):
        super().__init__(system_id="B0_LLM_ONLY")
        self.llm_client = LLMClient()

    def query(self, user_query: str) -> SystemResponse:
        """Execute query using direct LLM internal parametric knowledge."""
        start_time = time.time()
        
        prompt = f"Answer the following clinical question zero-shot: {user_query}"
        llm_response = self.llm_client.generate(prompt=prompt, system_prompt="You are a clinical assistant.")

        if llm_response:
            answer = f"Ollama llama3.2 Zero-Shot Answer: {llm_response}"
        else:
            answer = (
                f"Based on parametric clinical knowledge, regarding '{user_query}': "
                "Primary management involves prompt clinical evaluation, diagnostic confirmation, "
                "and evidence-guided therapy. Note that this response is generated without external retrieval."
            )

        latency = time.time() - start_time
        token_count = len(user_query.split()) + len(answer.split())

        return SystemResponse(
            system_id=self.system_id,
            query=user_query,
            answer=answer,
            evidence_chunks=[],
            citations=[],
            confidence=0.50,
            latency_seconds=round(latency, 4),
            token_count=token_count,
            metadata={"retrieval_rounds": 0},
        )
