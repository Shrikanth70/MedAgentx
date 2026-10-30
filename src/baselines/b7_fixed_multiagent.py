"""Baseline B7: Fixed 4-Agent Sequential Pipeline (Non-Adaptive Benchmark)."""

import time
import structlog
from src.baselines.base import BaseSystem, SystemResponse
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.verifier import VerifierAgent
from src.agents.reasoner import ReasonerAgent

logger = structlog.get_logger()


class B7FixedMultiAgent(BaseSystem):
    """Baseline B7: Fixed 4-Agent Pipeline executing all 4 agents sequentially without adaptive routing."""

    def __init__(self):
        super().__init__(system_id="B7_FIXED_MULTIAGENT")
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.verifier = VerifierAgent()
        self.reasoner = ReasonerAgent()

    def query(self, user_query: str) -> SystemResponse:
        """Execute static 4-agent sequential pass."""
        start_time = time.time()

        # Step 1: Agent 1 - Planner
        plan_out = self.planner.plan(user_query)

        # Step 2: Agent 2 - Retriever (always executed)
        ret_out = self.retriever.retrieve(
            query_text=user_query,
            entities=plan_out.extracted_entities,
            top_k=5,
        )

        # Step 3: Agent 3 - Verifier (always executed)
        ver_out = self.verifier.verify(
            query_text=user_query,
            vector_evidence=ret_out.vector_evidence,
            graph_evidence=ret_out.graph_evidence,
        )

        # Static confidence calculation without early stopping
        c_fixed = 0.75

        # Step 4: Agent 4 - Reasoner (always executed)
        reas_out = self.reasoner.synthesize(
            query_text=user_query,
            verified_evidence=ver_out.verified_evidence,
            detected_contradictions=ver_out.detected_contradictions,
            c_total=c_fixed,
        )

        latency = time.time() - start_time
        token_count = (
            len(user_query.split())
            + len(str(plan_out).split())
            + len(str(ret_out).split())
            + len(str(ver_out).split())
            + len(reas_out.clinical_report.split())
        )

        return SystemResponse(
            system_id=self.system_id,
            query=user_query,
            answer=reas_out.clinical_report,
            evidence_chunks=ver_out.verified_evidence,
            citations=reas_out.citations,
            confidence=c_fixed,
            latency_seconds=round(latency, 4),
            token_count=token_count,
            metadata={
                "retrieval_rounds": 1,
                "active_agent_count": 4,
                "is_adaptive": False,
            },
        )
