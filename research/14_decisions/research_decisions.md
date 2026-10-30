# 14 Research Decision Log

## Decision Record 1: Streamline Architecture to 4 Core Agents + ACMR
- **Date:** 2026-08-13
- **Decision:** Consolidate the original 8 agents into 4 core functional modules (Planner/Rewriter, Hybrid Retrieval Engine, Evidence Verifier/Critic, Clinical Reasoner/Synthesizer) governed by the ACMR Orchestrator.
- **Reason:** Literature review proves that 8 fixed agents introduce severe latency overhead (20s+ per query), token inefficiency, and inter-agent error propagation without accuracy gains.
- **Evidence:** DyLAN (ICLR 2024), SEMA-RAG (ACL 2026).
- **Alternatives Considered:** Maintaining fixed 8-agent pipeline.
- **Why Rejected:** High rejection risk in IEEE peer review due to over-engineering.
- **Impact:** Shifted research focus toward mathematical confidence calibration ($C_{total}$) and dynamic ACMR cost routing.

---

## Decision Record 2: Replace Verbalized Confidence with Mathematical Formula ($C_{total}$)
- **Date:** 2026-08-13
- **Decision:** Replace prompt-based LLM self-confidence ("Rate confidence 0-100%") with a mathematical composite score combining vector similarity, graph path validity, verifier consensus, and claim contradiction rates.
- **Reason:** LLM verbalized confidence is uncalibrated and prone to overconfident hallucinations on edge cases (MedCoAct 2025 limitation).
- **Evidence:** CONFRAG (2026), RAG-Checker (2024).
- **Impact:** Establishes defensible algorithmic novelty for IEEE publication.
