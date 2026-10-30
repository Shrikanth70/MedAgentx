# 10 Baselines Specification

## 1. Selected Baseline Paradigms

To prove the superiority of MedAgentX (ACMR), we evaluate against 6 fair baseline implementations using identical base LLMs (BioMistral-7B / LLaMA-3-8B / GPT-4o-mini):

| Baseline | Architecture | Retrieval Subsystem | Control Flow | Implementation Difficulty |
| :--- | :--- | :--- | :--- | :--- |
| **B1: Standalone LLM** | Direct Prompting | None (Parametric only) | Single-pass | Low |
| **B2: Standard Vector RAG** | Retrieve-Then-Generate | Dense Vector (Qdrant) | Fixed single-pass | Low |
| **B3: Medical RAG** | Advanced RAG | Dense Vector + Reranking | Fixed single-pass | Moderate |
| **B4: Corrective RAG (CRAG)** | Evaluator RAG | Vector + Web Search | Evaluator triggered | Moderate |
| **B5: Adaptive RAG** | Complexity Routing | Vector (Single/Multi-hop) | Classifier triggered | Moderate |
| **B6: SEMA-RAG** | Decoupled 3-Agent | Dense Vector | Fixed 3-agent pass | High |
| **B7: Fixed 8-Agent Pipeline** | Fixed Multi-Agent | Vector + Graph | Sequential 8-agent pass | High |
| **Proposed: MedAgentX** | Adaptive 4-Agent (ACMR) | Hybrid Vector + KG | Composite Confidence ACMR | High |
