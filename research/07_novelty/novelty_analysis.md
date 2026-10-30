# 07 Novelty Analysis & Architectural Component Audit

## 1. Executive Summary

This document performs a granular novelty audit of all 18 proposed features and components in the MedAgentX project. Each feature is audited against SOTA literature (2020–2026) and classified as:
- 🟢 **Novel:** Scientifically defensible contribution.
- 🟡 **Partially Novel / Combination:** Novel integration of existing ideas.
- 🔴 **Already Established:** Well-known in literature; cannot be claimed as a primary novelty.

---

## 2. Component-by-Component Novelty Audit

| Component / Feature | Existing SOTA Papers | SOTA Status | Novelty Rating | Risk & Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Agent Architecture** | SEMA-RAG (2026), AutoGen (2023), MetaGPT (2024) | Standard engineering pattern | 🔴 Established | Do NOT claim basic multi-agent setup as novel. |
| **Planner Agent** | AgentVerse (2023), SEMA-RAG (2026) | Standard task decomposition | 🔴 Established | Treat as a functional component, not a research contribution. |
| **Query Rewriting Agent** | Adaptive-RAG (2024), CRAG (2024) | Standard RAG pre-retrieval step | 🔴 Established | Standard utility module. |
| **Vector Retrieval** | DPR (2020), Qdrant / FAISS | Foundational dense retrieval | 🔴 Established | Baseline component. |
| **Knowledge Graph Traversal** | Microsoft GraphRAG (2024), Neo4j | Established in Graph RAG | 🟡 Combination | Novel only when integrated into dynamic ACMR loops. |
| **Hybrid Vector + KG Retrieval** | PathRAG (2025), Hybrid RAG (2024) | Emerging paradigm | 🟡 Combination | Strong baseline; claim synergy in multi-hop clinical QA. |
| **Evidence Verification Agent** | CRAG (2024), ArbGraph (2026) | Evaluator/Verifier pattern | 🟡 Combination | Differentiate by implementing clinical contradiction filtering. |
| **Reflection Agent** | Reflexion (2023), Self-Refine (2023) | Established RL/LLM memory | 🔴 Established | Use as component; do not claim isolation novelty. |
| **Clinical Reasoning Agent** | Med-PaLM 2 (2025), AMIE (2025) | Established clinical LLM task | 🔴 Established | Benchmark task. |
| **Explainability Agent** | RAG-Checker (2024) | Standard attribution tracing | 🔴 Established | Secondary feature. |
| **Memory / Context Store** | AutoGen (2023), LangChain State | Standard state management | 🔴 Established | Utility infrastructure. |
| **Verbalized LLM Confidence** | MedCoAct (2025), Self-RAG (2024) | Subjective LLM self-report | 🔴 Established | FLAWED in literature; replace with mathematical formula. |
| **Composite Mathematical Confidence ($C_{total}$)** | CONFRAG (2026), MedAgentX | **Novel formulation** | 🟢 **NOVEL** | **Primary Novelty Claim 1:** Mathematical multi-factor score. |
| **Adaptive Confidence Orchestration (ACMR)** | MedAgentX | **Novel policy** | 🟢 **NOVEL** | **Primary Novelty Claim 2:** Dynamic agent state routing. |
| **Evidence Sufficiency Estimation** | CONFRAG (2026), SEMA-RAG (2026) | Emerging | 🟢 **NOVEL** | Core component of ACMR. |
| **Adaptive Early Stopping** | Adaptive-RAG (2024) | Complexity-only in SOTA | 🟢 **NOVEL** | Extend to post-retrieval evidence sufficiency stopping. |
| **Uncertainty Handling / Abstention** | Clinical AI Safety literature | High clinical priority | 🟢 **NOVEL** | Abstention trigger based on high contradiction rate. |
| **Cost & Latency-Aware Agent Routing** | DyLAN (2024) | Agent pruning in general QA | 🟢 **NOVEL** | First application of cost-aware agent routing in Clinical RAG. |

---

## 3. Summary Classification Matrix

- 🟢 **Genuinely Novel Contributions (4):**
  1. Composite Mathematical Evidence Confidence State ($C_{total}$).
  2. Adaptive Confidence-Guided Multi-Agent Routing Policy (ACMR).
  3. Post-Retrieval Evidence Sufficiency Early Stopping & Safe Abstention.
  4. Latency- and Token-Cost-Aware Agent Activation in Clinical Decision Support.

- 🟡 **Partially Novel / Synergistic Combinations (3):**
  1. Dynamic Hybrid Vector (Qdrant) + Knowledge Graph (Neo4j) Triples Integration.
  2. Clinical Contradiction Filtering in Evidence Verification.
  3. Evidence-Grounded Differential Diagnosis Traceability.

- 🔴 **Already Established Baseline Features (11):**
  1. Multi-Agent Setup, Planner Agent, Query Rewrite Agent, Vector Retrieval, Knowledge Graph Indexing, Reflection Agent, Clinical Reasoning Prompting, Explainability Traces, Persistent Memory, Chat UI, PDF Parsing.
