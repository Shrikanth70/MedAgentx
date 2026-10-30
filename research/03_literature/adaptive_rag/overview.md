# Literature Overview: Adaptive RAG & Dynamic Retrieval

## 1. Domain Summary
Adaptive RAG dynamically adjusts retrieval strategies (no retrieval, single-step retrieval, multi-hop retrieval, or web search) based on incoming query complexity and model confidence states.

## 2. Key Foundational Papers

### Jeong et al. (2024) — *Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity*
- **Venue:** NAACL 2024 | **arXiv:** 2403.14403
- **Core Mechanism:** Uses a lightweight classifier trained on query complexity automatically labeled via model outcome biases. Routes simple queries directly to LLM, medium queries to single-pass RAG, and complex queries to multi-hop RAG.
- **Performance:** Achieves higher accuracy while reducing unnecessary retrieval calls by up to 35%.
- **Impact on MedAgentX:** Proves that query-level complexity routing saves compute. MedAgentX extends this from query classification to real-time clinical evidence confidence routing (ACMR).

### Jiang et al. (2023) — *Active Retrieval Augmented Generation (FLARE)*
- **Venue:** EMNLP 2023 | **arXiv:** 2305.06983
- **Core Mechanism:** Forward-Looking Active REtrieval (FLARE) continuously predicts upcoming sentences; if confidence in predicted tokens falls below a threshold, it uses the candidate sentence as a query to retrieve relevant passages.
- **Impact on MedAgentX:** Demonstrates token-level and sentence-level confidence tracking as a trigger for retrieval.

## 3. Comparative Summary: Adaptive Mechanisms

| System | Trigger Mechanism | Routing Target | Domain Focus |
| :--- | :--- | :--- | :--- |
| **Adaptive-RAG** | Query Complexity Classifier | No-RAG / Single-RAG / Multi-Hop | General QA |
| **FLARE** | Token Logprob Confidence | Dynamic Mid-Generation Retrieval | Open-Domain Text |
| **Self-RAG** | Learned Critique Tokens | Retrieve / Evaluate / Reject | Open-Domain QA |
| **MedAgentX (ACMR)** | Composite Evidence Confidence State | Selective Agent Activation (Retrieve/Verify/Reflect/Abstain) | Clinical Decision Support |
