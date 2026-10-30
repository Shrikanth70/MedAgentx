# State of the Art: Retrieval-Augmented Generation (RAG)

## 1. Evolution Timeline (2020–2026)

```
2020: Standard Dense RAG (Lewis et al., DPR)
  │
2022: Hybrid Dense-Sparse RAG (BM25 + Contriever)
  │
2023: Active & Mid-Generation RAG (FLARE, Self-RAG)
  │
2024: Corrective RAG (CRAG) & Adaptive RAG (Jeong et al.)
  │
2024: Graph RAG (Microsoft GraphRAG)
  │
2025–2026: Multi-Agent & Confidence-Guided RAG (SEMA-RAG, MedCoAct, CONFRAG, MedAgentX)
```

## 2. Key Paradigm Comparison

| Paradigm | Control Logic | Retrieval Strategy | Verification | Compute Cost | Clinical Safety |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive RAG** | Single-pass static | Dense Vector | None | Low | Low (High hallucination risk) |
| **Advanced RAG** | Pre/Post processing | Vector + Reranking | None | Moderate | Moderate |
| **Corrective RAG** | Evaluator confidence | Vector + Web Search | Document score | Moderate | Low (Open web noise) |
| **Adaptive RAG** | Pre-retrieval classifier | Complexity-based | None | Optimized | Moderate |
| **Graph RAG** | Graph communities | Knowledge Graph | Structure-based | High | High (Relational accuracy) |
| **SEMA-RAG** | Decoupled 3-Agent | Iterative Dense | Arbiter agent | High | High |
| **MedAgentX (ACMR)** | Composite Evidence Confidence State | Hybrid Vector + KG | Verification Agent + Reflection | Dynamically Optimized | Maximum (Strict clinical grounding) |

## 3. Position of MedAgentX
MedAgentX sits at the intersection of **Adaptive RAG, Graph RAG, and Multi-Agent Systems**, using real-time confidence states to dynamically route clinical queries across specialized agent nodes.
