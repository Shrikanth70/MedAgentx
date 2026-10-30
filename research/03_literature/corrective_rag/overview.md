# Literature Overview: Corrective RAG & Evidence Verification

## 1. Domain Summary
Corrective RAG (CRAG) and evidence verification paradigms introduce explicit evaluation modules to check retrieved passage relevance, detect low-confidence or contradictory context, and trigger corrective fallback actions before response generation.

## 2. Key Foundational Papers

### Yan et al. (2024) — *Corrective Retrieval Augmented Generation (CRAG)*
- **Venue:** arXiv:2401.15884 (2024)
- **Core Mechanism:** Introduces a lightweight Retrieval Evaluator that assigns a confidence score to retrieved documents.
  - **Correct:** Document confidence is high $\rightarrow$ proceed with generation.
  - **Incorrect:** Document confidence is low $\rightarrow$ discard documents, query web search engines.
  - **Ambiguous:** Confidence is moderate $\rightarrow$ combine retrieved documents with web search results.
- **Decompose-Then-Recompose:** Extracts key passage strips, discarding irrelevant noise.
- **Relevance to MedAgentX:** Validates the necessity of a dedicated Evidence Verification module. In clinical settings, web search cannot be a blind fallback due to unvetted medical sources; MedAgentX replaces open web search with structured Knowledge Graph traversal and guideline verification.

### RAG-Checker (2024) & Speculative RAG (2024)
- Focus on evaluating claim-level faithfulness and fine-grained claim attribution.
