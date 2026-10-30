# Literature Overview: Graph RAG & Knowledge Graph Integration

## 1. Domain Summary
Graph RAG combines unstructured text vector search with structured Knowledge Graph (KG) representations (entities, relationships, medical ontologies like UMLS, SNOMED-CT, MeSH) to support multi-hop reasoning, relationship discovery, and global text summarization.

## 2. Key Foundational Papers

### Edge et al. (Microsoft Research, 2024) — *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*
- **Venue:** arXiv:2404.16130 (2024)
- **Core Mechanism:** Uses an LLM to build entity-relationship graphs from text, partitions graphs into hierarchical communities, generates community summaries, and synthesizes answers across global topics.
- **Key Advantage over Vector RAG:** Handles global sensemaking and multi-document synthesis where standard vector search fails.

### Hybrid Graph-Vector Retrieval in Healthcare (2024–2026)
- Integrates semantic embeddings with medical ontology triples (`[Disease] --(has_symptom)--> [Symptom]`, `[Drug] --(contraindicated_with)--> [Condition]`).
- Reduces false clinical association risks by grounding reasoning in verified medical knowledge graph triples.
