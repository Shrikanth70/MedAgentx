# Literature Overview: Standard Retrieval-Augmented Generation (RAG)

## 1. Domain Summary
Retrieval-Augmented Generation (RAG) combines parametric knowledge stored in neural network weights with non-parametric memory retrieved from external knowledge bases. Introduced by Lewis et al. (2020), RAG addresses knowledge cutoffs and factual hallucinations in Large Language Models (LLMs).

## 2. Evolution of RAG Paradigms

```
Naive RAG (Single-pass Retrieve -> Read -> Generate)
   │
   ▼
Advanced RAG (Pre-retrieval query expansion, Post-retrieval reranking)
   │
   ▼
Modular / Agentic RAG (Dynamic routing, verification, iterative loops)
```

## 3. Key Foundational Papers

### Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- **Venue:** NeurIPS 2020
- **Key Insight:** End-to-end training of retriever (DPR) and generator (BART) models. Introduced RAG-Sequence and RAG-Token models.
- **Limitation for Medicine:** Single-pass retrieval is insufficient for complex clinical differential diagnosis requiring multi-source evidence synthesis.

### Karpukhin et al. (2020) — *Dense Passage Retrieval for Open-Domain Question Answering (DPR)*
- **Venue:** EMNLP 2020
- **Key Insight:** Dense embeddings using dual-encoder BERT architectures outperform traditional sparse BM25 lexical retrieval.
- **Relevance to MedAgentX:** Provides foundational dense retrieval mechanisms used in hybrid retrieval components.

## 4. Key Limitations of Standard RAG in Clinical AI
1. **Single-Pass Bottleneck:** Retrieves documents once before generation without validating whether retrieved passages contain adequate clinical evidence.
2. **Noise & Irrelevance Propagation:** Irrelevant passages retrieved due to semantic similarity mislead the generator, producing dangerous clinical hallucinations.
3. **Lack of Reasoning Structure:** Standard RAG treats clinical query answering as flat document matching rather than multi-step diagnostic reasoning.
