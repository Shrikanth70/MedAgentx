# 12 Detailed Experimental Protocol

## 1. Step-by-Step Execution Protocol

1. **Environment Setup:** Lock python environment (`Python 3.11`), random seeds (`seed=42`), vector DB parameters (`Qdrant HNSW M=16, ef_construct=100`), and model temperature (`temp=0.0`).
2. **Dataset Preprocessing:** Standardize MedQA test set (1,273 questions) into jsonl format with query IDs.
3. **Retrieval Indexing:** Index PubMed OA abstract vector embeddings (MedCPT) and build Neo4j Knowledge Graph triples.
4. **Automated Evaluation Execution:**
   - Execute B1–B7 and Proposed MedAgentX across all test queries.
   - Record accuracy, tokens, latency, retrieved passages, and confidence scores.
5. **Statistical Significance Testing:** Perform paired t-tests and McNemar's test ($\alpha=0.05$).
