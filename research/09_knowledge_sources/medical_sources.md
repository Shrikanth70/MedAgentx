# 09 Medical Knowledge Sources Inventory

## 1. Overview
MedAgentX integrates three tiers of clinical knowledge sources for hybrid vector-graph indexing:

```
                  ┌───────────────────────────────┐
                  │    Clinical Guidelines        │
                  │   (WHO, NICE, CDC, ACC/AHA)   │
                  └───────────────┬───────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
  ┌──────────────────────────────┐┌──────────────────────────────┐
  │   Biomedical Literature      ││   Medical Knowledge Graphs   │
  │    (PubMed Open Access)      ││   (UMLS, SNOMED-CT, MeSH)    │
  └──────────────────────────────┘└──────────────────────────────┘
```

## 2. Source Details

### Tier 1: Clinical Practice Guidelines (CPGs)
- **Sources:** WHO Guidelines, NICE Guidelines (UK), CDC Recommendations, ACC/AHA Cardiology Guidelines.
- **Role:** Gold-standard ground truth for therapeutic protocols and investigation workflows.
- **Indexing:** Hybrid Chunking + Vector Database (Qdrant).

### Tier 2: Biomedical Literature
- **Sources:** PubMed Open Access Subset (PMC OA).
- **Role:** Peer-reviewed clinical trial evidence and recent biomedical findings.
- **Indexing:** Dense semantic embeddings (MedCPT / BioClinical-BERT).

### Tier 3: Structured Medical Knowledge Graphs
- **Sources:** UMLS (Unified Medical Language System), SNOMED-CT subset, MeSH descriptors.
- **Role:** Entity-relationship triple store (`Disease -[HAS_SYMPTOM]-> Symptom`, `Drug -[TREATS]-> Condition`).
- **Indexing:** Neo4j Graph Database.
