# MedAgentX Technical Implementation Blueprint

## Executive Summary
This blueprint outlines the complete system architecture, technology stack, data pipelines, containerization, and evaluation runner for implementing the MedAgentX research prototype post-research phase.

---

## 1. Technology Stack Specification

| Component | Selected Technology | Alternative Options | Justification |
| :--- | :--- | :--- | :--- |
| **Language & Runtime** | Python 3.11 | Python 3.10 | Native async performance, typing support. |
| **Primary LLM Backbone** | GPT-4o / LLaMA-3-70B | BioMistral-7B | SOTA medical reasoning capability. |
| **Local LLM Backbone** | OpenBioLLM-70B / BioMistral-7B | LLaMA-3-8B-Instruct | High open-weights medical QA benchmark score. |
| **Embedding Model** | `ncbi/MedCPT-Article-Encoder` | `BAAI/bge-large-en-v1.5` | Dense biomedical article embedding specialized on PubMed. |
| **Vector Database** | Qdrant (Docker `v1.8.0`) | Milvus / Chroma | High HNSW vector indexing performance, metadata filtering. |
| **Knowledge Graph DB** | Neo4j Community (Docker `v5.17`) | Memgraph | Native Cypher graph queries, APOC procedure support. |
| **Agent Orchestration** | LangGraph (`v0.0.26`) | AutoGen / CrewAI | State graph control, explicit node routing, persistence. |
| **Backend API** | FastAPI (`v0.110.0`) | Flask | Asynchronous SSE streaming, Pydantic validation. |
| **Data Validation** | Pydantic v2 | Marshmallow | Type enforcement for structured agent state schemas. |
| **State Memory Store** | PostgreSQL 16 + Redis | SQLite | Persistent conversation memory & vector caching. |
| **Containerization** | Docker Compose | Kubernetes | One-click local reproducibility for evaluation. |

---

## 2. Directory & Module Architecture

```text
medagentx/
├── config/
│   ├── settings.py            # Pydantic environment configuration
│   └── logging.py             # Structlog configuration
├── data/
│   ├── raw/                   # Raw guidelines & PubMed XMLs
│   ├── processed/             # Cleaned JSON chunks & triples
│   └── embeddings/            # Embedded vector caches
├── src/
│   ├── agents/                # Core Agent Modules
│   │   ├── planner.py         # Agent 1: Query Planner & Reformulator
│   │   ├── retriever.py       # Agent 2: Hybrid Retrieval Engine (Qdrant + Neo4j)
│   │   ├── verifier.py        # Agent 3: Evidence Verifier & Critic
│   │   └── reasoner.py        # Agent 4: Clinical Reasoner & Synthesizer
│   ├── orchestrator/          # ACMR Orchestration Logic
│   │   ├── state.py           # MedAgentXState Pydantic schema
│   │   ├── confidence.py      # Composite C_total formula calculator
│   │   └── router.py          # ACMR dynamic routing graph definition
│   ├── knowledge/             # Database Interfaces
│   │   ├── vector_store.py    # Qdrant client wrapper
│   │   └── graph_store.py     # Neo4j Cypher driver wrapper
│   ├── api/                   # FastAPI Web Endpoints
│   │   ├── app.py             # FastAPI app initialization
│   │   └── routes.py          # /api/v1/consult endpoints
│   └── evaluation/            # Automated Experiment Runner
│       ├── test_runner.py     # Runs MedQA benchmark tests
│       └── metrics_eval.py    # Calculates ECE, F1, Recall, Tokens
├── docker-compose.yml         # Container orchestration manifest
├── requirements.txt           # Python dependencies
└── README.md                  # Project execution guide
```

---

## 3. Data & Indexing Pipeline

```
Raw Clinical Guidelines / PubMed Abstracts
   │
   ▼
[Preprocessing & Chunking Pipeline] ──► Recursive Character Splitter (500 tokens, 50 overlap)
   │
   ├───────────────────────────────────────────┐
   ▼                                           ▼
[Vector Embedding Pipeline]             [Knowledge Graph Extractor]
- MedCPT Embeddings                      - MedLINK Entity-Relation Extractor
- Store in Qdrant Vector Collection      - Store Triples in Neo4j Graph
```

---

## 4. Evaluation Runner Implementation Plan

The automated experiment runner (`src/evaluation/test_runner.py`) will:
1. Load test queries from `MedQA` (1,273 questions) and `PubMedQA` (500 questions).
2. Execute benchmark runs across B1–B7 and Proposed MedAgentX under locked seeds.
3. Compute Top-1 Accuracy, DDx Recall@3, Faithfulness Score, ECE Calibration, Token Count, and Latency.
4. Output structured CSV reports to `research/04_literature_matrix/results.csv`.

---

## 5. Compute & Resource Requirements

- **GPU Requirement:** Minimum 1x NVIDIA RTX 4090 (24GB VRAM) or A10G for local 7B/70B model execution.
- **CPU & RAM:** 8-core CPU, 32GB RAM for Qdrant and Neo4j parallel Docker services.
- **Disk Storage:** 100GB SSD storage for vector collection and graph database indices.
