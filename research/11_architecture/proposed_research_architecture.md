# 11 Proposed Research Architecture & Data Flow

## 1. Technical System Architecture

MedAgentX is implemented as an event-driven, graph-based multi-agent state machine utilizing **LangGraph**, **FastAPI**, **Qdrant**, and **Neo4j**.

```
[User Clinical Query]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│               ACMR Central Orchestrator                │
│ 1. Initialize State & Compute Initial Query Complexity │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│     Agent 1: Query Planner & Reformulator              │
│ - Decomposes clinical query into sub-questions         │
│ - Maps entities to MeSH / ICD-10 medical terms         │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│     Agent 2: Hybrid Retrieval Engine                   │
│ - Parallel Dense Vector Search (Qdrant)                │
│ - Parallel Entity-Relation Graph Traversal (Neo4j)     │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│     ACMR State Evaluator & Verification Loop           │
│ - Agent 3: Evidence Verifier checks relevance/conflicts│
│ - Compute Composite Confidence Score C_total           │
└────────────────────────┬───────────────────────────────┘
                         │
       ┌─────────────────┴─────────────────┐
       │ C_total Evaluation                │
       ├───────────────────────────────────┤
       │ If C_total >= 0.80  ──► STOP & Generate Report  │
       │ If 0.50 <= C_total < 0.80 ──► Re-query / Reflect│
       │ If C_total < 0.50   ──► Safe Abstention Warning │
       └─────────────────┬─────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│     Agent 4: Clinical Reasoner & Synthesizer           │
│ - Generates Differential Diagnoses (DDx)               │
│ - Orders Investigations & Evidence-Backed Treatments   │
│ - Formulates Structured Report + Citations             │
└────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Confidence Formulation ($C_{total}$)

The Composite Evidence Confidence Score $C_{total} \in [0, 1]$ is formally defined as:

$$C_{total} = w_1 \cdot S_{vec} + w_2 \cdot S_{graph} + w_3 \cdot V_{agree} - w_4 \cdot R_{contra}$$

Where:
- $S_{vec} \in [0, 1]$: Mean cosine similarity of top-k vector retrieved passages.
- $S_{graph} \in [0, 1]$: Relational path completeness score from Neo4j Knowledge Graph.
- $V_{agree} \in [0, 1]$: Verifier agent consensus score on claim relevance.
- $R_{contra} \in [0, 1]$: Measure of contradictory evidence detected across sources.
- Weights: $w_1 = 0.35, w_2 = 0.25, w_3 = 0.25, w_4 = 0.15$ (constrained such that $\sum w = 1.0$).
