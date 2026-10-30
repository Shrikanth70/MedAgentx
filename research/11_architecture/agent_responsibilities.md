# 11 Agent Responsibilities & State Schemas

## 1. Agent State Definition (Pydantic / LangGraph State)

```python
class MedAgentXState(TypedDict):
  query: str
  patient_context: dict
  sub_queries: list[str]
  mesh_terms: list[str]
  vector_evidence: list[dict]
  graph_evidence: list[dict]
  verified_evidence: list[dict]
  contradictions_found: list[str]
  confidence_scores: dict  # S_vec, S_graph, V_agree, R_contra, C_total
  iteration_count: int
  max_iterations: int
  reasoning_trace: list[str]
  final_report: dict
  abstain: bool
```

## 2. Granular Module Specifications

### Module 1: Query Planner & Reformulator
- **Input:** Raw clinical query + patient record.
- **Output:** Structured sub-questions, standardized MeSH concept terms.

### Module 2: Hybrid Retrieval Engine
- **Input:** Expanded queries + MeSH terms.
- **Output:** Vector passages from Qdrant + Graph triples from Neo4j.

### Module 3: Evidence Verifier & Critic
- **Input:** Retrieved vector/graph passages.
- **Output:** Verified context subset, contradiction flags, $C_{total}$ calculation.

### Module 4: Clinical Reasoner & Synthesizer
- **Input:** Verified evidence + confidence state.
- **Output:** Differential Diagnosis, Investigations, Structured Report, Citations.
