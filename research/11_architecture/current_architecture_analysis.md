# 11 Current vs. Proposed Architecture Analysis

## 1. Evaluation of Original 8-Agent Architecture

The original proposal specifies 8 agents + Central Orchestrator + Memory:
1. Planner Agent
2. Query Rewrite Agent
3. Hybrid Retrieval Agent
4. Evidence Verification Agent
5. Clinical Reasoning Agent
6. Reflection Agent
7. Report Generation Agent
8. Explainability Agent

### Critical Research Critique
- **Redundancy:** Having separate Report Generation and Explainability agents as autonomous graph nodes adds unnecessary LLM hops. Report structuring and reasoning trace formatting can be handled directly by the Clinical Reasoning agent or final output formatter.
- **Latency & Error Cascade:** 8 sequential steps multiply API latency (8 x 2.5s = 20s per turn) and increase risk of inter-agent context drift.

---

## 2. Refined Streamlined 4-Core Agent Architecture

To maximize scientific defensibility, reproducibility, and computational efficiency for IEEE publication, we streamline MedAgentX into **4 Specialized Core Agent Modules** governed by the **ACMR Central Orchestrator**:

```
                              ┌───────────────────────────────────┐
                              │    ACMR Central Orchestrator      │
                              │   (State & Confidence Manager)    │
                              └─────────────────┬─────────────────┘
                                                │
         ┌──────────────────────┬───────────────┴───────────────┬──────────────────────┐
         ▼                      ▼                               ▼                      ▼
┌─────────────────┐    ┌─────────────────┐             ┌─────────────────┐    ┌─────────────────┐
│   Agent 1:      │    │   Agent 2:      │             │   Agent 3:      │    │   Agent 4:      │
│ Query Planner   │    │ Hybrid Retrieval│             │ Evidence        │    │ Clinical        │
│ & Reformulator  │    │ Engine (Vec+KG) │             │ Verifier        │    │ Reasoner        │
└─────────────────┘    └─────────────────┘             └─────────────────┘    └─────────────────┘
```

### Streamlined Agent Consolidation Strategy
1. **Query Planner & Reformulator (Agent 1):** Combines original Planner and Query Rewrite agents.
2. **Hybrid Retrieval Engine (Agent 2):** Retains hybrid vector (Qdrant) + graph (Neo4j) retrieval.
3. **Evidence Verifier & Critic (Agent 3):** Combines original Evidence Verification and Reflection agents.
4. **Clinical Reasoner & Synthesizer (Agent 4):** Synthesizes differential diagnosis, generates report, and outputs explainable evidence traces (combining Clinical Reasoning, Report Generation, and Explainability).
