# 06 Competitor Analysis: MedAgentX vs. Closest Academic & Industry Systems

## 1. Competitive Overview

To defend the research novelty of MedAgentX for IEEE publication, we must compare it against the closest existing systems in literature (2024–2026).

```
                            [RAG Paradigm]
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
 [Static / Fixed Systems]                       [Dynamic / Adaptive Systems]
 ├── SEMA-RAG (ACL 2026)                        ├── MedCoAct (MDPI 2025)
 ├── Med-Gemini (Google 2024)                   ├── DeepRare (2025)
 └── Fixed 8-Agent MedAgentX (Initial)           └── MedAgentX (ACMR Refined)
```

---

## 2. Granular Competitor Matrix

| Feature / Dimension | SEMA-RAG (ACL 2026) | MedCoAct (2025) | DeepRare (2025) | ArbGraph (2026) | Med-Gemini (2024) | **MedAgentX (Proposed)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Domain Scope** | General Medical QA | Clinical Cases | Rare Genetic Diseases | Long-Form RAG | Multimodal Medical AI | **Clinical Decision Support (CDSS)** |
| **Multi-Agent Setup** | 3 Fixed Agents (Interpreter, Explorer, Arbiter) | Collaborative Agents | Triage + Diagnostic Agents | Arbiter Agents | Unified Model + API Tools | **4 Specialized Core Agents (ACMR)** |
| **Routing Policy** | Fixed 3-agent pass | Confidence feedback loops | Confidence triage | Fixed Graph Arbiter | Model internal routing | **Adaptive Confidence-Guided Routing (ACMR)** |
| **Confidence Scoring** | Output score | Verbalized LLM score | Triage threshold | Graph conflict score | Logprob uncertainty | **Composite Mathematical Formula ($C_{total}$)** |
| **Retrieval Strategy** | Dense Vector | Dense Vector | Phenotype Vector | Graph + Vector | Web Search + Vector | **Hybrid Vector (Qdrant) + Knowledge Graph (Neo4j)** |
| **Evidence Verification** | Arbiter Agent | Verifier Agent | None | Graph Contradiction | Web search matching | **Dedicated Evidence Verification Agent** |
| **Cost / Token Control** | None (Runs all agents) | Moderate | High (Rare disease prune) | None | High (Proprietary) | **Explicit Token & Latency Cost-Aware Routing** |
| **Abstention Safety** | Low | Low | Moderate | Low | Moderate | **High (Explicit Abstention Threshold)** |

---

## 3. In-Depth Analysis of Closest Competitors

### 3.1 SEMA-RAG (Findings of ACL 2026)
- **What it does:** Decouples medical retrieval into three agents: Interpreter (expands queries), Explorer (retrieves passages), and Arbiter (selects final answers).
- **Where it fails:** Runs all three agents for every query regardless of query difficulty or initial evidence sufficiency. Lacks Knowledge Graph integration.
- **MedAgentX Differentiation:** MedAgentX introduces dynamic early-stopping and selective agent activation (ACMR), skipping reflection/re-retrieval when initial evidence is sufficient, and integrating Knowledge Graph path traversal.

### 3.2 MedCoAct (MDPI Healthcare 2025)
- **What it does:** Uses confidence-aware collaboration between medical agents to improve clinical case diagnostic completeness.
- **Where it fails:** Relies on subjective verbalized confidence prompts ("Rate your confidence 0-100%"), which are notoriously miscalibrated in LLMs. Does not use Knowledge Graphs.
- **MedAgentX Differentiation:** MedAgentX formalizes an objective mathematical composite confidence score ($C_{total}$) incorporating retrieval similarity, verifier consensus, and claim contradiction rates.

### 3.3 DeepRare (arXiv 2025)
- **What it does:** Employs confidence triage to route rare disease diagnostic cases.
- **Where it fails:** Narrowly focused on rare genetic diseases using HPO ontologies; cannot handle general clinical differential diagnosis or multi-symptom outpatient presentations.
- **MedAgentX Differentiation:** MedAgentX covers broad clinical decision support across general medicine, internal medicine, and emergency triage.

---

## 4. Threat Summary & Strategic Positioning

> **Strategic Finding:** Claiming "Multi-Agent Medical RAG" alone is **NO LONGER NOVEL** due to SEMA-RAG (2026) and MedCoAct (2025).  
> MedAgentX's primary novelty must be positioned squarely on:
> 1. **Mathematically Formulated Composite Confidence Calibration ($C_{total}$).**
> 2. **Cost-Aware Dynamic Agent Routing (ACMR Policy).**
> 3. **Synergistic Hybrid Vector + Knowledge Graph Verification.**
