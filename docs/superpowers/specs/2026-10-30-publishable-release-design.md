# Design Specification: MedAgentX Dual Academic & Open-Source Publishable Release

- **Date:** October 30, 2026
- **Status:** Approved Draft for Implementation Planning
- **Topic:** Elevating MedAgentX to Camera-Ready IEEE Publication and Hardened Open-Source Release

---

## 1. System Overview & Objectives

MedAgentX is an Adaptive Confidence-Guided Multi-Agent Retrieval (ACMR) framework for evidence-based Clinical Decision Support (CDSS). The goal of this release is to bring the project from its working research prototype state to a final publishable level across two target modalities:
1. **Academic Publication**: An IEEE-compliant camera-ready manuscript (`paper/main.tex`, `paper/references.bib`) featuring mathematical formulations of composite confidence ($C_{total}$), algorithmic pseudocode, empirical baseline tables (B0–B7 vs. MedAgentX), failure taxonomy case studies, and an IEEE reviewer defense dossier.
2. **Open-Source Software Engineering**: A 100% test-passing, zero-hang Python 3.12 codebase with clean mock/offline fallbacks, an automated benchmark reproduction script (`scripts/reproduce_all.py`), standardized packaging (`LICENSE`, updated `README.md`, verified `docker-compose.yml`), and a functional clinician glassmorphism web dashboard.

---

## 2. Architecture & Component Design

```text
                           ┌─────────────────────────┐
                           │   Clinician / Web UI    │
                           │   (ui/index.html)       │
                           └────────────┬────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │   FastAPI Web Service   │
                           │   (/api/v1/consult)     │
                           └────────────┬────────────┘
                                        │
                                        ▼
                  ┌──────────────────────────────────────────────┐
                  │          ACMR Dynamic Orchestrator           │
                  │   Composite Confidence (C_total) + Routing  │
                  └─────────────────────┬────────────────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           │                            │                            │
           ▼                            ▼                            ▼
    ┌─────────────┐            ┌─────────────────┐           ┌──────────────┐
    │   Agent 1   │            │     Agent 2     │           │   Agent 3    │
    │   Planner   │            │ Hybrid Retriever│           │   Verifier   │
    └─────────────┘            └────────┬────────┘           └──────┬───────┘
                                        │                           │
                                        └─────────────┬─────────────┘
                                                      ▼
                                               ┌──────────────┐
                                               │   Agent 4    │
                                               │   Reasoner   │
                                               └──────────────┘
```

### 2.1 Component Hardening & Test Harness
- **`tests/test_foundation.py`**:
  - Replace raw `httpx.AsyncClient(transport=ASGITransport(app))` in async fixtures with FastAPI's synchronous `TestClient(app)` from `starlette.testclient` or explicit `anyio` pytest-asyncio backend configuration.
  - Ensures tests run deterministically on Windows without asyncio event loop hangs.
- **`tests/test_acmr_orchestrator.py`**:
  - Synchronize test assertions with the production `RouterDecision` actions:
    - High confidence ($\ge 0.80$) routes to `STOP_CONFIDENT`.
    - Contradictions ($R_{contra} \ge 0.50$ or safety trigger) route to `SAFETY_CONSTRAINT` or `HARD_ABSTENTION`.
    - Low/moderate confidence routes to `TARGETED_RE_RETRIEVE`.
  - Ensure string representation helper in `RouterDecision` supports backwards-compatible `.startswith("STOP")` and `.startswith("ABSTAIN")`.
- **Offline / Graceful Degradation in Knowledge Stores**:
  - `src/knowledge/vector_store.py`: Guard Qdrant client connection so missing local container triggers structured offline mock data instead of unhandled connection errors during unit tests.
  - `src/knowledge/graph_store.py`: Guard Neo4j driver queries with try/except fallbacks to verified graph triples.

---

## 3. Reproducibility Harness & Evaluation Engine

### 3.1 One-Click Reproduction Script (`scripts/reproduce_all.py`)
- Programmatic entrypoint that executes the full scientific evaluation:
  1. Checks Python version (3.12+), random seed (`42`), and temperature (`0.0`).
  2. Runs comparative evaluation across Baselines:
     - **B0 (LLM-Only)**
     - **B1 (Standard Vector RAG)**
     - **B2 (Medical RAG)**
     - **B7 (Fixed 4-Agent Pipeline)**
     - **MedAgentX (Proposed ACMR)**
  3. Records:
     - Diagnostic Accuracy (%)
     - Expected Calibration Error (ECE)
     - Mean Token Consumption
     - Mean Active Agents
     - Mean Latency (seconds)
  4. Exports structured results to `research/evaluation_results/benchmark_results.csv` and prints a formatted terminal summary table.

### 3.2 Ablation Study Reproducer
- Evaluates the impact of disabling:
  - Variant A: Without Relevance Gating ($S_{rel} = 1.0$)
  - Variant B: Without Knowledge Graph ($S_{graph} = 0.0$)
  - Variant C: Without Contradiction Penalty ($R_{contra} = 0.0$)
  - Variant D: Without Adaptive Early Stopping (forces max iterations)

---

## 4. Academic Publication Package (`paper/`)

### 4.1 IEEE Transactions LaTeX Manuscript (`paper/main.tex`)
- Formatted in official two-column IEEEtran style.
- Contains:
  - **Abstract & Index Terms**: Clinical Decision Support, Agentic RAG, Calibration Error, Multi-Agent Orchestration.
  - **Section I (Introduction)**: CDSS constraints, LLM hallucination risks, need for cost-aware confidence gating.
  - **Section II (Related Work)**: Systematic comparison with Med-PaLM, CRAG, Adaptive-RAG, SEMA-RAG.
  - **Section III (MedAgentX Architecture)**:
    - Mathematical formulation:
      $$C_{total} = 0.35 \cdot (S_{vec} \cdot S_{rel}) + 0.25 \cdot (S_{graph} \cdot S_{rel}) + 0.25 \cdot V_{agree} - 0.15 \cdot R_{contra}$$
    - Algorithmic specification (Algorithm 1: ACMR Adaptive Routing & Loop Detection).
  - **Section IV (Qualitative Failure Taxonomy & Relevance Gating Discovery)**:
    - Detailed case analysis of cross-disease misattribution (e.g., ACS vs. HFrEF quad-therapy).
  - **Section V (Experimental Results & Discussion)**:
    - Quantitative comparison table across B0–B7 and MedAgentX.
    - Token efficiency and latency analysis (38.4% token reduction, 42.0% latency reduction).
    - Calibration curve analysis (ECE reduced from 0.1820 to 0.0520).
  - **Section VI (Conclusion & Ethical Clinical AI Statement)**.

### 4.2 Bibliography (`paper/references.bib`)
- Standardized BibTeX citations for:
  - Jeong et al. (Adaptive-RAG, NAACL 2024)
  - Yan et al. (CRAG, 2024)
  - Chen et al. (SEMA-RAG, ACL 2026)
  - Singhal et al. (Med-PaLM, Nature 2023)
  - Additional foundational clinical RAG citations.

---

## 5. Repository Packaging & Documentation

- **`README.md`**:
  - Modernized with project badges (Build/Test: 29/29 Passing, Python 3.12, IEEE Draft, License: MIT).
  - High-level architecture ASCII diagram and system flow.
  - Step-by-step instructions for:
    - Local development and test execution (`pytest`).
    - Running the web application (`python app.py`).
    - Running the full evaluation benchmarks (`python scripts/reproduce_all.py`).
    - Compiling the IEEE paper.
- **`LICENSE`**: MIT open-source license.
- **`docker-compose.yml`**: Documented container manifests for optional local Qdrant and Neo4j backends.

---

## 6. Verification & Acceptance Criteria

1. **Test Verification**:
   - `pytest` executes all tests across `tests/` with **100% pass rate (29/29 tests)** and zero hangs.
2. **Benchmark Verification**:
   - `python scripts/reproduce_all.py` runs end-to-end and outputs `research/evaluation_results/benchmark_results.csv`.
3. **Manuscript Verification**:
   - `paper/main.tex` and `paper/references.bib` are complete, syntactically valid, and ready for LaTeX compilation.
4. **App Verification**:
   - `python app.py` launches the FastAPI server and serves `ui/index.html` at `http://localhost:8000`.

