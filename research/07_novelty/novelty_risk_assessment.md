# 07 Novelty Risk Assessment

## Risk Level Classification: 🟡 **MEDIUM RISK (VIABLE WITH REFINED CONTRIBUTION)**

### Rationale
- **High Overlap Area:** Generic "Multi-Agent Medical RAG" and "LLM Clinical Decision Support" are heavily published in 2024–2026 (SEMA-RAG in ACL 2026, MedCoAct in 2025, Med-Gemini in 2024). Submitting MedAgentX claiming basic multi-agent architecture would result in **HIGH REJECTION RISK** by IEEE reviewers.
- **Mitigation Strategy (Pivot Focus):** By refining MedAgentX to focus on **Adaptive Confidence-Guided Routing (ACMR), Mathematical Calibration ($C_{total}$), and Cost-Aware Early Stopping**, the project transitions into a **LOW TO MEDIUM RISK, HIGH IEEE READINESS** status.

---

## Action Plan to Maintain IEEE Defensibility

| Risk Item | Identified Threat | Mitigation Strategy |
| :--- | :--- | :--- |
| **Agent Redundancy** | 8 fixed agents look over-engineered | Streamline architecture to 4 specialized core agent modules + ACMR Orchestrator. |
| **Confidence Miscalibration** | LLMs self-report inaccurate confidence | Use mathematical composite formula $C_{total}$ combining vector sim, KG paths, and verifier agreement. |
| **Competing SEMA-RAG** | Decoupled medical RAG published in ACL 2026 | Highlight ACMR dynamic early-stopping and Hybrid Knowledge Graph traversal (absent in SEMA-RAG). |
| **High Token Costs** | Full agent loops are slow and expensive | Prove ACMR achieves $\ge 35\%$ token savings via early-stopping evaluation. |
