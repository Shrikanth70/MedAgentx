# State of the Art: Clinical Reasoning & Differential Diagnosis

## 1. Domain Summary
Clinical reasoning involves synthesizing multi-modal clinical signals (history, physical findings, lab results, imaging) into differential diagnoses, ordering target investigations, and recommending evidence-based treatment plans.

## 2. Key Frameworks Comparison

| System | Clinical Reasoning Methodology | External Evidence Grounding | Diagnostic Output Structure |
| :--- | :--- | :--- | :--- |
| **AMIE (Nature 2025)** | Simulated self-play OSCE dialogue | None (Parametric only) | Natural language dialogue |
| **MECR-RAG (2025)** | Case + guideline dual reasoning | Guidelines | Triage classification |
| **CLI-RAG (2025)** | Contextual EHR note mapping | EHR notes | Structured SOAP report |
| **MedAgentX** | Multi-step reasoning + KG path traversal | PubMed + Guidelines + KG | Structured Clinical Report with DDx, Investigations & Citations |
