# State of the Art: Agentic RAG

## 1. Executive Summary
Agentic RAG transforms static retrieval into an autonomous tool-calling workflow where agents plan queries, inspect context, critique relevance, re-retrieve missing information, and synthesize evidence-backed conclusions.

## 2. Key Frameworks Comparison

| System | Agent Architecture | Routing Mechanism | Domain | Target Benchmarks |
| :--- | :--- | :--- | :--- | :--- |
| **Self-RAG (2024)** | Single LLM with Special Tokens | Logprob of reflection tokens | Open Domain | PopQA, BioASQ, ARC |
| **DyLAN (2024)** | Dynamic Agent Network | Importance score pruning | General Reasoning | MMLU, GSM8K |
| **SEMA-RAG (2026)** | Decoupled 3-Agent Triplet | Fixed 3-agent graph | Healthcare QA | MedQA, BioASQ |
| **MedCoAct (2025)** | Collaborative Multi-Agent | Confidence feedback loops | Clinical Cases | Case Reports |
| **DeepRare (2025)** | Triage + Diagnostic Agents | Confidence triage | Rare Diseases | PubCaseFinder |
| **MedAgentX** | Adaptive 4-Core Agent Framework (ACMR) | Composite Mathematical Confidence Score | Clinical Decision Support | MedQA, MedMCQA, PubMedQA |

## 3. SOTA Gaps Identified
Current agentic RAG frameworks suffer from two major trade-offs:
1. **Fixed Triplet/Multi-Agent Overheads:** Systems like SEMA-RAG execute all agent passes for every query, incurring redundant latency and token costs.
2. **Subjective Self-Reported Confidence:** Systems like MedCoAct rely on uncalibrated LLM prompts to judge confidence, leading to overconfident hallucinations on edge cases.
