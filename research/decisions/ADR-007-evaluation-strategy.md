# ADR-007: Baseline Comparison & Ablation Evaluation Strategy

## Context
Demonstrating the research value of ACMR requires comparative evaluation against 7 established baseline paradigms (B1–B7) and 6 ablation variants (A–F).

## Decision
Implement an automated, reproducible experiment runner (`src/evaluation/test_runner.py`) executing unified system invocations across:
- **Baselines:** B1 (LLM-only), B2 (Standard Vector RAG), B3 (Medical RAG), B4 (CRAG), B5 (Adaptive RAG), B6 (SEMA-RAG), B7 (Fixed 4-Agent Pipeline).
- **Proposed:** MedAgentX (ACMR).
- **Ablation Matrix:**
  - Variant A: Full MedAgentX
  - Variant B: w/o ACMR (Fixed 4-Agent Pass)
  - Variant C: w/o Evidence Verification Agent
  - Variant D: w/o Knowledge Graph (Vector Search Only)
  - Variant E: w/o Reflection Agent
  - Variant F: w/o Mathematical Composite Confidence

## Status
**ACCEPTED** (Date: August 13, 2026)
