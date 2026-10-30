# 10 Ablation Experimentation Plan

## 1. Objective
Systematically strip individual components from MedAgentX to isolate and quantify the exact contribution of each architectural module.

## 2. Ablation Variants Matrix

| Variant ID | Architecture Configuration | Hypotheses Tested | Expected Impact |
| :--- | :--- | :--- | :--- |
| **Ablation A** | Full MedAgentX (ACMR + Hybrid KG + Verification + Reflection) | All (H1–H5) | Peak Performance (SOTA) |
| **Ablation B** | MedAgentX **w/o ACMR** (Fixed 4-Agent Sequential Pass) | H1, H4 | Latency & Token consumption spike (+40%) |
| **Ablation C** | MedAgentX **w/o Verification Agent** | H2 | Hallucination rate increases (+30%) |
| **Ablation D** | MedAgentX **w/o Knowledge Graph** (Vector Search Only) | H3 | Multi-hop recall@5 drops (-15%) |
| **Ablation E** | MedAgentX **w/o Reflection Agent** | H1 | Complex case accuracy drops (-8%) |
| **Ablation F** | MedAgentX **w/o Mathematical Confidence** (Verbalized Confidence) | H5 | Expected Calibration Error (ECE) degrades |
