# Hypothesis Traceability Matrix: MedAgentX

## Overview
This matrix maps every scientific research hypothesis (H1–H5) to its implementing component, experimental protocol, target metrics, comparison baselines, and precise falsification conditions.

---

## Traceability Mapping Table

| Hypothesis ID | Statement | System Component | Experimental Protocol | Target Metrics | Comparison Baselines | Falsification Condition |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H1: Clinical Accuracy** | Adaptive confidence-guided orchestration improves evidence-grounded clinical answer quality compared to standard RAG and fixed pipelines. | ACMR Orchestrator + Agent 4 Reasoner | Run MedQA test set (1,273 queries) across systems. | Top-1 Accuracy, DDx Recall@3, Macro-F1 | B1 (LLM-only), B2 (Standard RAG), B7 (Fixed 4-Agent Pipeline) | H1 is rejected if ACMR accuracy is not statistically superior to B2 or B7 ($p \ge 0.05$ via paired t-test). |
| **H2: Verification Grounding** | Evidence verification reduces unsupported clinical claims and hallucinated assertions by $\ge 40\%$. | Agent 3 Evidence Verifier & Critic | Evaluate claim attribution on PubMedQA dataset. | RAG-Checker Faithfulness, Hallucination Rate (%), Citation Precision | Ablation Variant C (MedAgentX w/o Verifier Agent) | H2 is rejected if reduction in unsupported claims is $< 20\%$ or statistically insignificant ($p \ge 0.05$). |
| **H3: Hybrid Synergy** | Combining dense vector search (Qdrant) with Knowledge Graph traversal (Neo4j) improves multi-hop evidence retrieval recall over vector search alone. | Agent 2 Hybrid Retrieval Engine | Multi-hop clinical case queries requiring disease-symptom-drug paths. | Recall@5, Recall@10, nDCG@10, Multi-hop Coverage | Vector-only Retrieval (Qdrant), Graph-only Retrieval (Neo4j) | H3 is rejected if Hybrid Retrieval Recall@5 does not exceed Vector-only search by a statistically significant margin ($p < 0.05$). |
| **H4: Cost & Efficiency** | ACMR dynamic routing achieves $\ge 30\%$ reduction in token consumption and latency while preserving $\ge 95\%$ of full pipeline diagnostic accuracy. | ACMR Router + Early Stopping Policy | Measure token count, API calls, and latency per query across benchmark runs. | Prompt/Completion Tokens, Latency (seconds), Active Agent Count | Fixed 4-Agent Sequential Pipeline (B7), SEMA-RAG (B6) | H4 is rejected if token savings are $< 15\%$ or if diagnostic accuracy drops by $> 5\%$ relative to full pipeline. |
| **H5: Confidence Calibration** | Composite evidence confidence state ($C_{total}$) improves calibration (reduces ECE) and enables safe abstention on unanswerable queries compared to LLM self-reports. | ACMR Composite Confidence Calculator ($C_{total}$) | Evaluate confidence score calibration on answerable vs unanswerable/contradictory queries. | Expected Calibration Error (ECE), Brier Score, Abstention F1-score | LLM Verbalized Confidence ("rate 0-100%"), Ablation Variant F | H5 is rejected if $C_{total}$ yields higher ECE than self-reported confidence or fails to abstain on $> 30\%$ of contradictory queries. |
