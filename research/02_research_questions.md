# 02 Research Questions & Formal Hypotheses

## 1. Central Research Question

> **Central Research Question (CRQ):**  
> *Can adaptive, confidence-guided orchestration of heterogeneous retrieval and clinical reasoning agents improve diagnostic correctness, evidence grounding, and confidence calibration while reducing computational and retrieval overhead compared to fixed multi-agent architectures and static RAG pipelines?*

---

## 2. Granular Research Questions (RQs)

### RQ1: Diagnostic Correctness & Reasoning Quality
**Does adaptive confidence-guided orchestration (ACMR) improve multi-step clinical reasoning and differential diagnosis accuracy compared to standalone LLMs, static RAG, and fixed multi-agent pipelines?**
- *Sub-question 1.1:* How does performance scale with clinical query complexity (simple single-symptom vs. complex multi-symptom presentations)?
- *Sub-question 1.2:* Does dynamic query decomposition by a Planner Agent yield measurable accuracy gains on clinical benchmark datasets (MedQA, MedMCQA)?

### RQ2: Evidence Verification & Hallucination Mitigation
**To what extent does an explicit Evidence Verification agent reduce unsupported clinical claims, hallucinated medical assertions, and citation inaccuracies compared to unverified RAG outputs?**
- *Sub-question 2.1:* How effectively can a verification agent detect and resolve conflicting evidence from disparate clinical sources (e.g., guidelines vs. PubMed case reports)?
- *Sub-question 2.2:* What is the impact of automated evidence verification on citation precision and recall?

### RQ3: Knowledge Graph + Vector Hybrid Retrieval
**Does combining dense vector retrieval with structured Knowledge Graph (KG) traversal improve multi-hop evidence retrieval and diagnostic coverage compared to dense vector retrieval alone?**
- *Sub-question 3.1:* How does KG entity-relation expansion impact retrieval recall@k for complex disease-symptom-drug queries?
- *Sub-question 3.2:* What is the optimal balance between vector semantic search and KG graph traversal in clinical contexts?

### RQ4: Computational Efficiency & Cost Control
**Can confidence-guided adaptive agent selection reduce token consumption, API calls, latency, and agent activations without degrading clinical reasoning performance compared to fixed multi-agent pipelines?**
- *Sub-question 4.1:* What percentage of clinical queries can be resolved using early-stopping criteria without invoking full multi-agent reflection loops?
- *Sub-question 4.2:* What is the exact trade-off curve between token budget and diagnostic F1-score?

### RQ5: Confidence Calibration & Abstention Safety
**Can a mathematical confidence state model reliably distinguish between evidence-sufficient and evidence-insufficient clinical queries, enabling safe abstention when appropriate?**
- *Sub-question 5.1:* How does MedAgentX perform on Expected Calibration Error (ECE) and Brier Score compared to uncalibrated LLM confidence self-reports?
- *Sub-question 5.2:* Can the system effectively refrain from generating dangerous medical recommendations when retrieved evidence is contradictory or absent?

---

## 3. Formal Scientific Hypotheses (H1–H5)

### Hypothesis H1: Adaptive Orchestration Superiority
- **Formal Statement:** *An adaptive confidence-guided multi-agent architecture achieves higher diagnostic accuracy (Top-1 and Top-3 accuracy on MedQA/MedMCQA) than standard single-pass RAG and fixed-sequence multi-agent pipelines.*
- **Independent Variable:** Orchestration Strategy (Static Single-Pass RAG vs. Fixed Multi-Agent Pipeline vs. Adaptive ACMR).
- **Dependent Variable:** Diagnostic Accuracy (%), Differential Diagnosis Recall.
- **Baseline:** Fixed 8-agent pipeline and Standard Vector RAG.
- **Metric:** Accuracy, Macro-F1, Differential Diagnosis Recall@3.
- **Falsification Condition:** H1 is rejected if ACMR accuracy is statistically indistinguishable from (or worse than) fixed multi-agent pipelines ($p > 0.05$ via paired t-test / McNemar's test).

### Hypothesis H2: Evidence Grounding & Verification Effect
- **Formal Statement:** *Integrating an explicit Evidence Verification Agent reduces unsupported clinical claims and hallucination rates by at least 40% relative to unverified multi-agent generation.*
- **Independent Variable:** Presence/Absence of Evidence Verification Agent in the pipeline.
- **Dependent Variable:** Hallucination Rate (%), Unsupported Claim Rate (%), Citation Precision.
- **Baseline:** MedAgentX pipeline without Verification Agent (Ablation Variant C).
- **Metric:** Factual Consistency Score (RAG-Checker / MedHALT), Citation Precision@k.
- **Falsification Condition:** H2 is rejected if the reduction in unsupported claims is $< 20\%$ or statistically insignificant ($p \ge 0.05$).

### Hypothesis H3: Hybrid Retrieval Synergy
- **Formal Statement:** *Hybrid retrieval combining vector semantic search (Qdrant) and Knowledge Graph path traversal (Neo4j) achieves significantly higher multi-hop recall@k ($k=5, 10$) on complex clinical queries than vector-only or graph-only retrieval.*
- **Independent Variable:** Retrieval Subsystem (Vector-only vs. Graph-only vs. Hybrid Vector+KG).
- **Dependent Variable:** Retrieval Recall@k, Mean Reciprocal Rank (MRR), Evidence Coverage.
- **Baseline:** Dense Passage Retrieval (DPR) with BioClinical-BERT / MedCPT embeddings.
- **Metric:** Recall@5, Recall@10, nDCG@10, Multi-hop Evidence Coverage.
- **Falsification Condition:** H3 is rejected if Hybrid Retrieval does not achieve a statistically significant improvement in Recall@5 ($p < 0.05$) over dense vector search.

### Hypothesis H4: Efficiency-Quality Trade-off Optimization
- **Formal Statement:** *Adaptive confidence-guided agent routing reduces overall token consumption and execution latency by at least 30% compared to a fixed multi-agent pipeline while maintaining $\ge 95\%$ of the full multi-agent pipeline's diagnostic accuracy.*
- **Independent Variable:** Routing Policy (Fixed All-Agent Activation vs. Adaptive ACMR Routing).
- **Dependent Variable:** Total Tokens per Query, Execution Latency (seconds), Total Agent Activations, Diagnostic Accuracy (%).
- **Baseline:** Fixed 8-agent sequential execution.
- **Metric:** Tokens/Query, Latency (s), Accuracy Preservation Ratio ($\text{Acc}_{\text{ACMR}} / \text{Acc}_{\text{Fixed}}$).
- **Falsification Condition:** H4 is rejected if token savings are $< 15\%$ or if diagnostic accuracy drops by $> 5\%$ relative to the full fixed pipeline.

### Hypothesis H5: Confidence Calibration & Safe Abstention
- **Formal Statement:** *A composite multi-factor confidence model improves confidence calibration (reducing Expected Calibration Error) and enables reliable abstention/clarification requests on low-evidence queries compared to LLM verbalized confidence.*
- **Independent Variable:** Confidence Calculation Method (LLM Self-Reported Verbalized Confidence vs. ACMR Composite Mathematical Confidence Score).
- **Dependent Variable:** Expected Calibration Error (ECE), Brier Score, Abstention Precision/Recall on Out-of-Domain / Insufficient-Evidence Queries.
- **Baseline:** Standard LLM self-confidence prompt ("Rate confidence 0–100%").
- **Metric:** ECE, Brier Score, Abstention F1-score.
- **Falsification Condition:** H5 is rejected if the composite confidence model yields higher ECE than self-reported confidence or fails to abstain on $> 30\%$ of unanswerable queries.
