# 10 Scientific Evaluation Metrics Specification

## 1. Clinical Correctness Metrics
- **Top-1 Accuracy (%):** Percentage of queries where the primary diagnostic recommendation matches the ground-truth key.
- **Differential Diagnosis Recall@3 (DDx-Recall@3):** Proportion of true target conditions included within the top-3 differential recommendations.
- **Clinical F1-Score:** Harmonic mean of diagnostic precision and recall across multi-label clinical cases.

## 2. Retrieval & Grounding Metrics
- **Precision@k & Recall@k (k=5, 10):** Proportion of retrieved passages containing ground-truth PubMed / Guideline citations.
- **Faithfulness Score (RAG-Checker):** Proportion of atomic generated claims directly supported by retrieved passage context.
- **Citation Precision & Recall:** Accuracy of linked PMC / Guideline citation tags.

## 3. Safety & Hallucination Metrics
- **Hallucination Rate (%):** Percentage of outputs containing unsupported medical assertions.
- **Contradiction Detection Rate (%):** Accuracy of detecting conflicting clinical guidelines.
- **Safe Abstention F1-Score:** Precision and recall of abstaining/flagging low-confidence or contradictory queries.

## 4. Confidence Calibration Metrics
- **Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Brier Score:** Mean squared error between confidence probability and actual binary correctness.

## 5. Efficiency & Cost Metrics
- **Token Consumption / Query:** Total prompt + completion tokens consumed per query.
- **Latency (seconds):** End-to-end response time.
- **Agent Activations:** Average count of active agent nodes invoked per query.
