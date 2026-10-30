# 10 Evaluation Framework & Experimental Protocol

## 1. Overview
The MedAgentX evaluation framework evaluates system performance across 6 dimensions:
1. Clinical Correctness & Differential Diagnosis
2. Evidence Retrieval & Grounding Quality
3. Verification & Hallucination Mitigation
4. Confidence Calibration & Abstention Safety
5. Computational & Token Efficiency
6. Explainability & Clinician Usability

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MedAgentX Test Runner                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│  Accuracy    │             │ Grounding &  │             │  Efficiency  │
│  Evaluation  │             │ Verification │             │  & Cost      │
│(Top-1, DDx-3)│             │(Faithfulness)│             │(Tokens, Secs)│
└──────────────┘             └──────────────┘             └──────────────┘
```

---

## 2. Experimental Benchmark Suite
- **Primary Benchmark:** MedQA USMLE (1,273 questions) for diagnostic correctness.
- **Secondary Benchmark:** PubMedQA (500 expert pairs) for evidence grounding.
- **Third Benchmark:** Synthetic Multi-hop Clinical Cases (100 cases with contradictory guidelines) to test ACMR abstention safety.
