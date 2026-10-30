# 08 Dataset Inventory & Evaluation Suite

## 1. Overview
Evaluating MedAgentX requires public, standardized, peer-reviewed medical QA and clinical reasoning datasets.

## 2. Granular Inventory of Candidate Datasets

### A. MedQA (USMLE)
- **URL:** https://github.com/jindingxin/MedQA
- **License:** MIT License / Open Research
- **Access:** Publicly accessible
- **Size:** 12,723 multiple-choice questions (USMLE Step 1, Step 2, Step 3)
- **Format:** JSON lines (Question, Options A-D, Answer Key, Explanation)
- **Task:** Multi-step clinical reasoning and medical knowledge retrieval.
- **Clinical Domain:** General Medicine, Internal Medicine, Pathology, Pharmacology.
- **Suitability:** **PRIMARY BENCHMARK** for diagnostic accuracy and RAG grounding.
- **Leakage Risk:** Medium (frequently used in LLM pre-training; mitigation via subset evaluation and retrieval grounding verification).

### B. MedMCQA
- **URL:** https://github.com/medmcqa/medmcqa
- **License:** Apache 2.0
- **Access:** Publicly accessible (HuggingFace Datasets)
- **Size:** 194,000+ entrance exam questions (AIIMS / NEET PG)
- **Format:** JSON lines
- **Task:** Medical subject QA across 24 clinical specialties.
- **Suitability:** High-volume evaluation of clinical domain coverage.

### C. PubMedQA
- **URL:** https://pubmedqa.github.io/
- **License:** MIT License
- **Access:** Publicly accessible
- **Size:** 1,000 expert-labeled QA pairs (1000 PQA-L, 61.2k unlabelled PQA-U)
- **Format:** JSON (Abstract context, Question, Yes/No/Maybe answer, Long answer)
- **Task:** Evidence-based biomedical text reasoning.
- **Suitability:** Ideal for evaluating Evidence Verification and Citation Attribution.

### D. MMLU Medical Subsets
- **Subsets:** Clinical Knowledge, Medical Genetics, Anatomy, Professional Medicine, College Biology.
- **Size:** ~3,000 questions
- **Task:** Domain-specific multiple choice medical assessment.
