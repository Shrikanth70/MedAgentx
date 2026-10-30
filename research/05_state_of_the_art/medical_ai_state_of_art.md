# State of the Art: Medical AI & Foundation Models

## 1. Domain Overview
Medical AI has evolved rapidly from task-specific classifiers to general-purpose medical LLMs (Med-PaLM 2, Med-Gemini, AMIE, BioMistral, OpenBioLLM) capable of passing clinical licensing examinations (USMLE) and engaging in diagnostic dialogue.

## 2. Benchmark Performance Landscape (MedQA USMLE Accuracy)

```
Med-Gemini (2024): 91.1%  [Closed API / Proprietary]
Med-PaLM 2 (2025): 86.5%  [Closed API / Proprietary]
OpenBioLLM-70B (2024): 84.3%  [Open Weights]
GPT-4o + RAG (2024): 82.1%  [Commercial API]
BioMistral-7B (2024): 54.8%  [Open Weights 7B]
Standard LLaMA-3-8B (2024): 51.2%  [Open Base]
```

## 3. Primary Limitations of Current Medical Foundation Models
1. **Lack of Verifiable Grounding:** Foundation models generate answers from parametric memory without explicit link to trusted clinical guidelines or PubMed sources.
2. **Knowledge Cutoffs:** Cannot account for recent therapeutic approvals or updated clinical practice guidelines published post-training.
3. **Overconfidence & Miscalibration:** Models often express high verbal confidence even when making incorrect diagnostic assertions.
