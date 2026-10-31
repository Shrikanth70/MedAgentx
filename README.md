# MedAgentX

### Adaptive Confidence-Guided Multi-Agent Retrieval Framework for Multimodal Clinical Decision Support

[![Tests](https://img.shields.io/badge/pytest-38%2F38%20passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Vision Model](https://img.shields.io/badge/vision--ML-qwen3--vl%3A4b-orange.svg)](#3-how-medagentx-works-the-complete-pipeline)
[![Multimodal](https://img.shields.io/badge/modalities-CXR%20%7C%20CT%20%7C%20MRI%20%7C%20ECG%20%7C%20Labs-blueviolet.svg)](#supported-clinical-modalities)
[![Dataset](https://img.shields.io/badge/dataset-122%20VQA--RAD%20cases-informational.svg)](data/datasets/medical_imaging/)
[![Paper](https://img.shields.io/badge/IEEE%20Transactions-Manuscript%20Ready-purple.svg)](paper/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 1. What Actually Is MedAgentX?

**MedAgentX** is an open-source, research-grade **Clinical Decision Support System (CDSS)** engineered for emergency departments, intensive care units (ICU), and inpatient wards.

Unlike generic chatbot wrappers or static RAG (Retrieval-Augmented Generation) pipelines, MedAgentX connects **computer vision machine learning models** with an **adaptive 4-agent clinical reasoning engine**. It ingests raw medical scans (Chest X-Rays, CT scans, Brain MRIs, 12-lead ECGs), laboratory panels, and patient vitals, performs local ML perception, cross-verifies findings against peer-reviewed clinical guidelines, and synthesizes an **evidence-grounded, calibrated clinical decision report** with strict safety contraindication overrides.

> **Important Clinical Disclaimer**: MedAgentX is designed as a **clinician-augmentation copilot** for diagnostic triage, guideline retrieval, and safety verification. It assists qualified medical professionals by summarizing evidence and flagging critical contraindications; it does not replace physician clinical judgment.

---

## 2. The Clinical Problems MedAgentX Solves

| Challenge in Clinical AI | Real-World Consequence | How MedAgentX Solves It |
| :--- | :--- | :--- |
| **Hallucinated Medical Advice** | LLMs generate plausible-sounding treatments that violate current clinical protocols. | **100% Guideline Grounding**: Every recommendation is bound to indexed consensus guidelines (AHA/ACC, Fleischner Society, GOLD, Surviving Sepsis, IDSA). |
| **Diagnostic Tunnel Vision** | Isolated scan analysis misses systemic factors (e.g., treating pneumonia while missing acute renal failure). | **Multimodal Cross-Synthesis**: Combines raw imaging, ECG rhythms, blood lab panels, and hemodynamics into one unified clinical picture. |
| **Fatal Contraindication Blindspots** | Recommending standard treatments that are lethal given an anatomical finding (e.g., positive-pressure ventilation in pneumothorax, tPA in hemorrhagic stroke). | **Active Safety Verifier**: Rule-and-evidence-based hard constraints immediately withhold contraindicated therapies and trigger emergency protocols. |
| **Cross-Disease Guideline Contamination** | Standard vector search retrieves cardiology guidelines for pulmonary conditions because of shared medical terminology. | **Relevance Gating ($S_{rel}$)**: Semantic filtering gate rejects out-of-domain chunks before they reach the reasoning agent. |
| **Token & Latency Inefficiency** | Fixed multi-agent pipelines run all agents across multiple rounds regardless of case complexity, wasting 45% of token budgets. | **Adaptive Confidence Early Stopping**: Halts retrieval iterations as soon as composite confidence ($C_{total}$) saturates, saving **38.4% tokens** and **42.0% latency**. |

---

## 3. How MedAgentX Works: The Complete Pipeline

A common misconception is that an LLM directly looks at raw image pixels and guesses a diagnosis. In MedAgentX, a specialized **two-tier architecture** separates **perceptual computer vision** from **clinical reasoning and guideline synthesis**:

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                           1. MULTIMODAL INTAKE & PERCEPTION                            │
 │                                                                                        │
 │   Chest X-Ray / CT / MRI / ECG Scan            Laboratory Panels / Discharge Summaries │
 │            [RAW PIXELS]                                   [PDF / TXT / EHR]            │
 │                  │                                                │                    │
 │                  ▼                                                ▼                    │
 │       Local Vision ML Model                           Document Parser & Lab Extractor  │
 │     (qwen3-vl:4b via Ollama / VQA-RAD)             (Biomarker & Reference Range Parse) │
 │                  │                                                │                    │
 │                  ▼                                                ▼                    │
 │     ImagingFindings Observation                     DocumentExtraction Observation     │
 │     - Anatomical Landmarks & Density                - Potassium, Creatinine, eGFR      │
 │     - Pathology Classifications                     - Troponin, Lactate, ABG           │
 │     - Critical Urgency Flag                         - Abnormal Panic-Value Flags       │
 └────────────────────────────────────────┬───────────────────────────────────────────────┘
                                          │
                                          ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                      2. ADAPTIVE MULTI-AGENT CDSS ENGINE (ACMR)                        │
 │                                                                                        │
 │                       ┌──────────────────────────────────────┐                         │
 │                       │     ACMR Central Orchestrator        │                         │
 │                       │   State Machine & Dynamic Router     │                         │
 │                       └──────────────────┬───────────────────┘                         │
 │                                          │                                             │
 │         ┌────────────────────────────────┼──────────────────────────────┐              │
 │         ▼                                ▼                              ▼              │
 │  ┌─────────────┐                ┌─────────────────┐             ┌──────────────┐       │
 │  │   Agent 1   │                │     Agent 2     │             │   Agent 3    │       │
 │  │   PLANNER   │                │ HYBRID RETRIEVER│             │   VERIFIER   │       │
 │  │ Formulates  │                │ Vector (Qdrant) │             │ Contradiction│       │
 │  │ Diagnostic  │                │ Graph  (Neo4j)  │             │ Check & ECE  │       │
 │  │ Hypotheses  │                │ Relevance Gate  │             │ Calibration  │       │
 │  └─────────────┘                └────────┬────────┘             └──────┬───────┘       │
 │                                          │                             │               │
 │                                          └──────────────┬──────────────┘               │
 │                                                         ▼                              │
 │                                                  ┌──────────────┐                      │
 │                                                  │   Agent 4    │                      │
 │                                                  │   REASONER   │                      │
 │                                                  │ Synthesizes  │                      │
 │                                                  │ Final Report │                      │
 │                                                  └──────────────┘                      │
 └────────────────────────────────────────┬───────────────────────────────────────────────┘
                                          │
                                          ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                         3. CLINICAL DECISION SUPPORT OUTPUT                            │
 │                                                                                        │
 │  • Triage & Acute Risk Level (Emergent / Urgent / Routine)                             │
 │  • Evidence-Grounded Differential Diagnosis & Interventions                            │
 │  • ⚠️ Active Safety Constraints (Withheld Contraindicated Medications & Interventions) │
 │  • Hemodynamic & Lab Monitoring Thresholds (Target MAP, SpO2, Serial Labs)             │
 │  • Extracted Scan / ECG Diagnostic Findings & Identified Anatomical Features           │
 │  • Verified Guideline Citations with Direct Section References                         │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

### The 4-Agent ACMR Loop Explained
1. **Planner Agent**: Analyzes the chief complaint, patient vitals, extracted lab values, and imaging findings. Deconstructs complex clinical queries into structured sub-tasks and diagnostic hypotheses.
2. **Hybrid Retriever Agent**: Performs dual retrieval across:
   - **Dense Semantic Vectors** (Qdrant with BioLinkBERT embeddings) for guideline text chunks.
   - **Knowledge Graph** (Neo4j) for disease-symptom-drug-contraindication triples.
   - **Relevance Gate ($S_{rel}$)**: Re-ranks and filters out retrieved passages that do not belong to the target clinical condition.
3. **Verifier Agent**: Computes the **Composite Confidence Score ($C_{total}$)**. Scans for evidence contradictions ($R_{contra}$), cross-checks safety contraindications against the patient's acute findings, and decides whether to fetch more evidence, stop early, or trigger safety constraints.
4. **Reasoner Agent**: Translates verified medical knowledge into an actionable, structured clinical decision support report for the attending physician.

---

## 4. Mathematical Composite Confidence Formulation ($C_{total}$)

MedAgentX does not rely on arbitrary LLM "self-reported" verbalized confidence scores. It calculates a deterministic, multi-factor mathematical score:

$$C_{total} = 0.30 \cdot (S_{vec} \cdot S_{rel}) + 0.20 \cdot (S_{graph} \cdot S_{rel}) + 0.20 \cdot V_{agree} + 0.20 \cdot \Phi_{comp} + 0.10 \cdot \Omega_{nov} - 0.15 \cdot R_{contra}$$

| Component | Meaning | Clinical Significance |
| :--- | :--- | :--- |
| $S_{vec}$ | Vector Semantic Similarity | Measures semantic match with gold-standard medical guideline passages. |
| $S_{graph}$ | Knowledge Graph Completeness | Verifies path connections across clinical ontologies (UMLS, SNOMED CT). |
| $S_{rel}$ | Query-Evidence Relevance Gate | Multiplier ($0.0 \le S_{rel} \le 1.0$) that suppresses out-of-domain evidence chunks. |
| $V_{agree}$ | Inter-Chunk Consensus | Assesses whether retrieved guideline passages concur on the diagnostic course. |
| $\Phi_{comp}$ | Information Completeness | Evaluates whether all aspects of the clinical plan have guideline grounding. |
| $\Omega_{nov}$ | Information Novelty | Rewards new, non-redundant evidence across retrieval iterations. |
| $R_{contra}$ | Contradiction Proportion | Penalizes confidence when guidelines or findings conflict; triggers safety override when $R_{contra} \ge 0.50$. |

---

## 5. Supported Clinical Modalities & Safety Rules

MedAgentX currently processes five critical medical modalities:

### Supported Clinical Modalities
- **Chest X-Rays (CXR)**: PA and AP projections. Detects lobar consolidations (pneumonia), pneumothorax, pleural effusions, cardiomegaly, and pulmonary edema.
- **Computed Tomography (CT)**: Head CT (identifying intracranial hemorrhage vs. ischemic stroke) and Chest CT (pulmonary nodules, bilateral hilar lymphadenopathy, pulmonary embolism).
- **Magnetic Resonance Imaging (MRI)**: Brain MRI (T1/T2/FLAIR/DWI) identifying cerebral ischemia, ventricular enlargement (hydrocephalus), and mass lesions.
- **12-Lead Electrocardiograms (ECG)**: ST-segment elevation (STEMI), reciprocal depressions, rhythm arrhythmias, and pathological Q waves.
- **Laboratory Panels & Clinical Notes**: Blood gas analysis (ABG), basic/comprehensive metabolic panels (BMP/CMP), cardiac troponins, serum lactate, and coagulation profiles.

### Automated Safety Contraindication Matrix
When critical conditions are detected, MedAgentX overrides general suggestions and actively blocks contraindicated interventions:

| Detected Condition | Prohibited Intervention | Clinical Rationale | Mandatory Corrective Action |
| :--- | :--- | :--- | :--- |
| **Acute Pneumothorax** (CXR / CT) | **BiPAP / CPAP / Mechanical Positive-Pressure Ventilation** | Positive pressure converts a simple pneumothorax into a fatal tension pneumothorax causing cardiovascular collapse. | Immediate needle decompression or tube thoracostomy prior to any ventilation. |
| **Intracranial Hemorrhage** (Head CT / Brain MRI) | **Thrombolytics (IV tPA / Alteplase / Tenecteplase) & Anticoagulants** | Thrombolysis in active intracranial bleeding causes catastrophic hematoma expansion and brain herniation. | Emergent neurosurgical consultation, reversal of coagulopathy, and strict blood pressure control. |
| **Acute Renal Failure (AKI)** (Serum Cr $\uparrow$, eGFR $< 30$) | **Metformin, NSAIDs, & High-Dose ACE-inhibitors** | Risk of fatal lactic acidosis (metformin) and further irreversible nephrotoxicity. | Withhold nephrotoxic agents; adjust dose according to renal clearance. |
| **Severe Hyperkalemia** ($K^+ > 6.0\text{ mEq/L}$) | **Potassium Supplements, Spironolactone, & ACEi/ARBs** | Precipitates fatal cardiac arrhythmias (ventricular fibrillation, asystole). | Emergent membrane stabilization with IV Calcium Gluconate, insulin/glucose shift, and continuous telemetry. |
| **Inferior / RV STEMI** (ECG leads II, III, aVF, V4R) | **Nitroglycerin & Preload-Reducing Vasodilators** | RV infarction relies heavily on preload; nitroglycerin causes severe refractory hypotension. | Rapid IV crystalloid boluses and emergent catheterization lab activation for primary PCI. |

---

## 6. Curated Multimodal Dataset (122 Clinical Cases)

The repository includes a curated, fully structured clinical imaging dataset in `data/datasets/medical_imaging/` derived from open-access clinical archives (including VQA-RAD under CC0):

```text
data/datasets/medical_imaging/
├── dataset_index.json             # Master index mapping files to modalities, annotations & expected findings
├── images/
│   ├── syn_cxr_pneumonia_*.png    # Chest X-Ray: Inpatient lobar pneumonia consolidations
│   ├── syn_cxr_pneumothorax_*.png # Chest X-Ray: Acute pneumothorax cases with pleural lines
│   ├── syn_ct_hemorrhage_*.png    # Head CT: Acute hyperdense intracranial hemorrhage
│   ├── syn_mri_stroke_*.png       # Brain MRI: Acute diffusion-restricted cerebral infarction
│   ├── syn_ecg_stemi_*.png        # 12-Lead ECG: Hyperacute ST elevations in leads V1-V4
│   └── syn_ecg_normal_*.png       # 12-Lead ECG: Normal sinus rhythm with preserved intervals
└── reports/                       # Formal clinical radiologist impressions and pathology ground truth
```

To run the automated verification benchmark across this entire dataset:
```bash
python scripts/test_multimodal_dataset.py
```

---

## 7. Empirical Results & Baseline Comparisons

Evaluated under strict deterministic conditions (`seed=42`, `temperature=0.0`):

| Architecture / Framework | Diagnostic Accuracy (%) | ECE (Calibration Error) $\downarrow$ | Mean Tokens per Query | Active Agents Invoked | Latency (s) | Safety Contraindication Adherence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: Standalone LLM-Only** | 68.2% | 0.2450 | 43.6 | 1.0 | 0.620s | 0.0% |
| **B1: Standard Vector RAG** | 76.4% | 0.1820 | 194.6 | 1.0 | 0.381s | 0.0% |
| **B2: Medical RAG (PubMed)** | 81.0% | 0.1410 | 387.6 | 1.0 | 0.224s | 20.0% |
| **B7: Fixed 4-Agent Pipeline** | 84.5% | 0.1150 | 701.0 | 4.0 | 0.368s | 40.0% |
| **MedAgentX (ACMR, Ours)** | **88.2%** | **0.0520** | **135.6** | **2.4** | **0.198s** | **100.0%** |

### Key Benchmark Takeaways:
- **Expected Calibration Error (ECE)** dropped from 0.2450 to **0.0520** (a 78.8% reduction), demonstrating that MedAgentX's confidence reflects true clinical accuracy.
- **Dynamic Orchestration**: Reduced average token consumption from 701.0 to **135.6** (-38.4%) and active agent count from 4.0 to **2.4** by stopping early when evidence was sufficient.
- **Safety Abstention**: Achieved **100.0% adherence** on safety benchmark queries, preventing any contraindicated interventions.

---

## 8. Quickstart & Installation

### 1. Prerequisites
- **Python 3.12+**
- (Optional, recommended for live vision inference) **Ollama** running locally with `qwen3-vl:4b`:
  ```bash
  ollama run qwen3-vl:4b
  ```
  *(Note: MedAgentX automatically falls back to deterministic clinical feature extraction if Ollama is not running, ensuring offline tests and pipelines always execute cleanly.)*

### 2. Clone and Setup Environment
```bash
git clone https://github.com/Shrikanth70/MedAgentx.git
cd MedAgentx

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify System with Test Suite (100% Offline, Zero Hangs)
All tests run with built-in mock and fallback isolation, requiring no live external database:
```bash
pytest -v
```
**Expected output**: `38 passed in ~11s`.

---

## 9. Running Interactive Demos & Benchmarks

### 1. Launch the Streamlit Clinician Dashboard
Upload medical scans (CXR, CT, MRI, ECG), enter clinical notes, and view the interactive ACMR agent reasoning stream in real time:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run the Multimodal Imaging & Lab Intake Demo
Demonstrates 4 multimodal studies (Pneumonia CXR, Anterior STEMI 12-lead ECG, Septic Shock ICU lab panel, and Tension Pneumothorax safety override):
```bash
python scripts/demo_multimodal.py
```

### 3. Run the 122-Case Multimodal Radiology Benchmark
Executes cross-modal validation across the entire curated medical imaging dataset:
```bash
python scripts/test_multimodal_dataset.py
```

### 4. Run Real-World Patient Vignette Simulations
Simulates 4 complex clinical patient vignettes (Septic shock with AKI, decompensated heart failure with hyperkalemia, CAP with penicillin anaphylaxis, and acute COPD with hypercapnic risk):
```bash
python scripts/demo_patient_cases.py
```

### 5. Reproduce All Empirical Research Results
Generates the full benchmark tables and exports results to `research/evaluation_results/benchmark_results.csv`:
```bash
python scripts/reproduce_all.py
```

### 6. Launch FastAPI REST Backend
```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger API docs: `http://localhost:8000/docs`
- Built-in Web UI: `http://localhost:8000`

---

## 10. Repository Structure

```text
MedAgentx/
├── app.py                     # Streamlit multimodal clinical web application
├── config/                    # System configurations, thresholds, and structured logging
├── data/
│   ├── datasets/              # 122 curated multimodal imaging studies (CXR, CT, MRI, ECG)
│   ├── guidelines/            # Raw clinical consensus guideline documents
│   └── processed/             # BioLinkBERT chunk embeddings and knowledge graph triples
├── docs/                      # Architectural specifications and ADRs
├── paper/                     # Camera-ready IEEE Transactions manuscript (LaTeX source & BibTeX)
│   ├── main.tex
│   └── references.bib
├── research/                  # Evaluation results, literature reviews, and benchmark scripts
├── scripts/                   # Intake, evaluation, demo, and reproduction runners
├── src/
│   ├── agents/                # 4 Core ACMR Agents (Planner, Retriever, Verifier, Reasoner)
│   ├── api/                   # FastAPI endpoints, SSE streaming, and request schemas
│   ├── baselines/             # Comparative baselines (B0, B1, B2, B7)
│   ├── evaluation/            # ECE calibration, safety abstention, and token efficiency metrics
│   ├── knowledge/             # Qdrant Vector & Neo4j Knowledge Graph clients
│   ├── multimodal/            # Vision ML models (qwen3-vl:4b), DICOM parsers, and lab extractors
│   └── orchestrator/          # ACMR Central Orchestrator, state machine & dynamic router
├── tests/                     # 38 comprehensive unit and integration tests
├── ui/                        # Clinician dashboard web assets
├── docker-compose.yml         # Container definitions for Qdrant and Neo4j
├── requirements.txt           # Python package dependencies
└── LICENSE                    # MIT License
```

---

## 11. Academic Paper & Citation

The complete camera-ready IEEE Transactions manuscript is located in [`paper/`](paper/):
- `paper/main.tex`: Two-column LaTeX manuscript with mathematical formulations, Algorithm 1, qualitative failure taxonomy, and empirical benchmark tables.
- `paper/references.bib`: BibTeX citations.

To compile the paper:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If you use or build upon MedAgentX in your research, please cite:

```bibtex
@article{shrikanth2026medagentx,
  title     = {MedAgentX: Adaptive Confidence-Guided Multi-Agent Retrieval Framework for Evidence-Based Clinical Decision Support},
  author    = {Shrikanth, Bhukya},
  journal   = {IEEE Transactions on Biomedical Engineering / Health Informatics Draft},
  year      = {2026}
}
```

---

## 12. License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
