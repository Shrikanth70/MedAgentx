# Design Specification: MedAgentX Real-World Patient Clinical Engine

- **Date:** October 30, 2026
- **Status:** Approved Draft for Implementation Planning
- **Topic:** Generalizing MedAgentX for Real, Practical Patient Presentations, Diverse Guidelines, and Vital/Lab-Aware Safety Verification

---

## 1. Executive Summary & Objectives

The goal of this design is to transition MedAgentX from answering static, keyword-triggered medical queries to reasoning over authentic, complex, real-world clinical patient cases.

In clinical practice, patients do not arrive as single-sentence guideline queries. They present with multi-morbidity (e.g., Heart Failure co-occurring with Chronic Kidney Disease), acute hemodynamic compromise (hypotension, tachypnea, hypoxemia), abnormal metabolic laboratories (hyperkalemia, acidosis, elevated lactate), and drug allergies.

This specification establishes:
1. **`PatientCase` Schema & Automated Vignette Parser**: Native ingestion of both structured patient records and unstructured clinical admission notes (HPI, vitals, labs, medications, allergies).
2. **Generalized Multi-Domain Retrieval Engine**: Eliminating keyword shortcuts in favor of lexical BM25/TF-IDF token scoring fused with semantic vector similarity, backed by an expanded multi-specialty clinical guideline database (Cardiology, Critical Care/Sepsis, Pulmonology/COPD/Pneumonia, Nephrology/AKI, Endocrinology/DKA).
3. **Vital- and Laboratory-Aware Safety Verifier**: Dynamic cross-referencing of retrieved clinical interventions against the patient's acute physiological status (eGFR, potassium, blood pressure, drug allergies).
4. **Patient-Tailored Clinical Reasoner**: Structured synthesis delivering triage stratification, patient-adjusted guideline therapies, explicit contraindication overrides, and monitoring orders.
5. **Interactive Inspection Tooling**: A live CLI inspection runner (`scripts/demo_patient_cases.py`) with diverse clinical case presets and custom freeform patient entry, alongside an updated interactive Web Dashboard (`app.py`).

---

## 2. Architecture & Data Flow

```text
                           ┌──────────────────────────────────────────────┐
                           │   Real Patient Presentation / Clinical Note  │
                           │   (Freeform note OR Structured PatientCase)  │
                           └──────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │       Agent 1: Planner & Vignette Parser     │
                           │  - Extracts Vitals (BP, HR, SpO2) & Labs     │
                           │  - Identifies Acute Risk Flags               │
                           │  - Decomposes Targeted Clinical Sub-Queries  │
                           └──────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │      Agent 2: Generalized Hybrid Retriever   │
                           │  - BM25/TF-IDF Token Fusion + Vector Search  │
                           │  - Multi-Domain Corpus (Sepsis, AKI, COPD,   │
                           │    DKA, HFrEF, STEMI, Hyperkalemia)          │
                           └──────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │    Agent 3: Vital/Lab-Aware Safety Verifier  │
                           │  - Checks SBP < 90 (Hypotension contraind.)  │
                           │  - Checks eGFR < 30 (Renal dosing/toxicity)  │
                           │  - Checks K+ > 5.2 (Hyperkalemia guard)      │
                           │  - Checks Documented Drug Allergies          │
                           │  - Calculates S_rel, V_agree, R_contra       │
                           └──────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │          ACMR Dynamic Orchestrator           │
                           │  - Computes C_total Composite Confidence     │
                           │  - Routes: STOP_CONFIDENT / SAFETY_CONSTRAINT│
                           └──────────────────────┬───────────────────────┘
                                                  │
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │       Agent 4: Patient-Specific Reasoner     │
                           │  1. Clinical Triage & Risk Stratification    │
                           │  2. Patient-Specific Guideline Treatment     │
                           │  3. Safety Overrides & Withheld Therapies    │
                           │  4. Immediate Monitoring & Orders            │
                           └──────────────────────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Patient Representation & Vignette Parser (`src/orchestrator/patient.py`)
- **`PatientCase` Data Model**:
  ```python
  class PatientCase(BaseModel):
      patient_id: str = "PT-001"
      age: int | None = None
      sex: str | None = None
      chief_complaint: str
      hpi: str = ""
      sbp: float | None = None
      dbp: float | None = None
      hr: float | None = None
      rr: float | None = None
      spo2: float | None = None
      temp: float | None = None
      egfr: float | None = None
      creatinine: float | None = None
      potassium: float | None = None
      troponin: float | None = None
      lactate: float | None = None
      wbc: float | None = None
      glucose: float | None = None
      past_medical_history: list[str] = Field(default_factory=list)
      allergies: list[str] = Field(default_factory=list)
      current_medications: list[str] = Field(default_factory=list)
      clinical_question: str = ""
  ```
- **`PatientVignetteParser`**:
  - Regex-based extraction of vitals:
    - Blood pressure: `(?:BP|blood pressure)[:\s]*(\d{2,3})[/](\d{2,3})`
    - Heart rate: `(?:HR|pulse|heart rate)[:\s]*(\d{2,3})`
    - SpO2: `(?:SpO2|O2 sat|saturation)[:\s]*(\d{2,3})%?`
    - Temperature: `(?:Temp|temperature)[:\s]*(\d{2,3}(?:\.\d+)?)`
  - Lab extraction:
    - eGFR: `(?:eGFR|GFR)[:\s]*(\d{1,3})`
    - Potassium: `(?:K\+?|potassium)[:\s]*(\d+\.?\d*)`
    - Creatinine: `(?:Cr|creatinine)[:\s]*(\d+\.?\d*)`
    - Lactate: `(?:lactate)[:\s]*(\d+\.?\d*)`
    - Troponin: `(?:troponin|trop)[:\s]*(elevated|positive|\d+\.?\d*)`
  - Acute Risk Flags:
    - `is_hypotensive`: SBP < 90 mmHg or MAP < 65 mmHg
    - `is_hypoxemic`: SpO2 < 90%
    - `is_severe_renal_impairment`: eGFR < 30 mL/min
    - `is_hyperkalemic`: Serum K+ > 5.2 mEq/L
    - `is_in_septic_shock`: Lactate ≥ 2.0 mmol/L and SBP < 90 mmHg

### 3.2 Generalized Multi-Domain Retrieval Engine (`src/knowledge/hybrid_retriever.py`)
- **Corpus Expansion**:
  - Add guidelines for:
    - **Sepsis & Septic Shock** (Surviving Sepsis Campaign 2021)
    - **Acute Kidney Injury (AKI) & Nephrotoxins** (KDIGO)
    - **Severe Hyperkalemia Management** (Endocrine/Nephrology Guidelines)
    - **Acute COPD / Severe Asthma Exacerbation** (GOLD / GINA)
    - **Diabetic Ketoacidosis (DKA)** (ADA Guidelines)
    - **Atrial Fibrillation with RVR & DOAC Anticoagulation** (AHA/ACC)
- **Scoring Algorithm**:
  - Computes term frequency-inverse document frequency (TF-IDF / BM25 token overlap) between the combined patient context (chief complaint + acute symptoms + question) and all guideline chunks.
  - Returns top-$K$ chunks with normalized dense vector scores ($S_{vec}$) and relevant Knowledge Graph triples ($S_{graph}$).

### 3.3 Vital- and Lab-Aware Safety Verifier (`src/agents/verifier.py`)
- Cross-references evidence with patient state:
  1. **Hemodynamic Safety**: If `is_hypotensive` ($SBP < 90$), blocks vasodilators (Nitroglycerin, Nitroprusside, Hydralazine) and acute Beta-blockers. Triggers `CLINICAL_CONTRAINDICATION` alert for volume/vasopressor resuscitation.
  2. **Renal Clearance Safety**: If `is_severe_renal_impairment` ($eGFR < 30$), flags contraindications for Metformin, NSAIDs, high-dose DOACs, or SGLT2i glycemic initiation.
  3. **Electrolyte Safety**: If `is_hyperkalemic` ($K^+ > 5.2$), flags contraindications for MRAs (Spironolactone), ACE inhibitors, ARBs, and ARNI. Recommends urgent hyperkalemia shifting and stabilization.
  4. **Allergy Guard**: Checks whether first-line antibiotics or treatments match patient allergies (e.g., Penicillin/Cephalosporin allergy). If so, triggers `ALLERGY_CONTRAINDICATION` and mandates alternative antibiotic classes.
- Sets $R_{contra} \ge 0.80$ on any critical contraindication, driving ACMR Orchestrator into `SAFETY_CONSTRAINT` mode.

### 3.4 Patient-Tailored Reasoner (`src/agents/reasoner.py`)
- Generates a clinical consultation report structured as:
  ```text
  PATIENT CLINICAL ASSESSMENT & DECISION SUPPORT REPORT
  Patient: [Demographics, Chief Complaint]
  Vitals: [BP, HR, SpO2] | Key Labs: [eGFR, K+, Lactate, etc.]
  
  1. CLINICAL TRIAGE & RISK STRATIFICATION
  [Identifies acute instability: e.g., Septic Shock, Severe Hyperkalemia]
  
  2. EVIDENCE-GROUNDED TREATMENT PLAN
  [Tailored guideline-directed medical therapy for this patient]
  
  3. PATIENT SAFETY CONSTRAINTS & WITHHELD MEDICATIONS
  [Explicitly flags first-line medications contraindicated by this patient's vitals, labs, or allergies]
  
  4. MONITORING ORDERS & DIAGNOSTIC WORKUP
  [Target vitals, lab recheck schedule, and escalation triggers]
  ```

### 3.5 Interactive Inspection Suite
- **`scripts/demo_patient_cases.py`**:
  - Pre-loaded with 4 realistic acute clinical vignettes:
    - *Case 1*: Septic Shock with AKI (BP 82/48, Lactate 3.8, eGFR 28).
    - *Case 2*: Acute Decompensated Heart Failure with Severe Hyperkalemia (K+ 5.8, eGFR 22, orthopneic).
    - *Case 3*: Severe Inpatient CAP with Anaphylactic Penicillin Allergy.
    - *Case 4*: Acute Severe COPD Exacerbation with Hypercapnic Respiratory Risk.
  - Custom freeform clinical note entry allowing any user to type a patient chart note and immediately inspect the live execution trace and report.
- **Streamlit Application (`app.py`)**:
  - Adds a "Real Patient Case Intake" view with sample presets and real-time execution.

---

## 4. Testing & Verification Plan

1. **Unit Tests**:
   - `test_patient_parser`: Tests extraction of vitals, abnormal labs, and acute risk flags from messy clinical text notes.
   - `test_generalized_hybrid_retriever`: Tests that non-cardiology queries (Sepsis, AKI, DKA, COPD) retrieve accurate, disease-specific guideline chunks.
   - `test_vital_lab_safety_verifier`: Tests that hypotension, hyperkalemia, and drug allergies trigger safety overrides with $R_{contra} \ge 0.80$.
2. **Regression Testing**:
   - All existing 29 pytest tests continue to pass without regression.
3. **End-to-End Live Execution**:
   - `python scripts/demo_patient_cases.py` runs all 4 realistic clinical cases successfully.
