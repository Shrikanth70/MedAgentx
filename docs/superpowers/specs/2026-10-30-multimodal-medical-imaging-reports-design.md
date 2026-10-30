# Design Specification: MedAgentX Multimodal Medical Perception & Imaging Engine

- **Date:** October 30, 2026
- **Status:** Approved Draft for Implementation Planning
- **Topic:** Ingestion and Interpretation of Patient Reports (PDF/Lab Tables) and Diagnostic Scans (Chest X-Rays, 12-Lead ECGs, CT Scans) Driving the ACMR Multi-Agent Decision Support Pipeline

---

## 1. Executive Summary & Objectives

Clinicians evaluating acute and complex patients rely extensively on objective diagnostic studies:
1. **Clinical Documents & Laboratory Reports**: PDF/tabular metabolic panels, complete blood counts, arterial blood gas analyses, and cardiac biomarkers.
2. **Medical Diagnostic Imaging**: Chest Radiographs (CXR) for pneumonia, pulmonary edema, cardiomegaly, and pneumothorax; 12-Lead Electrocardiograms (ECG) for acute STEMI and arrhythmias.

This specification extends MedAgentX with a dedicated **Multimodal Perception Subsystem** (`src/multimodal/`) that converts uploaded scans and reports into structured clinical facts, feeding directly into the Adaptive Confidence-Guided Multi-Agent Retrieval (ACMR) pipeline.

---

## 2. Architecture & Multimodal Data Flow

```text
       ┌──────────────────────────────────────────────────────────────┐
       │   Clinician Upload: Medical Scans (CXR/ECG) + Lab PDF/Report │
       └──────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
       ┌──────────────────────────────────────────────────────────────┐
       │             src/multimodal/ Perception Subsystem             │
       │                                                              │
       │   1. ClinicalDocumentParser:                                 │
       │      • Tabular parsing of BMP, CBC, Lactate, Troponin, eGFR  │
       │                                                              │
       │   2. MedicalImagingAnalyzer:                                 │
       │      • Radiological sign detection (Consolidation, Edema,    │
       │        Cardiomegaly, Pneumothorax, ST-elevation)             │
       │      • Outputs typed ImagingAnalysis contract                │
       └──────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
       ┌──────────────────────────────────────────────────────────────┐
       │         MultimodalPatientCase (Structured Case State)        │
       │  - Demographic & Vitals + Parsed Labs + Visual Scan Findings │
       └──────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
       ┌──────────────────────────────────────────────────────────────┐
       │                   ACMR Multi-Agent Pipeline                  │
       │  • Agent 1 (Planner): Cross-modal subquery generation        │
       │  • Agent 2 (Retriever): Multi-domain guideline retrieval     │
       │  • Agent 3 (Verifier): Cross-checks imaging vs. drugs        │
       │    (e.g., Pneumothorax -> withhold positive pressure;        │
       │     RV Infarction ECG -> block Nitroglycerin)                │
       │  • Agent 4 (Reasoner): Synthesizes Multimodal CDSS Report    │
       └──────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details & Data Contracts

### 3.1 Data Contracts (`src/multimodal/contracts.py`)
```python
class ImagingAnalysis(BaseModel):
    modality: str  # "Chest X-Ray", "12-Lead ECG", "CT Scan"
    radiological_findings: list[str] = Field(default_factory=list)
    diagnostic_impression: str = ""
    critical_urgency: str = "ROUTINE"  # "ROUTINE", "URGENT", "EMERGENT"
    contraindicated_interventions: list[str] = Field(default_factory=list)
    confidence: float = 0.90


class MultimodalPatientCase(PatientCase):
    imaging_analysis: ImagingAnalysis | None = None
    uploaded_document_text: str = ""
    uploaded_file_name: str = ""
```

### 3.2 Clinical Document & Lab Parser (`src/multimodal/document_parser.py`)
- Ingests PDF bytes, TXT, or tabular lab data.
- Employs regex and key-value tabular extraction for:
  - Renal function: `eGFR`, `Creatinine`, `BUN`
  - Electrolytes: `Potassium (K+)`, `Sodium (Na+)`, `Bicarbonate`
  - Critical markers: `Lactate`, `Troponin`, `BNP`
  - Blood gas: `pH`, `PaCO2`, `PaO2`, `HCO3`
  - Hematology: `WBC`, `Hemoglobin`, `Platelets`
- Merges findings into `PatientCase` labs dictionary and properties.

### 3.3 Medical Imaging Analyzer (`src/multimodal/imaging_analyzer.py`)
- Analyzes diagnostic images (PNG, JPG, DICOM):
  - **Chest Radiographs (CXR)**:
    - *Lobar Pneumonia*: Unilateral dense consolidation, air bronchograms.
    - *Cardiogenic Pulmonary Edema*: Bilateral perihilar bat-wing infiltrates, Kerley B lines, cardiomegaly (cardiothoracic ratio > 0.50).
    - *Pneumothorax*: Absence of lung markings, pleural line, mediastinal shift.
  - **12-Lead Electrocardiograms (ECG)**:
    - *Anterior STEMI*: ST-segment elevation ≥ 2 mm in leads V1–V4 with reciprocal depressions.
    - *Right Ventricular / Inferior STEMI*: ST-segment elevation in leads II, III, aVF and V4R.
    - *Severe Hyperkalemia*: Peaked, symmetrical T waves, PR prolongation, QRS widening.
- Dual Execution Architecture:
  - If vision provider / API configured, generates prompt analysis.
  - Deterministic clinical feature classifier operating 100% offline with zero external network or GPU weight downloads.

### 3.4 ACMR Orchestrator Cross-Modal Safety Verification
- Updates `VerifierAgent.verify_patient` and `MedAgentXSystem.query_multimodal`:
  - **Pneumothorax finding on imaging**: Blocks positive pressure ventilation (BiPAP/CPAP) and flags immediate need for needle/tube decompression.
  - **RV Infarction finding on ECG**: Blocks Nitroglycerin; mandates IV normal saline preload maintenance.
  - **Acute Pulmonary Edema on CXR**: Confirms indication for IV loop diuretics (Furosemide); checks blood pressure to avoid cardiogenic shock.
  - **Lobar Consolidation on CXR + Leukocytosis**: Matches severe inpatient CAP guideline protocols, confirming antimicrobial coverage.

### 3.5 Interactive Inspection Suite
- **Streamlit Application (`app.py`)**:
  - Adds a dedicated tab **"🩻 Multimodal Patient & Medical Scan Intake"**:
    - File upload widget for medical images (CXR, ECG, CT) with interactive image preview.
    - File upload widget for lab PDF/reports with parsed tabular output.
    - One-click presets for sample clinical scans and lab sheets.
    - Visual rendering of the multimodal ACMR trace, composite confidence dial, and final consultation report.
- **CLI Runner (`scripts/demo_multimodal.py`)**:
  - Command-line runner demonstrating end-to-end processing of sample multimodal studies (CXR pneumonia, ECG anterior STEMI, Lab PDF with septic shock and AKI).
- **Sample Asset Library (`data/samples/`)**:
  - Pre-packaged sample medical images and lab reports for immediate evaluation.

---

## 4. Verification & Testing Strategy

1. **Unit Tests (`tests/test_multimodal.py`)**:
   - `test_document_parser_tabular_labs`: Validates extraction of eGFR, K+, Troponin, Lactate, and WBC from simulated hospital lab reports.
   - `test_imaging_analyzer_cxr_and_ecg`: Validates recognition of CXR lobar consolidation, cardiogenic pulmonary edema, and ECG STEMI patterns.
   - `test_cross_modal_safety_verification`: Verifies that imaging-detected pneumothorax and RV infarction trigger appropriate $R_{contra} \ge 0.85$ safety overrides.
2. **Regression Testing**:
   - All existing 33 pytest tests continue to pass with 100% success.
3. **End-to-End Execution**:
   - `python scripts/demo_multimodal.py` executes successfully and prints full multimodal reports.

