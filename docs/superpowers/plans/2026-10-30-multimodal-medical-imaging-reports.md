# MedAgentX Multimodal Medical Perception & Imaging Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Equip MedAgentX with multimodal perception to ingest and interpret uploaded patient documents (PDF/lab reports) and diagnostic imaging scans (Chest X-Rays, 12-lead ECGs, CT scans), connecting radiological findings and lab abnormalities directly into the ACMR multi-agent decision support loop.

**Architecture:** Five sequential tasks: (1) multimodal data contracts and clinical document parser, (2) medical imaging analyzer with radiological sign detection, (3) cross-modal ACMR safety verifier and reasoner integration, (4) sample asset library and Streamlit/FastAPI upload views, (5) live multimodal CLI inspection runner (`scripts/demo_multimodal.py`), regression verification, and GitHub synchronization.

**Tech Stack:** Python 3.12, Pydantic v2, Pillow/PIL, pytest, structlog, Streamlit, FastAPI.

**Spec:** `docs/superpowers/specs/2026-10-30-multimodal-medical-imaging-reports-design.md`

## Global Constraints

- Python 3.12 compatibility.
- Zero external network or heavyweight GPU weight requirements; operates 100% offline with optional online Vision LLM hooks.
- All existing 33 pytest tests must continue to pass with zero regression.
- Standard imaging formats (PNG, JPG, DICOM-converted) and document formats (PDF, TXT, CSV) must be cleanly parsed.
- Imaging-detected contraindications (Pneumothorax, RV Infarction on ECG) must strictly trigger $R_{contra} \ge 0.85$ safety overrides.

## Review Focus

1. **Document parser handles real lab panels**: Complex tabular text with reference ranges (e.g., `Serum Creatinine: 2.8 mg/dL [0.7-1.3]`, `Potassium: 5.8 mEq/L [3.5-5.0]`, `Lactate: 3.8 mmol/L [0.5-2.0]`) must be extracted into correct numeric values.
2. **Imaging analyzer produces structured radiological impressions**: Output `ImagingAnalysis` must include specific visual findings, diagnostic impressions, urgency, and contraindicated interventions.
3. **Cross-modal safety interception**: A patient with acute dyspnea whose CXR reveals a `Pneumothorax` must immediately have positive pressure ventilation contraindicated in Agent 3 and Agent 4.
4. **Offline sample testability**: Reviewers without patient DICOM files must be able to run and inspect pre-packaged sample cases immediately via `scripts/demo_multimodal.py`.
5. **Interactive UI drag-and-drop**: `app.py` must support uploading scans and reports with real-time visual display and ACMR execution.

---

## Task 1: Multimodal Contracts & Clinical Document Parser

**Files:**
- Create: `src/multimodal/__init__.py`
- Create: `src/multimodal/contracts.py`
- Create: `src/multimodal/document_parser.py`
- Create: `tests/test_multimodal.py`

**Interfaces:**
- Produces: `ImagingAnalysis`, `MultimodalPatientCase`, `ClinicalDocumentParser.parse_report(file_or_text: str | bytes) -> dict[str, Any]`

- [x] **Step 1: Write failing test in `tests/test_multimodal.py`**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement `src/multimodal/contracts.py`**
- [x] **Step 4: Implement `src/multimodal/document_parser.py`**
- [x] **Step 5: Run test to verify it passes**
- [x] **Step 6: Commit Task 1**


---

## Task 2: Medical Imaging Analyzer (Chest X-Rays, ECGs, CT Scans)

**Files:**
- Create: `src/multimodal/imaging_analyzer.py`
- Modify: `tests/test_multimodal.py`

**Interfaces:**
- Produces: `MedicalImagingAnalyzer.analyze(image_input: str | bytes, clinical_hint: str = "") -> ImagingAnalysis`

- [x] **Step 1: Write failing tests in `tests/test_multimodal.py`**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement `src/multimodal/imaging_analyzer.py`**
- [x] **Step 4: Run test to verify it passes**
- [x] **Step 5: Commit Task 2**

---

## Task 3: Cross-Modal ACMR Safety Verification & Reasoner Integration

**Files:**
- Modify: `src/agents/verifier.py`
- Modify: `src/agents/reasoner.py`
- Modify: `src/orchestrator/system.py`
- Modify: `tests/test_multimodal.py`

**Interfaces:**
- Produces: `MedAgentXSystem.query_multimodal(patient_text: str = "", report_input: Any = None, image_input: Any = None) -> SystemResponse`

- [x] **Step 1: Write failing cross-modal ACMR test in `tests/test_multimodal.py`**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Update `src/agents/verifier.py` with cross-modal safety checks**
- [x] **Step 4: Update `src/agents/reasoner.py` for multimodal diagnostic correlation**
- [x] **Step 5: Update `src/orchestrator/system.py` with `query_multimodal`**
- [x] **Step 6: Run test to verify it passes**
- [x] **Step 7: Commit Task 3**

---

## Task 4: Sample Asset Library & Interactive UI Upload View

**Files:**
- Create: `data/samples/sample_cxr_pneumonia.png` (verified diagnostic sample asset)
- Create: `data/samples/sample_ecg_stemi.png`
- Create: `data/samples/sample_cxr_pulmonary_edema.png`
- Create: `data/samples/sample_cxr_pneumothorax.png`
- Create: `data/samples/sample_lab_report.txt`
- Modify: `app.py` (add Multimodal Upload Tab)
- Modify: `src/api/routes.py` (add `/api/v1/multimodal-consult` endpoint)

- [x] **Step 1: Generate sample medical image assets in `data/samples/`**
- [x] **Step 2: Update `src/api/routes.py` with `/api/v1/multimodal-consult`**
- [x] **Step 3: Update `app.py` with dedicated Multimodal Upload Tab**
- [x] **Step 4: Commit Task 4**

---

## Task 5: Live Multimodal CLI Runner, Full Regression Verification & GitHub Push

**Files:**
- Create: `scripts/demo_multimodal.py`
- Modify: `README.md`

- [x] **Step 1: Create `scripts/demo_multimodal.py`**
- [x] **Step 2: Run live multimodal demo via CLI**
- [x] **Step 3: Run full pytest suite across entire repository (36/36 passing)**
- [x] **Step 4: Update `README.md` with multimodal capabilities and badges**
- [ ] **Step 5: Commit and push to GitHub repository**


