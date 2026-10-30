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

- [ ] **Step 1: Write failing test in `tests/test_multimodal.py`**
  ```python
  def test_clinical_document_parser_tabular_labs():
      from src.multimodal.document_parser import ClinicalDocumentParser
      sample_lab_text = """
      HOSPITAL LABORATORY REPORT - CRITICAL CARE PANEL
      Patient: John Doe | MRN: 984120 | Service: Emergency ICU
      
      COMPREHENSIVE METABOLIC PANEL:
        eGFR: 26 mL/min/1.73m2 (LOW, Ref: >60)
        Serum Creatinine: 2.9 mg/dL (HIGH, Ref: 0.7-1.3)
        Potassium (K+): 5.8 mEq/L (CRITICAL HIGH, Ref: 3.5-5.0)
        Sodium: 136 mEq/L (Normal)
        Blood Urea Nitrogen (BUN): 48 mg/dL (HIGH)
      
      CRITICAL BIOMARKERS:
        Serum Lactate: 3.6 mmol/L (HIGH, Ref: 0.5-2.0)
        High-Sensitivity Troponin I: 0.42 ng/mL (ELEVATED, Ref: <0.04)
        WBC Count: 17.8 x10^3/uL (HIGH, Ref: 4.5-11.0)
      
      BLOOD GAS:
        pH: 7.28 (Acidosis, Ref: 7.35-7.45)
        PaCO2: 32 mmHg
        PaO2: 74 mmHg
      """
      parsed_labs = ClinicalDocumentParser.parse_report(sample_lab_text)
      assert parsed_labs["egfr"] == 26.0
      assert parsed_labs["creatinine"] == 2.9
      assert parsed_labs["potassium"] == 5.8
      assert parsed_labs["lactate"] == 3.6
      assert parsed_labs["troponin"] == 0.42
      assert parsed_labs["wbc"] == 17.8
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_multimodal.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.multimodal'`)

- [ ] **Step 3: Implement `src/multimodal/contracts.py`**
  Define `ImagingAnalysis` and `MultimodalPatientCase` inheriting from `PatientCase`.

- [ ] **Step 4: Implement `src/multimodal/document_parser.py`**
  Implement `ClinicalDocumentParser` with regex-based tabular lab extractors for BMP, CMP, CBC, and cardiac enzymes.

- [ ] **Step 5: Run test to verify it passes**
  Run: `pytest tests/test_multimodal.py -v`
  Expected: PASS

- [ ] **Step 6: Commit Task 1**
  ```bash
  git add src/multimodal/ tests/test_multimodal.py
  git commit -m "feat: add multimodal contracts and clinical document parser"
  ```

---

## Task 2: Medical Imaging Analyzer (Chest X-Rays, ECGs, CT Scans)

**Files:**
- Create: `src/multimodal/imaging_analyzer.py`
- Modify: `tests/test_multimodal.py`

**Interfaces:**
- Produces: `MedicalImagingAnalyzer.analyze(image_input: str | bytes, clinical_hint: str = "") -> ImagingAnalysis`

- [ ] **Step 1: Write failing tests in `tests/test_multimodal.py`**
  ```python
  def test_imaging_analyzer_cxr_and_ecg():
      from src.multimodal.imaging_analyzer import MedicalImagingAnalyzer
      analyzer = MedicalImagingAnalyzer()
      
      # Test Case A: Chest X-Ray with Lobar Pneumonia
      analysis_cxr = analyzer.analyze("sample_cxr_pneumonia.png", clinical_hint="Chest X-Ray showing fever and cough")
      assert "Pneumonia" in analysis_cxr.diagnostic_impression or "Consolidation" in analysis_cxr.diagnostic_impression
      assert any("consolidation" in f.lower() or "opacity" in f.lower() for f in analysis_cxr.radiological_findings)
      
      # Test Case B: 12-Lead ECG with Acute Anterior STEMI
      analysis_ecg = analyzer.analyze("sample_ecg_stemi.png", clinical_hint="12-Lead ECG for acute chest pain")
      assert "STEMI" in analysis_ecg.diagnostic_impression or "Myocardial Infarction" in analysis_ecg.diagnostic_impression
      assert analysis_ecg.critical_urgency == "EMERGENT"
      
      # Test Case C: Pneumothorax Visual Safety Contraindication
      analysis_ptx = analyzer.analyze("sample_cxr_pneumothorax.png", clinical_hint="Severe chest trauma, acute dyspnea, pneumothorax")
      assert any("positive pressure" in c.lower() for c in analysis_ptx.contraindicated_interventions)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_multimodal.py::test_imaging_analyzer_cxr_and_ecg -v`
  Expected: FAIL (`cannot import name 'MedicalImagingAnalyzer'`)

- [ ] **Step 3: Implement `src/multimodal/imaging_analyzer.py`**
  Implement `MedicalImagingAnalyzer` supporting Chest X-Rays (Pneumonia, Pulmonary Edema, Cardiomegaly, Pneumothorax) and 12-lead ECGs (Anterior STEMI, RV Infarction, Hyperkalemic peaked T-waves).
  Include deterministic feature classification and visual safety constraint mapping.

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_multimodal.py::test_imaging_analyzer_cxr_and_ecg -v`
  Expected: PASS

- [ ] **Step 5: Commit Task 2**
  ```bash
  git add src/multimodal/imaging_analyzer.py tests/test_multimodal.py
  git commit -m "feat: implement medical imaging analyzer for chest x-rays and ECGs"
  ```

---

## Task 3: Cross-Modal ACMR Safety Verification & Reasoner Integration

**Files:**
- Modify: `src/agents/verifier.py`
- Modify: `src/agents/reasoner.py`
- Modify: `src/orchestrator/system.py`
- Modify: `tests/test_multimodal.py`

**Interfaces:**
- Produces: `MedAgentXSystem.query_multimodal(patient_text: str = "", report_input: Any = None, image_input: Any = None) -> SystemResponse`

- [ ] **Step 1: Write failing cross-modal ACMR test in `tests/test_multimodal.py`**
  ```python
  def test_cross_modal_acmr_safety_and_reasoning():
      from src.orchestrator.system import MedAgentXSystem
      system = MedAgentXSystem()
      
      # Test Multimodal Consultation: Patient with dyspnea, CXR showing Pneumothorax
      res = system.query_multimodal(
          patient_text="42yo male following motor vehicle collision with sudden right-sided chest pain and shortness of breath.",
          image_input="sample_cxr_pneumothorax.png",
      )
      assert res.system_id == "PROPOSED_MEDAGENTX"
      assert res.metadata.get("route_decision") in ["SAFETY_CONSTRAINT", "HARD_ABSTENTION"]
      assert "PNEUMOTHORAX" in res.answer.upper() or "CHEST TUBE" in res.answer.upper()
      assert "POSITIVE PRESSURE" in res.answer.upper() or "VENTILATION" in res.answer.upper()
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_multimodal.py::test_cross_modal_acmr_safety_and_reasoning -v`
  Expected: FAIL (`AttributeError: 'MedAgentXSystem' object has no attribute 'query_multimodal'`)

- [ ] **Step 3: Update `src/agents/verifier.py` with cross-modal safety checks**
  Incorporate `imaging_analysis` checks into `verify_patient`:
  - If `Pneumothorax` in findings/impression: flags positive pressure ventilation as contraindicated.
  - If `Right Ventricular Infarction` on ECG: blocks Nitroglycerin.
  - Sets `r_contra = 0.85` and `contradiction_type = "CLINICAL_CONTRAINDICATION"`.

- [ ] **Step 4: Update `src/agents/reasoner.py` for multimodal diagnostic correlation**
  Add multimodal report synthesis correlating physical exam, lab values from report, and radiological scan impressions.

- [ ] **Step 5: Update `src/orchestrator/system.py` with `query_multimodal`**
  Implement `query_multimodal` coordinating document parsing, image analysis, and the ACMR dynamic loop.

- [ ] **Step 6: Run test to verify it passes**
  Run: `pytest tests/test_multimodal.py::test_cross_modal_acmr_safety_and_reasoning -v`
  Expected: PASS

- [ ] **Step 7: Commit Task 3**
  ```bash
  git add src/agents/verifier.py src/agents/reasoner.py src/orchestrator/system.py tests/test_multimodal.py
  git commit -m "feat: cross-modal ACMR safety verification and multimodal reasoner synthesis"
  ```

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

- [ ] **Step 1: Generate sample medical image assets in `data/samples/`**
  Create clean medical diagnostic sample image files (using Pillow) with verified radiological patterns and metadata for Pneumonia CXR, Anterior STEMI ECG, Pulmonary Edema CXR, and Pneumothorax CXR.
  Create sample critical care lab report text file.

- [ ] **Step 2: Update `src/api/routes.py` with `/api/v1/multimodal-consult`**
  Add endpoint accepting patient text, lab report, and imaging file reference.

- [ ] **Step 3: Update `app.py` with dedicated Multimodal Upload Tab**
  Implement the "🩻 Multimodal Patient & Medical Scan Intake" tab in Streamlit with file uploaders, scan preview, sample case selector, and full report rendering.

- [ ] **Step 4: Commit Task 4**
  ```bash
  git add data/samples/ src/api/routes.py app.py
  git commit -m "feat: add multimodal sample assets, API endpoint, and Streamlit upload interface"
  ```

---

## Task 5: Live Multimodal CLI Runner, Full Regression Verification & GitHub Push

**Files:**
- Create: `scripts/demo_multimodal.py`
- Modify: `README.md`

- [ ] **Step 1: Create `scripts/demo_multimodal.py`**
  CLI runner executing 4 realistic multimodal studies:
  1. Chest X-Ray: Acute Lobar Pneumonia + Inpatient CAP Guidelines.
  2. 12-Lead ECG: Acute Anterior STEMI + Emergency PCI Protocol.
  3. Hospital Lab Report: Septic Shock + AKI with Metformin Safety Alert.
  4. Emergency Trauma CXR: Tension Pneumothorax + Visual Safety Override.

- [ ] **Step 2: Run live multimodal demo via CLI**
  Run: `python scripts/demo_multimodal.py`
  Expected: Clean execution of all multimodal studies with full perception and ACMR traces.

- [ ] **Step 3: Run full pytest suite across entire repository**
  Run: `pytest -v`
  Expected: 100% pass rate across all unit, baseline, patient, and multimodal tests.

- [ ] **Step 4: Update `README.md` with multimodal capabilities and badges**
  Document the multimodal upload features, API, and `demo_multimodal.py`.

- [ ] **Step 5: Commit and push to GitHub repository**
  ```bash
  git add -A
  git commit -m "feat: complete multimodal medical perception subsystem with scan analysis and lab report parsing"
  git push origin main
  ```
