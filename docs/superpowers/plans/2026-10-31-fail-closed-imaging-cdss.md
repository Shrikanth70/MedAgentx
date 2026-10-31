# Fail-Closed Medical Imaging CDSS & Grounded Perception Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor MedAgentX's multimodal medical imaging pipeline to eliminate fail-open hallucinations, enforcing strict fail-closed gating on vision timeouts/failures, structured finding-level visual grounding, DICOM de-identification, and formal radiology-style report generation.

**Architecture:** A pluggable medical vision engine (`VisionEngine`) wraps local/remote VLMs and clinical classifiers behind an exponential backoff retry policy and secondary fallback. The ACMR router enforces a hard safety gate (`vision_status != OK` prohibits `STOP_CONFIDENT`), Agent 2 retrieves evidence per structured finding, Agent 3 cross-checks contradictions, and Agent 4 generates structured radiology reports with visual evidence and guideline citations.

**Tech Stack:** Python 3.12, Pydantic v2, PyTest, Pillow, Requests, NumPy, (optional pydicom, nibabel), Streamlit, Qdrant, Neo4j.

**Spec:** `docs/specs/fail_closed_imaging_cdss_spec.md`

## Global Constraints
- Python 3.12 compatibility.
- Zero placeholder findings: normal examinations cannot be fabricated; they must be backed by explicitly scored pertinent negatives.
- Hard router gate: `requires_imaging AND vision_status != VisionStatus.OK` mandates `ABSTAIN_INSUFFICIENT_EVIDENCE` or `ESCALATE_TO_HUMAN`.
- DICOM files must be de-identified (scrubbed of patient and institution PHI tags) prior to any VLM or LLM call.
- All existing 38 unit and integration tests must remain passing.

## Review Focus
1. **Vision Timeout**: Model fails to respond within timeout window -> Must emit `VisionStatus.TIMEOUT`, trigger `ABSTAIN_INSUFFICIENT_EVIDENCE`, and produce zero fabricated findings. (Tested in Task 10).
2. **Corrupted / Motion-Blurred Images**: Image fails Laplacian variance or exposure thresholds -> Emits `VisionStatus.LOW_QUALITY`, router requests new acquisition via `REQUEST_MORE_INPUT`. (Tested in Task 10).
3. **Critical Pathology (Pneumothorax / ICH / STEMI)**: Finding has `is_critical=True` -> Router triggers `ESCALATE_TO_HUMAN` and reasoner flags emergency intervention. (Tested in Task 10).
4. **DICOM De-identification**: DICOM file with patient names or hospital identifiers -> Scrubber strips PHI and preserves anatomical acquisition parameters. (Tested in Task 10).
5. **No Synthetic Confidence**: Verifier and confidence calculator do not pad missing metrics to 1.0; compute dynamic penalties. (Tested in Task 5).

---

## Tasks

### Task 1: Pydantic Data Contracts & Vision Status Schema
**Files:**
- Modify: `src/multimodal/contracts.py`
- Test: `tests/test_vision_contracts.py`

- [ ] Write failing test in `tests/test_vision_contracts.py` asserting `VisionStatus`, `VisualFinding`, `NegativeFinding`, `QualityAssessment`, and `VisionResult` validation.
- [ ] Implement new contracts in `src/multimodal/contracts.py`:
  - `VisionStatus` enum: `OK`, `FAILED`, `TIMEOUT`, `LOW_QUALITY`, `UNSUPPORTED`.
  - `VisualEvidence`: `bbox`, `slice_index`, `heatmap_path`, `supporting_visual_description`.
  - `VisualFinding`: `finding_id`, `label`, `coding_system`, `code`, `anatomical_location`, `laterality`, `size_or_extent`, `probability`, `uncertainty`, `is_critical`, `evidence`.
  - `NegativeFinding`: `label`, `code`, `probability`, `checked`.
  - `QualityAssessment`: `is_acceptable`, `orientation_valid`, `exposure_quality`, `motion_artifact_detected`, `anatomy_matches_modality`, `rejection_reasons`.
  - `VisionResult`: `trace_id`, `status`, `modality`, `body_part`, `view`, `quality`, `findings`, `negative_findings`, `limitations`, `model_name`, `model_version`, `inference_duration_seconds`.
- [ ] Run `pytest tests/test_vision_contracts.py` and confirm pass.
- [ ] Commit: `git commit -m "feat(multimodal): add fail-closed VisionResult and VisualFinding schemas"`

---

### Task 2: Ingestion Pipeline & Quality Control Gate
**Files:**
- Create: `src/multimodal/ingestion.py`
- Test: `tests/test_ingestion_qc.py`

- [ ] Write failing test in `tests/test_ingestion_qc.py` testing image quality validation (contrast, brightness, Laplacian blur variance) and DICOM PHI stripping.
- [ ] Implement `MedicalImageIngestion`:
  - `validate_image_quality(image_input) -> QualityAssessment`: checks resolution ($> 256\times 256$), histogram distribution (underexposed/overexposed), and Laplacian variance for motion blur.
  - `anonymize_dicom(dicom_bytes) -> tuple[np.ndarray, dict[str, Any]]`: strips PatientName, PatientID, InstitutionName, while preserving Modality, PixelSpacing, and Window settings (using `pydicom` with graceful fallback for testing).
  - `extract_key_slices(volume_input) -> list[tuple[int, Image.Image]]`: multi-slice keyframe extraction for 3D CT/MRI volumes.
- [ ] Run `pytest tests/test_ingestion_qc.py` and confirm pass.
- [ ] Commit: `git commit -m "feat(multimodal): implement DICOM de-identification and image quality control gate"`

---

### Task 3: Pluggable Vision Engine with Exponential Backoff & Secondary Fallback
**Files:**
- Create: `src/multimodal/vision_engine.py`
- Test: `tests/test_vision_engine.py`

- [ ] Write failing test in `tests/test_vision_engine.py` simulating timeout on primary model, successful retry, and secondary fallback.
- [ ] Implement `BaseVisionModel` abstract class and `PluggableVisionEngine`:
  - Configurable primary model (`qwen3-vl:4b` or local VLM) with timeout and 3-attempt exponential backoff.
  - Automated secondary fallback (clinical rules/CNN classifier).
  - On persistent failure or timeout, return `VisionResult(status=VisionStatus.TIMEOUT/FAILED, findings=[])`.
  - Strict prohibition: Never return placeholder normal findings on failure.
- [ ] Run `pytest tests/test_vision_engine.py` and confirm pass.
- [ ] Commit: `git commit -m "feat(multimodal): implement pluggable vision engine with backoff and secondary fallback"`

---

### Task 4: Refactor Imaging Analyzer Adapter
**Files:**
- Modify: `src/multimodal/imaging_analyzer.py`
- Test: `tests/test_imaging_analyzer.py`

- [ ] Write failing test in `tests/test_imaging_analyzer.py` ensuring `MedicalImagingAnalyzer` returns `VisionResult` with accurate status and zero hallucinated findings on timeout.
- [ ] Refactor `MedicalImagingAnalyzer`:
  - Integrate `MedicalImageIngestion` and `PluggableVisionEngine`.
  - Execute QC checks first: if quality is unacceptable, immediately return `VisionStatus.LOW_QUALITY`.
  - Maintain compatibility with `ImagingAnalysis` adapter properties while providing full `VisionResult` backing.
- [ ] Run `pytest tests/test_imaging_analyzer.py` and confirm pass.
- [ ] Commit: `git commit -m "refactor(multimodal): wire analyzer through quality gate and vision engine"`

---

### Task 5: ACMR Router Hard-Gate & Confidence Penalization
**Files:**
- Modify: `src/orchestrator/router.py`
- Modify: `src/orchestrator/confidence.py`
- Test: `tests/test_router_safety.py`

- [ ] Write failing test asserting:
  - If required imaging has `vision_status != OK`, decision is `ABSTAIN_INSUFFICIENT_EVIDENCE`.
  - If critical finding detected (`is_critical=True`), decision is `ESCALATE_TO_HUMAN`.
  - If `status == LOW_QUALITY`, decision is `REQUEST_MORE_INPUT`.
  - Confidence calculation applies penalties for vision uncertainty and missing modalities.
- [ ] Update `ConfidenceCalculator` in `src/orchestrator/confidence.py`:
  - Add penalty terms for vision uncertainty, missing modalities, and low evidence count.
  - Do not default missing scores to 1.0.
- [ ] Update `ACMRRouter` in `src/orchestrator/router.py` to enforce the safety hard-gates and support `ACMRAction` enums.
- [ ] Run `pytest tests/test_router_safety.py` and confirm pass.
- [ ] Commit: `git commit -m "feat(orchestrator): add router hard-gate and confidence penalties"`

---

### Task 6: Grounded Finding-Level Query Construction in Hybrid Retriever
**Files:**
- Modify: `src/agents/retriever.py`
- Test: `tests/test_retriever_grounding.py`

- [ ] Write failing test in `tests/test_retriever_grounding.py` ensuring multi-finding query generation and provenance verification.
- [ ] Update `HybridRetrieverAgent` in `src/agents/retriever.py`:
  - Formulate 1 query per detected `VisualFinding` (incorporating anatomical location and laterality) + 1 combined clinical query.
  - Require evidence chunks to carry provenance: `source_id`, `publication_year`, `guideline_body`, and `section`.
- [ ] Run `pytest tests/test_retriever_grounding.py` and confirm pass.
- [ ] Commit: `git commit -m "feat(retriever): construct per-finding grounded queries with provenance"`

---

### Task 7: Verifier Contradiction & Critical Alert Escalation
**Files:**
- Modify: `src/agents/verifier.py`
- Test: `tests/test_verifier_refactor.py`

- [ ] Write failing test checking cross-modal contradiction detection (imaging findings vs clinical notes) and critical alert escalation.
- [ ] Update `ClinicalVerifierAgent` in `src/agents/verifier.py`:
  - Compare `VisualFinding` labels against patient notes/vitals for clinical discordance (e.g. consolidation vs "clear breath sounds").
  - Flag critical findings (pneumothorax, intracranial hemorrhage, acute aortic dissection) to trigger router human escalation.
- [ ] Run `pytest tests/test_verifier_refactor.py` and confirm pass.
- [ ] Commit: `git commit -m "feat(verifier): detect cross-modal contradictions and escalate critical alerts"`

---

### Task 8: Structured Radiology CDSS Reasoner & Multi-Format Exporters
**Files:**
- Modify: `src/agents/reasoner.py`
- Test: `tests/test_reasoner_radiology.py`

- [ ] Write failing test verifying standard radiology report formatting, visual citations, and JSON/HTML export.
- [ ] Update `ClinicalReasonerAgent` in `src/agents/reasoner.py`:
  - Structure report: Technique, Quality, Compartment Findings (with heatmap paths), Pertinent Negatives, Impression (ranked differentials with probabilities), Recommendations (guideline citations), Contraindications, and Mandatory Decision Support Disclaimer.
  - If `status == TIMEOUT` or `ABSTAIN_INSUFFICIENT_EVIDENCE`: Output clear diagnostic failure notice stating image analysis could not be safely completed.
  - Add exporters for JSON and HTML formats.
- [ ] Run `pytest tests/test_reasoner_radiology.py` and confirm pass.
- [ ] Commit: `git commit -m "feat(reasoner): generate structured radiology-style report with visual citations"`

---

### Task 9: System Integration & Streamlit Dashboard Updates
**Files:**
- Modify: `src/orchestrator/system.py`
- Modify: `app.py`
- Test: `tests/test_system_integration.py`

- [ ] Update `MedAgentXSystem.query_multimodal` to accept raw files, handle DICOM/PNG, execute ingestion QC, and route via the new fail-closed state machine.
- [ ] Update `app.py`:
  - Display `VisionStatus` pills (OK = green, TIMEOUT = red, LOW_QUALITY = orange).
  - Render bounding boxes and visual heatmap references if available.
  - Show clear abstain/escalate alert banners when vision fails or critical findings are detected.
- [ ] Run `pytest tests/test_system_integration.py` and confirm pass.
- [ ] Commit: `git commit -m "feat(ui): update system orchestrator and clinician dashboard with fail-closed observability"`

---

### Task 10: Regression Tests & Evaluation Harness
**Files:**
- Create: `tests/test_vision_refactor.py`
- Create: `scripts/eval_vision_cdss.py`

- [ ] Implement regression test `test_regression_vision_timeout_abtains` proving the fail-open bug cannot recur.
- [ ] Implement test cases for corrupted images, wrong body part, and critical pathologies.
- [ ] Create `scripts/eval_vision_cdss.py` to evaluate ECE, hallucination rate, and fail-closed compliance across the 122-case benchmark.
- [ ] Run full test suite: `pytest -v` (assert 100% pass rate).
- [ ] Commit: `git commit -m "test: add comprehensive fail-closed regression tests and evaluation harness"`

