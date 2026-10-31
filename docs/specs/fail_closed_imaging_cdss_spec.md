# Architectural Specification: Fail-Closed Medical Imaging CDSS & Grounded Perception Pipeline

**Author:** MedAgentX Core CDS Engineering Team  
**Status:** DRAFT (Under Review)  
**Target Milestone:** 2.0.0 Refactor  
**Tracking Issue / Incident:** Post-mortem: Vision timeout fail-open hallucinated normal exam  

---

## 1. Executive Summary & Root Cause Analysis

### 1.1 The Incident
During clinical testing with an uploaded Chest X-Ray exhibiting marked right middle lobe consolidation (`consolidation-rml.jpg`), the vision perception inference timed out (`HTTPConnectionPool ... Read timed out. (read timeout=60.0)`). 

Instead of failing closed, the legacy analyzer caught the exception and defaulted to:
```text
Impression: Diagnostic Chest Examination without Acute Pathological Findings
Findings: Normal cardiothoracic anatomy ... Clear lung fields
```
Downstream, Agent 2 retrieved evidence for this fabricated entity, Agent 3 assigned default confidence scores ($V_{agree}=0.85, \Phi=1.0$), and the ACMR router emitted `STOP_CONFIDENT` with $C_{total}=0.905$. An acute inpatient pathology was falsely stamped as "Normal" with high clinical confidence.

### 1.2 Systemic Defects Identified
1. **Permissive Fallback (Fail-Open)**: Catching model exceptions and returning a placeholder "Normal" examination violated core medical device software safety standards (IEC 62304 Class C).
2. **Missing Status State Machine**: The pipeline had no schema distinguishing `OK` perception from `TIMEOUT`, `FAILED`, `LOW_QUALITY`, or `UNSUPPORTED`.
3. **No Router Hard-Gate**: The ACMR state machine router evaluated mathematical confidence without verifying if requisite modalities actually succeeded.
4. **Synthetic Confidence Defaulting**: Verifier and confidence calculators filled missing or uncomputable metrics with `1.0` instead of propagating penalties or `None`.
5. **Monolithic Naive Queries**: Retrieval generated a single concatenated string rather than decomposing structured visual findings into ontology-grounded queries.

---

## 2. Core Architecture & Architectural Invariants

### 2.1 Non-Negotiable Invariants
- **Invariant I1 (Strict Fail-Closed)**: If imaging was submitted or required for the case and `vision_result.status != VisionStatus.OK`, the router is **strictly prohibited** from emitting `STOP_CONFIDENT`. The decision must be `ABSTAIN_INSUFFICIENT_EVIDENCE` or `ESCALATE_TO_HUMAN`.
- **Invariant I2 (Zero Placeholder Findings)**: The phrase "no acute findings" or any normal assertion is **strictly prohibited** unless the vision model explicitly checks and scores pertinent negatives with evidence.
- **Invariant I3 (Evidence Provenance)**: Every clinical claim in the final report must trace to an explicit visual evidence object (slice, bounding box, Grad-CAM heatmap path) or a retrieved consensus guideline chunk ID. No uncited assertions are permitted.
- **Invariant I4 (PHI De-Identification)**: All DICOM metadata must undergo automated de-identification before any downstream VLM or LLM call.

---

## 3. Data Contracts & Pydantic Schemas

### 3.1 Vision Status Enum
```python
class VisionStatus(str, Enum):
    OK = "OK"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    LOW_QUALITY = "LOW_QUALITY"
    UNSUPPORTED = "UNSUPPORTED"
```

### 3.2 Finding & Evidence Schemas
```python
class VisualEvidence(BaseModel):
    bbox: list[float] | None = Field(default=None, description="[ymin, xmin, ymax, xmax] normalized 0-1")
    slice_index: int | None = Field(default=None, description="Keyframe slice index for 3D CT/MRI volumes")
    heatmap_path: str | None = Field(default=None, description="Path to saved Grad-CAM / attention overlay")
    supporting_visual_description: str = Field(description="Clinical description of visible density, margins, and texture")

class VisualFinding(BaseModel):
    finding_id: str = Field(default_factory=lambda: f"VF-{uuid.uuid4().hex[:8]}")
    label: str = Field(description="Clinical label (e.g., Lobar Consolidation, Tension Pneumothorax)")
    coding_system: str = Field(default="RadLex", description="RadLex, SNOMED-CT, or UMLS")
    code: str = Field(description="Ontology concept identifier, e.g., RID49488")
    anatomical_location: str = Field(description="Anatomical location, e.g., Right Middle Lobe")
    laterality: Literal["right", "left", "bilateral", "midline", "unspecified"] = "unspecified"
    size_or_extent: str | None = Field(default=None, description="e.g., 3.2 x 2.8 cm, segmental, diffuse")
    probability: float = Field(ge=0.0, le=1.0, description="Calibrated pathology probability")
    uncertainty: float = Field(ge=0.0, le=1.0, description="Epistemic uncertainty / ensemble variance")
    is_critical: bool = Field(default=False, description="Urgent finding requiring immediate human notification")
    evidence: VisualEvidence

class NegativeFinding(BaseModel):
    label: str = Field(description="Clinical entity evaluated, e.g., Pneumothorax, Pleural Effusion")
    code: str = Field(description="Ontology identifier")
    probability: float = Field(ge=0.0, le=1.0, description="Probability of absence")
    checked: bool = Field(default=True, description="Explicitly verified negative finding")
```

### 3.3 Quality Assessment & Vision Result
```python
class QualityAssessment(BaseModel):
    is_acceptable: bool
    orientation_valid: bool
    exposure_quality: Literal["optimal", "underexposed", "overexposed"]
    motion_artifact_detected: bool
    anatomy_matches_modality: bool
    rejection_reasons: list[str] = Field(default_factory=list)

class VisionResult(BaseModel):
    trace_id: str = Field(default_factory=lambda: f"TR-{uuid.uuid4().hex[:8]}")
    status: VisionStatus
    modality: str
    body_part: str
    view: str = Field(default="PA", description="PA, AP, Lateral, Axial, Coronal, Sagittal")
    quality: QualityAssessment
    findings: list[VisualFinding] = Field(default_factory=list)
    negative_findings: list[NegativeFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    model_name: str
    model_version: str
    inference_duration_seconds: float
```

---

## 4. Ingestion, Quality Control & Medical Volume Handling

### 4.1 Ingestion Pipeline (`src/multimodal/ingestion.py`)
- **DICOM Handling (`pydicom`)**:
  - Automatically scrubs PHI tags: `(0010, 0010) PatientName`, `(0010, 0020) PatientID`, `(0010, 0030) PatientBirthDate`, `(0008, 0080) InstitutionName`, `(0008, 0090) ReferringPhysicianName`.
  - Preserves clinical image tags: `(0008, 0060) Modality`, `(0018, 0015) BodyPartExamined`, `(0018, 5101) ViewPosition`, `(0028, 0030) PixelSpacing`, `(0028, 1050) WindowCenter`, `(0028, 1051) WindowWidth`.
  - Converts pixel arrays using standard Hounsfield Unit (HU) rescaling: $HU = \text{pixel} \times \text{RescaleSlope} + \text{RescaleIntercept}$.
- **NIfTI 3D Volume Handling (`nibabel`)**:
  - Ingests 3D CT/MRI volumes.
  - Generates multi-planar reconstructions (MPR): identifies key informative slices by intensity histogram gradient and axial spacing.
  - Prohibits single-slice guessing on 3D volumes.
- **Image Quality Checks (QC Gate)**:
  - Validates minimum resolution ($> 256\times 256$).
  - Detects extreme underexposure/overexposure via histogram quantile analysis ($p_{10} < 5$ or $p_{90} > 250$).
  - Detects severe motion blur via Laplacian variance threshold ($\text{Var}(\Delta I) < 50$).
  - If QC fails: returns `VisionResult(status=VisionStatus.LOW_QUALITY, quality=QualityAssessment(is_acceptable=False, ...))` without invoking heavy VLM models.

---

## 5. Agent 1: Pluggable VLM & Classifier Ensemble

### 5.1 Architecture & Interface
```python
class BaseVisionModel(ABC):
    @abstractmethod
    def infer(self, image_data: bytes | np.ndarray, metadata: dict[str, Any]) -> VisionResult:
        pass
```

### 5.2 Resilience & Exponential Backoff Policy
- **Primary Model**: Local or Hosted VLM (`qwen3-vl:4b` / `medgemma` / `chexagent`).
- **Secondary Model / Fallback Classifier**: Medical CNN / ViT (`TorchXRayVision` ensemble, DenseNet-121-all, or MONAI volume classifier).
- **Execution Policy**:
  - Maximum 3 retry attempts with exponential backoff ($2s, 4s, 8s$).
  - Per-attempt configurable timeout (default 180s for local CPU execution).
  - If Primary fails after retries: automatically invoke Secondary Model.
  - If Secondary fails: return `VisionResult(status=VisionStatus.TIMEOUT, findings=[])` or `VisionStatus.FAILED`.
  - **Zero Fabrication**: Never return a synthetic "Normal" study on failure.

---

## 6. Agent 2: Finding-Level Grounded Retrieval

### 6.1 Query Formulation
Rather than passing an unstructured clinical string, Agent 2 generates:
1. **Per-Finding Queries**: For each finding in `vision_result.findings`:
   - Query format: `"{label} in {anatomical_location} {laterality} on {modality} diagnostic criteria and clinical management"`
2. **Combined Clinical Synthesis Query**:
   - Query format: `"{modality} showing {findings_summary} in patient with {patient_chief_complaint}"`

### 6.2 Hybrid Vector + Graph Retrieval
- **Knowledge Graph (Neo4j)**:
  - Traverses `(Finding)-[:INDICATES]->(DifferentialDiagnosis)-[:REQUIRES_WORKUP]->(DiagnosticTest)`
  - Traverses `(Finding)-[:CONTRAINDICATES]->(Intervention)`
- **Vector Store (Qdrant)**:
  - Embedded guideline chunks carrying strict metadata: `source_id`, `publication_year`, `guideline_body` (ACR, Fleischner, GOLD, Surviving Sepsis, ATS/IDSA), `section`, and `evidence_grade`.
- **Minimum Evidence Constraint**: Require at least 2 independent citations per active diagnostic claim.

---

## 7. Agent 3 & ACMR Router: Calibrated Verifier & Hard Gating

### 7.1 Router State Machine Actions
```python
class ACMRAction(str, Enum):
    STOP_CONFIDENT = "STOP_CONFIDENT"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    REQUEST_MORE_INPUT = "REQUEST_MORE_INPUT"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"
    ABSTAIN_INSUFFICIENT_EVIDENCE = "ABSTAIN_INSUFFICIENT_EVIDENCE"
```

### 7.2 Router Hard-Gate Formulation
```python
def route(self, state: ACMRState) -> RouterDecision:
    # Hard Gate 1: Required modality failure must fail closed
    if state.requires_imaging and state.vision_result.status != VisionStatus.OK:
        return RouterDecision(
            action=ACMRAction.ABSTAIN_INSUFFICIENT_EVIDENCE,
            reason=f"Imaging perception incomplete (status={state.vision_result.status}). Automated normal conclusion forbidden."
        )

    # Hard Gate 2: Critical finding requires human escalation
    if any(f.is_critical for f in state.vision_result.findings):
        return RouterDecision(
            action=ACMRAction.ESCALATE_TO_HUMAN,
            reason=f"Critical acute pathology detected ({', '.join(f.label for f in state.vision_result.findings if f.is_critical)})."
        )

    # Hard Gate 3: Low Quality imaging requires new acquisition
    if state.vision_result.status == VisionStatus.LOW_QUALITY:
        return RouterDecision(
            action=ACMRAction.REQUEST_MORE_INPUT,
            reason=f"Study rejected due to image quality limitations: {', '.join(state.vision_result.quality.rejection_reasons)}."
        )
```

### 7.3 Mathematical Composite Confidence ($C_{total}$) Penalization
Composite confidence must not assume $1.0$ completeness or agreement when metrics cannot be computed:
$$C_{total} = w_v (S_{vec} \cdot S_{rel}) + w_g (S_{graph} \cdot S_{rel}) + w_a V_{agree} + w_c \Phi_{comp} + w_n \Omega_{nov} - w_r R_{contra} - \Pi_{penalty}$$

Where the penalty term $\Pi_{penalty}$ is:
$$\Pi_{penalty} = 0.25 \cdot (1 - \text{is\_vision\_ok}) + 0.15 \cdot \sigma_{vision} + 0.10 \cdot \mathbb{I}(\text{evidence\_count} < 2)$$

---

## 8. Agent 4: Structured Radiology CDSS Report Generator

The Reasoner produces reports adhering to formal structured radiology templates:

```text
================================================================================
CLINICAL DECISION SUPPORT REPORT — RADIOLOGICAL CONSULTATION
================================================================================
EXAMINATION: [Modality, Projection, Technique]
INDICATION: [Patient Age, Sex, Chief Complaint, HPI]
COMPARISON: [Prior Study Date / None Available]
IMAGE QUALITY: [Optimal / Suboptimal with explanation]

FINDINGS:
  • CHEST WALL & PLEURA: [Finding or verified negative with evidence citation]
  • LUNGS & AIRWAYS: [Finding: e.g. Dense lobar consolidation in RML with air bronchograms (Evidence: Ref-VF01, heatmap: overlays/rml_gradcam.png)]
  • CARDIOMEDIASTINAL SILHOUETTE: [Normal CTR < 0.50]

PERTINENT NEGATIVES:
  [x] Pneumothorax (Probability of absence: 0.98, Negative Evidence Checked)
  [x] Large Pleural Effusion (Probability of absence: 0.95, Negative Evidence Checked)

IMPRESSION:
  1. Acute Right Middle Lobe Consolidation, most consistent with Community-Acquired Lobar Pneumonia (Calibrated Probability: 0.91 [0.85-0.96]).
  2. Differential consideration: Aspiration pneumonitis vs obstructive atelectasis.

EVIDENCE-GROUNDED RECOMMENDATIONS:
  1. Empiric Antimicrobial Therapy: Initiate inpatient CAP regimen per 2019 ATS/IDSA Guidelines (Ceftriaxone + Azithromycin). [Guideline: ATS/IDSA 2019, Section 4]
  2. Sputum Gram stain/culture and urinary antigen testing.

SAFETY CONTRAINDICATIONS & WITHHELD INTERVENTIONS:
  • No acute physiological contraindications detected.

DECISION SUPPORT DISCLAIMER:
  ⚠️ This report was generated by the MedAgentX Clinical Decision Support System. 
  It is intended exclusively for clinician augmentation and requires direct review 
  and validation by a board-certified radiologist or attending physician.
================================================================================
```

---

## 9. Test Suite, Verification & Evaluation Harness

### 9.1 Unit & Integration Test Plan (`tests/test_vision_refactor.py`)
1. **Test `test_regression_vision_timeout_abtains`**:
   - Mock a vision timeout (`VisionStatus.TIMEOUT`).
   - Run orchestrator.
   - Assert: Router action is `ABSTAIN_INSUFFICIENT_EVIDENCE`.
   - Assert: `findings` list is empty.
   - Assert: No fabricated "Normal" exam appears anywhere in the report.
2. **Test `test_corrupt_and_low_quality_rejection`**:
   - Provide motion-blurred and dark images.
   - Assert: `VisionStatus.LOW_QUALITY`, action is `REQUEST_MORE_INPUT`.
3. **Test `test_dicom_anonymization`**:
   - Feed synthetic DICOM with patient name "John Doe".
   - Assert: Output metadata strips "John Doe" and preserves pixel spacing / windowing.
4. **Test `test_pathology_grounded_citations`**:
   - Provide true consolidation image.
   - Assert: `VisualFinding` generated, CAP guidelines retrieved, evidence heatmap referenced.

### 9.2 Evaluation Runner (`scripts/eval_vision_cdss.py`)
- Evaluates 122 curated VQA-RAD cases:
  - Expected Calibration Error (ECE)
  - Hallucination Rate (claims lacking visual bounding boxes or retrieved IDs)
  - Sensitivity / Specificity on critical findings (Pneumothorax, ICH, Consolidation)
  - 100% adherence to Fail-Closed constraints.

---

## 10. Deliverables & File Layout

| Module / Component | Target File Path | Primary Function |
| :--- | :--- | :--- |
| **Contracts** | `src/multimodal/contracts.py` | `VisionResult`, `VisualFinding`, `VisionStatus`, `QualityAssessment` |
| **Ingestion & QC** | `src/multimodal/ingestion.py` | DICOM de-identification, NIfTI slicing, quality gate |
| **Vision Engine** | `src/multimodal/vision_engine.py` | Pluggable models, backoff retry, secondary fallback |
| **Analyzer Adapter** | `src/multimodal/imaging_analyzer.py` | Refactored fail-closed analyzer replacing legacy code |
| **ACMR Router** | `src/orchestrator/router.py` | Hard-gating, `ABSTAIN_INSUFFICIENT_EVIDENCE`, `ESCALATE_TO_HUMAN` |
| **Confidence Math**| `src/orchestrator/confidence.py` | Penalty formulation, real data computation |
| **Retriever** | `src/agents/retriever.py` | Multi-finding query formulation and provenance tracking |
| **Verifier** | `src/agents/verifier.py` | Cross-modal contradiction checks and critical alert flags |
| **Reasoner** | `src/agents/reasoner.py` | Structured radiology report generator + multi-format export |
| **Tests** | `tests/test_vision_refactor.py` | Regression test for fail-open bug, timeout, and DICOM tests |
| **Evaluation** | `scripts/eval_vision_cdss.py` | Benchmark runner for calibration and hallucination rates |
| **Web UI** | `app.py` | Glassmorphism dashboard reflecting fail-closed statuses and overlays |

