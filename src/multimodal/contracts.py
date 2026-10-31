"""Multimodal Medical Perception Contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
import uuid
from pydantic import BaseModel, Field

from src.orchestrator.patient import PatientCase


class VisionStatus(str, Enum):
    """Execution status for medical vision inference."""

    OK = "OK"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    LOW_QUALITY = "LOW_QUALITY"
    UNSUPPORTED = "UNSUPPORTED"


class VisualEvidence(BaseModel):
    """Spatial and volumetric localization supporting a radiological finding."""

    bbox: list[float] | None = Field(default=None, description="[ymin, xmin, ymax, xmax] normalized 0-1")
    slice_index: int | None = Field(default=None, description="Keyframe slice index for 3D CT/MRI volumes")
    heatmap_path: str | None = Field(default=None, description="Path to saved Grad-CAM / attention overlay")
    supporting_visual_description: str = Field(
        default="", description="Clinical description of visible density, margins, and texture"
    )


class VisualFinding(BaseModel):
    """Standardized finding extracted from diagnostic medical imagery."""

    finding_id: str = Field(default_factory=lambda: f"VF-{uuid.uuid4().hex[:8]}")
    label: str = Field(description="Clinical label (e.g., Lobar Consolidation, Tension Pneumothorax)")
    coding_system: str = Field(default="RadLex", description="RadLex, SNOMED-CT, or UMLS")
    code: str = Field(description="Ontology concept identifier, e.g., RID49488")
    anatomical_location: str = Field(description="Anatomical location, e.g., Right Middle Lobe")
    laterality: Literal["right", "left", "bilateral", "midline", "unspecified"] | str = "unspecified"
    size_or_extent: str | None = Field(default=None, description="e.g., 3.2 x 2.8 cm, segmental, diffuse")
    probability: float = Field(ge=0.0, le=1.0, description="Calibrated pathology probability")
    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0, description="Epistemic uncertainty / ensemble variance")
    is_critical: bool = Field(default=False, description="Urgent finding requiring immediate human notification")
    evidence: VisualEvidence = Field(default_factory=VisualEvidence)


class NegativeFinding(BaseModel):
    """Explicitly verified absence of a key pathological feature (pertinent negative)."""

    label: str = Field(description="Clinical entity evaluated, e.g., Pneumothorax, Pleural Effusion")
    code: str = Field(description="Ontology identifier")
    probability: float = Field(ge=0.0, le=1.0, description="Probability of absence")
    checked: bool = Field(default=True, description="Explicitly verified negative finding")


class QualityAssessment(BaseModel):
    """Quality control metrics and diagnostic adequacy evaluation."""

    is_acceptable: bool = Field(description="Whether the image meets minimal diagnostic standards")
    orientation_valid: bool = Field(default=True, description="Whether image orientation is standard")
    exposure_quality: Literal["optimal", "underexposed", "overexposed"] | str = Field(
        default="optimal", description="Exposure assessment"
    )
    motion_artifact_detected: bool = Field(default=False, description="Whether motion blur is detected")
    anatomy_matches_modality: bool = Field(default=True, description="Whether anatomy aligns with study modality")
    rejection_reasons: list[str] = Field(default_factory=list, description="Reasons for rejection if is_acceptable=False")


class VisionResult(BaseModel):
    """Complete perception bundle for a medical imaging study."""

    trace_id: str = Field(default_factory=lambda: f"TR-{uuid.uuid4().hex[:8]}")
    status: VisionStatus = Field(description="Outcome status of the vision pipeline")
    modality: str = Field(default="Chest X-Ray")
    body_part: str = Field(default="Chest")
    view: str = Field(default="PA", description="PA, AP, Lateral, Axial, Coronal, Sagittal")
    quality: QualityAssessment = Field(default_factory=lambda: QualityAssessment(is_acceptable=True))
    findings: list[VisualFinding] = Field(default_factory=list)
    negative_findings: list[NegativeFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    model_name: str = Field(default="", description="Name of the vision model or ensemble used")
    model_version: str = Field(default="", description="Version identifier of the vision model")
    inference_duration_seconds: float = Field(default=0.0, description="End-to-end inference execution time")


class DocumentExtractionResult(BaseModel):
    """Structured extraction result from uploaded medical documents."""

    document_text: str = ""
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImagingAnalysis(BaseModel):
    """Structured perception findings from diagnostic imaging analysis (legacy adapter)."""

    modality: str = "Chest X-Ray"  # "Chest X-Ray", "12-Lead ECG", "CT Scan", "Ultrasound"
    radiological_findings: list[str] = Field(default_factory=list)
    diagnostic_impression: str = ""
    critical_urgency: str = "ROUTINE"  # "ROUTINE", "URGENT", "EMERGENT"
    contraindicated_interventions: list[str] = Field(default_factory=list)
    confidence: float = 0.90
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultimodalPatientCase(PatientCase):
    """Extended patient case combining clinical text, lab records, and diagnostic scans."""

    imaging_analysis: ImagingAnalysis | None = None
    uploaded_document_text: str = ""
    uploaded_file_name: str = ""
