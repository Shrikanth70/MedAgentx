"""Unit tests for multimodal vision contracts and fail-closed schemas."""

import json
import pytest
from pydantic import ValidationError

from src.multimodal.contracts import (
    VisionStatus,
    VisualEvidence,
    VisualFinding,
    NegativeFinding,
    QualityAssessment,
    VisionResult,
    ImagingAnalysis,
    MultimodalPatientCase,
    DocumentExtractionResult,
)


def test_vision_status_enum_values():
    """Verify all required VisionStatus enum values are present."""
    assert VisionStatus.OK == "OK"
    assert VisionStatus.FAILED == "FAILED"
    assert VisionStatus.TIMEOUT == "TIMEOUT"
    assert VisionStatus.LOW_QUALITY == "LOW_QUALITY"
    assert VisionStatus.UNSUPPORTED == "UNSUPPORTED"
    assert len(VisionStatus) == 5


def test_visual_evidence_defaults_and_custom():
    """Verify VisualEvidence fields and default values."""
    evidence = VisualEvidence()
    assert evidence.bbox is None
    assert evidence.slice_index is None
    assert evidence.heatmap_path is None
    assert evidence.supporting_visual_description == ""

    custom_evidence = VisualEvidence(
        bbox=[0.1, 0.2, 0.5, 0.6],
        slice_index=42,
        heatmap_path="/tmp/heatmap_01.png",
        supporting_visual_description="Dense opacification in right middle lobe",
    )
    assert custom_evidence.bbox == [0.1, 0.2, 0.5, 0.6]
    assert custom_evidence.slice_index == 42
    assert custom_evidence.heatmap_path == "/tmp/heatmap_01.png"
    assert custom_evidence.supporting_visual_description == "Dense opacification in right middle lobe"


def test_visual_finding_validation():
    """Verify VisualFinding required fields, defaults, and probability bounds."""
    evidence = VisualEvidence(bbox=[0.2, 0.3, 0.6, 0.7])
    finding = VisualFinding(
        label="Lobar Consolidation",
        code="RID49488",
        anatomical_location="Right Middle Lobe",
        probability=0.92,
        uncertainty=0.08,
        evidence=evidence,
    )

    assert finding.finding_id.startswith("VF-")
    assert finding.coding_system == "RadLex"
    assert finding.laterality == "unspecified"
    assert finding.is_critical is False
    assert finding.probability == 0.92
    assert finding.uncertainty == 0.08

    # Probability bounds
    with pytest.raises(ValidationError):
        VisualFinding(
            label="Invalid",
            code="RID000",
            anatomical_location="Lung",
            probability=1.5,
            uncertainty=0.1,
            evidence=evidence,
        )

    with pytest.raises(ValidationError):
        VisualFinding(
            label="Invalid",
            code="RID000",
            anatomical_location="Lung",
            probability=-0.1,
            uncertainty=0.1,
            evidence=evidence,
        )

    # Uncertainty bounds
    with pytest.raises(ValidationError):
        VisualFinding(
            label="Invalid",
            code="RID000",
            anatomical_location="Lung",
            probability=0.5,
            uncertainty=1.2,
            evidence=evidence,
        )


def test_negative_finding_validation():
    """Verify NegativeFinding fields and probability bounds."""
    nf = NegativeFinding(
        label="Pneumothorax",
        code="RID5342",
        probability=0.98,
    )
    assert nf.checked is True
    assert nf.probability == 0.98

    with pytest.raises(ValidationError):
        NegativeFinding(
            label="Pneumothorax",
            code="RID5342",
            probability=-0.05,
        )

    with pytest.raises(ValidationError):
        NegativeFinding(
            label="Pneumothorax",
            code="RID5342",
            probability=1.05,
        )


def test_quality_assessment_defaults_and_rejection():
    """Verify QualityAssessment fields, defaults, and rejection recording."""
    qc_pass = QualityAssessment(is_acceptable=True)
    assert qc_pass.orientation_valid is True
    assert qc_pass.exposure_quality == "optimal"
    assert qc_pass.motion_artifact_detected is False
    assert qc_pass.anatomy_matches_modality is True
    assert qc_pass.rejection_reasons == []

    qc_fail = QualityAssessment(
        is_acceptable=False,
        exposure_quality="underexposed",
        motion_artifact_detected=True,
        rejection_reasons=["Severe respiratory motion blur", "Histogram underexposure"],
    )
    assert qc_fail.is_acceptable is False
    assert len(qc_fail.rejection_reasons) == 2


def test_vision_result_creation_and_serialization():
    """Verify VisionResult assembly and JSON serialization roundtrip."""
    qc = QualityAssessment(is_acceptable=True)
    evidence = VisualEvidence(
        bbox=[0.2, 0.3, 0.6, 0.7],
        supporting_visual_description="Opacification",
    )
    finding = VisualFinding(
        label="Lobar Consolidation",
        code="RID49488",
        anatomical_location="Right Middle Lobe",
        probability=0.88,
        uncertainty=0.12,
        is_critical=True,
        evidence=evidence,
    )
    negative = NegativeFinding(
        label="Pleural Effusion",
        code="RID4883",
        probability=0.95,
    )

    result = VisionResult(
        status=VisionStatus.OK,
        modality="Chest X-Ray",
        body_part="Chest",
        view="PA",
        quality=qc,
        findings=[finding],
        negative_findings=[negative],
        limitations=["AP portable exam with suboptimal lung expansion"],
        model_name="qwen3-vl:4b",
        model_version="1.0.0",
        inference_duration_seconds=1.23,
    )

    assert result.trace_id.startswith("TR-")
    assert result.status == VisionStatus.OK
    assert len(result.findings) == 1
    assert result.findings[0].is_critical is True

    # JSON serialization roundtrip
    dumped = result.model_dump()
    assert dumped["status"] == "OK"
    json_str = result.model_dump_json()
    loaded_data = json.loads(json_str)
    assert loaded_data["findings"][0]["label"] == "Lobar Consolidation"

    roundtrip = VisionResult.model_validate_json(json_str)
    assert roundtrip.trace_id == result.trace_id
    assert roundtrip.status == VisionStatus.OK
    assert roundtrip.findings[0].probability == 0.88


def test_backward_compatibility():
    """Verify legacy contracts remain functional."""
    legacy = ImagingAnalysis(
        modality="Chest X-Ray",
        radiological_findings=["Consolidation in RML"],
        diagnostic_impression="Right middle lobe pneumonia",
    )
    assert legacy.modality == "Chest X-Ray"
    assert legacy.confidence == 0.90

    case = MultimodalPatientCase(
        case_id="CASE-001",
        chief_complaint="Productive cough and fever",
        imaging_analysis=legacy,
    )
    assert case.imaging_analysis is not None
    assert case.imaging_analysis.diagnostic_impression == "Right middle lobe pneumonia"

    doc = DocumentExtractionResult(
        document_text="Discharge summary...",
        extracted_fields={"diagnosis": "Community-acquired pneumonia"},
    )
    assert doc.document_text == "Discharge summary..."
    assert doc.extracted_fields["diagnosis"] == "Community-acquired pneumonia"
