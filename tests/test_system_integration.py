"""End-to-End System Integration Tests for MedAgentX Multimodal Pipeline."""

from unittest.mock import patch
import pytest

from src.multimodal.contracts import (
    QualityAssessment,
    VisionResult,
    VisionStatus,
    VisualEvidence,
    VisualFinding,
)
from src.orchestrator.system import MedAgentXSystem


@pytest.fixture
def system():
    return MedAgentXSystem()


def test_end_to_end_multimodal_vision_timeout(system):
    """Scenario 1: Vision timeout triggers fail-closed ABSTAIN_INSUFFICIENT_EVIDENCE with zero fabricated findings."""
    timeout_result = VisionResult(
        status=VisionStatus.TIMEOUT,
        modality="Chest X-Ray",
        body_part="Chest",
        quality=QualityAssessment(is_acceptable=False, rejection_reasons=["Inference read timeout"]),
        findings=[],
        negative_findings=[],
        limitations=["Vision model timed out after 3 retries."],
    )

    with patch("src.multimodal.imaging_analyzer.MedicalImagingAnalyzer.analyze_vision", return_value=timeout_result):
        res = system.query_multimodal(
            patient_text="62-year-old male with productive cough, dyspnea, and fever.",
            image_input="data/samples/sample_cxr_pneumonia.png",
        )

    assert res.metadata.get("route_decision") == "ABSTAIN_INSUFFICIENT_EVIDENCE"
    assert res.metadata.get("vision_status") == "TIMEOUT"
    # Confidence must reflect penalty (score should be penalized below high threshold)
    assert res.confidence < 0.75

    # Report checks: Must contain fail-closed audit notice and zero fabricated normal findings
    report_lower = res.answer.lower()
    assert "timeout" in report_lower
    assert "manual radiologist" in report_lower or "aborted" in report_lower or "insufficient" in report_lower
    assert "no acute findings" not in report_lower
    assert "clear lung fields" not in report_lower


def test_end_to_end_multimodal_low_quality_image(system):
    """Scenario 2: Low-quality image triggers REQUEST_MORE_INPUT for re-acquisition."""
    low_quality_result = VisionResult(
        status=VisionStatus.LOW_QUALITY,
        modality="Chest X-Ray",
        body_part="Chest",
        quality=QualityAssessment(
            is_acceptable=False,
            rejection_reasons=["Severe motion blur (Laplacian variance 12.4 < 100.0)"],
        ),
        findings=[],
        negative_findings=[],
        limitations=["Suboptimal study: Severe motion blur"],
    )

    with patch("src.multimodal.imaging_analyzer.MedicalImagingAnalyzer.analyze_vision", return_value=low_quality_result):
        res = system.query_multimodal(
            patient_text="45-year-old female with pleuritic chest pain.",
            image_input="data/samples/sample_cxr_pneumonia.png",
        )

    assert res.metadata.get("route_decision") == "REQUEST_MORE_INPUT"
    assert res.metadata.get("vision_status") == "LOW_QUALITY"
    assert "quality" in res.metadata
    assert "Severe motion blur" in str(res.metadata["quality"].get("rejection_reasons", []))

    report_lower = res.answer.lower()
    assert "suboptimal" in report_lower or "repeat" in report_lower or "re-acquisition" in report_lower


def test_end_to_end_multimodal_critical_pneumothorax(system):
    """Scenario 3: Critical pneumothorax detection triggers ESCALATE_TO_HUMAN and emergency procedures."""
    ptx_finding = VisualFinding(
        label="Tension Pneumothorax",
        coding_system="RadLex",
        code="RID50041",
        anatomical_location="Right pleural space",
        laterality="right",
        probability=0.96,
        uncertainty=0.04,
        is_critical=True,
        evidence=VisualEvidence(
            bbox=[0.05, 0.5, 0.45, 0.95],
            supporting_visual_description="Visceral pleural line with absent lung markings and contralateral mediastinal shift.",
        ),
    )
    ptx_result = VisionResult(
        status=VisionStatus.OK,
        modality="Chest X-Ray",
        body_part="Chest",
        view="PA",
        quality=QualityAssessment(is_acceptable=True),
        findings=[ptx_finding],
        negative_findings=[],
    )

    with patch("src.multimodal.imaging_analyzer.MedicalImagingAnalyzer.analyze_vision", return_value=ptx_result):
        res = system.query_multimodal(
            patient_text="34-year-old male post blunt chest trauma with acute tachycardia, hypotension, and severe dyspnea.",
            image_input="data/samples/sample_cxr_pneumothorax.png",
        )

    assert res.metadata.get("route_decision") == "ESCALATE_TO_HUMAN"
    assert res.metadata.get("vision_status") == "OK"
    assert res.metadata.get("is_critical_pathology") is True

    # Report should include emergency decompression / chest tube guidance
    report_upper = res.answer.upper()
    assert "PNEUMOTHORAX" in report_upper
    assert "CHEST TUBE" in report_upper or "DECOMPRESSION" in report_upper or "THORACOSTOMY" in report_upper


def test_end_to_end_multimodal_valid_consolidation_cxr(system):
    """Scenario 4: Valid consolidation Chest X-Ray routes to STOP_CONFIDENT with guideline-grounded CAP report."""
    consolidation_finding = VisualFinding(
        label="Lobar Consolidation",
        coding_system="RadLex",
        code="RID49488",
        anatomical_location="Right Lower Lobe",
        laterality="right",
        probability=0.92,
        uncertainty=0.08,
        is_critical=False,
        evidence=VisualEvidence(
            bbox=[0.45, 0.50, 0.85, 0.90],
            supporting_visual_description="Dense alveolar opacity with air bronchograms in RLL.",
        ),
    )
    ok_result = VisionResult(
        status=VisionStatus.OK,
        modality="Chest X-Ray",
        body_part="Chest",
        view="PA",
        quality=QualityAssessment(is_acceptable=True),
        findings=[consolidation_finding],
        negative_findings=[],
    )

    with patch("src.multimodal.imaging_analyzer.MedicalImagingAnalyzer.analyze_vision", return_value=ok_result):
        res = system.query_multimodal(
            patient_text="54-year-old male with 4-day history of high fever, productive rust-colored sputum, and right pleuritic pain.",
            image_input="data/samples/sample_cxr_pneumonia.png",
        )

    assert res.metadata.get("route_decision") == "STOP_CONFIDENT"
    assert res.metadata.get("vision_status") == "OK"
    assert res.confidence >= 0.80

    # Guideline grounded CAP recommendations
    report_upper = res.answer.upper()
    assert "PNEUMONIA" in report_upper or "CAP" in report_upper or "CONSOLIDATION" in report_upper
    assert "ANTIMICROBIAL" in report_upper or "CEFTRIAXONE" in report_upper or "AZITHROMYCIN" in report_upper

    # Check exporters metadata
    assert "json_report" in res.metadata
    assert "html_report" in res.metadata
    assert isinstance(res.metadata["json_report"], str)
    assert "Lobar Consolidation" in res.metadata["json_report"]
    assert "<!DOCTYPE html>" in res.metadata["html_report"] or "<html" in res.metadata["html_report"]
