"""Tests for Agent 4: Structured Radiology CDSS Reasoner & Multi-Format Exporters."""

import json
import pytest
from src.agents.reasoner import ReasonerAgent, ReasonerOutput
from src.multimodal.contracts import (
    NegativeFinding,
    QualityAssessment,
    VisionResult,
    VisionStatus,
    VisualEvidence,
    VisualFinding,
)
from src.orchestrator.patient import PatientCase


@pytest.fixture
def reasoner():
    return ReasonerAgent()


@pytest.fixture
def mock_patient():
    return PatientCase(
        patient_id="PT-RAD-001",
        age=62,
        sex="M",
        chief_complaint="Severe right-sided pleuritic chest pain and fever",
        hpi="Patient presents with acute productive cough and dyspnea.",
        sbp=118,
        dbp=72,
        spo2=94,
    )


def test_fail_closed_vision_timeout(reasoner, mock_patient):
    """Test fail-closed reporting when vision inference times out."""
    vision_result = VisionResult(
        status=VisionStatus.TIMEOUT,
        modality="Chest X-Ray",
        body_part="Chest",
        quality=QualityAssessment(is_acceptable=False, rejection_reasons=["Inference read timeout"]),
    )

    out = reasoner.synthesize_patient_report(
        patient=mock_patient,
        verified_evidence=[],
        detected_contradictions=[],
        c_total=0.0,
        mode="ABSTAIN_INSUFFICIENT_EVIDENCE",
        safety_reason="Vision inference timed out; automated interpretation aborted.",
        vision_result=vision_result,
    )

    assert isinstance(out, ReasonerOutput)
    report = out.clinical_report

    # Section I: Triage
    assert "• Diagnostic workup incomplete — insufficient clinical or imaging evidence to safely confirm disposition." in report

    # Section II: Recommendations
    assert "1. MANUAL DIAGNOSTIC IMAGING REVIEW: Perform urgent radiologist review of acquired study or repeat imaging acquisition." in report

    # Section VI: Multimodal Scan Interpretation
    assert "VI. MULTIMODAL DIAGNOSTIC SCAN INTERPRETATION:" in report
    assert "Modality: Chest X-Ray | Status: TIMEOUT (Perception Incomplete / Failed) | AI Confidence: 0.00" in report
    assert "⚠️ CLINICAL AUDIT NOTICE: Image analysis could not be safely or reliably completed (Status: TIMEOUT)." in report
    assert "Manual radiologist evaluation required." in report
    assert "• None — Automated interpretation aborted due to technical timeout / failure." in report

    # Zero placeholder findings check
    assert "no acute findings" not in report.lower()
    assert "clear lung fields" not in report.lower()


def test_request_more_input_mode(reasoner, mock_patient):
    """Test reporting when image quality is suboptimal requesting repeat acquisition."""
    vision_result = VisionResult(
        status=VisionStatus.LOW_QUALITY,
        modality="Chest X-Ray",
        body_part="Chest",
        quality=QualityAssessment(
            is_acceptable=False,
            rejection_reasons=["Severe motion artifact detected", "Severe underexposure"],
        ),
    )

    out = reasoner.synthesize_patient_report(
        patient=mock_patient,
        verified_evidence=[],
        detected_contradictions=[],
        c_total=0.35,
        mode="REQUEST_MORE_INPUT",
        safety_reason="Severe motion artifact detected.",
        vision_result=vision_result,
    )

    report = out.clinical_report
    assert "• Suboptimal image acquisition — requires repeat or supplemental imaging." in report
    assert "1. REPEAT IMAGING ACQUISITION: Re-acquire study addressing quality limitations." in report


def test_escalate_to_human_critical_alert(reasoner, mock_patient):
    """Test human escalation report when critical pathology alert is detected."""
    critical_finding = VisualFinding(
        label="Tension Pneumothorax",
        code="RID49488",
        anatomical_location="Right Hemithorax",
        probability=0.96,
        is_critical=True,
        evidence=VisualEvidence(
            heatmap_path="overlays/ptx_cam.png",
            supporting_visual_description="Visceral pleural line with absent peripheral lung markings",
        ),
    )
    vision_result = VisionResult(
        status=VisionStatus.OK,
        modality="Chest X-Ray",
        findings=[critical_finding],
    )

    critical_alerts = ["CRITICAL: Tension Pneumothorax detected requiring immediate bedside decompression."]

    out = reasoner.synthesize_patient_report(
        patient=mock_patient,
        verified_evidence=[],
        detected_contradictions=[],
        c_total=0.91,
        mode="ESCALATE_TO_HUMAN",
        vision_result=vision_result,
        visual_findings=[critical_finding],
        critical_alerts=critical_alerts,
    )

    report = out.clinical_report
    # Section I: Critical pathology alert
    assert "• 🚨 CRITICAL PATHOLOGY ALERT: Emergent physician bedside evaluation required immediately." in report
    # Section III: Critical alert prepended
    assert "CRITICAL: Tension Pneumothorax detected requiring immediate bedside decompression." in report


def test_structured_radiology_layout_ok_status(reasoner, mock_patient):
    """Test structured radiology findings formatting with heatmap/bbox and pertinent negatives."""
    finding = VisualFinding(
        label="Lobar Consolidation",
        code="RID49488",
        anatomical_location="Right Middle Lobe",
        probability=0.92,
        evidence=VisualEvidence(
            bbox=[0.3, 0.4, 0.6, 0.7],
            heatmap_path="overlays/rml_consolidation_cam.png",
            supporting_visual_description="Dense alveolar consolidation with visible air bronchograms",
        ),
    )
    neg_finding = NegativeFinding(
        label="Pneumothorax",
        code="RID5324",
        probability=0.98,
        checked=True,
    )
    vision_result = VisionResult(
        status=VisionStatus.OK,
        modality="Chest X-Ray",
        findings=[finding],
        negative_findings=[neg_finding],
    )

    out = reasoner.synthesize_patient_report(
        patient=mock_patient,
        verified_evidence=[{"source": "ATS/IDSA 2019", "content": "CAP guideline recommendation"}],
        detected_contradictions=[],
        c_total=0.93,
        mode="NORMAL",
        vision_result=vision_result,
    )

    report = out.clinical_report

    # Section VI structured findings format
    expected_finding_fragment = "• Lobar Consolidation: Right Middle Lobe (Probability: 0.92, Evidence: overlays/rml_consolidation_cam.png)"
    assert expected_finding_fragment in report

    # Pertinent negatives format
    assert "• Pertinent Negatives Evaluated: [x] Pneumothorax (Checked, Prob: 0.98)" in report

    # Decision Support Disclaimer
    disclaimer = (
        "⚠️ DECISION SUPPORT DISCLAIMER: This report was generated by the MedAgentX Clinical Decision Support System. "
        "It is intended exclusively for clinician augmentation and requires direct review and validation by a board-certified radiologist or attending physician."
    )
    assert disclaimer in report


def test_export_report_json(reasoner, mock_patient):
    """Test JSON exporter returning structured dictionary contract."""
    finding = VisualFinding(
        label="Lobar Consolidation",
        code="RID49488",
        anatomical_location="Right Middle Lobe",
        probability=0.92,
        evidence=VisualEvidence(
            heatmap_path="overlays/rml_consolidation_cam.png",
            supporting_visual_description="Dense alveolar consolidation",
        ),
    )
    vision_result = VisionResult(
        status=VisionStatus.OK,
        modality="Chest X-Ray",
        findings=[finding],
    )

    out = reasoner.synthesize_patient_report(
        patient=mock_patient,
        verified_evidence=[{"source": "ATS/IDSA Guidelines", "content": "Antibiotic regimen"}],
        detected_contradictions=[],
        c_total=0.91,
        mode="NORMAL",
        vision_result=vision_result,
    )

    json_str = reasoner.export_report_json(
        patient=mock_patient,
        reasoner_output=out,
        vision_result=vision_result,
    )

    data = json.loads(json_str)
    assert data["patient_id"] == "PT-RAD-001"
    assert "triage_level" in data
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert "contraindications" in data
    assert isinstance(data["contraindications"], list)
    assert "findings" in data
    assert isinstance(data["findings"], list)
    assert len(data["findings"]) == 1
    assert data["findings"][0]["label"] == "Lobar Consolidation"
    assert "citations" in data
    assert "disclaimer" in data
    assert "board-certified radiologist" in data["disclaimer"]


def test_export_report_html(reasoner, mock_patient):
    """Test HTML exporter rendering glassmorphic UI, badges, pills, and disclaimer."""
    finding = VisualFinding(
        label="Lobar Consolidation",
        code="RID49488",
        anatomical_location="Right Middle Lobe",
        probability=0.92,
        evidence=VisualEvidence(
            heatmap_path="overlays/rml_cam.png",
            supporting_visual_description="Consolidation in RML",
        ),
    )
    vision_result = VisionResult(
        status=VisionStatus.OK,
        modality="Chest X-Ray",
        findings=[finding],
    )

    out = reasoner.synthesize_patient_report(
        patient=mock_patient,
        verified_evidence=[{"source": "ATS 2019", "content": "CAP protocol"}],
        detected_contradictions=[],
        c_total=0.92,
        mode="NORMAL",
        vision_result=vision_result,
    )

    html_str = reasoner.export_report_html(
        patient=mock_patient,
        reasoner_output=out,
        vision_result=vision_result,
    )

    assert "<!DOCTYPE html>" in html_str or "<html" in html_str
    # Glassmorphic style elements
    assert "backdrop-filter" in html_str or "glass" in html_str
    # Status badges
    assert "badge" in html_str
    # Evidence pills
    assert "evidence-pill" in html_str
    # Disclaimer box
    assert "disclaimer-box" in html_str
    assert "DECISION SUPPORT DISCLAIMER" in html_str
    assert "PT-RAD-001" in html_str


def test_standard_patient_backward_compatibility(reasoner, mock_patient):
    """Ensure standard synthesize_patient_report works without vision_result."""
    out = reasoner.synthesize_patient_report(
        patient=mock_patient,
        verified_evidence=[{"source": "AHA Guidelines", "content": "Cardiovascular stability"}],
        detected_contradictions=[],
        c_total=0.88,
        mode="NORMAL",
    )
    assert isinstance(out, ReasonerOutput)
    assert "PATIENT-SPECIFIC CLINICAL DECISION SUPPORT REPORT" in out.clinical_report
    assert "PT-RAD-001" in out.clinical_report
