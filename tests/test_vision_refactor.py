"""Comprehensive Fail-Closed Regression & Safety Verification Test Suite.

Spec Reference: docs/specs/fail_closed_imaging_cdss_spec.md (Sections 9 & 10)
Plan Reference: docs/superpowers/plans/2026-10-31-fail-closed-imaging-cdss.md (Task 10)

Tests:
1. test_regression_fail_open_vision_timeout_never_recurs:
   Explicitly pins the original bug from user logs: a vision timeout must fail-closed,
   never fabricating a normal report ("Clear lung fields") or high confidence.
2. test_corrupt_and_suboptimal_image_rejection:
   Validates pre-inference QC gate rejecting low-resolution (<256x256), pitch-black,
   and motion-blurred studies with route REQUEST_MORE_INPUT.
3. test_critical_pathology_escalation_and_contraindication:
   Validates critical pathology hard-gating to ESCALATE_TO_HUMAN and contraindication
   flagging for positive-pressure ventilation in pneumothorax.
4. test_dicom_anonymization_pipeline:
   Validates DICOM de-identification scrubbing direct PHI tags while preserving
   clinical acquisition tags and pixel data.
5. test_grounded_pathology_and_evidence_citations:
   Validates calibrated visual finding perception, ATS/IDSA guideline citation,
   and visual finding evidence tracking.
"""

from __future__ import annotations

import json
from unittest.mock import patch
import numpy as np
from PIL import Image
import pytest

from src.multimodal.contracts import (
    QualityAssessment,
    VisionResult,
    VisionStatus,
    VisualEvidence,
    VisualFinding,
)
from src.multimodal.ingestion import MedicalImageIngestion
from src.orchestrator.system import MedAgentXSystem


@pytest.fixture
def system() -> MedAgentXSystem:
    return MedAgentXSystem()


def test_regression_fail_open_vision_timeout_never_recurs(system: MedAgentXSystem):
    """Explicitly pins original bug: vision timeout must fail-closed.

    When vision perception times out, the system must:
    1. Set vision_status == 'TIMEOUT'
    2. Route to 'ABSTAIN_INSUFFICIENT_EVIDENCE'
    3. Penalize confidence score below 0.80
    4. Never include fabricated 'Diagnostic Chest Examination without Acute Pathological Findings'
    5. Never include 'Clear lung fields' in the report
    6. Include 'CLINICAL AUDIT NOTICE' and state that automated interpretation was aborted
    7. Ensure findings in report and metadata are empty / none.
    """
    timeout_result = VisionResult(
        status=VisionStatus.TIMEOUT,
        modality="Chest X-Ray",
        body_part="Chest",
        quality=QualityAssessment(
            is_acceptable=False,
            rejection_reasons=["Inference read timeout after 3 retries (10.0s elapsed)"],
        ),
        findings=[],
        negative_findings=[],
        limitations=["Vision perception timed out."],
        model_name="Qwen2-VL-7B-Instruct",
    )

    with patch("src.multimodal.imaging_analyzer.MedicalImagingAnalyzer.analyze_vision", return_value=timeout_result):
        res = system.query_multimodal(
            patient_text="68-year-old male presenting with acute dyspnea and fever. Stat chest radiograph requested.",
            image_input="data/samples/sample_cxr_pneumonia.png",
        )

    # 1. Vision status
    assert res.metadata.get("vision_status") == "TIMEOUT"

    # 2. Hard-gate router decision
    assert res.metadata.get("route_decision") == "ABSTAIN_INSUFFICIENT_EVIDENCE"

    # 3. Penalized composite confidence
    assert res.confidence < 0.80

    # 4 & 5. Zero hallucinated/fabricated normal findings
    assert "Diagnostic Chest Examination without Acute Pathological Findings" not in res.answer
    assert "Clear lung fields" not in res.answer
    assert "clear lung fields" not in res.answer.lower()
    assert "no acute findings" not in res.answer.lower()

    # 6. Audit notice and aborted interpretation announcement
    assert "CLINICAL AUDIT NOTICE" in res.answer
    assert "aborted" in res.answer.lower()

    # 7. Findings payload is empty
    assert len(res.metadata["vision_result"]["findings"]) == 0
    assert len(res.metadata.get("radiological_findings", [])) == 0

    json_report = json.loads(res.metadata["json_report"])
    assert json_report["findings"] == []
    assert "None — Automated interpretation aborted" in res.answer or "None" in res.answer


@pytest.mark.parametrize(
    "corrupt_type,img_factory,expected_reason_substr",
    [
        (
            "low_resolution",
            lambda: Image.fromarray(np.random.randint(50, 200, (120, 120), dtype=np.uint8)).convert("L"),
            "Insufficient spatial resolution",
        ),
        (
            "pitch_black",
            lambda: Image.fromarray(np.zeros((300, 300), dtype=np.uint8)).convert("L"),
            "Severe underexposure",
        ),
        (
            "motion_blurred",
            lambda: Image.fromarray(
                np.tile(np.linspace(50, 180, 300, dtype=np.float32), (300, 1)).astype(np.uint8)
            ).convert("L"),
            "Severe motion blur",
        ),
    ],
)
def test_corrupt_and_suboptimal_image_rejection(
    system: MedAgentXSystem,
    corrupt_type: str,
    img_factory,
    expected_reason_substr: str,
):
    """Validates pre-inference QC gate rejecting suboptimal/corrupted scans to REQUEST_MORE_INPUT."""
    corrupt_image = img_factory()

    res = system.query_multimodal(
        patient_text=f"Clinical evaluation for {corrupt_type} image quality test.",
        image_input=corrupt_image,
        image_name=f"test_{corrupt_type}.png",
    )

    # 1. Vision status must be LOW_QUALITY
    assert res.metadata.get("vision_status") == "LOW_QUALITY"

    # 2. Route decision must demand repeat acquisition
    assert res.metadata.get("route_decision") == "REQUEST_MORE_INPUT"

    # 3. Rejection reasons must be clearly enumerated in metadata and report
    quality_meta = res.metadata.get("quality", {})
    rejection_reasons = quality_meta.get("rejection_reasons", [])
    assert len(rejection_reasons) > 0
    assert any(expected_reason_substr.lower() in r.lower() for r in rejection_reasons)

    # Router reason and answer must articulate why the image was rejected
    assert expected_reason_substr.lower() in res.metadata.get("router_reason", "").lower()
    assert "re-acquisition" in res.answer.lower() or "repeat" in res.answer.lower() or "suboptimal" in res.answer.lower()


def test_critical_pathology_escalation_and_contraindication(system: MedAgentXSystem):
    """Validates critical pathology hard-gating to ESCALATE_TO_HUMAN and contraindication identification."""
    res = system.query_multimodal(
        patient_text="32-year-old male following blunt motor vehicle trauma with right-sided chest pain, tachypnea, and hypotension. Tension pneumothorax suspected.",
        image_input="data/samples/sample_cxr_pneumothorax.png",
        image_name="sample_cxr_pneumothorax.png",
    )

    # 1. Route decision must escalate to human clinician
    assert res.metadata.get("route_decision") == "ESCALATE_TO_HUMAN"
    assert res.metadata.get("is_critical_pathology") is True

    # 2. Contraindications must include positive pressure ventilation without chest tube
    contraindications = res.metadata.get("contraindicated_interventions", [])
    assert len(contraindications) > 0
    assert any(
        "positive" in ci.lower() and "pressure" in ci.lower() for ci in contraindications
    )

    # 3. Clinical report must state positive-pressure ventilation contraindication
    report_lower = res.answer.lower()
    assert "pneumothorax" in report_lower
    assert "positive-pressure" in report_lower or "positive pressure" in report_lower
    assert "contraindication" in report_lower or "withheld" in report_lower


def test_dicom_anonymization_pipeline():
    """Validates DICOM de-identification scrubbing direct PHI while preserving acquisition and pixels."""
    ingestion = MedicalImageIngestion()

    # Synthetic DICOM payload with high-risk PHI tags and standard acquisition tags
    pixel_array = np.full((320, 320), 128, dtype=np.uint8).tolist()
    dicom_dict = {
        "PatientName": "O'CONNOR^TIMOTHY",
        "PatientID": "MRN-00918234",
        "PatientBirthDate": "19540822",
        "InstitutionName": "Central Regional Medical Center",
        "ReferringPhysicianName": "Dr. House, Gregory",
        "Modality": "DX",
        "BodyPartExamined": "CHEST",
        "ViewPosition": "PA",
        "PixelSpacing": [0.143, 0.143],
        "WindowCenter": 2048,
        "WindowWidth": 4096,
        "pixel_array": pixel_array,
    }
    raw_dicom_bytes = json.dumps(dicom_dict).encode("utf-8")

    pil_img, sanitized_meta = ingestion.anonymize_dicom(raw_dicom_bytes)

    # Verified output types
    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (320, 320)

    # Strict PHI scrubbing assertions
    assert "PatientName" not in sanitized_meta
    assert "PatientID" not in sanitized_meta
    assert "PatientBirthDate" not in sanitized_meta
    assert "InstitutionName" not in sanitized_meta
    assert "ReferringPhysicianName" not in sanitized_meta

    # Clinical acquisition preservation assertions
    assert sanitized_meta.get("Modality") == "DX"
    assert sanitized_meta.get("BodyPartExamined") == "CHEST"
    assert sanitized_meta.get("ViewPosition") == "PA"
    assert sanitized_meta.get("PixelSpacing") == [0.143, 0.143]
    assert sanitized_meta.get("WindowCenter") == 2048
    assert sanitized_meta.get("WindowWidth") == 4096


def test_grounded_pathology_and_evidence_citations(system: MedAgentXSystem):
    """Validates grounded RML consolidation CXR routes to STOP_CONFIDENT with ATS/IDSA guidelines."""
    res = system.query_multimodal(
        patient_text="59-year-old female presenting with 5-day fever, purulent sputum, and right middle lobe (RML) consolidation on chest radiograph.",
        image_input="data/samples/sample_cxr_pneumonia.png",
        image_name="sample_cxr_pneumonia.png",
    )

    # 1. Routing decision
    assert res.metadata.get("route_decision") == "STOP_CONFIDENT"
    assert res.metadata.get("vision_status") == "OK"
    assert res.confidence >= 0.80

    # 2. Visual finding extraction and evidence
    findings = res.metadata.get("vision_result", {}).get("findings", [])
    assert len(findings) > 0
    consolidation_finding = next(
        (f for f in findings if "consolidation" in f["label"].lower()),
        None,
    )
    assert consolidation_finding is not None
    assert "Right Middle Lobe" in consolidation_finding["anatomical_location"]
    assert consolidation_finding.get("evidence") is not None
    assert consolidation_finding["evidence"]["bbox"] is not None

    # 3. Guideline retrieval and evidence citations
    citations = res.citations
    assert len(citations) > 0
    assert any("ATS/IDSA" in c or "Pneumonia" in c for c in citations)

    # 4. Structured report includes visual finding evidence description and guideline citations
    report_upper = res.answer.upper()
    assert "LOBAR CONSOLIDATION" in report_upper or "CONSOLIDATION" in report_upper
    assert "ATS/IDSA" in report_upper or "COMMUNITY-ACQUIRED PNEUMONIA" in report_upper
    assert "REF-VF" in report_upper or "EVIDENCE" in report_upper or "BBOX" in report_upper or "ALVEOLAR" in report_upper
