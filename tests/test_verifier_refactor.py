"""Tests for Refactored Verifier Agent with Multimodal Contradictions and Critical Alert Escalation."""

import pytest
from src.agents.verifier import VerifierAgent, VerifierOutput
from src.multimodal.contracts import VisualFinding, VisionResult, VisualEvidence, QualityAssessment, VisionStatus
from src.orchestrator.patient import PatientCase


def test_cross_modal_contradiction_consolidation_clear_lungs():
    """Verify CXR consolidation contradicted by physical exam of 'clear breath sounds'."""
    verifier = VerifierAgent()
    finding = VisualFinding(
        label="Acute Right Middle Lobe Consolidation",
        code="RID49488",
        anatomical_location="Right Middle Lobe",
        probability=0.92,
        uncertainty=0.08,
        evidence=VisualEvidence(supporting_visual_description="Dense alveolar consolidation with air bronchograms"),
    )
    patient = PatientCase(
        chief_complaint="Productive cough and fever",
        hpi="Patient reports acute shortness of breath. Physical examination: lungs clear to auscultation bilaterally.",
    )
    out = verifier.verify(
        query_text="Assess chest radiograph and recommend antimicrobial regimen",
        vector_evidence=[{"content": "Community-acquired pneumonia with consolidation", "disease_category": "pneumonia"}],
        graph_evidence=[],
        visual_findings=[finding],
        patient=patient,
    )
    expected_msg = "Discordance: Imaging demonstrates acute consolidation/infiltrate while physical examination reports clear lung fields."
    assert expected_msg in out.detected_contradictions
    assert expected_msg in out.cross_modal_contradictions


def test_critical_pathology_detection():
    """Verify critical pathology sets is_critical_pathology=True, safety_severity='CRITICAL', safety_signal='YES'."""
    verifier = VerifierAgent()
    finding = VisualFinding(
        label="Tension Pneumothorax",
        code="RID5050",
        anatomical_location="Right Hemithorax",
        probability=0.96,
        uncertainty=0.14,
        is_critical=True,
    )
    out = verifier.verify(
        query_text="Evaluate upright chest radiograph",
        vector_evidence=[{"content": "Pneumothorax requires immediate tube thoracostomy"}],
        graph_evidence=[],
        visual_findings=[finding],
    )
    assert out.is_critical_pathology is True
    assert out.safety_severity == "CRITICAL"
    assert out.safety_signal == "YES"
    assert len(out.critical_alerts) > 0
    assert out.vision_uncertainty == 0.14


def test_intracranial_hemorrhage_vs_tpa_contraindication():
    """Verify acute intracranial hemorrhage vs tPA / alteplase order triggers critical contradiction."""
    verifier = VerifierAgent()
    finding = VisualFinding(
        label="Acute Intracranial Hemorrhage",
        code="RID1234",
        anatomical_location="Left Basal Ganglia",
        probability=0.98,
        uncertainty=0.05,
    )
    out = verifier.verify(
        query_text="Administer IV alteplase (tPA) for acute neurological deficit",
        vector_evidence=[{"content": "tPA dose for acute ischemic stroke is 0.9 mg/kg"}],
        graph_evidence=[],
        visual_findings=[finding],
    )
    expected_contra = "Critical Contraindication: Acute Intracranial Hemorrhage strictly contraindicates thrombolytic (tPA) or full anticoagulation therapy."
    assert expected_contra in out.detected_contradictions
    assert out.is_critical_pathology is True
    assert out.safety_severity == "CRITICAL"
    assert out.safety_signal == "YES"


def test_pneumothorax_vs_bipap_contraindication():
    """Verify acute pneumothorax vs BiPAP / positive pressure ventilation triggers critical contraindication."""
    verifier = VerifierAgent()
    finding = VisualFinding(
        label="Pneumothorax",
        code="RID5050",
        anatomical_location="Left Pleural Space",
        probability=0.88,
        uncertainty=0.10,
    )
    out = verifier.verify(
        query_text="Initiate BiPAP positive pressure ventilation for acute respiratory failure",
        vector_evidence=[{"content": "BiPAP settings for hypercapnic respiratory failure"}],
        graph_evidence=[],
        visual_findings=[finding],
    )
    expected_contra = "Critical Contraindication: Acute Pneumothorax strictly contraindicates positive pressure ventilation prior to needle/tube thoracostomy."
    assert expected_contra in out.detected_contradictions
    assert out.is_critical_pathology is True
    assert out.safety_severity == "CRITICAL"


def test_backward_compatibility_standard_verify():
    """Verify existing verify call signature and behavior remains 100% functional."""
    verifier = VerifierAgent()
    out = verifier.verify(
        query_text="Is sublingual nitroglycerin safe in STEMI?",
        vector_evidence=[{"content": "Aspirin 325 mg initial loading dose."}],
        graph_evidence=[{"subject": "STEMI"}],
    )
    assert isinstance(out, VerifierOutput)
    assert out.v_agree > 0.0
    assert out.r_contra >= 0.0
    assert out.is_critical_pathology is False
    assert out.critical_alerts == []
    assert out.vision_uncertainty == 0.0
    assert out.cross_modal_contradictions == []


def test_vision_result_container_and_uncertainty_aggregation():
    """Verify passing VisionResult directly unpacks findings and aggregates maximum epistemic uncertainty."""
    verifier = VerifierAgent()
    vr = VisionResult(
        status=VisionStatus.OK,
        modality="Chest X-Ray",
        body_part="Chest",
        quality=QualityAssessment(is_acceptable=True),
        findings=[
            VisualFinding(
                label="Bilateral Apical Bullae",
                code="RID123",
                anatomical_location="Apices",
                probability=0.70,
                uncertainty=0.12,
            ),
            VisualFinding(
                label="Acute Lobar Pneumonia",
                code="RID456",
                anatomical_location="Right Lower Lobe",
                probability=0.85,
                uncertainty=0.25,
            ),
        ],
    )
    out = verifier.verify(
        query_text="Assess pulmonary opacity",
        vector_evidence=[{"content": "Pneumonia antimicrobial guideline", "disease_category": "pneumonia"}],
        graph_evidence=[],
        vision_result=vr,
    )
    assert out.vision_uncertainty == 0.25
    assert out.is_critical_pathology is False
