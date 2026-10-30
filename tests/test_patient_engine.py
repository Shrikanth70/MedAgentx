"""Tests for MedAgentX Real-World Patient Clinical Engine."""

import pytest
from src.orchestrator.patient import PatientCase, PatientVignetteParser


def test_patient_vignette_parser():
    """Verify regex vignette parser extracts vitals, abnormal labs, and risk flags."""
    raw_note = (
        "72yo male presenting with acute severe dyspnea, orthopnea, and bilateral pedal edema. "
        "BP 84/50 mmHg, HR 118 bpm, SpO2 86% on room air, RR 26. "
        "Labs: eGFR 24 mL/min, Creatinine 2.8 mg/dL, K+ 5.7 mEq/L, elevated BNP. "
        "Allergies: Penicillin. History of HFrEF and CKD Stage 4. "
        "Question: What acute medications should be initiated or avoided?"
    )
    patient = PatientVignetteParser.parse(raw_note)
    assert patient.age == 72
    assert patient.sex == "Male"
    assert patient.sbp == 84.0
    assert patient.dbp == 50.0
    assert patient.hr == 118.0
    assert patient.spo2 == 86.0
    assert patient.rr == 26.0
    assert patient.egfr == 24.0
    assert patient.creatinine == 2.8
    assert patient.potassium == 5.7
    assert patient.is_hypotensive is True
    assert patient.is_hypoxemic is True
    assert patient.is_severe_renal_impairment is True
    assert patient.is_hyperkalemic is True
    assert any("penicillin" in a.lower() for a in patient.allergies)
    assert any("hfref" in h.lower() for h in patient.past_medical_history)
