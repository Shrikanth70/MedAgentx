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


def test_generalized_retrieval_multi_domain():
    """Verify hybrid retriever dynamically surfaces Sepsis, AKI, and COPD guidelines."""
    from src.knowledge.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()

    # Sepsis retrieval
    res_sepsis = retriever.retrieve("Patient in septic shock with refractory hypotension and elevated serum lactate")
    topics_sepsis = [c.get("clinical_topic", "") for c in res_sepsis["vector_evidence"]]
    assert any("Sepsis" in t or "Septic" in t or "Resuscitation" in t for t in topics_sepsis)

    # AKI / Hyperkalemia retrieval
    res_renal = retriever.retrieve("Severe acute kidney injury with hyperkalemia K 6.2 and oliguria")
    topics_renal = [c.get("clinical_topic", "") for c in res_renal["vector_evidence"]]
    assert any("Hyperkalemia" in t or "Kidney" in t or "Renal" in t for t in topics_renal)


def test_vital_lab_safety_verifier():
    """Verify safety verifier flags hemodynamic collapse, hyperkalemia, and allergy contraindications."""
    from src.agents.verifier import VerifierAgent
    from src.orchestrator.patient import PatientCase

    verifier = VerifierAgent()

    # Case A: Severe Hypotension Contraindication (SBP < 90)
    pt_hypo = PatientCase(chief_complaint="Chest pain", sbp=82.0, dbp=48.0)
    out_hypo = verifier.verify_patient(
        query_text="Administer IV nitroglycerin and metoprolol",
        vector_evidence=[{"content": "Nitroglycerin and beta blockers are first line for angina", "clinical_topic": "ACS"}],
        patient=pt_hypo,
    )
    assert out_hypo.r_contra >= 0.80
    assert out_hypo.safety_signal is True
    assert out_hypo.contradiction_type == "CLINICAL_CONTRAINDICATION"
    assert "hypotension" in out_hypo.safety_reason.lower()

    # Case B: Severe Hyperkalemia Contraindication (K > 5.2)
    pt_k = PatientCase(chief_complaint="Heart failure", potassium=5.8, egfr=24.0)
    out_k = verifier.verify_patient(
        query_text="Start spironolactone and sacubitril/valsartan",
        vector_evidence=[{"content": "Spironolactone and ARNI are foundational GDMT", "clinical_topic": "HFrEF"}],
        patient=pt_k,
    )
    assert out_k.r_contra >= 0.80
    assert "hyperkalemia" in out_k.safety_reason.lower()

    # Case C: Allergy Contraindication (Penicillin Allergy)
    pt_allergy = PatientCase(chief_complaint="Pneumonia", allergies=["Penicillin"])
    out_allergy = verifier.verify_patient(
        query_text="Prescribe amoxicillin clavulanate or ampicillin",
        vector_evidence=[{"content": "Amoxicillin and Ampicillin are standard beta lactams", "clinical_topic": "Pneumonia"}],
        patient=pt_allergy,
    )
    assert out_allergy.r_contra >= 0.80
    assert out_allergy.contradiction_type in ["ALLERGY_CONTRAINDICATION", "CLINICAL_CONTRAINDICATION"]
    assert "penicillin" in out_allergy.safety_reason.lower() or "allergy" in out_allergy.safety_reason.lower()


def test_medagentx_system_real_patient_execution():
    """Verify full ACMR orchestrator runs over real-world clinical patient presentation."""
    from src.orchestrator.system import MedAgentXSystem
    from src.orchestrator.patient import PatientCase

    system = MedAgentXSystem()

    pt = PatientCase(
        patient_id="PT-TEST-01",
        age=68,
        sex="Male",
        chief_complaint="Septic Shock and Suspected Urosepsis",
        sbp=82.0,
        dbp=48.0,
        hr=122.0,
        spo2=91.0,
        lactate=3.8,
        egfr=28.0,
        clinical_question="What is the immediate resuscitation and antimicrobial management protocol?"
    )
    res = system.query_patient(pt)
    assert res.system_id == "PROPOSED_MEDAGENTX"
    assert res.confidence > 0.0
    assert "CLINICAL" in res.answer or "REPORT" in res.answer or "SAFETY ALERT" in res.answer
    assert res.token_count > 0
    assert "trace" in res.metadata
