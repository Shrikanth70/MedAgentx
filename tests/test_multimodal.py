from __future__ import annotations

import pytest


def test_clinical_document_parser_tabular_labs():
    from src.multimodal.document_parser import ClinicalDocumentParser

    sample_lab_text = """
    HOSPITAL LABORATORY REPORT - CRITICAL CARE PANEL
    Patient: John Doe | MRN: 984120 | Service: Emergency ICU
    
    COMPREHENSIVE METABOLIC PANEL:
      eGFR: 26 mL/min/1.73m2 (LOW, Ref: >60)
      Serum Creatinine: 2.9 mg/dL (HIGH, Ref: 0.7-1.3)
      Potassium (K+): 5.8 mEq/L (CRITICAL HIGH, Ref: 3.5-5.0)
      Sodium: 136 mEq/L (Normal)
      Blood Urea Nitrogen (BUN): 48 mg/dL (HIGH)
    
    CRITICAL BIOMARKERS:
      Serum Lactate: 3.6 mmol/L (HIGH, Ref: 0.5-2.0)
      High-Sensitivity Troponin I: 0.42 ng/mL (ELEVATED, Ref: <0.04)
      WBC Count: 17.8 x10^3/uL (HIGH, Ref: 4.5-11.0)
    
    BLOOD GAS:
      pH: 7.28 (Acidosis, Ref: 7.35-7.45)
      PaCO2: 32 mmHg
      PaO2: 74 mmHg
    """
    parsed_labs = ClinicalDocumentParser.parse_report(sample_lab_text)
    assert parsed_labs["egfr"] == 26.0
    assert parsed_labs["creatinine"] == 2.9
    assert parsed_labs["potassium"] == 5.8
    assert parsed_labs["lactate"] == 3.6
    assert parsed_labs["troponin"] == 0.42
    assert parsed_labs["wbc"] == 17.8


def test_imaging_analyzer_cxr_and_ecg():
    from src.multimodal.imaging_analyzer import MedicalImagingAnalyzer

    analyzer = MedicalImagingAnalyzer()

    # Test Case A: Chest X-Ray with Lobar Pneumonia
    analysis_cxr = analyzer.analyze("sample_cxr_pneumonia.png", clinical_hint="Chest X-Ray showing fever and cough")
    assert "Pneumonia" in analysis_cxr.diagnostic_impression or "Consolidation" in analysis_cxr.diagnostic_impression
    assert any("consolidation" in f.lower() or "opacity" in f.lower() for f in analysis_cxr.radiological_findings)

    # Test Case B: 12-Lead ECG with Acute Anterior STEMI
    analysis_ecg = analyzer.analyze("sample_ecg_stemi.png", clinical_hint="12-Lead ECG for acute chest pain")
    assert "STEMI" in analysis_ecg.diagnostic_impression or "Myocardial Infarction" in analysis_ecg.diagnostic_impression
    assert analysis_ecg.critical_urgency == "EMERGENT"

    # Test Case C: Pneumothorax Visual Safety Contraindication
    analysis_ptx = analyzer.analyze("sample_cxr_pneumothorax.png", clinical_hint="Severe chest trauma, acute dyspnea, pneumothorax")
    assert any("positive pressure" in c.lower() or "positive-pressure" in c.lower() for c in analysis_ptx.contraindicated_interventions)


def test_cross_modal_acmr_safety_and_reasoning():
    from src.orchestrator.system import MedAgentXSystem

    system = MedAgentXSystem()

    # Test Multimodal Consultation: Patient with dyspnea, CXR showing Pneumothorax
    res = system.query_multimodal(
        patient_text="42yo male following motor vehicle collision with sudden right-sided chest pain and shortness of breath.",
        image_input="sample_cxr_pneumothorax.png",
    )
    assert res.system_id == "PROPOSED_MEDAGENTX"
    assert res.metadata.get("route_decision") in ["SAFETY_CONSTRAINT", "HARD_ABSTENTION"]
    assert "PNEUMOTHORAX" in res.answer.upper() or "CHEST TUBE" in res.answer.upper()
    assert "POSITIVE PRESSURE" in res.answer.upper() or "VENTILATION" in res.answer.upper()


def test_imaging_analyzer_ct_and_mri():
    from src.multimodal.imaging_analyzer import MedicalImagingAnalyzer

    analyzer = MedicalImagingAnalyzer()

    # Test Case A: Head CT with Intracranial Hemorrhage
    analysis_ct = analyzer.analyze("ct_scan_head_bleed.png", clinical_hint="Head CT showing hyperdense acute intracranial hemorrhage with mass effect")
    assert "CT Scan" in analysis_ct.modality
    assert "Hemorrhage" in analysis_ct.diagnostic_impression
    assert analysis_ct.critical_urgency == "EMERGENT"
    assert any("thrombolytic" in c.lower() or "alteplase" in c.lower() for c in analysis_ct.contraindicated_interventions)

    # Test Case B: Brain MRI with Acute Stroke (DWI Restriction)
    analysis_mri = analyzer.analyze("brain_mri_dwi_stroke.png", clinical_hint="Brain MRI showing DWI diffusion restriction in left MCA territory without bleed")
    assert "MRI" in analysis_mri.modality
    assert "Infarction" in analysis_mri.diagnostic_impression or "Ischemic" in analysis_mri.diagnostic_impression
    assert analysis_mri.critical_urgency == "EMERGENT"

    # Test Case C: Abdominal CT with Bowel Obstruction
    analysis_sbo = analyzer.analyze("abdomen_ct_axial.png", clinical_hint="Abdomen CT showing dilated bowel loops and high-grade mechanical bowel obstruction")
    assert "CT Scan" in analysis_sbo.modality
    assert "Obstruction" in analysis_sbo.diagnostic_impression
    assert any("npo" in c.lower() or "oral intake" in c.lower() for c in analysis_sbo.contraindicated_interventions)


def test_cross_modal_head_ct_hemorrhage_contraindication():
    from src.orchestrator.system import MedAgentXSystem

    system = MedAgentXSystem()

    # Acute Head CT Hemorrhage: Clinician considering IV Alteplase (tPA)
    res = system.query_multimodal(
        patient_text="67yo female presenting with sudden right hemiplegia and aphasia. Emergency physician orders Head CT and prepares IV Alteplase (tPA).",
        image_input="head_ct_acute_hemorrhage.png",
    )
    assert res.system_id == "PROPOSED_MEDAGENTX"
    assert res.metadata.get("route_decision") in ["SAFETY_CONSTRAINT", "HARD_ABSTENTION"]
    assert "HEMORRHAGE" in res.answer.upper()
    assert "THROMBOLYTIC" in res.answer.upper() or "ALTEPLASE" in res.answer.upper() or "CONTRAINDICATED" in res.answer.upper()




