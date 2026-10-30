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

