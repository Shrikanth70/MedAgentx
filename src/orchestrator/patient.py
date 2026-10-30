"""Patient Case Representation & Automated Clinical Vignette Parser."""

import re
from typing import Any
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


class PatientCase(BaseModel):
    """Rich structured clinical patient case profile."""

    patient_id: str = "PT-001"
    age: int | None = None
    sex: str | None = None
    chief_complaint: str = ""
    hpi: str = ""
    
    # Vital Signs
    sbp: float | None = None
    dbp: float | None = None
    hr: float | None = None
    rr: float | None = None
    spo2: float | None = None
    temp: float | None = None
    
    # Key Laboratory Biomarkers
    egfr: float | None = None
    creatinine: float | None = None
    potassium: float | None = None
    troponin: float | None = None
    lactate: float | None = None
    wbc: float | None = None
    glucose: float | None = None
    platelets: float | None = None
    
    # History & Clinical Context
    past_medical_history: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    clinical_question: str = ""

    # Physiological Risk Flags
    @property
    def is_hypotensive(self) -> bool:
        """True if SBP < 90 mmHg or MAP < 65 mmHg."""
        if self.sbp is not None and self.sbp < 90.0:
            return True
        if self.sbp is not None and self.dbp is not None:
            map_val = (self.sbp + 2 * self.dbp) / 3.0
            if map_val < 65.0:
                return True
        return False

    @property
    def is_hypoxemic(self) -> bool:
        """True if SpO2 < 90%."""
        return self.spo2 is not None and self.spo2 < 90.0

    @property
    def is_severe_renal_impairment(self) -> bool:
        """True if eGFR < 30 mL/min/1.73m^2."""
        return self.egfr is not None and self.egfr < 30.0

    @property
    def is_hyperkalemic(self) -> bool:
        """True if Serum Potassium > 5.2 mEq/L."""
        return self.potassium is not None and self.potassium > 5.2

    @property
    def is_in_septic_shock(self) -> bool:
        """True if Lactate >= 2.0 and Hypotensive."""
        return (self.lactate is not None and self.lactate >= 2.0) and self.is_hypotensive

    def summarize_clinical_state(self) -> str:
        """Render a concise clinical summary string for downstream reasoning."""
        parts = []
        if self.age or self.sex:
            parts.append(f"Demographics: {self.age or 'Unknown'}yo {self.sex or ''}".strip())
        if self.chief_complaint:
            parts.append(f"Chief Complaint: {self.chief_complaint}")
        
        vitals = []
        if self.sbp and self.dbp:
            vitals.append(f"BP {int(self.sbp)}/{int(self.dbp)} mmHg")
        if self.hr:
            vitals.append(f"HR {int(self.hr)} bpm")
        if self.spo2:
            vitals.append(f"SpO2 {int(self.spo2)}%")
        if self.rr:
            vitals.append(f"RR {int(self.rr)}")
        if vitals:
            parts.append(f"Vitals: {', '.join(vitals)}")

        labs = []
        if self.egfr:
            labs.append(f"eGFR {self.egfr} mL/min")
        if self.creatinine:
            labs.append(f"Cr {self.creatinine} mg/dL")
        if self.potassium:
            labs.append(f"K+ {self.potassium} mEq/L")
        if self.lactate:
            labs.append(f"Lactate {self.lactate} mmol/L")
        if labs:
            parts.append(f"Key Labs: {', '.join(labs)}")

        if self.past_medical_history:
            parts.append(f"PMHx: {', '.join(self.past_medical_history)}")
        if self.allergies:
            parts.append(f"Allergies: {', '.join(self.allergies)}")
        if self.clinical_question:
            parts.append(f"Question: {self.clinical_question}")

        return "\n".join(parts)


class PatientVignetteParser:
    """Automated clinical entity and numerical biomarker extractor for raw patient text."""

    @classmethod
    def parse(cls, raw_text: str) -> PatientCase:
        """Extract structured PatientCase from unstructured clinical note or emergency triage."""
        text = raw_text.strip()
        case_dict: dict[str, Any] = {}

        # 1. Demographics
        age_m = re.search(r"\b(\d{1,3})\s*(?:yo|y/o|year old|years old)\b", text, re.IGNORECASE)
        if age_m:
            case_dict["age"] = int(age_m.group(1))

        if re.search(r"\b(?:male|man|gentleman)\b", text, re.IGNORECASE):
            case_dict["sex"] = "Male"
        elif re.search(r"\b(?:female|woman|lady)\b", text, re.IGNORECASE):
            case_dict["sex"] = "Female"

        # 2. Vital Signs
        bp_m = re.search(r"(?:BP|blood pressure)[:\s]*(\d{2,3})[/\\](\d{2,3})", text, re.IGNORECASE)
        if bp_m:
            case_dict["sbp"] = float(bp_m.group(1))
            case_dict["dbp"] = float(bp_m.group(2))

        hr_m = re.search(r"(?:HR|pulse|heart rate)[:\s]*(\d{2,3})", text, re.IGNORECASE)
        if hr_m:
            case_dict["hr"] = float(hr_m.group(1))

        rr_m = re.search(r"(?:RR|respiratory rate)[:\s]*(\d{1,2})", text, re.IGNORECASE)
        if rr_m:
            case_dict["rr"] = float(rr_m.group(1))

        spo2_m = re.search(r"(?:SpO2|O2 sat|saturation)[:\s]*(\d{2,3})%?", text, re.IGNORECASE)
        if spo2_m:
            case_dict["spo2"] = float(spo2_m.group(1))

        temp_m = re.search(r"(?:Temp|temperature)[:\s]*(\d{2,3}(?:\.\d+)?)", text, re.IGNORECASE)
        if temp_m:
            case_dict["temp"] = float(temp_m.group(1))

        # 3. Key Laboratories
        egfr_m = re.search(r"(?:eGFR|GFR)[:\s]*(\d{1,3})", text, re.IGNORECASE)
        if egfr_m:
            case_dict["egfr"] = float(egfr_m.group(1))

        cr_m = re.search(r"(?:Creatinine|Cr)[:\s]*(\d+\.?\d*)", text, re.IGNORECASE)
        if cr_m:
            case_dict["creatinine"] = float(cr_m.group(1))

        k_m = re.search(r"(?:K\+?|potassium)[:\s]*(\d+\.?\d*)", text, re.IGNORECASE)
        if k_m:
            case_dict["potassium"] = float(k_m.group(1))

        lactate_m = re.search(r"(?:lactate)[:\s]*(\d+\.?\d*)", text, re.IGNORECASE)
        if lactate_m:
            case_dict["lactate"] = float(lactate_m.group(1))

        trop_m = re.search(r"(?:troponin|trop)[:\s]*(\d+\.?\d*)", text, re.IGNORECASE)
        if trop_m:
            case_dict["troponin"] = float(trop_m.group(1))

        # 4. Allergies
        allergy_m = re.search(r"(?:Allergies|Allergy)[:\s]*([^\.\n]+)", text, re.IGNORECASE)
        if allergy_m:
            allergies_raw = allergy_m.group(1).split(",")
            case_dict["allergies"] = [a.strip() for a in allergies_raw if a.strip()]

        # 5. Past Medical History
        pmh_m = re.search(r"(?:History of|PMHx|PMH|Past Medical History)[:\s]*([^\.\n]+)", text, re.IGNORECASE)
        if pmh_m:
            pmh_raw = re.split(r"[,;]|and", pmh_m.group(1))
            case_dict["past_medical_history"] = [h.strip() for h in pmh_raw if h.strip()]

        # 6. Clinical Question
        q_m = re.search(r"(?:Question|Management|Plan)[:\s]*([^\n]+)", text, re.IGNORECASE)
        if q_m:
            case_dict["clinical_question"] = q_m.group(1).strip()
        else:
            case_dict["clinical_question"] = text

        case_dict["chief_complaint"] = text[:150]
        case_dict["hpi"] = text

        return PatientCase(**case_dict)
