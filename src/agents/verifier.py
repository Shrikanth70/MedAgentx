"""Agent 3: Evidence Verifier & Critic Module with Safety Classification."""

import re
from typing import Any
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


class VerifierOutput(BaseModel):
    """Output contract for Agent 3 (Verifier)."""

    v_agree: float = 0.0
    r_contra: float = 0.0
    s_rel: float = 1.0  # Evidence-Query Relevance Gate Score
    retrieval_gap: str = "NONE"
    missing_evidence: list[str] = Field(default_factory=list)
    confidence_limitation: str = ""
    evidence_completeness: float = 1.0
    evidence_novelty: float = 1.0
    verified_evidence: list[dict[str, Any]] = Field(default_factory=list)
    detected_contradictions: list[str] = Field(default_factory=list)
    is_sufficient: bool = True

    # Structured Safety Classification Fields
    safety_signal: bool = False
    contradiction_type: str = "NONE"  # NONE, CLINICAL_CONTRAINDICATION, DRUG_INTERACTION, ALLERGY_CONTRAINDICATION, EVIDENCE_CONFLICT, INSUFFICIENT_EVIDENCE, UNRESOLVED_UNCERTAINTY
    safety_severity: str = "NONE"  # NONE, LOW, MEDIUM, HIGH, CRITICAL
    evidence_conflict: bool = False
    clinical_context: str = ""
    intervention: str = ""
    safety_reason: str = ""


class VerifierAgent:
    """Agent 3: Evidence Verifier & Critic with Relevance Gating and Safety Classification."""

    def __init__(self):
        self.name = "Agent 3 - Evidence Verifier & Critic"

    def _calculate_evidence_relevance(self, query_text: str, vector_evidence: list[dict[str, Any]]) -> float:
        """Calculate Query-Evidence Relevance Gate Score (S_rel) to prevent cross-disease misattribution."""
        if not vector_evidence:
            return 0.0

        q_lower = query_text.lower()

        # Extract Query Disease Intent
        query_diseases = []
        if any(k in q_lower for k in ["hfref", "heart failure", "ejection fraction"]):
            query_diseases.append("heart failure")
        if any(k in q_lower for k in ["stemi", "st-segment", "dapt", "acs", "angina"]):
            query_diseases.append("acute coronary")
        if any(k in q_lower for k in ["cap", "pneumonia"]):
            query_diseases.append("pneumonia")
        if any(k in q_lower for k in ["rv infarction", "right ventricular"]):
            query_diseases.append("right ventricular")
        if any(k in q_lower for k in ["oxygen", "spo2", "normoxic"]):
            query_diseases.append("oxygen")
        if any(k in q_lower for k in ["stroke", "ischemic", "thrombolysis", "thrombectomy", "alteplase"]):
            query_diseases.append("stroke")
        if any(k in q_lower for k in ["hydrocephalus", "ventriculomegaly", "ventricle"]):
            query_diseases.append("hydrocephalus")
        if any(k in q_lower for k in ["hemorrhage", "hematoma", "bleed"]):
            query_diseases.append("hemorrhage")
        if any(k in q_lower for k in ["nodule", "nodules", "adenopathy", "lymphadenopathy", "sarcoidosis"]):
            query_diseases.append("nodule")



        if not query_diseases:
            return 1.0

        # Evaluate Disease Match across retrieved evidence chunks
        relevant_chunks = 0
        for doc in vector_evidence:
            doc_category = doc.get("disease_category", "").lower()
            doc_content = doc.get("content", "").lower()
            doc_topic = doc.get("clinical_topic", "").lower()

            if not doc_category and not doc_content:
                relevant_chunks += 1
            else:
                disease_matched = any(qd in doc_category or qd in doc_content or qd in doc_topic for qd in query_diseases)
                if disease_matched or ("stemi" in q_lower and ("aspirin" in doc_content or "oxygen" in doc_content)):
                    relevant_chunks += 1

        s_rel = relevant_chunks / len(vector_evidence) if vector_evidence else 0.0
        return s_rel

    def _classify_safety_signal(
        self,
        query_text: str,
        vector_evidence: list[dict[str, Any]],
        s_rel: float,
    ) -> tuple[bool, str, str, bool, str, str, str, float, list[str]]:
        """Perform dynamic clinical safety classification from query and retrieved evidence."""
        q_lower = query_text.lower()
        full_text = " ".join([c.get("content", "").lower() for c in vector_evidence])

        detected_contradictions = []
        intervention = "Requested Clinical Intervention"
        clinical_context = "Specified Patient Condition"

        # Dynamic Extraction of Intervention
        if "nitroglycerin" in q_lower or "nitrates" in q_lower:
            intervention = "Sublingual / IV Nitroglycerin"
        elif "oxygen" in q_lower or "spo2" in q_lower:
            intervention = "Routine Supplemental Oxygen Therapy"
        elif "prasugrel" in q_lower:
            intervention = "Prasugrel P2Y12 Inhibitor"

        # Dynamic Extraction of Clinical Context
        if "right ventricular" in q_lower or "rv infarction" in q_lower:
            clinical_context = "Right Ventricular Myocardial Infarction"
        elif "stroke" in q_lower or "tia" in q_lower:
            clinical_context = "History of Stroke / TIA"
        elif "normoxic" in q_lower or "spo2 >= 90" in q_lower:
            clinical_context = "Normoxic STEMI (SpO2 >= 90%)"

        if not vector_evidence:
            return True, "INSUFFICIENT_EVIDENCE", "LOW", False, clinical_context, intervention, "No matching evidence chunks retrieved from knowledge base.", 0.0, []

        # Check A: Genuine Evidence Conflict across multiple sources
        has_pos_rec = any("recommended" in c.get("content", "").lower() and "not recommended" not in c.get("content", "").lower() for c in vector_evidence)
        has_neg_rec = any("not recommended" in c.get("content", "").lower() or "contraindicated" in c.get("content", "").lower() for c in vector_evidence)

        if len(vector_evidence) >= 2 and has_pos_rec and has_neg_rec and ("oxygen" in q_lower or "conflict" in q_lower):
            detected_contradictions.append("EVIDENCE CONFLICT: Retrieved guidelines provide opposing recommendations regarding routine oxygen in normoxic STEMI.")
            reason = "Retrieved guidelines provide conflicting evidence regarding routine supplemental oxygen therapy."
            return True, "EVIDENCE_CONFLICT", "MEDIUM", True, clinical_context, intervention, reason, 0.60, detected_contradictions

        # Check B: Supported Clinical Contraindication / Safety Constraint
        if ("contraindicated" in full_text or "contraindication" in full_text or "do not" in full_text) and (
            ("nitroglycerin" in q_lower and "right ventricular" in q_lower) or
            ("prasugrel" in q_lower and "stroke" in q_lower) or
            ("nitrates" in full_text and "preload" in full_text)
        ):
            msg = f"CLINICAL CONTRAINDICATION DETECTED: {intervention} is strictly contraindicated in {clinical_context}."
            detected_contradictions.append(msg)
            reason = f"Retrieved evidence indicates {intervention} is contraindicated in {clinical_context} due to risk of severe hemodynamic collapse or adverse clinical events."
            return True, "CLINICAL_CONTRAINDICATION", "HIGH", False, clinical_context, intervention, reason, 0.80, detected_contradictions

        return False, "NONE", "NONE", False, clinical_context, intervention, "", 0.0, []

    def _analyze_retrieval_gap(
        self,
        query_text: str,
        vector_evidence: list[dict[str, Any]],
        s_rel: float,
        r_contra: float,
        contradiction_type: str,
        previous_chunks: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[str], str, float, float]:
        """Analyze evidence sufficiency and identify specific clinical gaps."""
        q_lower = query_text.lower()
        missing_evidence = []
        limitation = ""
        gap = "NONE"
        completeness = 1.0

        # Calculate Novelty
        if previous_chunks:
            prev_ids = {c.get("chunk_id") for c in previous_chunks if "chunk_id" in c}
            curr_ids = {c.get("chunk_id") for c in vector_evidence if "chunk_id" in c}
            new_ids = curr_ids - prev_ids
            novelty = len(new_ids) / max(1, len(curr_ids))
        else:
            novelty = 1.0

        if contradiction_type == "CLINICAL_CONTRAINDICATION":
            gap = "CONTRADICTION"
            limitation = "Retrieved evidence establishes a clinical safety constraint."
            completeness = 1.0
        elif contradiction_type == "EVIDENCE_CONFLICT":
            gap = "CONTRADICTION"
            limitation = "Retrieved evidence contains opposing guideline recommendations."
            completeness = 0.70
        elif contradiction_type == "INSUFFICIENT_EVIDENCE":
            gap = "INSUFFICIENT_EVIDENCE"
            limitation = "No evidence found in knowledge base."
            completeness = 0.0
        elif s_rel < 0.50:
            gap = "INSUFFICIENT_RELEVANCE"
            missing_evidence.append("Relevant guidelines matching query disease intent")
            limitation = "Retrieved evidence category does not align with user query disease."
            completeness = 0.20
        else:
            full_text = " ".join([c.get("content", "") for c in vector_evidence]).lower()

            if "stemi" in q_lower or "dapt" in q_lower:
                has_dapt = "aspirin" in full_text and ("p2y12" in full_text or "ticagrelor" in full_text or "clopidogrel" in full_text)
                has_reperfusion = "pci" in full_text or "fibrinolytic" in full_text or "reperfusion" in full_text
                
                if not has_reperfusion:
                    gap = "MISSING_EVIDENCE"
                    missing_evidence.append("P2Y12 selection based on reperfusion strategy (PCI vs Fibrinolysis)")
                    limitation = "Retrieved evidence supports DAPT loading doses but does not fully distinguish PCI versus fibrinolysis pathways."
                    completeness = 0.70
                elif not has_dapt:
                    gap = "MISSING_EVIDENCE"
                    missing_evidence.append("DAPT loading dose specifications")
                    limitation = "Retrieved evidence lacks specific loading dose parameters."
                    completeness = 0.60

            elif "hfref" in q_lower or "heart failure" in q_lower:
                pillars = ["arni", "beta-blocker", "mra", "sglt2"]
                found_pillars = [p for p in pillars if p in full_text or ("sacubitril" in full_text if p == "arni" else False)]
                if len(found_pillars) < 3:
                    gap = "MISSING_EVIDENCE"
                    missing_evidence.append("Complete 4-pillar GDMT regimen details (ARNI, Beta-blocker, MRA, SGLT2i)")
                    limitation = "Incomplete evidence on foundational 4-pillar GDMT classes."
                    completeness = 0.65

        return gap, missing_evidence, limitation, completeness, round(novelty, 4)

    def verify(
        self,
        query_text: str,
        vector_evidence: list[dict[str, Any]],
        graph_evidence: list[dict[str, Any]],
        previous_chunks: list[dict[str, Any]] | None = None,
    ) -> VerifierOutput:
        """Verify evidence relevance, consensus agreement, clinical contraindications, and safety classification."""
        logger.info("Executing Agent 3 (Verifier)", vector_count=len(vector_evidence), graph_count=len(graph_evidence))

        # Calculate Evidence Relevance Gate Score (S_rel)
        s_rel = self._calculate_evidence_relevance(query_text, vector_evidence)

        # Safety Classification Analysis
        safety_signal, contradiction_type, severity, is_conflict, context, intervention, safety_reason, r_contra, detected_contradictions = self._classify_safety_signal(
            query_text=query_text,
            vector_evidence=vector_evidence,
            s_rel=s_rel,
        )

        # Gap Analysis
        gap, missing_ev, limitation, completeness, novelty = self._analyze_retrieval_gap(
            query_text=query_text,
            vector_evidence=vector_evidence,
            s_rel=s_rel,
            r_contra=r_contra,
            contradiction_type=contradiction_type,
            previous_chunks=previous_chunks,
        )

        verified_docs = vector_evidence if s_rel >= 0.50 else []
        v_agree = (0.85 if vector_evidence and graph_evidence else (0.70 if vector_evidence else 0.40)) * s_rel
        is_sufficient = bool(verified_docs) and contradiction_type not in ["EVIDENCE_CONFLICT", "INSUFFICIENT_EVIDENCE"] and s_rel >= 0.50

        return VerifierOutput(
            v_agree=round(v_agree, 4),
            r_contra=round(r_contra, 4),
            s_rel=round(s_rel, 4),
            retrieval_gap=gap,
            missing_evidence=missing_ev,
            confidence_limitation=limitation,
            evidence_completeness=completeness,
            evidence_novelty=novelty,
            verified_evidence=verified_docs,
            detected_contradictions=detected_contradictions,
            is_sufficient=is_sufficient,
            safety_signal=safety_signal,
            contradiction_type=contradiction_type,
            safety_severity=severity,
            evidence_conflict=is_conflict,
            clinical_context=context,
            intervention=intervention,
            safety_reason=safety_reason,
        )

    def verify_patient(
        self,
        query_text: str,
        vector_evidence: list[dict[str, Any]],
        patient: Any,
        graph_evidence: list[dict[str, Any]] | None = None,
        previous_chunks: list[dict[str, Any]] | None = None,
    ) -> VerifierOutput:
        """Verify evidence and assess acute vital sign, laboratory, and drug allergy contraindications."""
        out = self.verify(
            query_text=query_text,
            vector_evidence=vector_evidence,
            graph_evidence=graph_evidence or [],
            previous_chunks=previous_chunks,
        )

        q_lower = query_text.lower()
        full_text = " ".join([c.get("content", "").lower() for c in vector_evidence])

        # 1. Vital Sign Check: Acute Hypotension (SBP < 90 or MAP < 65)
        if getattr(patient, "is_hypotensive", False):
            vasodilators = ["nitroglycerin", "nitrate", "nitroprusside", "hydralazine", "metoprolol", "carvedilol", "beta blocker", "acei", "lisinopril"]
            if any(v in q_lower or v in full_text for v in vasodilators):
                out.safety_signal = True
                out.contradiction_type = "CLINICAL_CONTRAINDICATION"
                out.safety_severity = "HIGH"
                out.r_contra = 0.85
                out.safety_reason = (
                    f"Severe Acute Hypotension detected (BP {getattr(patient, 'sbp', 'N/A')}/{getattr(patient, 'dbp', 'N/A')} mmHg). "
                    "Vasodilators, nitrates, and acute beta-blockers are CONTRAINDICATED due to high risk of refractory cardiogenic shock. "
                    "Prioritize IV crystalloid fluid resuscitation and early vasopressors (Norepinephrine)."
                )
                out.intervention = "Vasodilators / Antihypertensives"
                out.clinical_context = f"Severe Hypotension (BP {getattr(patient, 'sbp', '')}/{getattr(patient, 'dbp', '')} mmHg)"
                out.detected_contradictions.append(out.safety_reason)

        # 2. Laboratory Check: Severe Hyperkalemia (Serum K+ > 5.2 mEq/L)
        if getattr(patient, "is_hyperkalemic", False):
            k_sparing = ["spironolactone", "eplerenone", "mra", "arni", "sacubitril", "lisinopril", "enalapril", "ramipril", "losartan", "valsartan", "ace inhibitor", "arb"]
            if any(k in q_lower or k in full_text for k in k_sparing):
                out.safety_signal = True
                out.contradiction_type = "CLINICAL_CONTRAINDICATION"
                out.safety_severity = "HIGH"
                out.r_contra = 0.85
                out.safety_reason = (
                    f"Severe Hyperkalemia detected (Serum K+ {getattr(patient, 'potassium', 'N/A')} mEq/L). "
                    "Aldosterone antagonists (Spironolactone), ARNI, and ACEi/ARBs are CONTRAINDICATED due to risk of fatal cardiac arrest. "
                    "Initiate emergent cardiac membrane stabilization (IV Calcium Gluconate) and intracellular potassium shifting (Insulin + D50)."
                )
                out.intervention = "Potassium-Sparing Agents (MRA / ARNI / ACEi)"
                out.clinical_context = f"Severe Hyperkalemia (K+ {getattr(patient, 'potassium', '')} mEq/L)"
                out.detected_contradictions.append(out.safety_reason)

        # 3. Laboratory Check: Severe Renal Impairment (eGFR < 30 mL/min)
        if getattr(patient, "is_severe_renal_impairment", False):
            nephrotoxic = ["metformin", "nsaid", "ibuprofen", "naproxen", "ketorolac", "sglt2", "dapagliflozin", "empagliflozin"]
            if any(n in q_lower or n in full_text for n in nephrotoxic):
                out.safety_signal = True
                out.contradiction_type = "CLINICAL_CONTRAINDICATION"
                out.safety_severity = "HIGH"
                out.r_contra = 0.80
                out.safety_reason = (
                    f"Severe Renal Impairment detected (eGFR {getattr(patient, 'egfr', 'N/A')} mL/min). "
                    "Metformin is contraindicated due to lactic acidosis risk. NSAIDs and SGLT2 inhibitors are withheld during acute renal hypoperfusion."
                )
                out.detected_contradictions.append(out.safety_reason)

        # 4. Drug Allergy Cross-Checking
        allergies = [a.lower() for a in getattr(patient, "allergies", [])]
        if any("penicillin" in a or "amoxicillin" in a or "beta-lactam" in a for a in allergies):
            penicillins = ["penicillin", "amoxicillin", "ampicillin", "augmentin", "piperacillin", "ceftriaxone", "cephalosporin"]
            if any(p in q_lower or p in full_text for p in penicillins):
                out.safety_signal = True
                out.contradiction_type = "ALLERGY_CONTRAINDICATION"
                out.safety_severity = "HIGH"
                out.r_contra = 0.85
                out.safety_reason = (
                    f"Documented Severe Drug Allergy: Penicillin / Beta-lactam. "
                    "Beta-lactam antibiotics are strictly CONTRAINDICATED. "
                    "Use guideline-directed non-beta-lactam alternatives (Respiratory fluoroquinolones like Levofloxacin or Macrolides/Aztreonam)."
                )
                out.intervention = "Penicillin / Beta-lactam Antibiotics"
                out.clinical_context = "Documented Penicillin Anaphylaxis/Allergy"
                out.detected_contradictions.append(out.safety_reason)

        # 5. Multimodal Scan & Radiological Cross-Modal Safety Checking
        imaging = getattr(patient, "imaging_analysis", None)
        if imaging is not None:
            imp_lower = imaging.diagnostic_impression.lower()
            findings_lower = " ".join([f.lower() for f in imaging.radiological_findings])

            # A. Pneumothorax on CXR -> Absolute Contraindication to Positive Pressure without Chest Tube
            has_ptx_finding = (
                ("pneumothorax" in imp_lower or "visceral pleural line" in findings_lower)
                and not any(neg in imp_lower or neg in findings_lower for neg in ["no pneumothorax", "without pneumothorax", "clear lung", "normal"])
            )
            if has_ptx_finding:
                out.safety_signal = True
                out.contradiction_type = "CLINICAL_CONTRAINDICATION"
                out.safety_severity = "HIGH"
                out.r_contra = 0.85
                out.safety_reason = (
                    "CRITICAL RADIOLOGICAL CONTRAINDICATION: Acute Pneumothorax detected on Chest Radiograph. "
                    "Non-invasive positive-pressure ventilation (BiPAP/CPAP) and positive-pressure mechanical ventilation "
                    "without prior tube thoracostomy are STRICTLY CONTRAINDICATED due to imminent tension physiology "
                    "and fatal cardiovascular collapse. Immediate needle decompression or chest tube placement is required."
                )
                out.intervention = "Positive-Pressure Ventilation (BiPAP/CPAP)"
                out.clinical_context = "Acute Pneumothorax on CXR"
                out.detected_contradictions.append(out.safety_reason)


            # B. Right Ventricular / Inferior STEMI on ECG -> Contraindicates Nitroglycerin / Preload Reducers
            elif "rv infarction" in imp_lower or "right ventricular" in imp_lower or "v4r" in findings_lower:
                out.safety_signal = True
                out.contradiction_type = "CLINICAL_CONTRAINDICATION"
                out.safety_severity = "HIGH"
                out.r_contra = 0.85
                out.safety_reason = (
                    "CRITICAL CARDIOGRAPHIC CONTRAINDICATION: Acute Right Ventricular Myocardial Infarction detected on 12-lead ECG. "
                    "Nitroglycerin, nitrates, and preload-reducing vasodilators are STRICTLY CONTRAINDICATED due to risk of "
                    "profound refractory hypotension and shock. Maintain preload with intravenous normal saline resuscitation."
                )
                out.intervention = "Nitroglycerin / Nitrates"
                out.clinical_context = "Acute RV Infarction on 12-Lead ECG"
                out.detected_contradictions.append(out.safety_reason)

            # C. Intracranial Hemorrhage on Head CT -> Contraindicates Thrombolytics & Anticoagulants
            elif "hemorrhage" in imp_lower or "bleed" in imp_lower or "hematoma" in imp_lower or "ich" in imp_lower:
                out.safety_signal = True
                out.contradiction_type = "CLINICAL_CONTRAINDICATION"
                out.safety_severity = "HIGH"
                out.r_contra = 0.85
                out.safety_reason = (
                    "CRITICAL RADIOLOGICAL CONTRAINDICATION: Acute Intracranial Hemorrhage detected on Head CT Scan. "
                    "Intravenous Thrombolytics (Alteplase / Tenecteplase / tPA), therapeutic anticoagulants (Heparin / DOACs), "
                    "and antiplatelet agents are STRICTLY CONTRAINDICATED due to imminent risk of fatal hematoma expansion. "
                    "Mandatory immediate neurosurgical consultation and strict blood pressure reduction."
                )
                out.intervention = "Thrombolytics / Anticoagulation (tPA / Heparin)"
                out.clinical_context = "Acute Intracranial Hemorrhage on Head CT"
                out.detected_contradictions.append(out.safety_reason)

            # D. Check general contraindicated interventions specified in scan analysis
            elif imaging.contraindicated_interventions and out.r_contra < 0.70:
                out.safety_signal = True
                out.safety_severity = "MEDIUM"
                out.r_contra = max(out.r_contra, 0.75)
                for ci in imaging.contraindicated_interventions:
                    out.detected_contradictions.append(f"SCAN SAFETY ALERT: {ci}")


        return out
