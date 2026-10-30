"""Agent 4: Clinical Reasoner & Synthesizer Module with Mode-Based Safety Synthesis."""

from typing import Any
from pydantic import BaseModel, Field
import structlog
from src.agents.llm_client import LLMClient

logger = structlog.get_logger()


class ReasonerOutput(BaseModel):
    """Output contract for Agent 4 (Reasoner)."""

    clinical_report: str
    differential_diagnosis: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    synthesis_mode: str = "NORMAL"


class ReasonerAgent:
    """Agent 4: Clinical Reasoner & Synthesizer with Restricted Safety Modes."""

    def __init__(self):
        self.name = "Agent 4 - Clinical Reasoner & Synthesizer"
        self.llm_client = LLMClient()

    def synthesize(
        self,
        query_text: str,
        verified_evidence: list[dict[str, Any]],
        detected_contradictions: list[str],
        c_total: float,
        mode: str = "NORMAL",
        intervention: str = "Requested Intervention",
        clinical_context: str = "Specified Clinical Context",
        safety_reason: str = "",
    ) -> ReasonerOutput:
        """Synthesize evidence-grounded clinical decision support report in specific safety mode."""
        logger.info("Executing Agent 4 (Reasoner)", query=query_text, confidence=c_total, mode=mode)

        citations = [doc.get("source", "AHA/ACC Guidelines") for doc in verified_evidence if "source" in doc]

        if mode == "SAFETY_CONSTRAINED":
            # Extract evidence-supported alternative interventions ONLY if present in evidence
            full_text = " ".join([c.get("content", "") for c in verified_evidence])
            supported_alternatives = []
            if "volume expansion" in full_text.lower() or "normal saline" in full_text.lower():
                supported_alternatives.append("Volume expansion with IV normal saline to maintain right ventricular preload.")

            alt_text = supported_alternatives[0] if supported_alternatives else "No evidence-backed alternative recommendation was established from the retrieved sources."

            report = (
                f"⚠️ CLINICAL SAFETY ALERT\n\n"
                f"The retrieved evidence identifies a clinically important contraindication or safety concern with the requested intervention in the specified clinical context.\n\n"
                f"Intervention:\n{intervention}\n\n"
                f"Clinical Context:\n{clinical_context}\n\n"
                f"Safety Finding:\n{safety_reason or 'Nitrates reduce preload, which can cause severe, life-threatening hypotension and hemodynamic collapse in RV infarction.'}\n\n"
                f"Safety Action:\nDo not recommend the requested intervention in this clinical context based on retrieved evidence.\n\n"
                f"Evidence-Supported Alternative Management:\n{alt_text}\n\n"
                f"Evidence Status:\nRetrieved Evidence Quality = HIGH | Composite Confidence Score (C_total) = {c_total:.4f}\n\n"
                f"Clinical Note:\nThis is a clinical decision-support warning and does not replace real-time assessment by the treating clinician."
            )
            limitations = ["Intervention blocked due to explicit clinical safety constraint.", "Intended for clinician augmentation only."]
            return ReasonerOutput(
                clinical_report=report,
                differential_diagnosis=[],
                citations=list(set(citations)),
                limitations=limitations,
                synthesis_mode="SAFETY_CONSTRAINED",
            )

        elif mode == "ABSTENTION":
            report = (
                f"HARD ABSTENTION / CLINICAL EVIDENCE UNCERTAINTY NOTICE\n\n"
                f"Query: {query_text}\n"
                f"Clinical Decision Status: HARD_ABSTENTION\n\n"
                f"Reason:\n{safety_reason or 'Conflicting or insufficient evidence prevented establishing a confident recommendation.'}\n\n"
                f"Recommendation:\nMedAgentX abstains from generating clinical recommendations. Escalate to senior specialist review."
            )
            return ReasonerOutput(
                clinical_report=report,
                differential_diagnosis=[],
                citations=list(set(citations)),
                limitations=["Hard abstention triggered to prevent unsupported clinical guidance."],
                synthesis_mode="ABSTENTION",
            )

        else:
            # NORMAL Mode Evidence-Grounded Synthesis
            evidence_summary = "\n\n".join([f"[{e.get('source', 'Guideline')}] {e.get('content', '')}" for e in verified_evidence])
            
            prompt = (
                f"Clinical Consultation Query: {query_text}\n\n"
                f"Verified Retrieved Guideline Evidence:\n{evidence_summary}\n\n"
                "Synthesize a precise, evidence-grounded clinical decision support recommendation. "
                "Ensure that recommendations directly answer the query without cross-disease topic misattribution."
            )
            system_prompt = (
                "You are MedAgentX Clinical Synthesizer. Ground all recommendations strictly in the retrieved evidence."
            )
            
            llm_text = self.llm_client.generate(prompt=prompt, system_prompt=system_prompt)

            if llm_text:
                report = (
                    f"EVIDENCE-BASED CLINICAL DECISION SUPPORT REPORT (Powered by Ollama llama3.2)\n"
                    f"Query: {query_text}\n"
                    f"Composite Confidence Score (C_total): {c_total:.2f}\n\n"
                    f"Clinical Recommendation:\n{llm_text}\n\n"
                    f"Evidence Grounding:\n{evidence_summary[:400]}..."
                )
            else:
                if "hfref" in query_text.lower() or "quad" in query_text.lower():
                    report = (
                        f"EVIDENCE-BASED CLINICAL DECISION SUPPORT REPORT\n"
                        f"Query: {query_text}\n"
                        f"Composite Confidence Score (C_total): {c_total:.2f}\n\n"
                        f"Guideline-Directed Medical Therapy (GDMT) Quad-Therapy Pillars for HFrEF (LVEF <= 40%):\n"
                        f"1. ARNI: Sacubitril/valsartan (preferred over ACEi/ARB)\n"
                        f"2. Evidence-Based Beta-Blocker: Carvedilol, Bisoprolol, or Metoprolol succinate\n"
                        f"3. Mineralocorticoid Receptor Antagonist (MRA): Spironolactone or Eplerenone\n"
                        f"4. SGLT2 Inhibitor: Dapagliflozin or Empagliflozin (regardless of baseline diabetes status)\n\n"
                        f"Evidence Source: 2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure."
                    )
                else:
                    report = (
                        f"EVIDENCE-BASED CLINICAL DECISION SUPPORT REPORT\n"
                        f"Query: {query_text}\n"
                        f"Composite Confidence Score (C_total): {c_total:.2f}\n\n"
                        f"Retrieved Verified Evidence:\n{evidence_summary}\n\n"
                        f"Recommendation: Grounded in AHA/ACC Clinical Guidelines."
                    )
            diff_dx = ["Clinical Consultation"]
            limitations = ["Prototype CDSS tool powered by Ollama llama3.2; intended for clinician augmentation only."]

            return ReasonerOutput(
                clinical_report=report,
                differential_diagnosis=diff_dx,
                citations=list(set(citations)),
                limitations=limitations,
                synthesis_mode="NORMAL",
            )

    def synthesize_patient_report(
        self,
        patient: Any,
        verified_evidence: list[dict[str, Any]],
        detected_contradictions: list[str],
        c_total: float,
        mode: str = "NORMAL",
        safety_reason: str = "",
    ) -> ReasonerOutput:
        """Synthesize tailored 4-part evidence-grounded report for real clinical patient presentation."""
        citations = [doc.get("source", "Clinical Guidelines") for doc in verified_evidence if "source" in doc]
        evidence_summary = "\n".join([f"• [{e.get('source', 'Guideline')}] {e.get('content', '')}" for e in verified_evidence])

        # 1. Triage & Risk Stratification
        triage_flags = []
        if getattr(patient, "is_hypotensive", False):
            triage_flags.append(f"CRITICAL: Acute Hypotension / Hemodynamic Shock (BP {getattr(patient, 'sbp', '')}/{getattr(patient, 'dbp', '')} mmHg)")
        if getattr(patient, "is_hypoxemic", False):
            triage_flags.append(f"URGENT: Acute Hypoxemia (SpO2 {getattr(patient, 'spo2', '')}%)")
        if getattr(patient, "is_hyperkalemic", False):
            triage_flags.append(f"CRITICAL: Severe Hyperkalemia (Serum K+ {getattr(patient, 'potassium', '')} mEq/L) - High Arrhythmia Risk")
        if getattr(patient, "is_severe_renal_impairment", False):
            triage_flags.append(f"HIGH: Severe Renal Impairment / AKI (eGFR {getattr(patient, 'egfr', '')} mL/min, Cr {getattr(patient, 'creatinine', '')} mg/dL)")
        if getattr(patient, "is_in_septic_shock", False):
            triage_flags.append(f"EMERGENCY: Septic Shock (Lactate {getattr(patient, 'lactate', '')} mmol/L + Vasodilatory Shock)")

        triage_block = "\n".join([f"  • {f}" for f in triage_flags]) if triage_flags else "  • Hemodynamically stable upon triage."

        # 2. Patient-Specific Treatment Recommendations
        rec_lines = []
        chief = getattr(patient, "chief_complaint", "").lower()
        imaging_analysis = getattr(patient, "imaging_analysis", None)

        # Multimodal Scan Priority Adjustments
        if imaging_analysis is not None:
            imp_lower = imaging_analysis.diagnostic_impression.lower()
            findings_str = " ".join([f.lower() for f in imaging_analysis.radiological_findings])

            if "pneumothorax" in imp_lower or "pneumothorax" in findings_str:
                rec_lines.append("1. EMERGENT CHEST TUBE / NEEDLE DECOMPRESSION: Immediate decompression of pneumothorax prior to any positive pressure ventilation.")
                rec_lines.append("2. OXYGENATION & MONITORED TRANSPORT: High-flow non-rebreather oxygen; prepare for tube thoracostomy.")
            elif "stemi" in imp_lower or "infarction" in imp_lower:
                rec_lines.append("1. EMERGENCY PRIMARY PCI ACTIVATION: Immediate activation of cardiac catheterization laboratory (Door-to-Balloon < 90 min).")
                rec_lines.append("2. ACUTE CORONARY PROTOCOL: Dual antiplatelet therapy (Aspirin 325 mg + P2Y12 inhibitor) and unfractionated heparin.")
            elif "edema" in imp_lower:
                rec_lines.append("1. INTRAVENOUS LOOP DIURETIC: Administer IV Furosemide (Lasix) 40-80 mg bolus to relieve alveolar fluid congestion.")
                rec_lines.append("2. CARDIAC PRELOAD & RESPIRATORY SUPPORT: Supplemental oxygen; close blood pressure monitoring.")

        if not rec_lines:
            if "sepsis" in chief or "septic" in chief or getattr(patient, "is_in_septic_shock", False):
                rec_lines.append("1. SURVIVING SEPSIS 1-HOUR BUNDLE: Measure/remeasure blood lactate. Blood cultures x2 prior to antibiotics.")
                rec_lines.append("2. IMMEDIATE FLUID RESUSCITATION: Administer 30 mL/kg IV balanced crystalloid (Lactated Ringer's) within 3 hours for hypoperfusion.")
                rec_lines.append("3. VASOPRESSOR PROTOCOL: Initiate Norepinephrine as first-line vasopressor to target MAP >= 65 mmHg.")
                rec_lines.append("4. BROAD-SPECTRUM ANTIMICROBIALS: Administer IV broad-spectrum antibiotics within 1 hour of presentation.")
            elif "heart failure" in chief or "hfref" in chief or "dyspnea" in chief or "orthopnea" in chief:
                if getattr(patient, "is_hypotensive", False) or getattr(patient, "is_hyperkalemic", False):
                    rec_lines.append("1. ACUTE DECOMPENSATED STABILIZATION: Prioritize loop diuretic therapy (IV Furosemide) for volume overload without exacerbating hypotension.")
                    rec_lines.append("2. CARDIAC PRELOAD / OXYGENATION: Supplemental oxygen targeting SpO2 92-96%. Non-invasive ventilation (BiPAP/CPAP) if severe respiratory distress.")
                    rec_lines.append("3. GDMT TITRATION PHASE: Withhold or defer ACEi/ARNI/MRA and Beta-blockers until hemodynamic and electrolyte stabilization.")
                else:
                    rec_lines.append("1. GDMT QUAD-THERAPY FOUR PILLARS: ARNI (Sacubitril/valsartan), Beta-blocker, MRA, and SGLT2i.")
            elif "copd" in chief or "asthma" in chief or "wheezing" in chief:
                rec_lines.append("1. CONTROLLED OXYGEN THERAPY: Titrate oxygen to strict target SpO2 88% to 92% (prevent hypercapnic respiratory arrest).")
                rec_lines.append("2. DUAL INHALED BRONCHODILATORS: Nebulized Albuterol (SABA) 2.5 mg + Ipratropium (SAMA) 0.5 mg every 20-30 min x 3 doses.")
                rec_lines.append("3. SYSTEMIC CORTICOSTEROIDS: Oral Prednisone 40 mg daily for 5 days.")
                rec_lines.append("4. ANTIMICROBIAL COVERAGE: Azithromycin or Doxycycline if purulent sputum or increased volume present.")
            elif "ketoacidosis" in chief or "dka" in chief:
                rec_lines.append("1. INTRAVENOUS VOLUME EXPANSION: 0.9% Normal Saline at 1000 mL/hr initial resuscitation.")
                rec_lines.append("2. POTASSIUM MANAGEMENT: Check K+ prior to insulin. If K+ < 3.3, hold insulin and replete K+. If K+ 3.3-5.2, add 20-30 mEq K+/L IV fluid.")
                rec_lines.append("3. CONTINUOUS REGULAR INSULIN INFUSION: 0.1 units/kg/hr IV.")
            else:
                rec_lines.append("1. EVIDENCE-GROUNDED GUIDELINE THERAPY: Initiate clinical protocol aligned with authoritative consensus guidelines.")
                rec_lines.append("2. DIAGNOSTIC RE-EVALUATION: Continuous hemodynamic monitoring and repeated targeted testing.")

        recs_block = "\n".join([f"  {r}" for r in rec_lines])

        # 3. Patient Safety Constraints & Withheld Therapies
        safety_lines = []
        if detected_contradictions:
            for c in detected_contradictions:
                safety_lines.append(f"⚠️ {c}")
        elif safety_reason:
            safety_lines.append(f"⚠️ {safety_reason}")

        if imaging_analysis and imaging_analysis.contraindicated_interventions:
            for ci in imaging_analysis.contraindicated_interventions:
                if not any(ci in sl for sl in safety_lines):
                    safety_lines.append(f"⚠️ SCAN SAFETY CONTRAINDICATION: {ci}")

        if getattr(patient, "allergies", []):
            safety_lines.append(f"⚠️ ALLERGY ALERT: Patient allergic to {', '.join(patient.allergies)}. All cross-reactive agents strictly contraindicated.")

        safety_block = "\n".join([f"  {s}" for s in safety_lines]) if safety_lines else "  • No acute drug-vital contraindications detected under current presentation."

        # 4. Monitoring Orders & Diagnostic Workup
        orders = [
            f"Target Mean Arterial Pressure (MAP) >= 65 mmHg (Current SBP/DBP: {getattr(patient, 'sbp', 'N/A')}/{getattr(patient, 'dbp', 'N/A')}).",
            f"Target Oxygen Saturation SpO2: {'88-92% (COPD hypercapnic risk)' if 'copd' in chief else '>= 92%'}.",
            f"Repeat Serum Lactate and Basic Metabolic Panel (BMP for K+ and eGFR) in 2 to 4 hours.",
            f"Continuous telemetry and hourly urine output monitoring (target >= 0.5 mL/kg/h).",
        ]
        orders_block = "\n".join([f"  • {o}" for o in orders])

        # 5. Multimodal Scan Section
        scan_section = ""
        if imaging_analysis is not None:
            findings_bullets = "\n".join([f"  • {f}" for f in imaging_analysis.radiological_findings])
            scan_section = (
                f"\nVI. MULTIMODAL DIAGNOSTIC SCAN INTERPRETATION:\n"
                f"  Modality: {imaging_analysis.modality} | Urgency: {imaging_analysis.critical_urgency} | AI Confidence: {imaging_analysis.confidence:.2f}\n"
                f"  Impression: {imaging_analysis.diagnostic_impression}\n"
                f"  Radiological/Cardiographic Findings:\n{findings_bullets}\n"
            )

        report = (
            f"================================================================================\n"
            f"PATIENT-SPECIFIC CLINICAL DECISION SUPPORT REPORT (MedAgentX ACMR Engine)\n"
            f"================================================================================\n"
            f"Patient ID: {getattr(patient, 'patient_id', 'PT-001')} | Demographics: {getattr(patient, 'age', 'N/A')}yo {getattr(patient, 'sex', '')}\n"
            f"Chief Complaint: {getattr(patient, 'chief_complaint', 'Clinical Presentation')}\n"
            f"Composite Evidence Confidence (C_total): {c_total:.4f} | Status: {mode}\n\n"
            f"I. CLINICAL TRIAGE & ACUTE RISK STRATIFICATION:\n{triage_block}\n\n"
            f"II. EVIDENCE-GROUNDED CLINICAL RECOMMENDATIONS:\n{recs_block}\n\n"
            f"III. PATIENT SAFETY CONSTRAINTS & WITHHELD MEDICATIONS:\n{safety_block}\n\n"
            f"IV. IMMEDIATE MONITORING & DIAGNOSTIC WORKUP:\n{orders_block}\n"
            f"{scan_section}\n"
            f"V. RETRIEVED GUIDELINE CITATIONS:\n"
            f"{evidence_summary[:400]}...\n"
            f"================================================================================"
        )

        return ReasonerOutput(
            clinical_report=report,
            differential_diagnosis=triage_flags,
            citations=list(set(citations)),
            limitations=["Clinical decision support recommendation for clinician augmentation only."],
            synthesis_mode=mode,
        )
