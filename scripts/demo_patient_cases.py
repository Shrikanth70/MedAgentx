"""Live Real-World Patient Clinical Case Demonstration for MedAgentX.

Runs authentic, complex clinical vignettes featuring vital sign derangements,
abnormal laboratories, and drug allergies across multi-specialty guidelines.
"""

import sys
import os

# Configure stdout UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.orchestrator.system import MedAgentXSystem
from src.orchestrator.patient import PatientCase, PatientVignetteParser

REAL_WORLD_CASES = [
    {
        "id": "CASE-1",
        "title": "Emergency Septic Shock with Acute Kidney Injury",
        "raw_vignette": (
            "68yo male presenting with acute altered mental status, severe rigors, and suspected urosepsis. "
            "Vitals: BP 82/48 mmHg, HR 124 bpm, RR 28, SpO2 91% on room air, Temp 39.1 C. "
            "Labs: Serum Lactate 3.8 mmol/L, eGFR 28 mL/min, Creatinine 2.6 mg/dL, WBC 18,500. "
            "History of Type 2 Diabetes, CKD Stage 3b. Current meds: Metformin 1000mg BID. "
            "Question: What is the immediate resuscitation, vasopressor choice, and antimicrobial protocol?"
        )
    },
    {
        "id": "CASE-2",
        "title": "Decompensated Heart Failure with Severe Hyperkalemia & Hypotension",
        "raw_vignette": (
            "72yo male presenting with acute worsening dyspnea, orthopnea, and severe bilateral lower extremity edema. "
            "Vitals: BP 86/52 mmHg, HR 112 bpm, SpO2 87% on room air. "
            "Labs: Serum Potassium K+ 5.8 mEq/L, eGFR 22 mL/min, Creatinine 3.1 mg/dL, elevated BNP. "
            "History of HFrEF (LVEF 25%), CKD Stage 4. "
            "Question: What GDMT medications should be withheld, and what is the acute stabilization protocol?"
        )
    },
    {
        "id": "CASE-3",
        "title": "Severe Inpatient CAP with Anaphylactic Penicillin Allergy",
        "raw_vignette": (
            "58yo female admitted for severe community-acquired pneumonia with right lower lobe consolidation. "
            "Vitals: BP 118/74 mmHg, HR 98 bpm, SpO2 92% on 2L nasal cannula, Temp 38.8 C. "
            "Allergies: Anaphylactic Penicillin Allergy (airway edema and urticaria). "
            "Question: What is the recommended inpatient empiric antimicrobial regimen avoiding beta-lactams?"
        )
    },
    {
        "id": "CASE-4",
        "title": "Acute Severe COPD Exacerbation with Hypercapnic Failure Risk",
        "raw_vignette": (
            "66yo male with severe COPD presenting with acute breathlessness, wheezing, and purulent sputum. "
            "Vitals: BP 136/82 mmHg, HR 102 bpm, SpO2 84% on room air. Arterial blood gas shows PaCO2 56 mmHg. "
            "Question: What is the targeted oxygen saturation range, bronchodilator regimen, and systemic steroid protocol?"
        )
    }
]


def run_patient_demonstration():
    print("=" * 80)
    print("   MEDAGENTX — REAL-WORLD CLINICAL PATIENT DECISION SUPPORT ENGINE   ")
    print("      Adaptive ACMR Dynamic Multi-Agent Orchestration with Safety Bounds     ")
    print("=" * 80)

    system = MedAgentXSystem()

    for item in REAL_WORLD_CASES:
        print("\n" + "=" * 80)
        print(f" [{item['id']}] {item['title'].upper()}")
        print("=" * 80)
        print(f"CLINICAL ADMISSION NOTE:\n{item['raw_vignette']}\n")

        # Parse patient
        patient = PatientVignetteParser.parse(item["raw_vignette"])
        patient.patient_id = item["id"]

        print(f"EXTRACTED PATIENT PROFILE:")
        print(f"  • Demographics: {patient.age}yo {patient.sex or 'Unknown'}")
        if patient.sbp and patient.dbp:
            print(f"  • Blood Pressure: {patient.sbp:.0f}/{patient.dbp:.0f} mmHg {'(HYPOTENSION ALERT)' if patient.is_hypotensive else ''}")
        if patient.spo2:
            print(f"  • SpO2: {patient.spo2:.0f}% {'(HYPOXEMIA ALERT)' if patient.is_hypoxemic else ''}")
        if patient.potassium:
            print(f"  • Serum K+: {patient.potassium:.1f} mEq/L {'(HYPERKALEMIA ALERT)' if patient.is_hyperkalemic else ''}")
        if patient.egfr:
            print(f"  • eGFR: {patient.egfr:.0f} mL/min {'(SEVERE RENAL IMPAIRMENT)' if patient.is_severe_renal_impairment else ''}")
        if patient.lactate:
            print(f"  • Serum Lactate: {patient.lactate:.1f} mmol/L {'(SEPTIC SHOCK CRITERIA)' if patient.is_in_septic_shock else ''}")
        if patient.allergies:
            print(f"  • Documented Allergies: {', '.join(patient.allergies)}")

        print("\n--- EXECUTING ACMR DYNAMIC ORCHESTRATION ---")
        res = system.query_patient(patient)

        print(f"  • Route Decision: {res.metadata.get('route_decision')}")
        print(f"  • Composite Confidence (C_total): {res.confidence:.4f}")
        print(f"  • Active Agent Passes: {res.metadata.get('active_agent_count')}")
        print(f"  • Safety Classification: {res.metadata.get('contradiction_type')} (Severity: {res.metadata.get('safety_severity')})")
        print(f"  • Latency: {res.latency_seconds:.4f}s | Tokens: {res.token_count}")

        print("\n--- AGENT EXECUTION TRACE ---")
        for trace_step in res.metadata.get("trace", []):
            print(f"  • {trace_step}")

        print("\n" + res.answer)
        print("-" * 80)

    print("\nReal patient clinical engine demonstration completed successfully.")


if __name__ == "__main__":
    run_patient_demonstration()
