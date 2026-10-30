"""Live Multimodal Clinical Perception & ACMR Multi-Agent Demonstration Runner.

Demonstrates end-to-end ingestion and synthesis of:
  1. Chest X-Ray (Lobar Consolidation / CAP Protocol)
  2. 12-Lead ECG (Acute Anterior STEMI / Emergency PCI)
  3. Hospital Laboratory Report (Severe Sepsis + AKI + Drug Safety Override)
  4. Emergency Trauma CXR (Tension Pneumothorax / Positive Pressure Safety Block)
"""

from __future__ import annotations

import os
import sys
import time

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.orchestrator.system import MedAgentXSystem



def print_banner(title: str):
    print("\n" + "=" * 90)
    print(f"  {title.upper()}")
    print("=" * 90)


def run_multimodal_demonstration():
    print_banner("MedAgentX Multimodal Clinical Decision Support System — Live Demonstration")
    print("Initializing MedAgentX System & ACMR Multi-Agent Orchestrator...")
    system = MedAgentXSystem()
    print("System loaded successfully.\n")

    cases = [
        {
            "id": "MM-01",
            "name": "Case 1: Chest X-Ray — Acute Lobar Pneumonia (Inpatient CAP Protocol)",
            "text": "54-year-old male presenting with 4-day history of high fever (39.0 C), productive cough with rust-colored sputum, and right-sided pleuritic chest pain.",
            "image": "data/samples/sample_cxr_pneumonia.png",
            "report": None,
        },
        {
            "id": "MM-02",
            "name": "Case 2: 12-Lead ECG — Acute Anterior STEMI (Emergency PCI Protocol)",
            "text": "61-year-old female presenting with acute crushing substernal chest pressure radiating to the left arm and jaw, diaphoresis, and nausea.",
            "image": "data/samples/sample_ecg_stemi.png",
            "report": None,
        },
        {
            "id": "MM-03",
            "name": "Case 3: Hospital Critical Care Lab Report — Septic Shock & KDIGO AKI",
            "text": "72-year-old female with suspected severe urosepsis and altered mental status. History of Type 2 Diabetes on Metformin.",
            "image": None,
            "report": "data/samples/sample_lab_report.txt",
        },
        {
            "id": "MM-04",
            "name": "Case 4: Trauma CXR — Acute Pneumothorax with Positive-Pressure Safety Override",
            "text": "34-year-old male status post motor vehicle collision with blunt thoracic trauma, acute dyspnea, and right chest pain. Clinician considering non-invasive positive pressure ventilation (BiPAP).",
            "image": "data/samples/sample_cxr_pneumothorax.png",
            "report": None,
        },
    ]

    for idx, c in enumerate(cases, 1):
        print_banner(f"Study [{idx}/4] — {c['name']}")
        print(f"  • Patient Presentation: {c['text']}")
        if c["image"]:
            print(f"  • Ingested Diagnostic Scan: {c['image']}")
        if c["report"]:
            print(f"  • Ingested Lab Document: {c['report']}")

        t0 = time.time()
        res = system.query_multimodal(
            patient_text=c["text"],
            report_input=c["report"],
            image_input=c["image"],
        )
        elapsed = time.time() - t0

        print("\n  [PERCEPTION & ACMR PIPELINE METRICS]")
        print(f"    - Modality:             {res.metadata.get('imaging_modality', 'Laboratory Document')}")
        print(f"    - Scan Impression:      {res.metadata.get('imaging_impression', 'N/A')}")
        print(f"    - Urgency Level:        {res.metadata.get('imaging_urgency', 'ROUTINE')}")
        print(f"    - Route Action:         {res.metadata.get('route_decision', 'STOP')}")
        print(f"    - Composite Confidence: {res.confidence:.4f}")
        print(f"    - Latency:              {elapsed:.3f} s")
        print(f"    - Active Agents:        {res.metadata.get('active_agent_count', 4)}")

        contraindications = res.metadata.get("contraindicated_interventions", [])
        if contraindications:
            print("\n  [DETECTED SAFETY CONTRAINDICATIONS]")
            for ci in contraindications:
                print(f"    ⚠️  {ci}")

        parsed_labs = res.metadata.get("parsed_labs", {})
        if parsed_labs:
            print("\n  [PARSED TABULAR LAB BIOMARKERS]")
            for k, v in parsed_labs.items():
                print(f"    • {k.upper()}: {v}")

        print("\n  [SYNTHESIZED CLINICAL DECISION SUPPORT REPORT SUMMARY]")
        # Print first 14 lines of answer
        report_preview = "\n".join(res.answer.split("\n")[:18])
        print(f"{report_preview}\n    [... full report retained ...]")

    print_banner("Multimodal Demonstration Completed Successfully (4/4 Studies Processed)")


if __name__ == "__main__":
    run_multimodal_demonstration()
