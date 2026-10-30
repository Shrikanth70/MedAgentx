"""Live Execution Demo Script for MedAgentX with Ollama llama3.2."""

import sys
import os

# Configure stdout UTF-8 encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestrator.system import MedAgentXSystem

def main():
    print("=" * 80)
    print("           MEDAGENTX — ADAPTIVE CONFIDENCE-GUIDED MULTI-AGENT CDSS          ")
    print("                 POWERED BY OLLAMA (LLAMA 3.2) LOCAL LLM                    ")
    print("=" * 80)

    system = MedAgentXSystem()

    queries = [
        "What is the baseline quad-therapy for heart failure with reduced ejection fraction HFrEF?",
        "Administer sublingual nitroglycerin for severe chest pain in a patient with right ventricular infarction."
    ]

    for idx, q in enumerate(queries, 1):
        print(f"\n--------------------------------------------------------------------------------")
        print(f"CASE {idx}: {q}")
        print("--------------------------------------------------------------------------------")

        res = system.query(q)

        print(f"\n[SYSTEM ID]: {res.system_id}")
        print(f"[COMPOSITE CONFIDENCE (C_total)]: {res.confidence:.4f}")
        print(f"[ROUTE DECISION]: {res.metadata.get('route_decision')}")
        print(f"[ACTIVE AGENT COUNT]: {res.metadata.get('active_agent_count')}")
        print(f"[EXECUTION LATENCY]: {res.latency_seconds:.4f} seconds")

        print("\n--- AGENT EXECUTION TRACE ---")
        for step in res.metadata.get("trace", []):
            print(f"  • {step}")

        print("\n--- CLINICAL DECISION SUPPORT REPORT ---")
        print(res.answer)
        print("=" * 80)

if __name__ == "__main__":
    main()
