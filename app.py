"""Streamlit Web Application for MedAgentX Clinical Decision Support System."""

import os
import sys
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.settings import settings
from src.orchestrator.system import MedAgentXSystem

# Configure Streamlit page layout and theme
st.set_page_config(
    page_title="MedAgentX — Clinical Decision Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern dark theme and card styling
st.markdown(
    """
    <style>
    .main {
        background-color: #090d16;
    }
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(59, 130, 246, 0.08) 0%, transparent 40%);
    }
    .stButton>button {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 1.2rem;
        width: 100%;
    }
    .safety-alert-banner {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
    }
    .safety-constraint-banner {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #1a1a2e;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
    }
    .evidence-conflict-banner {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_system():
    """Instantiate MedAgentX system once."""
    return MedAgentXSystem()


def main():
    # Sidebar
    st.sidebar.title("MedAgentX")
    st.sidebar.caption("Adaptive Confidence-Guided Multi-Agent RAG")
    st.sidebar.markdown("---")

    st.sidebar.subheader("⚙️ System Status")
    st.sidebar.markdown(f"**LLM Provider:** `{settings.LLM_PROVIDER.upper()}`")
    st.sidebar.markdown(f"**Model Engine:** `{settings.LLM_MODEL}`")
    st.sidebar.markdown(f"**Ollama Base URL:** `{settings.OLLAMA_BASE_URL}`")
    st.sidebar.markdown("---")

    st.sidebar.subheader("📐 ACMR Parameters")
    st.sidebar.markdown(f"- **Confidence High ($C_{{high}}$):** `{settings.CONFIDENCE_HIGH_THRESHOLD}`")
    st.sidebar.markdown(f"- **Confidence Low ($C_{{low}}$):** `{settings.CONFIDENCE_LOW_THRESHOLD}`")
    st.sidebar.markdown(f"- **Max Retrieval Rounds:** `{settings.MAX_RETRIEVAL_ITERATIONS}`")

    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("🩺 MedAgentX Clinical Decision Support System")
        st.caption("Genuinely Adaptive, Gap-Guided Multi-Agent RAG with Safety Classification")
    with col_h2:
        st.info("🟢 System Ready\nLLM: Ollama llama3.2")

    # Scenario Presets — Diverse Real-World Patient Cases & Clinical Vignettes
    scenarios = {
        "Select a Realistic Patient Presentation...": "",
        "Case 1 (Emergency): Septic Shock with Acute Kidney Injury (Lactate 3.8, BP 82/48, eGFR 28)": (
            "68-year-old male presenting with acute altered mental status, rigors, and suspected urosepsis. "
            "Vitals: BP 82/48 mmHg, HR 124 bpm, RR 28, SpO2 91% on room air, Temp 39.1 C. "
            "Labs: Serum Lactate 3.8 mmol/L, eGFR 28 mL/min, Creatinine 2.6 mg/dL, WBC 18,500. "
            "History of Type 2 Diabetes, CKD Stage 3b. Current meds: Metformin 1000mg BID. "
            "Question: What is the immediate fluid resuscitation, vasopressor choice, and antimicrobial protocol?"
        ),
        "Case 2 (ICU/Cardio): Decompensated HFrEF with Severe Hyperkalemia (K+ 5.8, eGFR 22, BP 86/52)": (
            "72-year-old male with acute worsening dyspnea, orthopnea, and bilateral pedal edema. "
            "Vitals: BP 86/52 mmHg, HR 112 bpm, SpO2 87% on room air. "
            "Labs: Serum Potassium K+ 5.8 mEq/L, eGFR 22 mL/min, Creatinine 3.1 mg/dL, elevated BNP. "
            "History of HFrEF (LVEF 25%), CKD Stage 4. "
            "Question: What GDMT medications should be withheld or initiated for acute stabilization?"
        ),
        "Case 3 (Inpatient): Severe CAP with Documented Anaphylactic Penicillin Allergy": (
            "58-year-old female admitted with severe community-acquired pneumonia, pleuritic chest pain, and purulent sputum. "
            "Vitals: BP 118/74 mmHg, HR 98 bpm, SpO2 92% on 2L nasal cannula, Temp 38.8 C. "
            "Documented Allergies: Anaphylactic Penicillin Allergy (urticaria and airway swelling). "
            "Question: What is the recommended inpatient empiric antimicrobial regimen avoiding beta-lactams?"
        ),
        "Case 4 (Pulmonology): Acute COPD Exacerbation with Hypercapnic Respiratory Risk": (
            "66-year-old male with severe COPD presenting with acute breathlessness, wheezing, and purulent sputum. "
            "Vitals: BP 136/82 mmHg, HR 102 bpm, SpO2 84% on room air. Arterial blood gas shows PaCO2 56 mmHg. "
            "Question: What is the targeted oxygen saturation range, bronchodilator regimen, and systemic steroid protocol?"
        ),
        "Case 5 (Safety Alert): Right Ventricular STEMI with Hypotension Risk": (
            "54-year-old male presenting with acute inferior STEMI and Right Ventricular Infarction. "
            "BP 94/60 mmHg, HR 88 bpm. Clinician orders sublingual nitroglycerin for persistent chest discomfort."
        ),
    }

    selected_scenario = st.selectbox("📌 Select Realistic Clinical Patient Vignette or Enter Custom Below:", list(scenarios.keys()))
    default_text = scenarios[selected_scenario] if selected_scenario else ""

    # Query Input
    query_text = st.text_area("✍️ Patient Presentation / Clinical Admission Note / Vignette:", value=default_text, height=140)

    btn_run = st.button("⚡ Run ACMR Clinical Consultation & Patient Safety Evaluation")

    if btn_run:
        if not query_text.strip():
            st.warning("Please enter a clinical patient presentation or select a preset case.")
            return

        system = get_system()

        with st.spinner("Executing Adaptive ACMR dynamic loop across multi-specialty guidelines..."):
            res = system.query_patient(query_text)

        # Extract metadata
        c_total = res.confidence
        route_decision = res.metadata.get("route_decision", "N/A")
        active_agents = res.metadata.get("active_agent_count", 1)
        gap = res.metadata.get("retrieval_gap", "NONE")
        delta_c = res.metadata.get("delta_c", 0.0)
        novelty = res.metadata.get("evidence_novelty", 1.0)
        overlap = res.metadata.get("evidence_overlap", 0.0)
        contradiction_type = res.metadata.get("contradiction_type", "NONE")
        safety_severity = res.metadata.get("safety_severity", "NONE")
        safety_signal = res.metadata.get("safety_signal", False)
        evidence_conflict = res.metadata.get("evidence_conflict", False)

        # Safety Classification Banner (before metrics)
        if route_decision == "SAFETY_CONSTRAINT":
            st.markdown(
                '<div class="safety-constraint-banner">'
                '⚠️ CLINICAL SAFETY CONSTRAINT — The requested intervention is contraindicated in the specified clinical context. '
                'See Safety Alert below.</div>',
                unsafe_allow_html=True,
            )
        elif route_decision == "HARD_ABSTENTION":
            if contradiction_type == "EVIDENCE_CONFLICT":
                st.markdown(
                    '<div class="evidence-conflict-banner">'
                    '🔀 EVIDENCE CONFLICT — Contradictory guideline evidence detected. '
                    'MedAgentX abstains to prevent unsupported recommendations.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="safety-alert-banner">'
                    '🛑 HARD ABSTENTION — Insufficient or unresolvable evidence. '
                    'MedAgentX abstains from generating clinical recommendations.</div>',
                    unsafe_allow_html=True,
                )

        # Metrics Row 1: Confidence & Routing
        st.subheader("📊 Composite Evidence Confidence & Routing Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Composite C_total", f"{c_total:.4f}", delta=f"{delta_c:+.4f}" if delta_c != 0 else None)
        m2.metric("ACMR Decision", route_decision)
        m3.metric("Retrieval Gap", gap)
        m4.metric("Evidence Novelty", f"{novelty:.2f}")
        m5.metric("Evidence Overlap", f"{overlap:.2f}")
        m6.metric("Active Agents", f"{active_agents}")

        # Progress Bar for C_total
        st.progress(min(max(c_total, 0.0), 1.0))

        # Metrics Row 2: Safety Classification
        st.subheader("🛡️ Safety Classification Status")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Contradiction Type", contradiction_type)
        s2.metric("Safety Severity", safety_severity)
        s3.metric("Safety Signal", "YES" if safety_signal else "NO")
        s4.metric("Evidence Conflict", "YES" if evidence_conflict else "NO")

        if res.metadata.get("router_reason"):
            st.info(f"**ACMR Router Decision Rationale:** {res.metadata.get('router_reason')}")

        # Execution Trace
        with st.expander("📜 View Real-Time Multi-Agent Execution Trace & Gap Analysis", expanded=True):
            trace_steps = res.metadata.get("trace", [])
            for step in trace_steps:
                st.markdown(f"- {step}")

        # Final Report
        st.subheader("📄 Clinical Decision Support Report")
        st.text_area("Generated Output:", value=res.answer, height=320)


if __name__ == "__main__":
    main()
