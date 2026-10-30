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
        st.caption("Adaptive Multimodal Multi-Agent Clinical Decision Support with Safety Classification")
    with col_h2:
        st.info("🟢 System Ready\nLLM: Ollama llama3.2")

    tab1, tab2 = st.tabs(["📋 Clinical Vignette & Patient CDSS", "🩻 Multimodal Patient & Medical Scan Intake"])

    with tab1:
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

        selected_scenario = st.selectbox("📌 Select Realistic Clinical Patient Vignette or Enter Custom Below:", list(scenarios.keys()), key="sb_tab1")
        default_text = scenarios[selected_scenario] if selected_scenario else ""

        # Query Input
        query_text = st.text_area("✍️ Patient Presentation / Clinical Admission Note / Vignette:", value=default_text, height=140, key="ta_tab1")

        btn_run = st.button("⚡ Run ACMR Clinical Consultation & Patient Safety Evaluation", key="btn_tab1")

        if btn_run:
            if not query_text.strip():
                st.warning("Please enter a clinical patient presentation or select a preset case.")
            else:
                system = get_system()
                with st.spinner("Executing Adaptive ACMR dynamic loop across multi-specialty guidelines..."):
                    res = system.query_patient(query_text)

                render_results(res)

    with tab2:
        st.subheader("🩻 Multimodal Diagnostic Scan & Lab Document Analysis")
        st.caption("Upload patient radiological imaging (CXR, ECG, CT) and laboratory panels for cross-modal safety verification.")

        mm_presets = {
            "Choose a Sample Diagnostic Study...": {"text": "", "img": "", "report": ""},
            "Sample Study 1: Acute Lobar Pneumonia (Chest X-Ray)": {
                "text": "54-year-old male presenting with 4-day history of high fever, productive cough, and right pleuritic pain.",
                "img": "data/samples/sample_cxr_pneumonia.png",
                "report": "",
            },
            "Sample Study 2: Acute Anterior STEMI (12-Lead ECG)": {
                "text": "61-year-old female with sudden-onset crushing retrosternal chest pain radiating to left arm and diaphoresis.",
                "img": "data/samples/sample_ecg_stemi.png",
                "report": "",
            },
            "Sample Study 3: Acute Cardiogenic Pulmonary Edema (Chest X-Ray)": {
                "text": "70-year-old male with severe orthopnea, bilateral crackles, jugular venous distension, and tachypnea.",
                "img": "data/samples/sample_cxr_pulmonary_edema.png",
                "report": "",
            },
            "Sample Study 4: Tension Pneumothorax Safety Override (Trauma CXR)": {
                "text": "34-year-old male following blunt chest trauma with sudden right-sided dyspnea and tachycardia. Clinician considering positive-pressure ventilation.",
                "img": "data/samples/sample_cxr_pneumothorax.png",
                "report": "",
            },
            "Sample Study 5: Critical Care Sepsis & AKI Panel (Lab Report + Septic Shock)": {
                "text": "72-year-old female admitted to ICU for fever and severe lethargy.",
                "img": "",
                "report": "data/samples/sample_lab_report.txt",
            },
            "Sample Study 6: Head CT — Acute Intracranial Hemorrhage (tPA Contraindication)": {
                "text": "64-year-old male with acute collapse, right hemiparesis, and coma. Clinician orders Head CT and considers IV Alteplase (tPA).",
                "img": "data/datasets/medical_imaging/ct/ct_img_108.jpg",
                "report": "",
            },
            "Sample Study 7: Brain MRI — Acute Ischemic Stroke (DWI Restriction / Code Stroke)": {
                "text": "58-year-old female presenting with 90-minute history of sudden right facial droop and expressive aphasia. Brain MRI shows acute DWI restriction.",
                "img": "data/datasets/medical_imaging/mri/mri_img_389.jpg",
                "report": "",
            },
            "Sample Study 8: Brain MRI — Ventriculomegaly & Communicating Hydrocephalus": {
                "text": "73-year-old male with progressive gait instability, urinary incontinence, and memory decline. Brain MRI shows marked ventriculomegaly with Evans index > 0.32.",
                "img": "data/datasets/medical_imaging/mri/mri_img_170.jpg",
                "report": "",
            },
        }


        selected_mm = st.selectbox("🎯 Load Sample Diagnostic Scan & Lab Case:", list(mm_presets.keys()), key="sb_tab2")
        preset_info = mm_presets[selected_mm]

        col_up1, col_up2 = st.columns(2)
        with col_up1:
            uploaded_image = st.file_uploader(
                "📷 Upload Medical Diagnostic Image (CXR, ECG, CT)",
                type=["png", "jpg", "jpeg"],
                key="img_uploader",
            )
            # Image to display / use
            active_image = None
            if uploaded_image is not None:
                active_image = uploaded_image.read()
                st.image(active_image, caption="Uploaded Patient Diagnostic Image", use_container_width=True)
            elif preset_info["img"] and os.path.exists(preset_info["img"]):
                active_image = preset_info["img"]
                st.image(preset_info["img"], caption=f"Preset Image: {os.path.basename(preset_info['img'])}", use_container_width=True)

        with col_up2:
            uploaded_report = st.file_uploader(
                "📄 Upload Laboratory Report / Panel (TXT, CSV)",
                type=["txt", "csv"],
                key="doc_uploader",
            )
            report_content = ""
            if uploaded_report is not None:
                report_content = uploaded_report.read().decode("utf-8", errors="ignore")
                st.text_area("Uploaded Laboratory Report Content:", value=report_content, height=220)
            elif preset_info["report"] and os.path.exists(preset_info["report"]):
                with open(preset_info["report"], "r", encoding="utf-8") as f:
                    report_content = f.read()
                st.text_area("Preset Laboratory Report Content:", value=report_content, height=220)

        mm_text = st.text_area(
            "📝 Clinical Vignette / Presentation & Patient History:",
            value=preset_info["text"] if not uploaded_image and not uploaded_report else "",
            height=100,
            key="ta_tab2",
        )

        btn_mm_run = st.button("🩻 Run Multimodal ACMR Decision Support & Safety Analysis", key="btn_tab2")

        if btn_mm_run:
            if not active_image and not report_content and not mm_text.strip():
                st.warning("Please upload a medical scan, lab report, or enter a clinical presentation.")
            else:
                system = get_system()
                with st.spinner("Analyzing diagnostic image, extracting tabular lab biomarkers, and running ACMR loop..."):
                    res = system.query_multimodal(
                        patient_text=mm_text,
                        report_input=report_content,
                        image_input=active_image,
                    )

                render_results(res)


def render_results(res):
    """Helper to render ACMR decision, safety classification, and clinical report."""
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

    # Metrics Row 2: Safety Classification & Multimodal Perception
    st.subheader("🛡️ Safety Classification & Perception Status")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Contradiction Type", contradiction_type)
    s2.metric("Safety Severity", safety_severity)
    s3.metric("Safety Signal", "YES" if safety_signal else "NO")
    s4.metric("Evidence Conflict", "YES" if evidence_conflict else "NO")

    # If imaging metadata exists, show perception card
    if res.metadata.get("imaging_impression"):
        st.info(
            f"**🩻 Imaging Perception:** {res.metadata.get('imaging_modality', 'Scan')} — "
            f"**Impression:** {res.metadata.get('imaging_impression')} "
            f"(Urgency: `{res.metadata.get('imaging_urgency', 'ROUTINE')}`)"
        )

    if res.metadata.get("router_reason"):
        st.info(f"**ACMR Router Decision Rationale:** {res.metadata.get('router_reason')}")

    # Execution Trace
    with st.expander("📜 View Real-Time Multi-Agent Execution Trace & Gap Analysis", expanded=True):
        trace_steps = res.metadata.get("trace", [])
        for step in trace_steps:
            st.markdown(f"- {step}")

    # Final Report
    st.subheader("📄 Clinical Decision Support Report")
    st.text_area("Generated Output:", value=res.answer, height=360)



if __name__ == "__main__":
    main()
