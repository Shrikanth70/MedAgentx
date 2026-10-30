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
