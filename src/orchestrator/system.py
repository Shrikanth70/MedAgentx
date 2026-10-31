"""MedAgentX Orchestrator System Entrypoint with Adaptive State & Observability."""

import time
from typing import Any
import structlog
from config.settings import settings
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.verifier import VerifierAgent
from src.agents.reasoner import ReasonerAgent
from src.orchestrator.confidence import ConfidenceCalculator
from src.orchestrator.router import ACMRRouter, RouterDecision
from src.baselines.base import SystemResponse

logger = structlog.get_logger()


class MedAgentXSystem:
    """MedAgentX ACMR System Orchestrator with Stateful Adaptive Loop & Safety Policy."""

    def __init__(self):
        self.system_id = "PROPOSED_MEDAGENTX"
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.verifier = VerifierAgent()
        self.reasoner = ReasonerAgent()
        self.router = ACMRRouter()

    def query(self, user_query: str) -> SystemResponse:
        """Execute Adaptive Confidence-Guided Multi-Agent Retrieval (ACMR) workflow."""
        start_time = time.time()
        trace = []
        active_agents = 0

        # Initial Plan
        plan = self.planner.plan(user_query)
        active_agents += 1
        trace.append("Activated Agent 1: Planner & Reformulator")

        # Persistent ACMR Loop State Data Structure
        acmr_state = {
            "iteration": 0,
            "previous_query": None,
            "current_query": user_query,
            "previous_confidence": None,
            "current_confidence": None,
            "confidence_delta": None,
            "previous_document_ids": set(),
            "current_document_ids": set(),
            "previous_chunks": [],
            "previous_gaps": [],
            "evidence_overlap": 0.0,
            "evidence_novelty": 1.0,
            "retrieval_gap": "NONE",
            "router_reason": "",
            "contradiction_type": "NONE",
            "safety_severity": "NONE",
            "safety_signal": False,
            "evidence_conflict": False,
            "clinical_context": "",
            "intervention": "",
            "safety_reason": "",
        }

        iteration = 0
        c_total = 0.0
        prev_c = 0.0
        delta_c = 0.0
        current_query = user_query
        action = "RE_RETRIEVE"
        verifier_output = None
        router_decision = None

        while iteration <= settings.MAX_RETRIEVAL_ITERATIONS and action in ["RE_RETRIEVE", "TARGETED_RE_RETRIEVE", "QUERY_REWRITE_RE_RETRIEVE", "RESOLVE_CONTRADICTION"]:
            acmr_state["iteration"] = iteration
            acmr_state["previous_query"] = current_query

            # Step A: Adapt Query if iteration > 0
            if iteration > 0 and verifier_output:
                current_query = self.planner.adapt_query(
                    original_query=user_query,
                    retrieval_gap=verifier_output.retrieval_gap,
                    missing_evidence=verifier_output.missing_evidence,
                    previous_queries=list(acmr_state["previous_gaps"]),
                )
                trace.append(f"Iteration {iteration} - Adapted Query: '{current_query}' (Gap: {verifier_output.retrieval_gap})")

            acmr_state["current_query"] = current_query

            # Step B: Agent 2 (Retriever)
            retrieval_res = self.retriever.retrieve_evidence(
                query_text=current_query, entities=plan.extracted_entities
            )
            active_agents += 1

            curr_chunks = retrieval_res.vector_evidence
            curr_chunk_ids = {c.get("chunk_id", f"chunk_{idx}") for idx, c in enumerate(curr_chunks)}
            acmr_state["current_document_ids"] = curr_chunk_ids

            # Calculate Evidence Overlap
            if acmr_state["previous_document_ids"]:
                overlap_count = len(curr_chunk_ids & acmr_state["previous_document_ids"])
                evidence_overlap = round(overlap_count / max(1, len(curr_chunk_ids)), 4)
            else:
                evidence_overlap = 0.0

            acmr_state["evidence_overlap"] = evidence_overlap

            # Step C: Agent 3 (Verifier with Safety Classification)
            verifier_output = self.verifier.verify(
                query_text=user_query,
                vector_evidence=curr_chunks,
                graph_evidence=retrieval_res.graph_evidence,
                previous_chunks=acmr_state["previous_chunks"],
            )
            active_agents += 1
            acmr_state["evidence_novelty"] = verifier_output.evidence_novelty
            acmr_state["retrieval_gap"] = verifier_output.retrieval_gap
            acmr_state["contradiction_type"] = verifier_output.contradiction_type
            acmr_state["safety_severity"] = verifier_output.safety_severity
            acmr_state["safety_signal"] = verifier_output.safety_signal
            acmr_state["evidence_conflict"] = verifier_output.evidence_conflict
            acmr_state["clinical_context"] = verifier_output.clinical_context
            acmr_state["intervention"] = verifier_output.intervention
            acmr_state["safety_reason"] = verifier_output.safety_reason

            # Step D: Calculate Composite Confidence & Delta C
            c_total = ConfidenceCalculator.calculate_composite_confidence(
                s_vec=retrieval_res.s_vec,
                s_graph=retrieval_res.s_graph,
                v_agree=verifier_output.v_agree,
                r_contra=verifier_output.r_contra,
                s_rel=verifier_output.s_rel,
                completeness=verifier_output.evidence_completeness,
                novelty=verifier_output.evidence_novelty,
                contradiction_type=verifier_output.contradiction_type,
            )
            acmr_state["current_confidence"] = c_total

            if iteration > 0 and acmr_state["previous_confidence"] is not None:
                delta_c = round(c_total - acmr_state["previous_confidence"], 4)
            else:
                delta_c = 0.0

            acmr_state["confidence_delta"] = delta_c

            # Loop Detection Check
            is_loop_detected = (
                iteration > 0
                and (
                    (current_query == acmr_state["previous_query"] and evidence_overlap >= 0.85)
                    or (evidence_overlap >= 0.85 and abs(delta_c) < 0.01)
                )
            )

            if is_loop_detected:
                trace.append("ACMR Loop Detector: Identical retrieval state detected. Halting repetition.")

            # Step E: ACMR Router Decision
            router_decision = self.router.route_explainable(
                c_total=c_total,
                delta_c=delta_c,
                retrieval_gap=verifier_output.retrieval_gap,
                missing_evidence=verifier_output.missing_evidence,
                s_rel=verifier_output.s_rel,
                r_contra=verifier_output.r_contra,
                contradiction_type=verifier_output.contradiction_type,
                safety_severity=verifier_output.safety_severity,
                evidence_novelty=verifier_output.evidence_novelty,
                evidence_overlap=evidence_overlap,
                iteration=iteration,
                is_loop_detected=is_loop_detected,
            )
            action = router_decision.action
            acmr_state["router_reason"] = router_decision.reason

            # Rich Observability Execution Trace Formatting (ASCII Safe)
            obs_block = (
                f"\n==================================================\n"
                f"ACMR ITERATION {iteration}\n"
                f"==================================================\n"
                f"Original Query: {user_query}\n"
                f"Adapted Query:  {current_query}\n"
                f"Safety Classification: {verifier_output.contradiction_type}\n"
                f"Safety Severity:       {verifier_output.safety_severity}\n"
                f"Retrieval Gap:  {verifier_output.retrieval_gap}\n"
                f"Missing Evid:   {verifier_output.missing_evidence}\n"
                f"Retrieved Docs: {len(curr_chunks)}\n"
                f"Evidence Overlap: {evidence_overlap:.2f}\n"
                f"S_vec: {retrieval_res.s_vec:.2f} | S_graph: {retrieval_res.s_graph:.2f} | S_rel: {verifier_output.s_rel:.2f}\n"
                f"V_agree: {verifier_output.v_agree:.2f} | R_contra: {verifier_output.r_contra:.2f}\n"
                f"Novelty: {verifier_output.evidence_novelty:.2f} | Completeness: {verifier_output.evidence_completeness:.2f}\n"
                f"C_total: {c_total:.4f} (Delta C: {'N/A' if iteration == 0 else f'{delta_c:+.4f}'})\n"
                f"Router Action: {router_decision.action}\n"
                f"Reason: {router_decision.reason}\n"
                f"=================================================="
            )
            logger.info(obs_block)
            trace.append(f"Iteration {iteration} -> {router_decision.action}: {router_decision.reason}")

            # Update State
            acmr_state["previous_document_ids"].update(curr_chunk_ids)
            acmr_state["previous_chunks"].extend(curr_chunks)
            acmr_state["previous_gaps"].append(current_query)
            acmr_state["previous_confidence"] = c_total

            if action in ["STOP_CONFIDENT", "STOP_SATURATED", "STOP_MAX_ITERATIONS", "SAFETY_CONSTRAINT", "HARD_ABSTENTION"]:
                break

            iteration += 1

        # Step F: Agent 4 (Reasoner & Synthesizer) with Safety Mode Selection
        active_agents += 1
        trace.append(f"Activated Agent 4: Clinical Reasoner & Synthesizer (Action: {action})")

        if action == "SAFETY_CONSTRAINT":
            reasoner_output = self.reasoner.synthesize(
                query_text=user_query,
                verified_evidence=verifier_output.verified_evidence if verifier_output else [],
                detected_contradictions=verifier_output.detected_contradictions if verifier_output else [],
                c_total=c_total,
                mode="SAFETY_CONSTRAINED",
                intervention=acmr_state["intervention"],
                clinical_context=acmr_state["clinical_context"],
                safety_reason=acmr_state["safety_reason"],
            )
            answer = reasoner_output.clinical_report

        elif action == "HARD_ABSTENTION":
            reasoner_output = self.reasoner.synthesize(
                query_text=user_query,
                verified_evidence=verifier_output.verified_evidence if verifier_output else [],
                detected_contradictions=verifier_output.detected_contradictions if verifier_output else [],
                c_total=c_total,
                mode="ABSTENTION",
                safety_reason=acmr_state["safety_reason"] or router_decision.reason,
            )
            answer = reasoner_output.clinical_report

        else:
            reasoner_output = self.reasoner.synthesize(
                query_text=user_query,
                verified_evidence=verifier_output.verified_evidence if verifier_output else [],
                detected_contradictions=verifier_output.detected_contradictions if verifier_output else [],
                c_total=c_total,
                mode="NORMAL",
            )
            answer = reasoner_output.clinical_report

        latency = time.time() - start_time
        token_count = len(user_query.split()) + len(answer.split())

        return SystemResponse(
            system_id=self.system_id,
            query=user_query,
            answer=answer,
            evidence_chunks=verifier_output.verified_evidence if verifier_output else [],
            citations=reasoner_output.citations if reasoner_output else [],
            confidence=c_total,
            latency_seconds=round(latency, 4),
            token_count=token_count,
            metadata={
                "c_total": c_total,
                "delta_c": delta_c,
                "retrieval_rounds": iteration,
                "route_decision": action,
                "active_agent_count": active_agents,
                "trace": trace,
                "s_rel": verifier_output.s_rel if verifier_output else 0.0,
                "retrieval_gap": verifier_output.retrieval_gap if verifier_output else "NONE",
                "missing_evidence": verifier_output.missing_evidence if verifier_output else [],
                "router_reason": router_decision.reason if router_decision else "",
                "evidence_overlap": acmr_state["evidence_overlap"],
                "evidence_novelty": acmr_state["evidence_novelty"],
                "contradiction_type": acmr_state["contradiction_type"],
                "safety_severity": acmr_state["safety_severity"],
                "safety_signal": acmr_state["safety_signal"],
                "evidence_conflict": acmr_state["evidence_conflict"],
                "clinical_context": acmr_state["clinical_context"],
                "intervention": acmr_state["intervention"],
            },
        )

    def query_patient(self, patient_or_text: Any) -> SystemResponse:
        """Execute real-world patient clinical decision support through ACMR pipeline."""
        from src.orchestrator.patient import PatientCase, PatientVignetteParser

        if isinstance(patient_or_text, str):
            patient = PatientVignetteParser.parse(patient_or_text)
        elif isinstance(patient_or_text, dict):
            patient = PatientCase(**patient_or_text)
        else:
            patient = patient_or_text

        start_time = time.time()
        trace = []
        active_agents = 0

        # Step 1: Agent 1 (Planner for Patient)
        plan = self.planner.plan_patient(patient)
        active_agents += 1
        trace.append(f"Activated Agent 1: Clinical Vignette Planner for Patient {getattr(patient, 'patient_id', 'PT-001')}")

        iteration = 0
        c_total = 0.0
        delta_c = 0.0
        current_query = plan.original_query
        action = "RE_RETRIEVE"
        verifier_output = None
        router_decision = None
        previous_chunks: list[dict[str, Any]] = []

        while iteration <= settings.MAX_RETRIEVAL_ITERATIONS and action in ["RE_RETRIEVE", "TARGETED_RE_RETRIEVE", "QUERY_REWRITE_RE_RETRIEVE", "RESOLVE_CONTRADICTION"]:
            # Step B: Agent 2 (Retriever)
            retrieval_res = self.retriever.retrieve_evidence(
                query_text=current_query, entities=plan.extracted_entities
            )
            active_agents += 1
            curr_chunks = retrieval_res.vector_evidence

            # Step C: Agent 3 (Verifier with Patient Vital & Lab Safety Guard)
            verifier_output = self.verifier.verify_patient(
                query_text=current_query,
                vector_evidence=curr_chunks,
                patient=patient,
                graph_evidence=retrieval_res.graph_evidence,
                previous_chunks=previous_chunks,
            )
            active_agents += 1

            # Step D: Composite Confidence Calculation
            c_total = ConfidenceCalculator.calculate_composite_confidence(
                s_vec=retrieval_res.s_vec,
                s_graph=retrieval_res.s_graph,
                v_agree=verifier_output.v_agree,
                r_contra=verifier_output.r_contra,
                s_rel=verifier_output.s_rel,
                completeness=verifier_output.evidence_completeness,
                novelty=verifier_output.evidence_novelty,
                contradiction_type=verifier_output.contradiction_type,
            )

            # Step E: ACMR Dynamic Router
            router_decision = self.router.route_explainable(
                c_total=c_total,
                delta_c=0.0,
                retrieval_gap=verifier_output.retrieval_gap,
                missing_evidence=verifier_output.missing_evidence,
                s_rel=verifier_output.s_rel,
                r_contra=verifier_output.r_contra,
                contradiction_type=verifier_output.contradiction_type,
                safety_severity=verifier_output.safety_severity,
                evidence_novelty=verifier_output.evidence_novelty,
                evidence_overlap=0.0,
                iteration=iteration,
            )
            action = router_decision.action
            trace.append(f"Iteration {iteration} -> {action}: {router_decision.reason}")

            previous_chunks.extend(curr_chunks)

            if action in ["STOP_CONFIDENT", "STOP_SATURATED", "STOP_MAX_ITERATIONS", "SAFETY_CONSTRAINT", "HARD_ABSTENTION"]:
                break

            iteration += 1

        # Step F: Agent 4 (Reasoner with Tailored Patient Synthesis)
        active_agents += 1
        trace.append(f"Activated Agent 4: Patient-Tailored Reasoner (Action: {action})")

        safety_reason = (
            router_decision.reason
            if router_decision and action in ["SAFETY_CONSTRAINT", "HARD_ABSTENTION"]
            else ""
        )

        reasoner_output = self.reasoner.synthesize_patient_report(
            patient=patient,
            verified_evidence=verifier_output.verified_evidence if verifier_output else [],
            detected_contradictions=verifier_output.detected_contradictions if verifier_output else [],
            c_total=c_total,
            mode=action,
            safety_reason=safety_reason,
        )

        json_report = self.reasoner.export_report_json(
            patient=patient,
            reasoner_output=reasoner_output,
        )
        html_report = self.reasoner.export_report_html(
            patient=patient,
            reasoner_output=reasoner_output,
        )

        latency = time.time() - start_time
        token_count = len(patient.summarize_clinical_state().split()) + len(reasoner_output.clinical_report.split())

        return SystemResponse(
            system_id=self.system_id,
            query=getattr(patient, "chief_complaint", "") or current_query,
            answer=reasoner_output.clinical_report,
            evidence_chunks=verifier_output.verified_evidence if verifier_output else [],
            citations=reasoner_output.citations,
            confidence=c_total,
            latency_seconds=round(latency, 4),
            token_count=token_count,
            metadata={
                "c_total": c_total,
                "delta_c": delta_c,
                "retrieval_rounds": iteration,
                "route_decision": action,
                "active_agent_count": active_agents,
                "trace": trace,
                "s_rel": verifier_output.s_rel if verifier_output else 0.0,
                "contradiction_type": verifier_output.contradiction_type if verifier_output else "NONE",
                "safety_severity": verifier_output.safety_severity if verifier_output else "NONE",
                "router_reason": router_decision.reason if router_decision else "",
                "json_report": json_report,
                "html_report": html_report,
            },
        )

    def query_multimodal(
        self,
        patient_text: str = "",
        report_input: Any = None,
        image_input: Any = None,
        image_name: str = "",
    ) -> SystemResponse:
        """Execute multimodal clinical decision support synthesizing scans, lab reports, and vignettes."""
        from src.multimodal.contracts import MultimodalPatientCase
        from src.multimodal.document_parser import ClinicalDocumentParser
        from src.multimodal.imaging_analyzer import MedicalImagingAnalyzer
        from src.orchestrator.patient import PatientCase, PatientVignetteParser

        # 1. Parse patient clinical text or initialize baseline case
        if patient_text:
            base_patient = PatientVignetteParser.parse(patient_text)
        else:
            base_patient = PatientCase(
                patient_id="PT-MM-01",
                chief_complaint="Multimodal Diagnostic Consultation",
                hpi="Clinical scan and laboratory evaluation submitted for ACMR decision support.",
            )

        # 2. Extract and incorporate structured lab values if report is uploaded
        parsed_labs: dict[str, Any] = {}
        if report_input:
            parsed_labs = ClinicalDocumentParser.parse_report(report_input)
            for lab_key, val in parsed_labs.items():
                if hasattr(base_patient, lab_key):
                    setattr(base_patient, lab_key, val)

        # 3. Analyze medical imaging scans (Chest X-Rays, 12-lead ECGs, CTs)
        has_image = (
            image_input is not None
            and (not isinstance(image_input, str) or image_input.strip() != "")
        )
        if has_image:
            analyzer = MedicalImagingAnalyzer()
            hint = f"{base_patient.chief_complaint} {base_patient.hpi} {patient_text}".strip()
            vision_result = analyzer.analyze_vision(
                image_input=image_input,
                clinical_hint=hint,
                file_name=image_name,
            )
            imaging_analysis = analyzer.map_vision_result(vision_result)

            visual_findings = vision_result.findings
            vision_status = (
                vision_result.status.value
                if hasattr(vision_result.status, "value")
                else str(vision_result.status)
            )
            vision_uncertainty = max([f.uncertainty for f in visual_findings], default=0.0)
            requires_imaging = True
            is_critical_pathology = any(f.is_critical for f in visual_findings)
            quality_rejection_reasons = (
                vision_result.quality.rejection_reasons
                if (vision_result.quality and vision_result.quality.rejection_reasons)
                else []
            )
        else:
            vision_result = None
            imaging_analysis = None
            visual_findings = None
            vision_status = "OK"
            vision_uncertainty = 0.0
            requires_imaging = False
            is_critical_pathology = False
            quality_rejection_reasons = []

        # 4. Construct unified MultimodalPatientCase
        multimodal_case = MultimodalPatientCase(
            **base_patient.model_dump(),
            imaging_analysis=imaging_analysis,
            uploaded_document_text=str(report_input)[:1000] if report_input else "",
            uploaded_file_name=image_name or (str(image_input) if isinstance(image_input, str) else "uploaded_scan"),
        )

        start_time = time.time()
        trace = []
        active_agents = 0

        # Step 1: Agent 1 (Planner for Patient)
        plan = self.planner.plan_patient(multimodal_case)
        active_agents += 1
        trace.append(f"Activated Agent 1: Clinical Vignette Planner for Patient {getattr(base_patient, 'patient_id', 'PT-MM-01')}")

        iteration = 0
        c_total = 0.0
        delta_c = 0.0
        prev_c = 0.0
        current_query = plan.original_query
        action = "RE_RETRIEVE"
        verifier_output = None
        router_decision = None
        previous_chunks: list[dict[str, Any]] = []
        previous_document_ids: set[str] = set()

        while iteration <= settings.MAX_RETRIEVAL_ITERATIONS and action in [
            "RE_RETRIEVE",
            "TARGETED_RE_RETRIEVE",
            "QUERY_REWRITE_RE_RETRIEVE",
            "RESOLVE_CONTRADICTION",
        ]:
            # Step B: Agent 2 (Retriever)
            retrieval_res = self.retriever.retrieve(
                query_text=current_query,
                entities=plan.extracted_entities,
                visual_findings=visual_findings,
                patient_context=multimodal_case,
                modality=imaging_analysis.modality if imaging_analysis else "",
            )
            active_agents += 1
            curr_chunks = retrieval_res.vector_evidence
            curr_chunk_ids = {c.get("chunk_id", f"chunk_{idx}") for idx, c in enumerate(curr_chunks)}

            if previous_document_ids:
                overlap_count = len(curr_chunk_ids & previous_document_ids)
                evidence_overlap = round(overlap_count / max(1, len(curr_chunk_ids)), 4)
            else:
                evidence_overlap = 0.0

            # Step C: Agent 3 (Verifier with Patient Vital & Lab Safety Guard)
            verifier_output = self.verifier.verify(
                query_text=current_query,
                vector_evidence=curr_chunks,
                graph_evidence=retrieval_res.graph_evidence,
                previous_chunks=previous_chunks,
                visual_findings=visual_findings,
                vision_result=vision_result,
                patient=base_patient,
            )
            active_agents += 1

            # Step D: Composite Confidence Calculation
            c_total = ConfidenceCalculator.calculate_composite_confidence(
                s_vec=retrieval_res.s_vec,
                s_graph=retrieval_res.s_graph,
                v_agree=verifier_output.v_agree,
                r_contra=verifier_output.r_contra,
                s_rel=verifier_output.s_rel,
                completeness=verifier_output.evidence_completeness,
                novelty=verifier_output.evidence_novelty,
                contradiction_type=verifier_output.contradiction_type,
                vision_status=vision_status,
                vision_uncertainty=vision_uncertainty,
                evidence_count=retrieval_res.evidence_count,
            )

            if iteration > 0:
                delta_c = round(c_total - prev_c, 4)
            else:
                delta_c = 0.0

            # Step E: ACMR Dynamic Router
            effective_critical = (
                (verifier_output.is_critical_pathology if verifier_output else False)
                or is_critical_pathology
            )
            router_decision = self.router.route_explainable(
                c_total=c_total,
                delta_c=delta_c,
                retrieval_gap=verifier_output.retrieval_gap,
                missing_evidence=verifier_output.missing_evidence,
                s_rel=verifier_output.s_rel,
                r_contra=verifier_output.r_contra,
                contradiction_type=verifier_output.contradiction_type,
                safety_severity=verifier_output.safety_severity,
                evidence_novelty=verifier_output.evidence_novelty,
                evidence_overlap=evidence_overlap,
                iteration=iteration,
                vision_status=vision_status,
                requires_imaging=requires_imaging,
                is_critical_pathology=effective_critical,
                quality_rejection_reasons=quality_rejection_reasons,
                evidence_count=retrieval_res.evidence_count,
            )
            action = router_decision.action
            trace.append(f"Iteration {iteration} -> {action}: {router_decision.reason}")

            previous_chunks.extend(curr_chunks)
            previous_document_ids.update(curr_chunk_ids)
            prev_c = c_total

            if action in [
                "STOP_CONFIDENT",
                "STOP_SATURATED",
                "STOP_MAX_ITERATIONS",
                "SAFETY_CONSTRAINT",
                "HARD_ABSTENTION",
                "ABSTAIN_INSUFFICIENT_EVIDENCE",
                "ESCALATE_TO_HUMAN",
                "REQUEST_MORE_INPUT",
            ]:
                break

            iteration += 1

        # Step F: Agent 4 (Reasoner with Tailored Patient Synthesis)
        active_agents += 1
        trace.append(f"Activated Agent 4: Patient-Tailored Reasoner (Action: {action})")

        safety_reason = (
            router_decision.reason
            if router_decision and action in [
                "SAFETY_CONSTRAINT",
                "HARD_ABSTENTION",
                "ABSTAIN_INSUFFICIENT_EVIDENCE",
                "ESCALATE_TO_HUMAN",
                "REQUEST_MORE_INPUT",
            ]
            else ""
        )

        reasoner_output = self.reasoner.synthesize_patient_report(
            patient=multimodal_case,
            verified_evidence=verifier_output.verified_evidence if verifier_output else [],
            detected_contradictions=verifier_output.detected_contradictions if verifier_output else [],
            c_total=c_total,
            mode=action,
            safety_reason=safety_reason,
            vision_result=vision_result,
            visual_findings=visual_findings,
            critical_alerts=verifier_output.critical_alerts if verifier_output else [],
            cross_modal_contradictions=verifier_output.cross_modal_contradictions if verifier_output else [],
        )

        json_report = self.reasoner.export_report_json(
            patient=multimodal_case,
            reasoner_output=reasoner_output,
            vision_result=vision_result,
        )
        html_report = self.reasoner.export_report_html(
            patient=multimodal_case,
            reasoner_output=reasoner_output,
            vision_result=vision_result,
        )

        latency = time.time() - start_time
        token_count = len(multimodal_case.summarize_clinical_state().split()) + len(reasoner_output.clinical_report.split())

        res = SystemResponse(
            system_id=self.system_id,
            query=getattr(base_patient, "chief_complaint", "") or current_query,
            answer=reasoner_output.clinical_report,
            evidence_chunks=verifier_output.verified_evidence if verifier_output else [],
            citations=reasoner_output.citations,
            confidence=c_total,
            latency_seconds=round(latency, 4),
            token_count=token_count,
            metadata={
                "c_total": c_total,
                "delta_c": delta_c,
                "retrieval_rounds": iteration,
                "route_decision": action,
                "active_agent_count": active_agents,
                "trace": trace,
                "s_rel": verifier_output.s_rel if verifier_output else 0.0,
                "contradiction_type": verifier_output.contradiction_type if verifier_output else "NONE",
                "safety_severity": verifier_output.safety_severity if verifier_output else "NONE",
                "router_reason": router_decision.reason if router_decision else "",
                "evidence_overlap": evidence_overlap,
                "evidence_novelty": verifier_output.evidence_novelty if verifier_output else 1.0,
                "retrieval_gap": verifier_output.retrieval_gap if verifier_output else "NONE",
                "missing_evidence": verifier_output.missing_evidence if verifier_output else [],
                "safety_signal": verifier_output.safety_signal if verifier_output else False,
                "evidence_conflict": verifier_output.evidence_conflict if verifier_output else False,
                "vision_status": vision_status,
                "vision_result": vision_result.model_dump() if vision_result else None,
                "quality": vision_result.quality.model_dump() if (vision_result and vision_result.quality) else None,
                "requires_imaging": requires_imaging,
                "is_critical_pathology": effective_critical,
                "json_report": json_report,
                "html_report": html_report,
            },
        )

        # Inject multimodal metadata for API and Streamlit UI consumption
        if imaging_analysis is not None:
            res.metadata["imaging_modality"] = imaging_analysis.modality
            res.metadata["imaging_impression"] = imaging_analysis.diagnostic_impression
            res.metadata["imaging_urgency"] = imaging_analysis.critical_urgency
            res.metadata["contraindicated_interventions"] = imaging_analysis.contraindicated_interventions
            res.metadata["radiological_findings"] = imaging_analysis.radiological_findings

        if parsed_labs:
            res.metadata["parsed_labs"] = parsed_labs

        return res

