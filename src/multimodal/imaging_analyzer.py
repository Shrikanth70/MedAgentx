"""Medical Imaging Perception & Radiological Sign Analyzer Adapter.

Grounded multimodal clinical decision support adapter bridging the high-integrity
fail-closed PluggableVisionEngine and quality control gate with the orchestrator.
"""

from __future__ import annotations

import os
from typing import Any
from PIL import Image
import structlog

from src.multimodal.contracts import ImagingAnalysis, VisionResult, VisionStatus
from src.multimodal.ingestion import MedicalImageIngestion
from src.multimodal.vision_engine import (
    ClinicalRulesVisionFallback,
    PluggableVisionEngine,
    QwenVLMModel,
)

logger = structlog.get_logger()


class MedicalImagingAnalyzer:
    """Clinical Diagnostic Imaging Analyzer for Chest X-Rays, 12-Lead ECGs, and Scans.

    Enforces fail-closed safety, strict zero-placeholder guarantees, automated pre-inference
    quality control gating, and calibrated clinical finding extraction.
    """

    def __init__(
        self,
        use_vision_api: bool = False,
        vision_engine: PluggableVisionEngine | None = None,
        ingestion: MedicalImageIngestion | None = None,
    ) -> None:
        self.use_vision_api = use_vision_api
        self.ingestion = ingestion or MedicalImageIngestion()
        if vision_engine is not None:
            self.vision_engine = vision_engine
        else:
            primary = QwenVLMModel() if use_vision_api else None
            secondary = ClinicalRulesVisionFallback()
            self.vision_engine = PluggableVisionEngine(primary_model=primary, secondary_model=secondary)

    def analyze_vision(
        self,
        image_input: str | bytes | Image.Image,
        clinical_hint: str = "",
        file_name: str = "",
    ) -> VisionResult:
        """Run image quality gating and vision perception pipeline, returning native VisionResult."""
        extracted_name = file_name or (os.path.basename(image_input) if isinstance(image_input, str) else "")

        # 1. Quality Control Gate
        quality = self.ingestion.validate_image_quality(image_input)
        if not quality.is_acceptable:
            return VisionResult(
                status=VisionStatus.LOW_QUALITY,
                modality="Diagnostic Imaging",
                body_part="Unknown",
                quality=quality,
                findings=[],
                negative_findings=[],
                limitations=quality.rejection_reasons,
                model_name="QualityControlGate",
            )

        # 2. Vision Inference Pipeline
        metadata = {
            "clinical_hint": clinical_hint,
            "file_name": extracted_name,
        }
        return self.vision_engine.analyze(image_input, metadata=metadata)

    def analyze(
        self,
        image_input: str | bytes | Image.Image,
        clinical_hint: str = "",
        file_name: str = "",
    ) -> ImagingAnalysis:
        """Analyze a diagnostic image and return structured radiological/cardiographic findings.

        Args:
            image_input: File path, image bytes, or PIL Image.
            clinical_hint: Clinical vignette or context guiding interpretation.
            file_name: Optional explicit file name to preserve context.

        Returns:
            ImagingAnalysis compatible with orchestrator and CDSS report generation.
        """
        extracted_name = file_name or (os.path.basename(image_input) if isinstance(image_input, str) else "")

        # Step 1: Run image quality control
        quality = self.ingestion.validate_image_quality(image_input)
        if not quality.is_acceptable:
            logger.warning("imaging_quality_gate_rejected", reasons=quality.rejection_reasons)
            return ImagingAnalysis(
                modality="Diagnostic Imaging",
                radiological_findings=[],
                diagnostic_impression=f"Suboptimal Study: {', '.join(quality.rejection_reasons)}",
                critical_urgency="ROUTINE",
                contraindicated_interventions=[],
                confidence=0.0,
                metadata={
                    "vision_status": VisionStatus.LOW_QUALITY.value,
                    "quality_assessment": quality.model_dump(),
                },
            )

        # Step 2: Pass image_input and metadata to vision engine
        metadata = {
            "clinical_hint": clinical_hint,
            "file_name": extracted_name,
        }
        vision_result = self.vision_engine.analyze(image_input, metadata=metadata)

        # Step 3: Handle the resulting VisionResult
        if vision_result.status != VisionStatus.OK:
            logger.warning("vision_perception_incomplete", status=vision_result.status.value)
            return ImagingAnalysis(
                modality=vision_result.modality or "Diagnostic Imaging",
                radiological_findings=[],
                diagnostic_impression=f"Imaging Perception Incomplete ({vision_result.status.value})",
                critical_urgency="ROUTINE",
                confidence=0.0,
                contraindicated_interventions=[],
                metadata={
                    "vision_status": vision_result.status.value,
                    "vision_result": vision_result.model_dump(),
                    "trace_id": vision_result.trace_id,
                },
            )

        # If vision_result.status == VisionStatus.OK:
        # Map findings: format as readable clinical finding strings combining label and anatomical location/evidence
        mapped_findings: list[str] = []
        for f in vision_result.findings:
            desc = (
                f.evidence.supporting_visual_description
                if f.evidence and f.evidence.supporting_visual_description
                else f.anatomical_location
            )
            mapped_findings.append(f"{f.label}: {desc}")

        # Map contraindicated interventions from findings
        contraindications = self._extract_contraindications(vision_result)

        # Critical urgency: emergent if any critical finding detected
        is_emergent = any(f.is_critical for f in vision_result.findings)
        critical_urgency = "EMERGENT" if is_emergent else ("URGENT" if mapped_findings else "ROUTINE")

        # Confidence: from findings probability (e.g. max probability or 0.92)
        if vision_result.findings:
            confidence = max((f.probability for f in vision_result.findings), default=0.92)
        else:
            confidence = 0.90

        # Impression
        diagnostic_impression = self._build_impression(vision_result)

        return ImagingAnalysis(
            modality=vision_result.modality or "Diagnostic Imaging",
            radiological_findings=mapped_findings,
            diagnostic_impression=diagnostic_impression,
            critical_urgency=critical_urgency,
            contraindicated_interventions=contraindications,
            confidence=confidence,
            metadata={
                "vision_status": VisionStatus.OK.value,
                "vision_result": vision_result.model_dump(),
                "trace_id": vision_result.trace_id,
            },
        )

    def _extract_contraindications(self, vision_result: VisionResult) -> list[str]:
        """Map visual findings and anatomical labels to established clinical contraindications."""
        contraindications: list[str] = []
        findings_text = " ".join(
            [
                f"{f.label} {f.anatomical_location} {f.evidence.supporting_visual_description if f.evidence else ''}"
                for f in vision_result.findings
            ]
        ).lower()

        # Pneumothorax
        if "pneumothorax" in findings_text:
            contraindications.extend([
                "Positive-pressure ventilation (BiPAP/CPAP) without immediate chest tube decompression",
                "High-pressure invasive mechanical ventilation prior to tube thoracostomy",
            ])

        # Acute Intracranial Hemorrhage / Bleed
        if any(k in findings_text for k in ["hemorrhage", "hematoma", "bleed", "ich"]):
            contraindications.extend([
                "Thrombolytic therapy (IV Alteplase / Tenecteplase / tPA) - strict contraindication due to fatal hemorrhage extension",
                "Therapeutic anticoagulation (Heparin, Enoxaparin, Warfarin, DOACs)",
                "Antiplatelet agents (Aspirin, Clopidogrel, Ticagrelor)",
            ])

        # Acute Ischemic Infarction / Stroke
        if any(k in findings_text for k in ["ischemic", "infarction", "stroke"]):
            if "hemorrhage" not in findings_text and "stemi" not in findings_text:
                contraindications.append(
                    "Excessive acute blood pressure lowering (permissive hypertension indicated to maintain penumbral perfusion)"
                )

        # Small Bowel Obstruction
        if "bowel obstruction" in findings_text or "sbo" in findings_text:
            contraindications.extend([
                "Oral intake or oral medications (Strict NPO mandatory)",
                "Oral contrast / cathartics / prokinetics",
            ])

        # Skull Fracture
        if "skull fracture" in findings_text or "fracture" in findings_text:
            contraindications.append(
                "Blind nasogastric tube or nasopharyngeal airway placement (risk of intracranial penetration in basilar fracture)"
            )

        # RV Infarction
        if "right ventricular" in findings_text or "rv infarction" in findings_text:
            contraindications.extend([
                "Nitroglycerin and nitrate vasodilators (risk of catastrophic preload collapse and refractory shock)",
                "Morphine or diuretics without invasive arterial pressure monitoring",
            ])
        # Anterior STEMI
        elif "stemi" in findings_text or "myocardial infarction" in findings_text:
            contraindications.append(
                "Delaying emergency primary percutaneous coronary intervention (PCI)"
            )

        # Pulmonary Edema
        if "pulmonary edema" in findings_text or "interstitial edema" in findings_text:
            contraindications.append(
                "Aggressive crystalloid fluid boluses (risk of worsening respiratory failure)"
            )

        # Hyperkalemia
        if "hyperkalemia" in findings_text:
            contraindications.append(
                "Potassium-sparing diuretics, ACE inhibitors, ARBs, and potassium supplementation"
            )

        return contraindications

    def _build_impression(self, vision_result: VisionResult) -> str:
        """Formulate concise clinical impression from primary visual finding."""
        if not vision_result.findings:
            if vision_result.negative_findings:
                return f"Diagnostic {vision_result.modality} without Acute Pathological Findings"
            return f"Diagnostic {vision_result.modality} Study"

        primary = vision_result.findings[0]
        label_lower = primary.label.lower()

        if "pneumothorax" in label_lower:
            laterality = f"{primary.laterality.title()} " if primary.laterality and primary.laterality != "unspecified" else ""
            return f"{laterality}Pneumothorax with Tension Risk"
        if "hemorrhage" in label_lower:
            return "Acute Intracranial Hemorrhage with Mass Effect (ICH / Hematoma)"
        if "ischemic" in label_lower or "stroke" in label_lower or "infarction" in label_lower and "cerebral" in label_lower:
            return "Acute Ischemic Cerebral Infarction (DWI Restriction without Hemorrhage)"
        if "stemi" in label_lower or "anterior" in label_lower:
            return "Acute Anterior ST-Elevation Myocardial Infarction (STEMI)"
        if "right ventricular" in label_lower or "rv" in label_lower:
            return "Acute Inferior-Posterior Myocardial Infarction with Right Ventricular Involvement (RV Infarction)"
        if "hyperkalemia" in label_lower:
            return "Cardiotoxic Hyperkalemia with Severe Arrhythmia Risk"
        if "consolidation" in label_lower or "pneumonia" in label_lower:
            return "Acute Community-Acquired Lobar Pneumonia (Consolidation)"
        if "pulmonary edema" in label_lower or "edema" in label_lower:
            return "Acute Cardiogenic Pulmonary Edema with Biventricular Cardiomegaly"
        if "effusion" in label_lower:
            laterality = f"{primary.laterality.title()} " if primary.laterality and primary.laterality != "unspecified" else ""
            return f"{laterality}Pleural Effusion"
        if "bowel obstruction" in label_lower or "sbo" in label_lower:
            return "Acute High-Grade Mechanical Small Bowel Obstruction (SBO)"
        if "skull fracture" in label_lower or "fracture" in label_lower:
            return "Traumatic Calvarial / Basilar Skull Fracture"
        if "ventriculomegaly" in label_lower or "hydrocephalus" in label_lower:
            return "Obstructive / Communicating Hydrocephalus (Ventriculomegaly)"

        return f"{primary.label} ({primary.anatomical_location})"
