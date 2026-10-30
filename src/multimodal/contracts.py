"""Multimodal Medical Perception Contracts."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from src.orchestrator.patient import PatientCase


class ImagingAnalysis(BaseModel):
    """Structured perception findings from diagnostic imaging analysis."""

    modality: str = "Chest X-Ray"  # "Chest X-Ray", "12-Lead ECG", "CT Scan", "Ultrasound"
    radiological_findings: list[str] = Field(default_factory=list)
    diagnostic_impression: str = ""
    critical_urgency: str = "ROUTINE"  # "ROUTINE", "URGENT", "EMERGENT"
    contraindicated_interventions: list[str] = Field(default_factory=list)
    confidence: float = 0.90
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultimodalPatientCase(PatientCase):
    """Extended patient case combining clinical text, lab records, and diagnostic scans."""

    imaging_analysis: ImagingAnalysis | None = None
    uploaded_document_text: str = ""
    uploaded_file_name: str = ""

