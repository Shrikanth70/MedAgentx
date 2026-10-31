"""Unit tests for MedicalImagingAnalyzer adapter refactored with QC gate and VisionEngine."""

from __future__ import annotations

import io
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import pytest

from src.multimodal.contracts import (
    ImagingAnalysis,
    MultimodalPatientCase,
    QualityAssessment,
    VisionResult,
    VisionStatus,
    VisualEvidence,
    VisualFinding,
)
from src.multimodal.imaging_analyzer import MedicalImagingAnalyzer
from src.multimodal.vision_engine import BaseVisionModel, PluggableVisionEngine


def _make_valid_image(width: int = 300, height: int = 300) -> Image.Image:
    """Creates a high-contrast sharp image simulating a diagnostic scan."""
    arr = np.zeros((height, width), dtype=np.uint8)
    arr[::10, :] = 200
    arr[:, ::10] = 200
    arr[50:150, 50:150] = 160
    return Image.fromarray(arr).convert("L")


def _make_small_image(width: int = 100, height: int = 100) -> Image.Image:
    """Creates a low-resolution image under the 256x256 threshold."""
    arr = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    return Image.fromarray(arr).convert("L")


def test_low_quality_image_returns_low_quality_and_zero_findings() -> None:
    """Low quality image (<256x256) returns status LOW_QUALITY with 0 findings and 0.0 confidence."""
    analyzer = MedicalImagingAnalyzer()
    small_img = _make_small_image(128, 128)

    # 1. Test standard adapter analyze()
    analysis = analyzer.analyze(small_img, clinical_hint="CXR for pneumonia")

    assert isinstance(analysis, ImagingAnalysis)
    assert analysis.radiological_findings == []
    assert "Suboptimal Study" in analysis.diagnostic_impression
    assert analysis.confidence == 0.0
    assert analysis.critical_urgency == "ROUTINE"
    assert analysis.contraindicated_interventions == []
    assert analysis.metadata.get("vision_status") == VisionStatus.LOW_QUALITY.value
    assert "quality_assessment" in analysis.metadata

    # 2. Test native analyze_vision()
    v_res = analyzer.analyze_vision(small_img, clinical_hint="CXR for pneumonia")
    assert isinstance(v_res, VisionResult)
    assert v_res.status == VisionStatus.LOW_QUALITY
    assert v_res.findings == []
    assert v_res.quality.is_acceptable is False


def test_vision_timeout_returns_timeout_and_zero_findings_never_fabricating_normal() -> None:
    """On vision timeout/failure, return status TIMEOUT with 0 findings and 0.0 confidence (FAIL CLOSED)."""
    # Create a custom mock vision engine that returns TIMEOUT
    mock_engine = MagicMock(spec=PluggableVisionEngine)
    mock_engine.analyze.return_value = VisionResult(
        status=VisionStatus.TIMEOUT,
        modality="Chest X-Ray",
        body_part="Chest",
        findings=[],
        negative_findings=[],
        limitations=["Inference timeout after retries exhausted"],
        model_name="mock_timeout",
    )

    analyzer = MedicalImagingAnalyzer(vision_engine=mock_engine)
    valid_img = _make_valid_image(300, 300)

    analysis = analyzer.analyze(valid_img, clinical_hint="Chest X-Ray showing severe consolidation")

    # Strict zero-placeholder guarantee
    assert analysis.radiological_findings == []
    assert analysis.confidence == 0.0
    assert analysis.critical_urgency == "ROUTINE"
    assert analysis.contraindicated_interventions == []
    assert "Imaging Perception Incomplete (TIMEOUT)" in analysis.diagnostic_impression
    assert "Normal" not in analysis.diagnostic_impression
    assert "clear lung" not in str(analysis.radiological_findings).lower()
    assert analysis.metadata.get("vision_status") == VisionStatus.TIMEOUT.value


def test_successful_consolidation_study() -> None:
    """Successful consolidation study returns valid findings, impression, and proper confidence."""
    analyzer = MedicalImagingAnalyzer()
    valid_img = _make_valid_image(300, 300)

    analysis = analyzer.analyze(
        valid_img,
        clinical_hint="Chest X-Ray showing fever, cough, right middle lobe consolidation and infiltrate",
        file_name="cxr_consolidation_rml.png",
    )

    assert analysis.metadata.get("vision_status") == VisionStatus.OK.value
    assert len(analysis.radiological_findings) > 0
    assert any("consolidation" in f.lower() or "alveolar" in f.lower() for f in analysis.radiological_findings)
    assert "Pneumonia" in analysis.diagnostic_impression or "Consolidation" in analysis.diagnostic_impression
    assert analysis.confidence >= 0.90
    assert analysis.critical_urgency in ["URGENT", "EMERGENT"]


def test_successful_pneumothorax_contraindications() -> None:
    """Pneumothorax finding triggers emergent urgency and positive-pressure ventilation contraindication."""
    analyzer = MedicalImagingAnalyzer()
    valid_img = _make_valid_image(300, 300)

    analysis = analyzer.analyze(
        valid_img,
        clinical_hint="Trauma patient with sudden dyspnea, tension pneumothorax right lung",
        file_name="cxr_tension_ptx.png",
    )

    assert analysis.metadata.get("vision_status") == VisionStatus.OK.value
    assert analysis.critical_urgency == "EMERGENT"
    assert len(analysis.radiological_findings) > 0
    assert any(
        "positive pressure" in c.lower() or "positive-pressure" in c.lower()
        for c in analysis.contraindicated_interventions
    )


def test_head_ct_hemorrhage_contraindications() -> None:
    """Acute intracranial hemorrhage triggers emergent urgency and thrombolytic contraindications."""
    analyzer = MedicalImagingAnalyzer()
    valid_img = _make_valid_image(300, 300)

    analysis = analyzer.analyze(
        valid_img,
        clinical_hint="Head CT showing acute intracranial hemorrhage with mass effect",
        file_name="head_ct_bleed.png",
    )

    assert analysis.metadata.get("vision_status") == VisionStatus.OK.value
    assert analysis.critical_urgency == "EMERGENT"
    assert any("thrombolytic" in c.lower() or "alteplase" in c.lower() for c in analysis.contraindicated_interventions)
    assert any("anticoagulation" in c.lower() for c in analysis.contraindicated_interventions)


def test_custom_vision_engine_injection() -> None:
    """MedicalImagingAnalyzer accepts custom vision engine parameter for testing."""
    mock_engine = MagicMock(spec=PluggableVisionEngine)
    analyzer = MedicalImagingAnalyzer(vision_engine=mock_engine)

    assert analyzer.vision_engine is mock_engine


def test_analyze_vision_returns_native_vision_result() -> None:
    """analyze_vision returns native VisionResult directly."""
    analyzer = MedicalImagingAnalyzer()
    valid_img = _make_valid_image(300, 300)

    v_res = analyzer.analyze_vision(
        valid_img,
        clinical_hint="Chest X-Ray showing right middle lobe consolidation",
        file_name="test_rml.png",
    )

    assert isinstance(v_res, VisionResult)
    assert v_res.status == VisionStatus.OK
    assert len(v_res.findings) > 0
    assert v_res.findings[0].label == "Lobar Consolidation"


def test_preserves_imaging_analysis_contract_fields() -> None:
    """ImagingAnalysis output preserves all contract fields so orchestrator is unbroken."""
    analyzer = MedicalImagingAnalyzer()
    valid_img = _make_valid_image(300, 300)

    analysis = analyzer.analyze(valid_img, clinical_hint="Chest X-Ray")

    # Required fields on ImagingAnalysis
    assert hasattr(analysis, "modality")
    assert hasattr(analysis, "radiological_findings")
    assert hasattr(analysis, "diagnostic_impression")
    assert hasattr(analysis, "critical_urgency")
    assert hasattr(analysis, "contraindicated_interventions")
    assert hasattr(analysis, "confidence")
    assert hasattr(analysis, "metadata")

    # Works inside MultimodalPatientCase
    case = MultimodalPatientCase(
        patient_id="PT-TEST-01",
        chief_complaint="Chest pain",
        imaging_analysis=analysis,
    )
    assert case.imaging_analysis is not None
    assert case.imaging_analysis.confidence == analysis.confidence
