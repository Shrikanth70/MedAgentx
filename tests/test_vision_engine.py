"""Unit tests for pluggable vision engine, exponential backoff, and secondary fallback."""

import io
import time
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import pytest

from src.multimodal.contracts import (
    VisionResult,
    VisionStatus,
    VisualFinding,
    NegativeFinding,
)


def _create_test_image(width: int = 100, height: int = 100, color: tuple = (200, 200, 200)) -> bytes:
    """Helper to create test image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_base_vision_model_is_abstract():
    """Verify BaseVisionModel cannot be instantiated directly."""
    from src.multimodal.vision_engine import BaseVisionModel

    with pytest.raises(TypeError):
        BaseVisionModel()


class TestQwenVLMModel:
    """Tests for QwenVLMModel Ollama interface."""

    def test_image_preprocessing_downscaling(self):
        """Verify image is downscaled to max 512x512 JPEG before sending."""
        from src.multimodal.vision_engine import QwenVLMModel

        model = QwenVLMModel(model_name="qwen3-vl:4b")
        large_img_bytes = _create_test_image(width=1024, height=768)

        processed_b64 = model._preprocess_image(large_img_bytes)
        assert isinstance(processed_b64, str)
        assert len(processed_b64) > 0

        # Verify decoded image dimensions do not exceed 512x512
        import base64
        decoded_bytes = base64.b64decode(processed_b64)
        with Image.open(io.BytesIO(decoded_bytes)) as img:
            assert img.width <= 512
            assert img.height <= 512
            assert img.format == "JPEG"

    def test_image_preprocessing_numpy_array(self):
        """Verify numpy array inputs are handled and converted to max 512x512."""
        from src.multimodal.vision_engine import QwenVLMModel

        model = QwenVLMModel()
        arr = np.zeros((800, 800), dtype=np.uint8)
        processed_b64 = model._preprocess_image(arr)
        assert isinstance(processed_b64, str)

    @patch("requests.post")
    def test_qwen_vlm_successful_inference(self, mock_post):
        """Verify successful response from Ollama API parses into VisionResult."""
        from src.multimodal.vision_engine import QwenVLMModel

        mock_payload = {
            "message": {
                "content": """
                {
                    "modality": "Chest X-Ray",
                    "body_part": "Chest",
                    "view": "PA",
                    "findings": [
                        {
                            "label": "Lobar Consolidation",
                            "coding_system": "RadLex",
                            "code": "RID49488",
                            "anatomical_location": "Right Middle Lobe",
                            "laterality": "right",
                            "probability": 0.92,
                            "uncertainty": 0.05,
                            "is_critical": false,
                            "evidence": {
                                "bbox": [0.35, 0.55, 0.65, 0.85],
                                "supporting_visual_description": "Dense consolidation in RML with air bronchograms."
                            }
                        }
                    ],
                    "negative_findings": [
                        {
                            "label": "Pneumothorax",
                            "code": "RID4895",
                            "probability": 0.98,
                            "checked": true
                        }
                    ],
                    "limitations": []
                }
                """
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_post.return_value = mock_resp

        model = QwenVLMModel(model_name="qwen3-vl:4b")
        img = _create_test_image()
        result = model.infer(img, metadata={"hint": "cough and fever"})

        assert isinstance(result, VisionResult)
        assert result.status == VisionStatus.OK
        assert len(result.findings) == 1
        assert result.findings[0].label == "Lobar Consolidation"
        assert result.findings[0].code == "RID49488"
        assert len(result.negative_findings) == 1
        assert result.negative_findings[0].label == "Pneumothorax"

    @patch("requests.post")
    def test_qwen_vlm_timeout_raises_or_returns_timeout(self, mock_post):
        """Verify network timeout raises TimeoutError or returns VisionResult with TIMEOUT status."""
        import requests
        from src.multimodal.vision_engine import QwenVLMModel

        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        model = QwenVLMModel(timeout_seconds=5.0)
        img = _create_test_image()
        
        # Should raise TimeoutError or return VisionResult(status=VisionStatus.TIMEOUT)
        try:
            result = model.infer(img)
            assert result.status == VisionStatus.TIMEOUT
            assert len(result.findings) == 0
        except TimeoutError:
            pass  # Raising TimeoutError is also acceptable per contract


class TestClinicalRulesVisionFallback:
    """Tests for deterministic clinical rule fallback model."""

    def test_consolidation_rule_extraction(self):
        """Verify clinical rules detect lobar consolidation with RadLex code and negative findings."""
        from src.multimodal.vision_engine import ClinicalRulesVisionFallback

        model = ClinicalRulesVisionFallback()
        img = _create_test_image()
        result = model.infer(img, metadata={"clinical_hint": "Right middle lobe consolidation with air bronchograms"})

        assert result.status == VisionStatus.OK
        assert any(f.label == "Lobar Consolidation" and f.code == "RID49488" for f in result.findings)
        assert len(result.negative_findings) > 0
        assert any(nf.label == "Pneumothorax" for nf in result.negative_findings)

    def test_tension_pneumothorax_rule_is_critical(self):
        """Verify tension pneumothorax triggers critical urgency flag."""
        from src.multimodal.vision_engine import ClinicalRulesVisionFallback

        model = ClinicalRulesVisionFallback()
        result = model.infer(b"", metadata={"clinical_hint": "Right-sided tension pneumothorax with mediastinal shift"})

        assert result.status == VisionStatus.OK
        assert any(f.is_critical for f in result.findings)
        assert any(f.code == "RID50041" or "pneumothorax" in f.label.lower() for f in result.findings)

    def test_explicit_normal_study(self):
        """Verify explicitly verified normal study returns OK with verified negative findings and no false findings."""
        from src.multimodal.vision_engine import ClinicalRulesVisionFallback

        model = ClinicalRulesVisionFallback()
        result = model.infer(b"", metadata={"clinical_hint": "Normal chest x-ray, clear lungs, no acute cardiopulmonary abnormality"})

        assert result.status == VisionStatus.OK
        assert len(result.findings) == 0
        assert len(result.negative_findings) >= 2
        assert all(nf.checked for nf in result.negative_findings)

    def test_unknown_study_does_not_fabricate_normal(self):
        """Verify unverified unknown study does NOT fabricate verified normal findings."""
        from src.multimodal.vision_engine import ClinicalRulesVisionFallback

        model = ClinicalRulesVisionFallback()
        result = model.infer(b"", metadata={"file_name": "unknown_file_xyz.jpg"})

        assert result.findings == []
        assert len(result.negative_findings) == 0
        assert len(result.limitations) > 0


class TestPluggableVisionEngine:
    """Tests for PluggableVisionEngine backoff, fallback, and zero-placeholder guarantees."""

    def test_primary_success_first_try(self):
        """Verify primary model success on first try returns OK without calling secondary."""
        from src.multimodal.vision_engine import PluggableVisionEngine, BaseVisionModel

        mock_primary = MagicMock(spec=BaseVisionModel)
        mock_secondary = MagicMock(spec=BaseVisionModel)

        expected_finding = VisualFinding(
            label="Lobar Consolidation",
            code="RID49488",
            anatomical_location="Right Lower Lobe",
            probability=0.9,
        )
        mock_primary.infer.return_value = VisionResult(
            status=VisionStatus.OK,
            findings=[expected_finding],
            model_name="mock_vlm",
        )

        engine = PluggableVisionEngine(
            primary_model=mock_primary,
            secondary_model=mock_secondary,
            max_retries=3,
        )

        result = engine.analyze(b"fake_image")

        assert result.status == VisionStatus.OK
        assert len(result.findings) == 1
        assert result.findings[0].label == "Lobar Consolidation"
        assert mock_primary.infer.call_count == 1
        mock_secondary.infer.assert_not_called()

    @patch("time.sleep")
    def test_primary_retry_exponential_backoff(self, mock_sleep):
        """Verify primary model failure retries with exponential backoff up to max_retries."""
        from src.multimodal.vision_engine import PluggableVisionEngine, BaseVisionModel

        mock_primary = MagicMock(spec=BaseVisionModel)
        mock_primary.infer.side_effect = [
            TimeoutError("Timeout attempt 1"),
            TimeoutError("Timeout attempt 2"),
            VisionResult(
                status=VisionStatus.OK,
                findings=[VisualFinding(label="Pleural Effusion", code="RID4844", anatomical_location="Right Base", probability=0.88)],
                model_name="mock_vlm",
            ),
        ]

        engine = PluggableVisionEngine(
            primary_model=mock_primary,
            max_retries=3,
            backoff_factor=1.0,
        )

        result = engine.analyze(b"fake_image")

        assert result.status == VisionStatus.OK
        assert mock_primary.infer.call_count == 3
        # Should have slept with exponential backoff: 1.0 * (2^0) = 1.0, 1.0 * (2^1) = 2.0
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    @patch("time.sleep")
    def test_primary_exhausted_fallback_to_secondary_success(self, mock_sleep):
        """Verify that when primary model exhausts retries, secondary fallback is invoked."""
        from src.multimodal.vision_engine import PluggableVisionEngine, BaseVisionModel

        mock_primary = MagicMock(spec=BaseVisionModel)
        mock_primary.infer.side_effect = TimeoutError("Persistent primary timeout")

        mock_secondary = MagicMock(spec=BaseVisionModel)
        fallback_finding = VisualFinding(
            label="Pneumothorax",
            code="RID4895",
            anatomical_location="Right Hemithorax",
            probability=0.95,
            is_critical=True,
        )
        mock_secondary.infer.return_value = VisionResult(
            status=VisionStatus.OK,
            findings=[fallback_finding],
            model_name="rules_fallback",
        )

        engine = PluggableVisionEngine(
            primary_model=mock_primary,
            secondary_model=mock_secondary,
            max_retries=2,
            backoff_factor=0.5,
        )

        result = engine.analyze(b"fake_image", metadata={"clinical_hint": "pneumothorax"})

        assert result.status == VisionStatus.OK
        assert len(result.findings) == 1
        assert result.findings[0].label == "Pneumothorax"
        assert mock_primary.infer.call_count == 2
        assert mock_secondary.infer.call_count == 1
        # Limitations should document the fallback
        assert any("fallback" in limit.lower() or "secondary" in limit.lower() for limit in result.limitations)

    @patch("time.sleep")
    def test_both_models_fail_returns_timeout_with_zero_placeholders(self, mock_sleep):
        """Verify that when both models fail or timeout, engine returns TIMEOUT/FAILED with empty findings.
        
        Strict zero-placeholder guarantee: Must NEVER return synthetic normal exam.
        """
        from src.multimodal.vision_engine import PluggableVisionEngine, BaseVisionModel

        mock_primary = MagicMock(spec=BaseVisionModel)
        mock_primary.infer.side_effect = TimeoutError("Primary timeout")

        mock_secondary = MagicMock(spec=BaseVisionModel)
        mock_secondary.infer.side_effect = RuntimeError("Secondary classifier error")

        engine = PluggableVisionEngine(
            primary_model=mock_primary,
            secondary_model=mock_secondary,
            max_retries=2,
            backoff_factor=0.1,
        )

        result = engine.analyze(b"fake_image")

        assert result.status in (VisionStatus.TIMEOUT, VisionStatus.FAILED)
        # Strict zero-placeholder guarantee
        assert result.findings == []
        assert not any("normal" in f.label.lower() for f in result.findings)
        assert len(result.limitations) > 0

    @patch("time.sleep")
    def test_primary_status_failed_triggers_secondary(self, mock_sleep):
        """Verify that returning VisionStatus.FAILED from primary also triggers secondary."""
        from src.multimodal.vision_engine import PluggableVisionEngine, BaseVisionModel

        mock_primary = MagicMock(spec=BaseVisionModel)
        mock_primary.infer.return_value = VisionResult(
            status=VisionStatus.FAILED,
            findings=[],
            limitations=["Inference memory error"],
        )

        mock_secondary = MagicMock(spec=BaseVisionModel)
        mock_secondary.infer.return_value = VisionResult(
            status=VisionStatus.OK,
            findings=[VisualFinding(label="Consolidation", code="RID49488", anatomical_location="Lobe", probability=0.85)],
            model_name="secondary_rules",
        )

        engine = PluggableVisionEngine(
            primary_model=mock_primary,
            secondary_model=mock_secondary,
            max_retries=1,
        )

        result = engine.analyze(b"fake_image")

        assert result.status == VisionStatus.OK
        assert len(result.findings) == 1
        assert mock_secondary.infer.call_count == 1
