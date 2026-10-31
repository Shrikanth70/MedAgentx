"""Tests for Ingestion Pipeline & Quality Control Gate."""

from __future__ import annotations

import io
import json
import numpy as np
from PIL import Image
import pytest

from src.multimodal.contracts import QualityAssessment
from src.multimodal.ingestion import MedicalImageIngestion


@pytest.fixture
def ingestion() -> MedicalImageIngestion:
    return MedicalImageIngestion()


@pytest.fixture
def valid_sharp_image() -> Image.Image:
    """Creates a high-contrast sharp 512x512 image simulating a clear radiograph."""
    # Create high-contrast alternating checkerboard / grid texture
    arr = np.zeros((512, 512), dtype=np.uint8)
    # Background around 100-150
    arr[::16, :] = 200
    arr[:, ::16] = 200
    arr[128:384, 128:384] = 160
    arr[200:300, 200:300] = 70
    return Image.fromarray(arr).convert("L")


def test_valid_image_passes_quality_gate(ingestion: MedicalImageIngestion, valid_sharp_image: Image.Image) -> None:
    assessment = ingestion.validate_image_quality(valid_sharp_image)

    assert isinstance(assessment, QualityAssessment)
    assert assessment.is_acceptable is True
    assert assessment.exposure_quality == "optimal"
    assert assessment.motion_artifact_detected is False
    assert len(assessment.rejection_reasons) == 0


def test_small_image_rejected_for_spatial_resolution(ingestion: MedicalImageIngestion) -> None:
    small_arr = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    small_img = Image.fromarray(small_arr).convert("L")

    assessment = ingestion.validate_image_quality(small_img)

    assert assessment.is_acceptable is False
    assert any("Insufficient spatial resolution" in r for r in assessment.rejection_reasons)


def test_solid_black_image_rejected_as_underexposed(ingestion: MedicalImageIngestion) -> None:
    black_arr = np.zeros((300, 300), dtype=np.uint8)
    black_img = Image.fromarray(black_arr).convert("L")

    assessment = ingestion.validate_image_quality(black_img)

    assert assessment.is_acceptable is False
    assert assessment.exposure_quality == "underexposed"
    assert any("Severe underexposure" in r for r in assessment.rejection_reasons)


def test_solid_white_image_rejected_as_overexposed(ingestion: MedicalImageIngestion) -> None:
    white_arr = np.full((300, 300), 245, dtype=np.uint8)
    white_img = Image.fromarray(white_arr).convert("L")

    assessment = ingestion.validate_image_quality(white_img)

    assert assessment.is_acceptable is False
    assert assessment.exposure_quality == "overexposed"
    assert any("Severe overexposure" in r for r in assessment.rejection_reasons)


def test_motion_blurred_image_rejected(ingestion: MedicalImageIngestion) -> None:
    # A smooth gradient with reasonable std >= 10, but very low Laplacian variance (< 15)
    # np.linspace gives smooth variation without sharp edges
    x = np.linspace(50, 180, 300, dtype=np.float32)
    gradient = np.tile(x, (300, 1)).astype(np.uint8)
    blur_img = Image.fromarray(gradient).convert("L")

    assessment = ingestion.validate_image_quality(blur_img)

    assert assessment.is_acceptable is False
    assert assessment.motion_artifact_detected is True
    assert any("Severe motion blur" in r for r in assessment.rejection_reasons)


def test_validate_image_quality_supports_bytes_and_file_path(
    ingestion: MedicalImageIngestion, valid_sharp_image: Image.Image, tmp_path
) -> None:
    # Test bytes input
    buf = io.BytesIO()
    valid_sharp_image.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    assessment_from_bytes = ingestion.validate_image_quality(raw_bytes)
    assert assessment_from_bytes.is_acceptable is True

    # Test file path input
    file_path = tmp_path / "test_img.png"
    valid_sharp_image.save(file_path)

    assessment_from_path = ingestion.validate_image_quality(str(file_path))
    assert assessment_from_path.is_acceptable is True


def test_dicom_anonymization_strips_phi_and_preserves_acquisition(
    ingestion: MedicalImageIngestion,
) -> None:
    # Simulated DICOM payload with PHI and acquisition tags
    pixel_data = np.full((256, 256), 120, dtype=np.uint8).tolist()
    dicom_payload = {
        "PatientName": "DOE^JOHN",
        "PatientID": "MRN-984210",
        "PatientBirthDate": "19650412",
        "InstitutionName": "Metropolitan Health Hospital",
        "ReferringPhysicianName": "Dr. Smith, Jane",
        "Modality": "CT",
        "BodyPartExamined": "CHEST",
        "ViewPosition": "PA",
        "PixelSpacing": [0.75, 0.75],
        "WindowCenter": 40,
        "WindowWidth": 350,
        "pixel_array": pixel_data,
    }
    raw_bytes = json.dumps(dicom_payload).encode("utf-8")

    pil_img, sanitized_meta = ingestion.anonymize_dicom(raw_bytes)

    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (256, 256)

    # PHI tags must be stripped
    assert "PatientName" not in sanitized_meta
    assert "PatientID" not in sanitized_meta
    assert "PatientBirthDate" not in sanitized_meta
    assert "InstitutionName" not in sanitized_meta
    assert "ReferringPhysicianName" not in sanitized_meta

    # Clinical acquisition metadata must be preserved
    assert sanitized_meta["Modality"] == "CT"
    assert sanitized_meta["BodyPartExamined"] == "CHEST"
    assert sanitized_meta["ViewPosition"] == "PA"
    assert sanitized_meta["PixelSpacing"] == [0.75, 0.75]
    assert sanitized_meta["WindowCenter"] == 40
    assert sanitized_meta["WindowWidth"] == 350


def test_extract_key_slices_3d_volume(ingestion: MedicalImageIngestion) -> None:
    # 3D volume of shape (depth=60, height=256, width=256)
    volume = np.zeros((60, 256, 256), dtype=np.uint8)
    for i in range(60):
        volume[i, :, :] = i * 4

    slices = ingestion.extract_key_slices(volume)

    assert len(slices) >= 3
    # Expected key slices at 25%, 50%, 75% depth -> [15, 30, 45]
    indices = [s[0] for s in slices]
    assert indices == [15, 30, 45]

    for idx, img in slices:
        assert isinstance(img, Image.Image)
        assert img.size == (256, 256)
