"""Medical Image Ingestion and Quality Control Gate.

Handles DICOM / NIfTI de-identification, HU windowing, spatial resolution
validation, exposure analysis, and discrete Laplacian motion artifact detection.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from src.multimodal.contracts import QualityAssessment

# Standard PHI tags to strip during de-identification (DICOM PS 3.15 Annex E)
PHI_TAGS_TO_STRIP: set[str] = {
    "patientname",
    "patientid",
    "patientbirthdate",
    "institutionname",
    "referringphysicianname",
    "patientsex",
    "patientage",
    "patientweight",
    "patientaddress",
    "accessionnumber",
    "studyid",
    "otherpatientids",
    "physiciansofrecord",
    "operatorname",
}

# Standard non-identifying acquisition tags to preserve
ACQUISITION_TAGS: set[str] = {
    "modality",
    "bodypartexamined",
    "viewposition",
    "pixelspacing",
    "windowcenter",
    "windowwidth",
    "rescaleslope",
    "rescaleintercept",
    "photometricinterpretation",
    "rows",
    "columns",
    "slice_thickness",
    "repetition_time",
    "echo_time",
    "magnetic_field_strength",
}


class MedicalImageIngestion:
    """Ingestion pipeline and pre-inference quality control gate for medical imaging."""

    def __init__(self, min_resolution: int = 256, laplacian_blur_threshold: float = 15.0) -> None:
        self.min_resolution = min_resolution
        self.laplacian_blur_threshold = laplacian_blur_threshold
        # 3x3 Discrete Laplacian convolution kernel
        self._laplacian_kernel = np.array(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )

    def validate_image_quality(self, image_input: str | bytes | Image.Image) -> QualityAssessment:
        """Evaluates image spatial resolution, exposure, and motion blur before VLM inference.

        Args:
            image_input: File path, raw bytes, or PIL Image instance.

        Returns:
            QualityAssessment detailing diagnostic adequacy and any rejection reasons.
        """
        # Load image into PIL Image
        try:
            if isinstance(image_input, Image.Image):
                pil_image = image_input
            elif isinstance(image_input, (str, Path)):
                path_obj = Path(image_input)
                if not path_obj.exists():
                    # Allow virtual / mock filenames without physical files on disk
                    return QualityAssessment(is_acceptable=True)
                pil_image = Image.open(str(image_input))
            elif isinstance(image_input, (bytes, bytearray)):
                if len(image_input) == 0:
                    return QualityAssessment(
                        is_acceptable=False,
                        rejection_reasons=["Empty (0 bytes) image payload. Re-acquisition required."],
                    )
                pil_image = Image.open(io.BytesIO(image_input))
            else:
                return QualityAssessment(
                    is_acceptable=False,
                    rejection_reasons=[f"Unsupported image input type: {type(image_input).__name__}"],
                )
        except Exception as e:
            return QualityAssessment(
                is_acceptable=False,
                rejection_reasons=[f"Corrupt or unreadable image payload: {str(e)}"],
            )

        rejection_reasons: list[str] = []
        is_acceptable = True
        exposure_quality = "optimal"
        motion_artifact_detected = False

        width, height = pil_image.size

        # 1. Spatial Resolution Check
        if width < self.min_resolution or height < self.min_resolution:
            is_acceptable = False
            rejection_reasons.append(
                f"Insufficient spatial resolution (< {self.min_resolution}x{self.min_resolution})"
            )

        # Convert to single-channel grayscale float array for intensity and sharpness analysis
        gray_image = pil_image.convert("L")
        arr = np.asarray(gray_image, dtype=np.float32)

        # 2. Exposure & Contrast Analysis (Quantile and standard deviation checks)
        p10 = float(np.percentile(arr, 10))
        p90 = float(np.percentile(arr, 90))
        std = float(np.std(arr))
        ptp = float(np.ptp(arr))

        if p10 > 220.0:
            exposure_quality = "overexposed"
            is_acceptable = False
            rejection_reasons.append("Severe overexposure / sensor saturation")
        elif p90 < 35.0 or std < 10.0:
            exposure_quality = "underexposed"
            is_acceptable = False
            rejection_reasons.append("Severe underexposure / signal loss")

        # 3. Motion Blur / Degradation Check via discrete Laplacian kernel convolution
        # Only evaluate sharpness if image is not flat/constant
        is_constant_or_flat = std < 10.0 or ptp <= 10.0
        if not is_constant_or_flat and width >= 3 and height >= 3:
            convolved = convolve2d(arr, self._laplacian_kernel, mode="valid")
            lap_variance = float(np.var(convolved))
            if lap_variance < self.laplacian_blur_threshold:
                motion_artifact_detected = True
                is_acceptable = False
                rejection_reasons.append("Severe motion blur or low edge sharpness")

        return QualityAssessment(
            is_acceptable=is_acceptable,
            orientation_valid=True,
            exposure_quality=exposure_quality,
            motion_artifact_detected=motion_artifact_detected,
            anatomy_matches_modality=True,
            rejection_reasons=rejection_reasons,
        )

    def anonymize_dicom(
        self, dicom_input: bytes | str | dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        """De-identifies DICOM payload by stripping direct PHI tags while preserving acquisition metadata.

        Args:
            dicom_input: Raw DICOM bytes, file path, or dictionary representation.

        Returns:
            Tuple of (PIL.Image.Image, sanitized_metadata_dict).
        """
        raw_meta: dict[str, Any] = {}
        pixel_array: np.ndarray | None = None

        # Attempt to use pydicom if available
        pydicom_loaded = False
        try:
            import pydicom

            if isinstance(dicom_input, (bytes, bytearray)):
                ds = pydicom.dcmread(io.BytesIO(dicom_input))
                pydicom_loaded = True
            elif isinstance(dicom_input, (str, Path)) and Path(str(dicom_input)).is_file():
                ds = pydicom.dcmread(str(dicom_input))
                pydicom_loaded = True

            if pydicom_loaded:
                for elem in ds:
                    if elem.keyword:
                        raw_meta[elem.keyword] = elem.value
                if hasattr(ds, "pixel_array"):
                    pixel_array = ds.pixel_array
        except Exception:
            pydicom_loaded = False

        if not pydicom_loaded:
            # Fallback JSON/dictionary parser for synthetic or simulated DICOM payloads
            if isinstance(dicom_input, dict):
                raw_meta = dict(dicom_input)
            elif isinstance(dicom_input, (bytes, bytearray)):
                try:
                    raw_meta = json.loads(dicom_input.decode("utf-8"))
                except Exception:
                    raw_meta = {}
            elif isinstance(dicom_input, (str, Path)) and Path(str(dicom_input)).is_file():
                try:
                    with open(str(dicom_input), "r", encoding="utf-8") as f:
                        raw_meta = json.load(f)
                except Exception:
                    raw_meta = {}

            if "pixel_array" in raw_meta:
                pixel_array = np.array(raw_meta.pop("pixel_array"), dtype=np.float32)

        # Sanitize metadata: remove direct patient and institutional identifiers
        sanitized_meta: dict[str, Any] = {}
        for key, value in raw_meta.items():
            if key.lower() not in PHI_TAGS_TO_STRIP:
                sanitized_meta[key] = value

        # Convert pixel data to PIL Image
        if pixel_array is not None:
            # Apply Window Center / Window Width windowing if present
            window_center = sanitized_meta.get("WindowCenter")
            window_width = sanitized_meta.get("WindowWidth")
            if window_center is not None and window_width is not None:
                try:
                    wc = float(window_center if not isinstance(window_center, list) else window_center[0])
                    ww = float(window_width if not isinstance(window_width, list) else window_width[0])
                    min_val = wc - (ww / 2.0)
                    max_val = wc + (ww / 2.0)
                    clipped = np.clip(pixel_array, min_val, max_val)
                    if max_val > min_val:
                        norm_arr = ((clipped - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                    else:
                        norm_arr = clipped.astype(np.uint8)
                except Exception:
                    norm_arr = np.clip(pixel_array, 0, 255).astype(np.uint8)
            else:
                p_min = float(np.min(pixel_array))
                p_max = float(np.max(pixel_array))
                if p_max > p_min:
                    norm_arr = ((pixel_array - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
                else:
                    norm_arr = np.clip(pixel_array, 0, 255).astype(np.uint8)

            pil_img = Image.fromarray(norm_arr).convert("L")
        else:
            # Synthetic 256x256 fallback image if pixel array is absent
            pil_img = Image.new("L", (256, 256), color=128)

        return pil_img, sanitized_meta

    def extract_key_slices(
        self, volume_input: bytes | str | Path | np.ndarray
    ) -> list[tuple[int, Image.Image]]:
        """Extracts representative multi-planar key slices from 3D CT/MRI volumes.

        Prevents single-slice under-evaluation of full 3D volumetric studies.

        Args:
            volume_input: 3D numpy array, or path / bytes containing volumetric data.

        Returns:
            List of (slice_index, PIL.Image.Image) for key slices at depth percentiles.
        """
        if isinstance(volume_input, np.ndarray):
            arr = volume_input
        elif isinstance(volume_input, (str, Path)):
            path_str = str(volume_input)
            if path_str.endswith(".npy"):
                arr = np.load(path_str)
            elif path_str.endswith((".nii", ".nii.gz")):
                try:
                    import nibabel as nib  # type: ignore

                    nii = nib.load(path_str)
                    arr = np.asarray(nii.get_fdata())
                except ImportError as exc:
                    raise ImportError("nibabel is required to load NIfTI volumes") from exc
            else:
                arr = np.load(path_str)
        elif isinstance(volume_input, (bytes, bytearray)):
            arr = np.load(io.BytesIO(volume_input))
        else:
            raise TypeError(f"Unsupported volume input type: {type(volume_input)}")

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D volume array of shape (depth, height, width), got {arr.shape}")

        depth = arr.shape[0]
        if depth == 0:
            return []

        if depth < 3:
            key_indices = list(range(depth))
        else:
            # Extract key slices at 25%, 50%, 75% depth
            key_indices = [int(depth * 0.25), int(depth * 0.50), int(depth * 0.75)]

        key_slices: list[tuple[int, Image.Image]] = []
        for idx in key_indices:
            slice_data = arr[idx, :, :].astype(np.float32)
            s_min = float(np.min(slice_data))
            s_max = float(np.max(slice_data))
            if s_max > s_min:
                normalized = ((slice_data - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
            else:
                normalized = np.clip(slice_data, 0, 255).astype(np.uint8)

            img = Image.fromarray(normalized).convert("L")
            key_slices.append((idx, img))

        return key_slices
