"""Clinical Document & Tabular Lab Parser."""

from __future__ import annotations

import os
import re
from typing import Any
import structlog

logger = structlog.get_logger()


class ClinicalDocumentParser:
    """Extracts structured laboratory biomarkers, blood gases, and vitals from clinical documents."""

    @classmethod
    def parse_report(cls, file_or_text: str | bytes) -> dict[str, Any]:
        """Parse raw text, file path, or bytes into structured clinical dictionary.

        Args:
            file_or_text: Plain text string, file path, or bytes.

        Returns:
            Dictionary of extracted lab biomarker keys and float values.
        """
        text = ""
        if isinstance(file_or_text, bytes):
            text = file_or_text.decode("utf-8", errors="ignore")
        elif isinstance(file_or_text, str):
            if os.path.exists(file_or_text):
                try:
                    with open(file_or_text, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception as exc:
                    logger.warning("failed_to_read_file_path", path=file_or_text, error=str(exc))
                    text = file_or_text
            else:
                text = file_or_text

        extracted: dict[str, Any] = {}

        patterns = {
            "egfr": [
                r"eGFR\s*[:=]?\s*([><]?\s*\d+(?:\.\d+)?)",
                r"estimated\s+gfr\s*[:=]?\s*([><]?\s*\d+(?:\.\d+)?)",
            ],
            "creatinine": [
                r"(?:serum\s+)?creatinine\s*[:=]?\s*(\d+(?:\.\d+)?)",
                r"\bCr\b\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            "potassium": [
                r"potassium(?:\s*\([Kk]\+\))?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                r"\bK\+\b\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            "sodium": [
                r"sodium(?:\s*\([Nn]a\+\))?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                r"\bNa\+\b\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            "bun": [
                r"(?:blood\s+urea\s+nitrogen|bun)\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            "lactate": [
                r"(?:serum\s+)?lactate\s*[:=]?\s*(\d+(?:\.\d+)?)",
                r"lactic\s+acid\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            "troponin": [
                r"(?:high-sensitivity\s+|hs-)?troponin(?:\s*[itI-T])?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                r"\btroponin\b\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            "wbc": [
                r"wbc(?:\s+count)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                r"white\s+blood\s+cell(?:\s+count)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                r"leukocytes\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            "glucose": [
                r"(?:blood\s+|serum\s+)?glucose\s*[:=]?\s*(\d+(?:\.\d+)?)",
                r"\bbg\b\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            "platelets": [
                r"platelets?(?:\s+count)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                r"\bplt\b\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            "ph": [
                r"\bpH\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            "paco2": [
                r"paco2\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            "pao2": [
                r"pao2\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
            "hco3": [
                r"(?:bicarbonate|hco3)\s*[:=]?\s*(\d+(?:\.\d+)?)",
            ],
        }

        for biomarker, regex_list in patterns.items():
            for pat in regex_list:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    val_str = match.group(1).replace(">", "").replace("<", "").strip()
                    try:
                        extracted[biomarker] = float(val_str)
                        break
                    except ValueError:
                        continue

        return extracted

