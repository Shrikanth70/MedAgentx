"""Generate verified diagnostic sample medical images and lab reports."""

import os
from PIL import Image, ImageDraw, ImageFont


def create_sample_assets():
    os.makedirs("data/samples", exist_ok=True)

    # 1. Chest X-Ray: Lobar Pneumonia
    img_pneumonia = Image.new("RGB", (512, 512), color=(20, 24, 28))
    draw = ImageDraw.Draw(img_pneumonia)
    # Ribcage / Thorax outlines
    draw.ellipse([80, 80, 432, 460], outline=(70, 75, 80), width=4)
    # Mediastinum / Spine
    draw.rectangle([230, 80, 282, 460], fill=(60, 65, 70))
    # Cardiac silhouette (normal CTR < 0.5)
    draw.ellipse([210, 260, 310, 420], fill=(75, 80, 85))
    # Dense RLL Consolidation
    draw.ellipse([280, 300, 410, 420], fill=(180, 185, 190))
    draw.text((20, 20), "CHEST RADIOGRAPH (PA ERECT)", fill=(240, 240, 240))
    draw.text((20, 40), "FINDING: Dense right lower lobe alveolar consolidation", fill=(200, 220, 255))
    draw.text((20, 60), "IMPRESSION: Acute Community-Acquired Lobar Pneumonia", fill=(255, 200, 100))
    img_pneumonia.save("data/samples/sample_cxr_pneumonia.png")

    # 2. 12-Lead ECG: Acute Anterior STEMI
    img_stemi = Image.new("RGB", (512, 320), color=(253, 240, 240))
    draw_ecg = ImageDraw.Draw(img_stemi)
    # ECG grid lines
    for x in range(0, 512, 20):
        draw_ecg.line([(x, 0), (x, 320)], fill=(250, 210, 210), width=1)
    for y in range(0, 320, 20):
        draw_ecg.line([(0, y), (512, y)], fill=(250, 210, 210), width=1)
    # Waveform with ST elevation in V2-V3
    points = [
        (30, 160), (60, 160), (70, 175), (80, 70), (90, 180),
        (100, 110), (140, 110), (170, 160), (220, 160),
        (230, 175), (240, 60), (250, 180), (260, 100), (300, 100), (330, 160),
        (380, 160), (390, 175), (400, 65), (410, 180), (420, 105), (460, 105), (490, 160)
    ]
    draw_ecg.line(points, fill=(180, 20, 20), width=3)
    draw_ecg.text((20, 15), "12-LEAD ELECTROCARDIOGRAM (25mm/s, 10mm/mV)", fill=(40, 40, 40))
    draw_ecg.text((20, 35), "LEADS V1-V4: 4mm Tombstone ST-Segment Elevation", fill=(200, 0, 0))
    draw_ecg.text((20, 55), "IMPRESSION: Acute Anterior ST-Elevation Myocardial Infarction", fill=(180, 0, 0))
    img_stemi.save("data/samples/sample_ecg_stemi.png")

    # 3. Chest X-Ray: Acute Cardiogenic Pulmonary Edema
    img_edema = Image.new("RGB", (512, 512), color=(20, 24, 28))
    draw_edema = ImageDraw.Draw(img_edema)
    # Thorax
    draw_edema.ellipse([70, 70, 442, 470], outline=(70, 75, 80), width=4)
    # Marked Cardiomegaly (CTR > 0.60)
    draw_edema.ellipse([150, 240, 360, 430], fill=(85, 90, 95))
    # Bat-wing perihilar opacities
    draw_edema.ellipse([160, 210, 240, 320], fill=(150, 155, 160))
    draw_edema.ellipse([270, 210, 350, 320], fill=(150, 155, 160))
    # Kerley B lines at periphery
    for y_line in range(350, 420, 15):
        draw_edema.line([(90, y_line), (130, y_line)], fill=(170, 175, 180), width=2)
        draw_edema.line([(380, y_line), (420, y_line)], fill=(170, 175, 180), width=2)
    draw_edema.text((20, 20), "CHEST RADIOGRAPH: ACUTE DYSPNEA WORKUP", fill=(240, 240, 240))
    draw_edema.text((20, 40), "FINDING: Bilateral bat-wing perihilar alveolar infiltrates & Kerley B lines", fill=(200, 220, 255))
    draw_edema.text((20, 60), "IMPRESSION: Acute Cardiogenic Pulmonary Edema (CTR 0.62)", fill=(255, 160, 80))
    img_edema.save("data/samples/sample_cxr_pulmonary_edema.png")

    # 4. Chest X-Ray: Tension Pneumothorax
    img_ptx = Image.new("RGB", (512, 512), color=(15, 18, 22))
    draw_ptx = ImageDraw.Draw(img_ptx)
    # Thorax outline
    draw_ptx.ellipse([70, 70, 442, 470], outline=(70, 75, 80), width=4)
    # Left lung markings present
    draw_ptx.ellipse([100, 120, 220, 410], fill=(45, 50, 55))
    # Right hemithorax: complete hyperlucency / absence of lung markings
    draw_ptx.rectangle([290, 110, 420, 430], fill=(8, 10, 12))
    # Visceral pleural line (collapsed right lung)
    draw_ptx.line([(300, 140), (290, 250), (300, 380)], fill=(220, 225, 230), width=3)
    # Mediastinal shift to the left
    draw_ptx.rectangle([190, 120, 260, 430], fill=(70, 75, 80))
    draw_ptx.text((20, 20), "EMERGENCY PORTABLE TRAUMA CXR", fill=(255, 100, 100))
    draw_ptx.text((20, 40), "FINDING: Complete absence of peripheral lung markings on right", fill=(240, 240, 240))
    draw_ptx.text((20, 60), "ALERT: Visible visceral pleural line; contralateral mediastinal shift", fill=(255, 80, 80))
    draw_ptx.text((20, 80), "CONTRAINDICATION: Positive Pressure Ventilation strictly contraindicated", fill=(255, 60, 60))
    img_ptx.save("data/samples/sample_cxr_pneumothorax.png")

    # 5. Critical Care Hospital Laboratory Report Text
    lab_text = """METROPOLITAN HOSPITAL CENTER - EMERGENCY CRITICAL CARE LAB PANEL
Patient: Marcus Vance | MRN: 489201 | Location: Resuscitation Bay 2
Attending: Dr. S. Reynolds, MD | Service: Emergency Medicine / Critical Care

COMPREHENSIVE METABOLIC PANEL (CMP):
  eGFR: 24 mL/min/1.73m2 (CRITICAL LOW, Reference: >60 mL/min)
  Serum Creatinine: 3.1 mg/dL (CRITICAL HIGH, Reference: 0.7 - 1.3 mg/dL)
  Potassium (K+): 5.9 mEq/L (CRITICAL HIGH, Reference: 3.5 - 5.0 mEq/L)
  Sodium: 134 mEq/L (Reference: 135 - 145 mEq/L)
  Blood Urea Nitrogen (BUN): 52 mg/dL (HIGH, Reference: 8 - 20 mg/dL)
  Serum Bicarbonate (HCO3): 15 mEq/L (LOW, Reference: 22 - 29 mEq/L)
  Serum Glucose: 218 mg/dL (HIGH, Reference: 70 - 99 mg/dL)

CRITICAL CARE BIOMARKERS:
  Serum Lactate: 4.2 mmol/L (CRITICAL HIGH, Reference: 0.5 - 2.0 mmol/L)
  High-Sensitivity Troponin I: 0.58 ng/mL (ELEVATED, Reference: <0.04 ng/mL)
  WBC Count: 18.9 x10^3/uL (HIGH, Reference: 4.5 - 11.0 x10^3/uL)
  Platelet Count: 142 x10^3/uL (MILD THROMBOCYTOPENIA, Ref: 150 - 450)

ARTERIAL BLOOD GAS (ABG - Room Air):
  pH: 7.24 (Acidemia, Reference: 7.35 - 7.45)
  PaCO2: 30 mmHg (Respiratory compensation, Reference: 35 - 45 mmHg)
  PaO2: 68 mmHg (Moderate Hypoxemia, Reference: 80 - 100 mmHg)

INTERPRETATION & NOTIFICATIONS:
  • Severe metabolic acidosis with hyperlactatemia consistent with tissue hypoperfusion/sepsis.
  • Concomitant Acute Kidney Injury (Stage 3 KDIGO) with severe hyperkalemia.
  • Metformin and potassium-sparing diuretics must be immediately discontinued.
"""
    with open("data/samples/sample_lab_report.txt", "w", encoding="utf-8") as f:
        f.write(lab_text)

    print("All sample assets created in data/samples/ successfully.")


if __name__ == "__main__":
    create_sample_assets()

