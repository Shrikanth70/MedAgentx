import os
import json
import pandas as pd
from PIL import Image
import io

os.makedirs("data/datasets/medical_imaging/xray", exist_ok=True)
os.makedirs("data/datasets/medical_imaging/ct", exist_ok=True)
os.makedirs("data/datasets/medical_imaging/mri", exist_ok=True)
os.makedirs("data/datasets/medical_imaging/ecg", exist_ok=True)

df_test = pd.read_parquet("data/downloads/vqa_rad_test.parquet")
df_train = pd.read_parquet("data/downloads/vqa_rad_train.parquet")
df = pd.concat([df_test, df_train], ignore_index=True)

print(f"Total rows combined: {len(df)}")

catalog = []
saved_paths = set()

# Helper to save image
def save_img(img_bytes, filename, folder):
    filepath = os.path.join("data/datasets/medical_imaging", folder, filename)
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    return filepath

# Categorize and curate diverse images across modalities
for idx, row in df.iterrows():
    img_data = row["image"]
    if not isinstance(img_data, dict) or "bytes" not in img_data:
        continue
    
    img_bytes = img_data["bytes"]
    raw_path = img_data.get("path", f"img_{idx}.jpg")
    q = str(row["question"]).lower()
    ans = str(row["answer"])
    
    # Identify modality
    modality = None
    subfolder = None
    category = "Diagnostic Imaging"
    contraindications = []
    
    if any(k in q for k in ["mri", "t1", "t2", "flair", "dwi", "diffusion"]):
        modality = "Brain / Spine MRI"
        subfolder = "mri"
        if "dwi" in q or "diffusion" in q or "infarct" in q or "ischemi" in q:
            category = "Acute Ischemic Infarction"
        elif "ventricle" in q or "hydrocephalus" in q:
            category = "Ventriculomegaly / Hydrocephalus"
        elif "mass" in q or "tumor" in q or "glioma" in q:
            category = "Intracranial Mass / Neoplasm"
        else:
            category = "Magnetic Resonance Imaging"
            
    elif any(k in q for k in ["ct", "computed tomography", "axial", "coronal", "skull", "bone window", "temporal bone", "bowel obstruction", "pancreas"]):
        modality = "CT Scan (Head / Chest / Abdomen)"
        subfolder = "ct"
        if "hemorrhage" in q or "bleed" in q or "hematoma" in q:
            category = "Acute Intracranial Hemorrhage"
            contraindications = ["Thrombolytics (IV Alteplase / Tenecteplase)", "Anticoagulants (Heparin / Enoxaparin / DOACs)", "Antiplatelet agents"]
        elif "fracture" in q or "skull" in q:
            category = "Traumatic Craniofacial Fracture"
        elif "aneurysm" in q:
            category = "Aortic / Vascular Aneurysm"
        elif "obstruction" in q:
            category = "Acute Bowel Obstruction"
        else:
            category = "Computed Tomography Diagnostic Scan"
            
    elif any(k in q for k in ["chest", "lung", "x-ray", "xray", "radiograph", "edema", "consolidation", "cardiomegaly", "pneumothorax", "pleural", "bronchus", "infiltrate"]):
        modality = "Chest X-Ray (CXR)"
        subfolder = "xray"
        if "pneumothorax" in q:
            category = "Pneumothorax"
            contraindications = ["Positive-pressure ventilation (BiPAP/CPAP) without immediate chest tube"]
        elif "edema" in q or "heart enlarged" in q or "cardiomegaly" in q:
            category = "Cardiogenic Pulmonary Edema / Cardiomegaly"
            contraindications = ["Aggressive fluid boluses"]
        elif "consolidation" in q or "infiltrate" in q:
            category = "Lobar Pneumonia / Alveolar Consolidation"
        else:
            category = "Thoracic Radiography"

    if modality and subfolder:
        out_name = f"{subfolder}_{raw_path}"
        if out_name not in saved_paths and len(saved_paths) < 120:
            saved_paths.add(out_name)
            saved_file = save_img(img_bytes, out_name, subfolder)
            catalog.append({
                "id": f"IMG-{len(catalog)+1:03d}",
                "file_path": saved_file.replace("\\", "/"),
                "file_name": out_name,
                "modality": modality,
                "subfolder": subfolder,
                "clinical_category": category,
                "sample_question": row["question"],
                "ground_truth_answer": ans,
                "contraindicated_interventions": contraindications,
                "source_dataset": "VQA-RAD (CC0 Public Domain, Nature Scientific Data)",
            })

# Also include our existing verified ECG diagnostic scans
ecg_samples = [
    {
        "id": f"IMG-{len(catalog)+1:03d}",
        "file_path": "data/samples/sample_ecg_stemi.png",
        "file_name": "sample_ecg_stemi.png",
        "modality": "12-Lead ECG",
        "subfolder": "ecg",
        "clinical_category": "Acute Anterior STEMI",
        "sample_question": "Does this 12-lead ECG demonstrate acute ST elevation myocardial infarction in anterior leads?",
        "ground_truth_answer": "Yes, marked ST elevation in V1-V4 indicative of acute anterior STEMI requiring emergent PCI.",
        "contraindicated_interventions": ["Delaying primary percutaneous coronary intervention (PCI)"],
        "source_dataset": "MedAgentX Validated Diagnostic Asset Library",
    },
    {
        "id": f"IMG-{len(catalog)+2:03d}",
        "file_path": "data/samples/sample_cxr_pneumothorax.png",
        "file_name": "sample_cxr_pneumothorax.png",
        "modality": "Chest X-Ray (CXR)",
        "subfolder": "xray",
        "clinical_category": "Tension Pneumothorax",
        "sample_question": "Is positive pressure ventilation contraindicated in this patient?",
        "ground_truth_answer": "Yes, visceral pleural line and hyperlucency indicate pneumothorax; positive pressure is contraindicated prior to chest tube decompression.",
        "contraindicated_interventions": ["Positive-pressure ventilation (BiPAP/CPAP) without immediate chest tube decompression"],
        "source_dataset": "MedAgentX Validated Diagnostic Asset Library",
    }
]
catalog.extend(ecg_samples)

index_path = "data/datasets/medical_imaging/dataset_index.json"
with open(index_path, "w", encoding="utf-8") as f:
    json.dump(catalog, f, indent=2)

summary = {
    "total_studies": len(catalog),
    "modalities": {},
    "subfolders": {},
}
for c in catalog:
    summary["modalities"][c["modality"]] = summary["modalities"].get(c["modality"], 0) + 1
    summary["subfolders"][c["subfolder"]] = summary["subfolders"].get(c["subfolder"], 0) + 1

summary_path = "data/datasets/medical_imaging/summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nCurated Multimodal Medical Imaging Dataset Created Successfully:")
print(f"Total curated cases: {len(catalog)}")
print("Modalities breakdown:", summary["modalities"])
print("Subfolders breakdown:", summary["subfolders"])
print(f"Saved catalog to: {index_path}")

