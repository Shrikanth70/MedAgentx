# 08 Recommended Evaluation Dataset Suite

## Recommended Dataset Selection for MedAgentX Evaluation

To ensure statistically defensible, un-contaminated evaluation for IEEE submission, we recommend a balanced multi-dataset benchmark strategy:

1. **MedQA USMLE Test Split (1,273 questions):** Primary benchmark for multi-step clinical reasoning and differential diagnosis accuracy.
2. **PubMedQA Labeled Test Set (500 questions):** Primary benchmark for Evidence Verification, Faithfulness, and Citation Attribution.
3. **MMLU Clinical Knowledge Subset (265 questions):** Zero-shot domain knowledge comparison against baseline LLMs.
4. **VQA-RAD Multimodal Radiology Benchmark (2,244 real clinical scan QA pairs, CC0 Public Domain):**
   - **Coverage**: Authentic diagnostic imaging scans across **Chest X-Rays**, **Head/Chest/Abdomen CT Scans**, and **Brain/Spine MRI Scans**.
   - **Curated Dataset Archive**: Stored in `data/datasets/medical_imaging/`:
     - `xray/`: 45 thoracic radiographs (lobar pneumonia, pulmonary edema, tension pneumothorax, cardiomegaly).
     - `ct/`: 65 computed tomography scans (head CT acute intracranial hemorrhage, calvarial/facial fractures, abdominal mechanical bowel obstruction).
     - `mri/`: 11 magnetic resonance imaging studies (brain MRI acute ischemic stroke with DWI diffusion restriction, ventriculomegaly / hydrocephalus, mass lesions).
     - `ecg/`: Validated 12-lead electrocardiograms (anterior STEMI, right ventricular infarction, hyperkalemia).
   - **Metadata Index**: `data/datasets/medical_imaging/dataset_index.json` with ground truth diagnoses, clinical queries, and safety contraindications.
   - **Validation Runner**: `python scripts/test_multimodal_dataset.py` executing 100% automated verification.

