import pandas as pd

df = pd.read_parquet('data/downloads/vqa_rad_test.parquet')

print("--- CT Scans ---")
ct_df = df[df['question'].str.contains('ct|computed tomography', case=False)].head(8)
for idx, r in ct_df.iterrows():
    p = r['image']['path'] if isinstance(r['image'], dict) else ''
    print(f"Row {idx} ({p}): Q: {r['question']} | A: {r['answer']}")

print("\n--- MRI Scans ---")
mri_df = df[df['question'].str.contains('mri|magnetic resonance', case=False)].head(8)
for idx, r in mri_df.iterrows():
    p = r['image']['path'] if isinstance(r['image'], dict) else ''
    print(f"Row {idx} ({p}): Q: {r['question']} | A: {r['answer']}")

print("\n--- Chest / Lungs / Heart / X-Rays ---")
xray_df = df[df['question'].str.contains('lung|chest|edema|consolidation|heart|cardiomegaly|pneumo', case=False)].head(8)
for idx, r in xray_df.iterrows():
    p = r['image']['path'] if isinstance(r['image'], dict) else ''
    print(f"Row {idx} ({p}): Q: {r['question']} | A: {r['answer']}")

