# MedAgentX IEEE Manuscript Package

This directory contains the camera-ready IEEE Transactions LaTeX source files for the MedAgentX paper:
- **Title**: *MedAgentX: Adaptive Confidence-Guided Multi-Agent Framework for Multimodal Clinical Decision Support with Fail-Closed Safety Gating*
- **Authors**: Bhukya Srikanth, Banoth Elisharaj, and G. Baby Lakshmi Prasanna
- **Affiliation**: Department of CSE (AI & ML), Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad, India
- **Target Journal**: IEEE Transactions on Biomedical Engineering (TBME) / IEEE Journal of Biomedical and Health Informatics (JBHI)
- **`main.tex`**: Complete two-column IEEEtran manuscript with formal mathematical formulations ($C_{total}$, $\Pi_{penalty}$), Algorithm 1 (ACMR Multimodal Loop), 2 vector TikZ diagrams (System Architecture and Router Decision State Machine), and 7 publication-standard booktabs tables:
  1. **Table I**: Comprehensive Related-Work Comparison against CheXagent, Med-PaLM M, LLaVA-Med, Clinical Vector RAG, and Adaptive-RAG.
  2. **Table II**: MedAgentX Component and Agent Architecture Specification.
  3. **Table III**: Empirical Baseline Comparison across B0, B1, B2, B7, and MedAgentX.
  4. **Table IV**: Component Ablation Study ($S_{rel}$, Knowledge Graph, Contradiction Penalty, Fail-Closed Hard Gate, Early Stopping).
  5. **Table V**: Benchmark Datasets and Clinical Knowledge Sources (VQA-RAD, MIMIC-CXR subset, Cardiology and Infectious Disease guidelines).
  6. **Table VI**: Failure-Mode and Clinical Safety Analysis across Adverse Scenarios (including VLM timeout fail-closed case).
  7. **Table VII**: System Hyperparameters, Thresholds, and Model Configurations ($\tau_{high}$, $\tau_{low}$, $T_{max}$, backoff retries, penalty coefficients).
- **`references.bib`**: Curated, verified BibTeX bibliography database covering foundational clinical AI, CheXagent, LLaVA-Med, Med-PaLM, RadLex, IEC 62304 medical device software standards, Adaptive-RAG, TorchXRayVision, and calibration literature.

## Key Research Highlights Included in the Manuscript

1. **Two-Tier Multimodal Architecture**: Decouples pre-inference ingestion QC (DICOM de-identification and motion blur/exposure gating) and pluggable VLM perception from guideline-grounded multi-agent reasoning.
2. **Fail-Closed Safety Invariant**: Eliminates dangerous fail-open exception catching where vision timeouts fabricate "Normal" examination findings. Router enforces hard gates routing to `ABSTAIN_INSUFFICIENT_EVIDENCE` or `ESCALATE_TO_HUMAN`.
3. **Calibrated Composite Confidence ($C_{total}$)**: Deterministic formula penalizing vision uncertainty, unverified modalities, and thin evidence count.
4. **Empirical Benchmarks**: 88.2% diagnostic accuracy, Expected Calibration Error (ECE) reduced to **0.0378**, **100% fail-closed safety compliance**, **0.0% hallucination rate**, and a **38.4% token reduction** via adaptive early stopping.

## Compilation Instructions

### Option 1: Local pdflatex / latexmk
Ensure TeX Live / MiKTeX is installed with `IEEEtran.cls`.
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option 2: Overleaf
1. Create a **New Project** -> **Upload Project (ZIP)**.
2. Zip the `paper/` directory containing `main.tex` and `references.bib`.
3. Select the XeLaTeX or pdfLaTeX engine and click **Recompile**.
