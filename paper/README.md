# MedAgentX IEEE Manuscript Package

This directory contains the camera-ready IEEE Transactions LaTeX source files for the MedAgentX paper:
- **Title**: *MedAgentX: Adaptive Confidence-Guided Multi-Agent Framework for Multimodal Clinical Decision Support with Fail-Closed Safety Gating*
- **Target Journal**: IEEE Transactions on Biomedical Engineering (TBME) / IEEE Journal of Biomedical and Health Informatics (JBHI)
- **`main.tex`**: Complete two-column IEEEtran manuscript with formal mathematical formulations ($C_{total}$, $\Pi_{penalty}$), Algorithm 1 (ACMR Multimodal Loop), Table I (Clinical Baselines), Table II (Multimodal CDSS Benchmark across 122 Cases), Table III (Ablation Matrix), and Qualitative Failure Taxonomy (Cross-Disease Misattribution and Fail-Open vs Fail-Closed Vision Analysis).
- **`references.bib`**: BibTeX bibliography database covering foundational clinical AI, CheXagent, RadLex, IEC 62304 medical device software standards, Adaptive-RAG, and calibration literature.

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
