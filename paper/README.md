# MedAgentX IEEE Manuscript Package

This directory contains the camera-ready IEEE Transactions LaTeX source files for the MedAgentX paper:
- **`main.tex`**: Complete two-column IEEEtran manuscript with formal mathematical formulations ($C_{total}$), Algorithm 1 pseudocode, Table I (benchmark comparison), Table II (ablation matrix), and qualitative failure taxonomy case studies.
- **`references.bib`**: BibTeX bibliography database covering foundational clinical AI, Adaptive-RAG, CRAG, SEMA-RAG, and calibration literature.

## Compilation Instructions

### Option 1: Local pdflatex / latexmk
Ensure TeX Live / MiKTeX is installed.
```bash
# Obtain IEEEtran.cls if not already installed globally:
# Download IEEEtran.cls from CTAN: https://www.ctan.org/pkg/ieeetran

pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option 2: Overleaf
1. Create a **New Project** -> **Upload Project (ZIP)**.
2. Zip the `paper/` directory containing `main.tex` and `references.bib`.
3. Upload and click **Recompile**.
