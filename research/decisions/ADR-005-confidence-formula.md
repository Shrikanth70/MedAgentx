# ADR-005: Mathematical Composite Evidence Confidence State Formulation ($C_{total}$)

## Context
Standard LLMs and prior clinical agent frameworks (e.g., MedCoAct 2025) rely on subjective verbalized confidence prompts ("rate confidence 0-100%"), which are notoriously miscalibrated and prone to overconfident hallucinations.

## Decision
Implement an objective mathematical formulation for Composite Evidence Confidence ($C_{total}$):

$$C_{total} = w_1 \cdot S_{vec} + w_2 \cdot S_{graph} + w_3 \cdot V_{agree} - w_4 \cdot R_{contra}$$

Initial research parameters: $w_1 = 0.35, w_2 = 0.25, w_3 = 0.25, w_4 = 0.15$ (constrained such that $\sum w = 1.0$).

## Configurable Policy Thresholds
- **High Confidence ($C_{total} \ge 0.80$):** Direct synthesis, early stopping.
- **Moderate Confidence ($0.50 \le C_{total} < 0.80$):** Targeted re-retrieval (max 2 iterations).
- **Low Confidence ($C_{total} < 0.50$):** Explicit safety abstention warning.

## Status
**ACCEPTED** (Date: August 13, 2026)
