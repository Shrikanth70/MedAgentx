# ADR-008: Query-Evidence Relevance Gating ($S_{rel}$) to Prevent Cross-Disease Misattribution

## Context
During clinical evaluation of MedAgentX, a critical retrieval-verification failure was identified:
When queried for **"baseline quad-therapy for HFrEF"**, an early version of the hybrid retriever returned high-authority **Acute Coronary Syndrome (ACS)** guideline chunks (DAPT, Nitroglycerin, Statins, Angiography). Because the source document had high authority and similarity, the system assigned $C_{total} = 0.72$ and generated an ACS recommendation for an HFrEF query.

**Root Cause Analysis:** High evidence quality $\neq$ High evidence relevance. Relying solely on vector similarity ($S_{vec}$) and Knowledge Graph path presence ($S_{graph}$) allows authoritative documents from adjacent medical domains to contaminate generation.

## Decision
We introduce an explicit **Query-Evidence Relevance Gate Score ($S_{rel}$)** in Agent 3 (Verifier):

1. **Disease & Topic Matching:** Evaluate exact disease domain alignment (e.g., Query: Heart Failure vs Evidence: Acute Coronary Syndrome).
2. **Relevance Discounting in $C_{total}$:**
   $$C_{total} = 0.35 \cdot (S_{vec} \cdot S_{rel}) + 0.25 \cdot (S_{graph} \cdot S_{rel}) + 0.25 \cdot V_{agree} - 0.15 \cdot R_{contra}$$
3. **Safety Override ($S_{rel} < 0.50$):** If $S_{rel} < 0.50$, the system triggers `ABSTAIN` or `RE_RETRIEVE`, blocking Agent 4 (Reasoner) from synthesizing cross-disease misattributed reports.

## Consequences
- **Positive:** Completely eliminates cross-disease hallucination where ACS guidelines are misattributed to HFrEF queries.
- **Positive:** $C_{total}$ reflects clinical answer relevance rather than document retrieval confidence.
- **Verification:** Unit tests and live query traces confirm HFrEF quad-therapy queries synthesize the 4 pillars (ARNI, Beta-blocker, MRA, SGLT2i).
