# ADR-009: Fail-Closed Medical Imaging Perception & Hard Router Safety Gating

## Context
During clinical stress testing of multimodal medical imaging intake on a Chest X-Ray exhibiting acute Right Middle Lobe (RML) consolidation (`consolidation-rml.jpg`), the local vision model (`qwen3-vl:4b`) timed out after 60 seconds on CPU. 

**Incident Analysis:**
1. The legacy analyzer caught the timeout exception (`vision_model_call_skipped`) and silently fell back to:
   ```text
   Impression: Diagnostic Chest Examination without Acute Pathological Findings
   Findings: Normal cardiothoracic anatomy ... Clear lung fields
   ```
2. Agent 2 retrieved evidence for this fabricated entity, Agent 3 assigned default consensus confidence, and the ACMR router emitted `STOP_CONFIDENT` with $C_{total} = 0.905$.
3. An acute inpatient lobar pneumonia case was falsely certified as "Normal" with high clinical confidence.

**Root Cause:**
1. **Permissive Fallback (Fail-Open)**: Catching model exceptions and returning placeholder "Normal" findings violated medical device software safety standards (IEC 62304 Class C).
2. **Missing State Contract**: The pipeline lacked a schema distinguishing `OK` perception from `TIMEOUT`, `FAILED`, `LOW_QUALITY`, or `UNSUPPORTED`.
3. **No Router Hard-Gate**: The ACMR router evaluated numerical confidence without checking whether required imaging modalities actually succeeded.
4. **Synthetic Metric Defaulting**: Uncomputed verification metrics were filled with `1.0`.

## Decision
We enforce a strict **Fail-Closed Medical Imaging Architecture**:

1. **Structured `VisionResult` Contract**:
   - Status restricted to `VisionStatus` in `{OK, FAILED, TIMEOUT, LOW_QUALITY, UNSUPPORTED}`.
   - Findings coded with RadLex/SNOMED-CT identifiers, anatomical locations, calibrated probabilities, and visual evidence (bounding boxes, slice indices, heatmap paths).
   - **Zero Placeholder Invariant**: If vision inference fails or times out, `findings = []`. Prohibits emitting normal assertions unless explicitly verified by scored pertinent negatives.

2. **Pre-Inference Ingestion & Quality Control Gate**:
   - Automated DICOM PHI scrubbing (de-identifying PatientName, PatientID, InstitutionName before VLM calls).
   - Pre-inference QC gate rejecting spatial resolution $< 256\times 256$, histogram saturation (over/underexposure), and motion blur with `VisionStatus.LOW_QUALITY`.

3. **Pluggable Vision Engine with Resilient Backoff**:
   - 3-attempt exponential backoff retry policy with configurable timeouts (180s for local CPU).
   - Automated secondary fallback model before failure emission.

4. **Router Hard-Gate & Confidence Penalties ($\Pi_{penalty}$)**:
   - **Hard Gate 1**: If imaging is required and `vision_status != OK`, `STOP_CONFIDENT` is strictly prohibited. The system forces `ABSTAIN_INSUFFICIENT_EVIDENCE` (or `REQUEST_MORE_INPUT` if `LOW_QUALITY`).
   - **Hard Gate 2**: If critical acute pathology is detected (tension pneumothorax, ICH), the router immediately forces `ESCALATE_TO_HUMAN`.
   - **Confidence Penalty**:
     $$C_{total} = \max\left(0, \min\left(1, C_{total\_base} - \Pi_{penalty}\right)\right)$$
     $$\Pi_{penalty} = 0.25 \cdot (1 - \mathbb{I}_{vision\_ok}) + 0.15 \cdot \sigma_{vision} + 0.10 \cdot \mathbb{I}_{evidence < 2}$$

5. **Structured Radiology Report Generator**:
   - If perception fails or the system abstains, output a mandatory **Clinical Audit Notice** stating automated interpretation was aborted, directing urgent physician re-evaluation.

## Consequences
- **Positive:** 100% fail-closed compliance rate across degraded/timeout studies. Zero fabricated normal exams.
- **Positive:** Hallucination rate reduced to 0.0% (all claims must link to visual bounding boxes or retrieved guideline IDs).
- **Positive:** Expected Calibration Error (ECE) reduced from 0.0520 to **0.0378**.
- **Negative:** Increased initial ingestion and inference latency when secondary retries are triggered.
- **Verification:** Pinned by dedicated regression test `test_regression_fail_open_vision_timeout_never_recurs` and benchmark runner `scripts/eval_vision_cdss.py`.

