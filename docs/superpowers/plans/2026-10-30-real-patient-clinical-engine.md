# MedAgentX Real-World Patient Clinical Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform MedAgentX from answering static, keyword-triggered queries into a robust clinical engine capable of parsing real, unstructured patient presentations, retrieving multi-specialty guidelines dynamically, enforcing vital/lab-aware contraindication safety bounds, and synthesizing patient-tailored clinical guidance.

**Architecture:** Five sequential tasks: (1) Patient schema & clinical vignette regex parser, (2) multi-domain guideline expansion & generalized BM25/TF-IDF hybrid retriever, (3) vital/lab/allergy safety verifier, (4) patient-tailored reasoner & orchestrator integration, (5) interactive CLI inspection suite (`scripts/demo_patient_cases.py`), regression test verification, and GitHub synchronization.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, structlog, Streamlit, FastAPI.

**Spec:** `docs/superpowers/specs/2026-10-30-real-patient-clinical-engine-design.md`

## Global Constraints

- Python 3.12 compatibility.
- All existing 29 pytest tests in `tests/` must continue to pass without regression.
- All new tests run 100% offline with zero external network or container dependencies.
- Freeform text queries continue to work backward-compatibly, while rich `PatientCase` presentations unlock deep clinical reasoning.
- Real-time contraindication detection strictly sets $R_{contra} \ge 0.80$ when acute vitals, renal function ($eGFR < 30$), potassium ($K^+ > 5.2$), or allergies conflict with guideline therapy.

## Review Focus

1. **Backwards compatibility for simple queries**: `system.query("What is the initial dual antiplatelet dosage for STEMI?")` must still return the expected `SystemResponse` without error.
2. **Robust regex parsing of real clinical text**: Vitals written in diverse formats (`BP 84/50`, `BP: 84/50 mmHg`, `blood pressure 84 / 50`, `SpO2 86%`, `eGFR 22 mL/min`, `K 5.8 mEq/L`) must be cleanly parsed into numerical floats.
3. **Multi-domain retrieval without hardcoded keywords**: Queries for Sepsis, AKI, DKA, and COPD must retrieve their respective clinical guideline chunks dynamically without if/elif disease category checks.
4. **Offline synthesis reliability**: When Ollama is unavailable locally, the reasoner must dynamically generate rich, patient-specific 4-part clinical reports tailored to the patient's exact numbers, rather than static templates.
5. **Interactive UI & CLI execution**: `python scripts/demo_patient_cases.py` must run end-to-end and display all 4 realistic emergency patient cases cleanly.

---

## Task 1: Patient Model & Vignette Parser

**Files:**
- Create: `src/orchestrator/patient.py`
- Modify: `src/agents/planner.py`
- Create: `tests/test_patient_engine.py`

**Interfaces:**
- Produces: `PatientCase` schema, `PatientVignetteParser.parse(text: str) -> PatientCase`, `PlannerAgent.plan_patient(patient: PatientCase) -> PlannerOutput`

- [ ] **Step 1: Write failing tests in `tests/test_patient_engine.py`**
  ```python
  def test_patient_vignette_parser():
      raw_note = (
          "72yo male presenting with acute severe dyspnea, orthopnea, and bilateral pedal edema. "
          "BP 84/50 mmHg, HR 118 bpm, SpO2 86% on room air, RR 26. "
          "Labs: eGFR 24 mL/min, Creatinine 2.8 mg/dL, K+ 5.7 mEq/L, elevated BNP. "
          "Allergies: Penicillin. History of HFrEF and CKD Stage 4. "
          "Question: What acute medications should be initiated or avoided?"
      )
      from src.orchestrator.patient import PatientVignetteParser
      patient = PatientVignetteParser.parse(raw_note)
      assert patient.age == 72
      assert patient.sbp == 84.0
      assert patient.dbp == 50.0
      assert patient.hr == 118.0
      assert patient.spo2 == 86.0
      assert patient.egfr == 24.0
      assert patient.potassium == 5.7
      assert patient.is_hypotensive is True
      assert patient.is_hypoxemic is True
      assert patient.is_severe_renal_impairment is True
      assert patient.is_hyperkalemic is True
      assert "penicillin" in [a.lower() for a in patient.allergies]
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_patient_engine.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'src.orchestrator.patient'`

- [ ] **Step 3: Implement `PatientCase` and `PatientVignetteParser` in `src/orchestrator/patient.py`**
  Implement the Pydantic model with physiological properties (`is_hypotensive`, `is_hypoxemic`, `is_severe_renal_impairment`, `is_hyperkalemic`, `is_in_septic_shock`) and regular expressions for vitals, labs, and allergies.

- [ ] **Step 4: Update `src/agents/planner.py` to support `plan_patient(patient: PatientCase)`**
  Enhance Agent 1 to accept either raw string queries or `PatientCase` instances, extracting acute risk flags and formulating targeted clinical sub-queries.

- [ ] **Step 5: Run tests to verify they pass**
  Run: `pytest tests/test_patient_engine.py -v`
  Expected: PASS

- [ ] **Step 6: Commit Task 1**
  ```bash
  git add src/orchestrator/patient.py src/agents/planner.py tests/test_patient_engine.py
  git commit -m "feat: add PatientCase model and clinical vignette regex parser"
  ```

---

## Task 2: Multi-Domain Guideline Expansion & Generalized Hybrid Retrieval

**Files:**
- Modify: `data/processed/chunks.json` (add Sepsis, AKI, Hyperkalemia, COPD, DKA guidelines)
- Modify: `src/knowledge/hybrid_retriever.py` (replace hardcoded branching with dynamic token-overlap & vector fusion)
- Test: `tests/test_patient_engine.py` (add retrieval tests)

**Interfaces:**
- Produces: `HybridRetriever.retrieve(query: str, top_k: int = 5) -> dict[str, Any]` matching non-cardiology conditions dynamically.

- [ ] **Step 1: Write retrieval test in `tests/test_patient_engine.py`**
  ```python
  def test_generalized_retrieval_multi_domain():
      from src.knowledge.hybrid_retriever import HybridRetriever
      retriever = HybridRetriever()
      # Sepsis retrieval
      res_sepsis = retriever.retrieve("Patient in septic shock with refractory hypotension and elevated serum lactate")
      topics = [c.get("clinical_topic", "") for c in res_sepsis["vector_evidence"]]
      assert any("Sepsis" in t or "Septic" in t or "Resuscitation" in t for t in topics)
      # AKI / Hyperkalemia retrieval
      res_renal = retriever.retrieve("Severe acute kidney injury with hyperkalemia K 6.2 and oliguria")
      topics_renal = [c.get("clinical_topic", "") for c in res_renal["vector_evidence"]]
      assert any("Hyperkalemia" in t or "Kidney" in t or "Renal" in t for t in topics_renal)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_patient_engine.py::test_generalized_retrieval_multi_domain -v`
  Expected: FAIL (current hardcoded retriever falls back to general list without Sepsis/AKI chunks)

- [ ] **Step 3: Expand `data/processed/chunks.json` with multi-specialty clinical guidelines**
  Add authoritative chunks for:
  - Surviving Sepsis Campaign 2021 (1-hour bundle, IV fluid 30 mL/kg, norepinephrine, lactate)
  - KDIGO Acute Kidney Injury & Hyperkalemia management (Calcium gluconate, Insulin+D50, holding nephrotoxins)
  - GOLD / GINA COPD & Asthma (controlled oxygen 88-92%, SABA/SAMA, systemic corticosteroids)
  - ADA Diabetic Ketoacidosis (fluid resuscitation, regular insulin infusion, potassium repletion)
  - AHA/ACC Atrial Fibrillation (rate control with beta-blockers/diltiazem, DOAC anticoagulation, renal adjustment)

- [ ] **Step 4: Update `src/knowledge/hybrid_retriever.py` with generalized scoring**
  Remove hardcoded `if/elif` statements. Implement multi-field term scoring (chief complaint, clinical topic, content, disease category) combined with cosine vector similarity.

- [ ] **Step 5: Run test to verify it passes**
  Run: `pytest tests/test_patient_engine.py::test_generalized_retrieval_multi_domain -v`
  Expected: PASS

- [ ] **Step 6: Commit Task 2**
  ```bash
  git add data/processed/chunks.json src/knowledge/hybrid_retriever.py tests/test_patient_engine.py
  git commit -m "feat: expand guideline corpus and implement generalized hybrid retriever"
  ```

---

## Task 3: Vital- and Lab-Aware Safety Verifier

**Files:**
- Modify: `src/agents/verifier.py`
- Test: `tests/test_patient_engine.py`

**Interfaces:**
- Modifies: `VerifierAgent.verify(query_text: str, vector_evidence: list, graph_evidence: list, patient: PatientCase | None = None) -> VerifierOutput`

- [ ] **Step 1: Write safety verification tests in `tests/test_patient_engine.py`**
  ```python
  def test_vital_lab_safety_verifier():
      from src.agents.verifier import VerifierAgent
      from src.orchestrator.patient import PatientCase
      verifier = VerifierAgent()
      
      # Case A: Severe Hypotension Contraindication (SBP < 90)
      pt_hypo = PatientCase(chief_complaint="Chest pain", sbp=82.0, dbp=48.0)
      out_hypo = verifier.verify_patient(
          query_text="Administer IV nitroglycerin and metoprolol",
          vector_evidence=[{"content": "Nitroglycerin and beta blockers are first line for angina", "clinical_topic": "ACS"}],
          patient=pt_hypo,
      )
      assert out_hypo.r_contra >= 0.80
      assert out_hypo.safety_signal is True
      assert out_hypo.contradiction_type == "CLINICAL_CONTRAINDICATION"
      
      # Case B: Severe Hyperkalemia Contraindication (K > 5.2)
      pt_k = PatientCase(chief_complaint="Heart failure", potassium=5.8, egfr=24.0)
      out_k = verifier.verify_patient(
          query_text="Start spironolactone and sacubitril/valsartan",
          vector_evidence=[{"content": "Spironolactone and ARNI are foundational GDMT", "clinical_topic": "HFrEF"}],
          patient=pt_k,
      )
      assert out_k.r_contra >= 0.80
      assert "hyperkalemia" in out_k.safety_reason.lower()
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_patient_engine.py::test_vital_lab_safety_verifier -v`
  Expected: FAIL (`verify_patient` not defined on VerifierAgent)

- [ ] **Step 3: Implement vital-, lab-, and allergy-aware checks in `src/agents/verifier.py`**
  Add `verify_patient` method evaluating:
  - SBP < 90 mmHg / shock: flags vasodilators and acute beta-blockers.
  - eGFR < 30 mL/min: flags Metformin, NSAIDs, and high-dose renally excreted drugs.
  - K+ > 5.2 mEq/L: flags MRAs (Spironolactone), ACEi, ARB, and ARNI.
  - Allergies: flags beta-lactams if Penicillin/Cephalosporin allergy is documented.
  Sets `r_contra = 0.80`, `safety_signal = True`, and descriptive `safety_reason`.

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_patient_engine.py::test_vital_lab_safety_verifier -v`
  Expected: PASS

- [ ] **Step 5: Commit Task 3**
  ```bash
  git add src/agents/verifier.py tests/test_patient_engine.py
  git commit -m "feat: implement vital-, lab-, and allergy-aware clinical safety verification"
  ```

---

## Task 4: Patient-Tailored Reasoner & System Orchestration

**Files:**
- Modify: `src/agents/reasoner.py`
- Modify: `src/orchestrator/system.py`
- Modify: `src/api/routes.py`
- Modify: `app.py`
- Test: `tests/test_patient_engine.py`

**Interfaces:**
- Modifies: `MedAgentXSystem.query_patient(patient: PatientCase | str) -> SystemResponse`
- Produces: 4-part structured patient report with clinical triage, recommendations, safety constraints, and monitoring orders.

- [ ] **Step 1: Write end-to-end patient consultation test in `tests/test_patient_engine.py`**
  ```python
  def test_medagentx_system_real_patient_execution():
      from src.orchestrator.system import MedAgentXSystem
      from src.orchestrator.patient import PatientCase
      system = MedAgentXSystem()
      
      pt = PatientCase(
          patient_id="PT-TEST-01",
          age=68,
          sex="Male",
          chief_complaint="Septic Shock and Suspected Urosepsis",
          sbp=82.0,
          dbp=48.0,
          hr=122.0,
          spo2=91.0,
          lactate=3.8,
          egfr=28.0,
          clinical_question="What is the immediate resuscitation and antimicrobial management protocol?"
      )
      res = system.query_patient(pt)
      assert res.system_id == "PROPOSED_MEDAGENTX"
      assert res.confidence > 0.0
      assert "CLINICAL ASSESSMENT" in res.answer or "SAFETY ALERT" in res.answer
      assert "RESUSCITATION" in res.answer or "Crystalloid" in res.answer or "Norepinephrine" in res.answer
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_patient_engine.py::test_medagentx_system_real_patient_execution -v`
  Expected: FAIL (`query_patient` not defined on `MedAgentXSystem`)

- [ ] **Step 3: Update `src/agents/reasoner.py` to synthesize patient-tailored reports**
  Add `synthesize_patient_report` incorporating patient vitals, abnormal labs, contraindicated drugs, and guideline evidence into the 4 structured report sections.

- [ ] **Step 4: Update `src/orchestrator/system.py` with `query_patient(patient_or_text)`**
  Support passing either a `PatientCase` or a raw string note. When raw text is passed, automatically invoke `PatientVignetteParser.parse()`.

- [ ] **Step 5: Update `src/api/routes.py` and `app.py`**
  Add `/api/v1/patient-consult` route accepting `PatientCase` payloads. Update `app.py` with a clinical case intake tab with ER/ICU case presets.

- [ ] **Step 6: Run full test suite**
  Run: `pytest -v`
  Expected: All tests pass (original 29 + new patient engine tests).

- [ ] **Step 7: Commit Task 4**
  ```bash
  git add src/agents/reasoner.py src/orchestrator/system.py src/api/routes.py app.py tests/test_patient_engine.py
  git commit -m "feat: implement patient-tailored reasoner and system orchestration"
  ```

---

## Task 5: Live Demonstration Runner, Regression Verification & GitHub Push

**Files:**
- Create: `scripts/demo_patient_cases.py`
- Modify: `README.md`

- [ ] **Step 1: Create `scripts/demo_patient_cases.py`**
  CLI runner containing 4 pre-configured realistic emergency patient cases:
  1. Septic Shock with Acute Kidney Injury ($BP\ 82/48$, Lactate $3.8$, $eGFR\ 28$).
  2. Acute Decompensated Heart Failure with Severe Hyperkalemia ($K^+\ 5.8$, $eGFR\ 22$).
  3. Severe Inpatient Community-Acquired Pneumonia with Penicillin Allergy.
  4. Acute Severe COPD Exacerbation with Hypercapnic Failure Risk.
  Includes interactive mode allowing user to type or paste any real clinical presentation.

- [ ] **Step 2: Run live patient cases via CLI**
  Run: `python scripts/demo_patient_cases.py --all`
  Expected: Clean execution of all 4 clinical patient cases with complete traces and tailored reports.

- [ ] **Step 3: Run full pytest suite across entire repository**
  Run: `pytest -v`
  Expected: 100% passing tests (zero failures, zero hangs).

- [ ] **Step 4: Update `README.md`**
  Document the real patient engine, `scripts/demo_patient_cases.py`, and patient intake API.

- [ ] **Step 5: Commit and push to GitHub**
  ```bash
  git add -A
  git commit -m "feat: real patient clinical engine with multi-domain guidelines and live demo suite"
  git push origin main
  ```

