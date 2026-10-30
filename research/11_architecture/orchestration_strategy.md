# 11 Orchestration Strategy & Control Policy

## ACMR Dynamic Routing Rules

```
                      ┌─────────────────────────┐
                      │  Evaluate C_total State │
                      └────────────┬────────────┘
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      │                            │                            │
      ▼                            ▼                            ▼
C_total >= 0.80              0.50 <= C_total < 0.80       C_total < 0.50
[High Confidence]            [Moderate Confidence]        [Low Confidence / Conflict]
      │                            │                            │
      ▼                            ▼                            ▼
Skip Reflection              Trigger Re-Retrieval         Abstain & Prompt Clinician
Route to Reasoner Direct     Targeted Query Expansion     Display Uncertainty Warning
```

1. **High Confidence Threshold ($C_{total} \ge 0.80$):** Proceed immediately to Agent 4 (Clinical Reasoner). Skips redundant reflection and re-retrieval passes, saving tokens and latency.
2. **Moderate Confidence Threshold ($0.50 \le C_{total} < 0.80$):** Trigger targeted re-retrieval via Agent 2 using expanded graph queries. Increment `iteration_count` (max 2 loops).
3. **Low Confidence / High Contradiction Threshold ($C_{total} < 0.50$):** Trigger explicit safety abstention. Output structured warning highlighting evidence gaps or contradictory guidelines.
