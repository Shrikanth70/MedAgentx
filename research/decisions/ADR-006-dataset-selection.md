# ADR-006: Evaluation Benchmark Dataset Selection

## Context
Standardized, peer-reviewed medical QA benchmarks are required to evaluate diagnostic correctness, evidence grounding, and verification accuracy.

## Decision
Select a 3-tier evaluation benchmark suite:
1. **MedQA USMLE Test Split (1,273 questions):** Primary diagnostic accuracy benchmark.
2. **PubMedQA Expert-Labeled Test Split (500 questions):** Primary evidence verification and citation attribution benchmark.
3. **MMLU Clinical Knowledge Subset (265 questions):** Zero-shot domain knowledge comparison.

## Data Provenance Protocol
All evaluation runs will log exact commit hashes, dataset split versions, example IDs, random seeds, and preprocessing versions to prevent benchmark contamination.

## Status
**ACCEPTED** (Date: August 13, 2026)
