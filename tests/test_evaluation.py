"""Tests for Phase 8 Automated Evaluation Suite & Metrics Evaluator."""

import os
import pytest
from src.evaluation.metrics_eval import (
    calculate_ece,
    calculate_brier_score,
    summarize_benchmark_metrics,
)
from src.evaluation.test_runner import run_benchmark_experiment


def test_calculate_ece():
    """Test Expected Calibration Error calculation."""
    confidences = [0.90, 0.80, 0.70, 0.60, 0.50]
    correctness = [1, 1, 1, 0, 0]
    ece = calculate_ece(confidences, correctness, num_bins=5)
    assert 0.0 <= ece <= 1.0


def test_calculate_brier_score():
    """Test Brier Score calculation."""
    confidences = [1.0, 0.8, 0.6]
    correctness = [1, 1, 0]
    brier = calculate_brier_score(confidences, correctness)
    assert brier >= 0.0


def test_summarize_benchmark_metrics():
    """Test summary performance metrics aggregation."""
    sample_records = [
        {
            "confidence": 0.85,
            "is_correct": 1,
            "token_count": 150,
            "latency_seconds": 0.5,
            "active_agent_count": 3,
        },
        {
            "confidence": 0.40,
            "is_correct": 0,
            "token_count": 100,
            "latency_seconds": 0.3,
            "active_agent_count": 2,
        },
    ]
    summary = summarize_benchmark_metrics(sample_records)
    assert summary["sample_count"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["mean_tokens"] == 125.0
    assert summary["mean_active_agents"] == 2.5


def test_run_benchmark_experiment(tmp_path):
    """Test automated benchmark runner execution and CSV generation."""
    out_dir = str(tmp_path / "eval_results")
    res = run_benchmark_experiment(
        benchmark_name="medqa",
        system_name="b0",
        seed=42,
        output_dir=out_dir,
    )
    assert res["status"] == "success"
    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    assert os.path.exists(csv_path)
