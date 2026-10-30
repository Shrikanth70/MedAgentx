"""Metrics Evaluator for Clinical Accuracy, Calibration (ECE), and Efficiency."""

import math
from typing import Any


def calculate_ece(confidences: list[float], correctness: list[int], num_bins: int = 5) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    if not confidences or len(confidences) != len(correctness):
        return 0.0

    bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples falling into current bin
        bin_indices = [
            idx
            for idx, c in enumerate(confidences)
            if bin_lower <= c < bin_upper or (i == num_bins - 1 and c == bin_upper)
        ]

        if bin_indices:
            bin_size = len(bin_indices)
            avg_confidence = sum(confidences[idx] for idx in bin_indices) / bin_size
            avg_accuracy = sum(correctness[idx] for idx in bin_indices) / bin_size
            ece += (bin_size / total_samples) * abs(avg_accuracy - avg_confidence)

    return round(ece, 4)


def calculate_brier_score(confidences: list[float], correctness: list[int]) -> float:
    """Calculate Brier Score for confidence calibration."""
    if not confidences or len(confidences) != len(correctness):
        return 0.0
    total_sq_error = sum((c - y) ** 2 for c, y in zip(confidences, correctness))
    return round(total_sq_error / len(confidences), 4)


def summarize_benchmark_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate benchmark results list into summary performance metrics."""
    if not results:
        return {}

    confidences = [r.get("confidence", 0.0) for r in results]
    correctness = [r.get("is_correct", 1) for r in results]
    tokens = [r.get("token_count", 0) for r in results]
    latencies = [r.get("latency_seconds", 0.0) for r in results]
    agent_counts = [r.get("active_agent_count", 1) for r in results]

    accuracy = sum(correctness) / len(correctness) if correctness else 0.0
    ece = calculate_ece(confidences, correctness)
    brier = calculate_brier_score(confidences, correctness)
    avg_tokens = sum(tokens) / len(tokens) if tokens else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_agents = sum(agent_counts) / len(agent_counts) if agent_counts else 0.0

    return {
        "sample_count": len(results),
        "accuracy": round(accuracy, 4),
        "ece": ece,
        "brier_score": brier,
        "mean_tokens": round(avg_tokens, 2),
        "mean_latency_sec": round(avg_latency, 4),
        "mean_active_agents": round(avg_agents, 2),
    }
