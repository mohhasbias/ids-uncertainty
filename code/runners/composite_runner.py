#!/usr/bin/env python3
"""
Composite Confidence Runner for IDS Uncertainty Estimation

Post-processes Hybrid runner results to calculate composite confidence scores using penalty-based formula.
Makes ZERO API calls - purely computational post-processing.
Outputs results to scenario_results/composite/run_YYYYMMDD_HHMMSS/composite_results.jsonl

Formula: C_composite = C_self - β·(1 - C_conf_consistency) - γ·(1 - C_resp_consistency)
Where: β=0.5, γ=0.5, result clamped to [0.0, 1.0]

Usage:
    uv run python composite_runner.py --hybrid-results scenario_results/hybrid/run_20251128_123456/hybrid_results.jsonl
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from ids_models import (
    IDSStructuredResponse,
    HybridTestResult,
    CompositeTestResult,
    AttackType,
    ThreatLevel,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Composite Confidence Runner for IDS Uncertainty Estimation"
    )
    parser.add_argument(
        "--hybrid-results",
        type=str,
        required=True,
        help="Path to hybrid runner JSONL results file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generate timestamp)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Penalty coefficient for confidence consistency (default: 0.5)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Penalty coefficient for response consistency (default: 0.5)",
    )
    return parser.parse_args()


def load_hybrid_results(hybrid_file: Path) -> List[Dict[str, Any]]:
    """
    Load hybrid results from JSONL file.

    Args:
        hybrid_file: Path to hybrid JSONL results file

    Returns:
        List of hybrid result dictionaries
    """
    results = []

    if not hybrid_file.exists():
        raise FileNotFoundError(f"Hybrid results file not found: {hybrid_file}")

    with open(hybrid_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                result = json.loads(line)
                results.append(result)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping malformed JSON at line {line_num}: {e}")
                continue

    return results


def calculate_c_self(predictions: List[Dict[str, Any]]) -> float:
    """
    Calculate C_self: mean self-reported confidence across all predictions.

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Mean confidence score in [0.0, 1.0]
    """
    if not predictions:
        return 0.0

    confidence_scores = [p["confidence_score"] for p in predictions]
    return float(np.mean(confidence_scores))


def calculate_c_conf_consistency(predictions: List[Dict[str, Any]]) -> float:
    """
    Calculate C_conf_consistency: 1 - std_dev of confidence scores.

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Confidence consistency in [0.0, 1.0]
    """
    if not predictions or len(predictions) == 1:
        return 1.0  # Single prediction is perfectly consistent

    confidence_scores = [p["confidence_score"] for p in predictions]
    std_dev = float(np.std(confidence_scores))

    # Clamp to [0, 1] to handle edge cases
    return max(0.0, min(1.0, 1.0 - std_dev))


def calculate_c_resp_consistency(predictions: List[Dict[str, Any]]) -> float:
    """
    Calculate C_resp_consistency: agreement rate on is_malicious classification.

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Response consistency (agreement rate) in [0.0, 1.0]
    """
    if not predictions:
        return 0.0

    # Count majority vote
    malicious_votes = [p["is_malicious"] for p in predictions]
    most_common_vote = max(set(malicious_votes), key=malicious_votes.count)

    # Calculate agreement rate
    matches = sum(1 for vote in malicious_votes if vote == most_common_vote)
    return matches / len(predictions)


def calculate_composite_confidence(
    predictions: List[Dict[str, Any]],
    beta: float = 0.5,
    gamma: float = 0.5,
) -> tuple[float, Dict[str, float]]:
    """
    Calculate composite confidence using penalty-based formula.

    Formula: C_composite = C_self - β·(1 - C_conf_consistency) - γ·(1 - C_resp_consistency)

    Args:
        predictions: List of prediction dictionaries
        beta: Penalty coefficient for confidence consistency
        gamma: Penalty coefficient for response consistency

    Returns:
        Tuple of (composite_confidence, components_dict)
    """
    c_self = calculate_c_self(predictions)
    c_conf_consistency = calculate_c_conf_consistency(predictions)
    c_resp_consistency = calculate_c_resp_consistency(predictions)

    # Apply penalty-based formula
    composite = c_self - beta * (1.0 - c_conf_consistency) - gamma * (1.0 - c_resp_consistency)

    # Clamp to [0.0, 1.0]
    composite = max(0.0, min(1.0, composite))

    components = {
        "c_self": c_self,
        "c_conf_consistency": c_conf_consistency,
        "c_resp_consistency": c_resp_consistency,
    }

    return composite, components


def calculate_routing_decision(
    confidence: float,
    threshold_accept: float = 0.85,
    threshold_reject: float = 0.60,
) -> str:
    """
    Calculate routing decision based on confidence thresholds.

    Args:
        confidence: Composite confidence score
        threshold_accept: Minimum confidence for accept
        threshold_reject: Minimum confidence for escalate (below is reject)

    Returns:
        'accept', 'escalate', or 'reject'
    """
    if confidence >= threshold_accept:
        return "accept"
    elif confidence >= threshold_reject:
        return "escalate"
    else:
        return "reject"


def process_hybrid_result(
    hybrid_result: Dict[str, Any],
    hybrid_file_path: str,
    beta: float,
    gamma: float,
) -> CompositeTestResult:
    """
    Process a single hybrid result to calculate composite confidence.

    Args:
        hybrid_result: Hybrid result dictionary
        hybrid_file_path: Path to source hybrid JSONL file
        beta: Penalty coefficient for confidence consistency
        gamma: Penalty coefficient for response consistency

    Returns:
        CompositeTestResult
    """
    start_time = time.time()

    try:
        # Extract predictions from hybrid result
        all_predictions = hybrid_result.get("all_predictions", [])

        if not all_predictions:
            raise ValueError("No predictions found in hybrid result")

        # Calculate composite confidence
        composite_confidence, components = calculate_composite_confidence(
            all_predictions, beta=beta, gamma=gamma
        )

        # Copy final prediction and ground truth from hybrid
        final_pred_dict = hybrid_result["final_prediction"]
        final_prediction = IDSStructuredResponse(**final_pred_dict)
        
        ground_truth_dict = hybrid_result["ground_truth"]
        from ids_models import GroundTruth
        ground_truth = GroundTruth(**ground_truth_dict)

        # Recalculate routing decision with composite confidence
        routing_decision = calculate_routing_decision(composite_confidence)

        execution_time = time.time() - start_time

        return CompositeTestResult(
            sample_id=hybrid_result["sample_id"],
            model_name=hybrid_result["model_name"],
            timestamp=datetime.utcnow().isoformat() + "Z",
            ground_truth=ground_truth,
            composite_confidence=composite_confidence,
            composite_components=components,
            final_prediction=final_prediction,
            hybrid_run_reference=hybrid_file_path,
            is_correct=hybrid_result["is_correct"],
            routing_decision=routing_decision,
            execution_time=execution_time,
            error=None,
        )

    except Exception as e:
        execution_time = time.time() - start_time

        # Try to get ground truth from hybrid result, use default if not available
        from ids_models import GroundTruth
        try:
            ground_truth = GroundTruth(**hybrid_result.get("ground_truth", {}))
        except Exception:
            ground_truth = GroundTruth(
                is_malicious=False,
                attack_type=AttackType.UNKNOWN,
                threat_level=ThreatLevel.BENIGN,
            )

        # Return error result
        return CompositeTestResult(
            sample_id=hybrid_result.get("sample_id", "unknown"),
            model_name=hybrid_result.get("model_name", "unknown"),
            timestamp=datetime.utcnow().isoformat() + "Z",
            ground_truth=ground_truth,
            composite_confidence=0.0,
            composite_components={
                "c_self": 0.0,
                "c_conf_consistency": 0.0,
                "c_resp_consistency": 0.0,
            },
            final_prediction=IDSStructuredResponse(
                is_malicious=False,
                threat_level=ThreatLevel.BENIGN,
                attack_type=AttackType.UNKNOWN,
                confidence_score=0.0,
                reasoning="Error occurred during composite calculation",
                key_features=[],
            ),
            hybrid_run_reference=hybrid_file_path,
            is_correct=False,
            routing_decision="reject",
            execution_time=execution_time,
            error=str(e),
        )


def setup_output_directory(output_dir: Optional[str] = None) -> Path:
    """
    Setup output directory with timestamp.

    Args:
        output_dir: Custom output directory path (optional)

    Returns:
        Path to output directory
    """
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📂 Using output directory: {output_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"scenario_results/composite/run_{timestamp}")
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📂 Created output directory: {output_path}")

    return output_path


def main():
    """Main execution function"""
    args = parse_arguments()

    print("=" * 80)
    print("Composite Confidence Runner")
    print("=" * 80)
    print(f"Hybrid Results: {args.hybrid_results}")
    print(f"Beta (β): {args.beta}")
    print(f"Gamma (γ): {args.gamma}")
    print(f"Formula: C_composite = C_self - β·(1 - C_conf) - γ·(1 - C_resp)")
    print(f"API Calls: 0 (post-processing only)")
    print("=" * 80)

    # Load hybrid results
    hybrid_file = Path(args.hybrid_results)
    print(f"\n📊 Loading hybrid results from: {hybrid_file}")

    try:
        hybrid_results = load_hybrid_results(hybrid_file)
        print(f"✅ Loaded {len(hybrid_results)} hybrid results")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    except Exception as e:
        print(f"❌ Error loading hybrid results: {e}")
        return

    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    output_file = output_dir / "composite_results.jsonl"

    # Process each hybrid result
    print(f"\n🔄 Processing composite confidence calculations...")

    with open(output_file, "w") as f:
        total_processed = 0
        total_correct = 0
        total_errors = 0
        composite_scores = []

        for idx, hybrid_result in enumerate(hybrid_results, 1):
            sample_id = hybrid_result.get("sample_id", f"unknown_{idx}")

            # Skip error results from hybrid
            if hybrid_result.get("error"):
                print(f"⏭️  [{idx}/{len(hybrid_results)}] Skipping hybrid error: {sample_id}")
                continue

            print(f"[{idx}/{len(hybrid_results)}] Processing sample {sample_id}")

            # Process composite confidence
            result = process_hybrid_result(
                hybrid_result=hybrid_result,
                hybrid_file_path=str(hybrid_file),
                beta=args.beta,
                gamma=args.gamma,
            )

            # Write result to JSONL
            f.write(json.dumps(result.model_dump()) + "\n")
            f.flush()

            # Update statistics
            total_processed += 1
            if result.error:
                total_errors += 1
                print(f"  ❌ Error: {result.error}")
            else:
                if result.is_correct:
                    total_correct += 1

                composite_scores.append(result.composite_confidence)

                print(f"  Composite: {result.composite_confidence:.3f} "
                      f"(self={result.composite_components['c_self']:.3f}, "
                      f"conf_cons={result.composite_components['c_conf_consistency']:.3f}, "
                      f"resp_cons={result.composite_components['c_resp_consistency']:.3f})")
                print(f"  Routing: {result.routing_decision}, Correct: {result.is_correct}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total samples processed: {total_processed}")

    if total_processed > 0:
        accuracy = total_correct / total_processed
        print(f"Accuracy: {total_correct}/{total_processed} ({accuracy:.1%})")
        print(f"Errors: {total_errors}")

        if composite_scores:
            print(f"\nComposite Confidence Statistics:")
            print(f"  Mean: {np.mean(composite_scores):.3f}")
            print(f"  Std Dev: {np.std(composite_scores):.3f}")
            print(f"  Min: {min(composite_scores):.3f}")
            print(f"  Max: {max(composite_scores):.3f}")

    print(f"\nResults saved to: {output_file}")
    print(f"API Calls Made: 0 (100% cost efficient)")
    print("=" * 80)


if __name__ == "__main__":
    main()
