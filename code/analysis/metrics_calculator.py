#!/usr/bin/env python3
"""
Metrics Calculator for IDS Uncertainty Estimation
Implements key evaluation metrics:
- Accuracy, Precision, Recall, F1
- Expected Calibration Error (ECE)
- Confidence-Accuracy Correlation
- Selective Risk / Routing Accuracy
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class ClassificationMetrics:
    """Core classification performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total: int


@dataclass
class CalibrationMetrics:
    """Calibration metrics for uncertainty estimation"""
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float  # MCE
    brier_score: float
    confidence_bins: List[Dict[str, Any]]  # For plotting calibration curves


@dataclass
class CorrelationMetrics:
    """Correlation between uncertainty and accuracy"""
    confidence_accuracy_correlation: float
    correlation_p_value: float
    selective_risk: Dict[str, float]  # Risk at different coverage levels


@dataclass
class RoutingMetrics:
    """Metrics for confidence-based routing effectiveness"""
    automation_rate: float  # Percentage of samples accepted automatically
    escalation_rate: float  # Percentage requiring human review
    rejection_rate: float  # Percentage rejected
    routing_accuracy: float  # Accuracy of routing decisions
    accepted_accuracy: float  # Accuracy of auto-accepted samples
    escalated_accuracy: float  # Accuracy of escalated samples
    routing_distribution: Dict[str, int]


class MetricsCalculator:
    """
    Calculate comprehensive metrics for IDS uncertainty evaluation
    """

    def __init__(self, results_file: str, n_bins: int = 10):
        """
        Initialize metrics calculator

        Args:
            results_file: Path to JSONL results file from baseline_runner
            n_bins: Number of bins for calibration error calculation
        """
        self.results_file = Path(results_file)
        self.n_bins = n_bins
        self.results = self._load_results()

    def _load_results(self) -> List[Dict[str, Any]]:
        """Load results from JSONL file"""
        results = []
        with open(self.results_file, 'r') as f:
            for line in f:
                results.append(json.loads(line.strip()))
        return results

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate all metrics

        Returns:
            Dictionary with all metric categories
        """
        return {
            "classification": self.calculate_classification_metrics(),
            "calibration": self.calculate_calibration_metrics(),
            "correlation": self.calculate_correlation_metrics(),
            "routing": self.calculate_routing_metrics(),
            "summary": self._generate_summary()
        }

    def calculate_classification_metrics(self) -> ClassificationMetrics:
        """Calculate core classification metrics"""
        tp = tn = fp = fn = 0

        for result in self.results:
            # Skip error results
            if result.get('error'):
                continue

            is_correct = result['is_correct']
            gt = result['ground_truth']['is_malicious']

            # Handle both baseline and ensemble result formats
            if 'prediction' in result:
                pred = result['prediction'].get('is_malicious', False)
            elif 'final_prediction' in result:
                pred = result['final_prediction'].get('is_malicious', False)
            else:
                continue  # Skip if neither format is found

            if gt and pred:
                tp += 1
            elif not gt and not pred:
                tn += 1
            elif not gt and pred:
                fp += 1
            elif gt and not pred:
                fn += 1

        total = tp + tn + fp + fn

        # Calculate metrics (with division by zero protection)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            total=total
        )

    def calculate_calibration_metrics(self) -> CalibrationMetrics:
        """
        Calculate calibration metrics including ECE and MCE

        Expected Calibration Error (ECE): Average difference between
        confidence and accuracy across bins
        """
        # Extract confidence and correctness
        confidences = []
        correctness = []

        for result in self.results:
            if result.get('error'):
                continue

            # Handle both baseline and ensemble result formats
            if 'prediction' in result:
                conf = result['prediction'].get('confidence_score', 0.0)
            elif 'final_prediction' in result:
                conf = result['final_prediction'].get('confidence_score', 0.0)
            else:
                continue
            correct = 1.0 if result['is_correct'] else 0.0

            confidences.append(conf)
            correctness.append(correct)

        confidences = np.array(confidences)
        correctness = np.array(correctness)

        # Create bins
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidences, bins[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        # Calculate ECE and MCE
        ece = 0.0
        mce = 0.0
        confidence_bins = []

        for bin_idx in range(self.n_bins):
            bin_mask = bin_indices == bin_idx
            bin_size = np.sum(bin_mask)

            if bin_size == 0:
                confidence_bins.append({
                    'bin_idx': bin_idx,
                    'bin_range': (bins[bin_idx], bins[bin_idx + 1]),
                    'count': 0,
                    'avg_confidence': 0.0,
                    'accuracy': 0.0,
                    'calibration_error': 0.0
                })
                continue

            bin_confidences = confidences[bin_mask]
            bin_correctness = correctness[bin_mask]

            avg_confidence = np.mean(bin_confidences)
            avg_accuracy = np.mean(bin_correctness)
            calibration_error = abs(avg_confidence - avg_accuracy)

            # Update ECE (weighted by bin size)
            ece += (bin_size / len(confidences)) * calibration_error

            # Update MCE (maximum calibration error)
            mce = max(mce, calibration_error)

            confidence_bins.append({
                'bin_idx': bin_idx,
                'bin_range': (float(bins[bin_idx]), float(bins[bin_idx + 1])),
                'count': int(bin_size),
                'avg_confidence': float(avg_confidence),
                'accuracy': float(avg_accuracy),
                'calibration_error': float(calibration_error)
            })

        # Calculate Brier score
        brier_score = np.mean((confidences - correctness) ** 2)

        return CalibrationMetrics(
            expected_calibration_error=float(ece),
            maximum_calibration_error=float(mce),
            brier_score=float(brier_score),
            confidence_bins=confidence_bins
        )

    def calculate_correlation_metrics(self) -> CorrelationMetrics:
        """
        Calculate correlation between confidence and accuracy
        This is critical for validating the uncertainty hypothesis
        """
        # Extract confidence and correctness
        confidences = []
        correctness = []

        for result in self.results:
            if result.get('error'):
                continue

            # Handle both baseline and ensemble result formats
            if 'prediction' in result:
                conf = result['prediction'].get('confidence_score', 0.0)
            elif 'final_prediction' in result:
                conf = result['final_prediction'].get('confidence_score', 0.0)
            else:
                continue
            correct = 1.0 if result['is_correct'] else 0.0

            confidences.append(conf)
            correctness.append(correct)

        # Calculate Pearson correlation (requires n≥2)
        # Safety net: This should never happen due to main() validation, but handle gracefully
        if len(confidences) >= 2:
            correlation, p_value = stats.pearsonr(confidences, correctness)
        else:
            # Defensive fallback - indicates validation was bypassed
            print(f"WARNING: Bypassed validation - only {len(confidences)} samples for correlation")
            correlation = float('nan')
            p_value = float('nan')

        # Calculate selective risk at different coverage levels
        # (accuracy when accepting top X% by confidence)
        selective_risk = self._calculate_selective_risk(confidences, correctness)

        return CorrelationMetrics(
            confidence_accuracy_correlation=float(correlation),
            correlation_p_value=float(p_value),
            selective_risk=selective_risk
        )

    def _calculate_selective_risk(
        self,
        confidences: List[float],
        correctness: List[float]
    ) -> Dict[str, float]:
        """
        Calculate accuracy at different coverage levels
        (when accepting top X% most confident predictions)
        """
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        sorted_correctness = np.array(correctness)[sorted_indices]

        coverage_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        selective_risk = {}

        for coverage in coverage_levels:
            n_samples = int(len(sorted_correctness) * coverage)
            if n_samples == 0:
                continue

            top_correctness = sorted_correctness[:n_samples]
            accuracy = np.mean(top_correctness)

            selective_risk[f"coverage_{int(coverage*100)}"] = float(accuracy)

        return selective_risk

    def calculate_routing_metrics(self) -> RoutingMetrics:
        """
        Calculate metrics for confidence-based routing decisions
        This evaluates the practical deployment value
        """
        routing_counts = {'accept': 0, 'escalate': 0, 'reject': 0, 'error': 0}
        accepted_correct = 0
        escalated_correct = 0

        for result in self.results:
            routing = result['routing_decision']
            routing_counts[routing] = routing_counts.get(routing, 0) + 1

            if result.get('error'):
                continue

            is_correct = result['is_correct']

            if routing == 'accept' and is_correct:
                accepted_correct += 1
            elif routing == 'escalate' and is_correct:
                escalated_correct += 1

        total = sum(routing_counts.values())

        # Calculate rates
        automation_rate = routing_counts['accept'] / total if total > 0 else 0.0
        escalation_rate = routing_counts['escalate'] / total if total > 0 else 0.0
        rejection_rate = routing_counts['reject'] / total if total > 0 else 0.0

        # Calculate accuracies
        accepted_accuracy = (
            accepted_correct / routing_counts['accept']
            if routing_counts['accept'] > 0 else 0.0
        )
        escalated_accuracy = (
            escalated_correct / routing_counts['escalate']
            if routing_counts['escalate'] > 0 else 0.0
        )

        # Routing accuracy: correct decisions on accepted samples
        routing_accuracy = accepted_accuracy

        return RoutingMetrics(
            automation_rate=automation_rate,
            escalation_rate=escalation_rate,
            rejection_rate=rejection_rate,
            routing_accuracy=routing_accuracy,
            accepted_accuracy=accepted_accuracy,
            escalated_accuracy=escalated_accuracy,
            routing_distribution=routing_counts
        )

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate high-level summary"""
        total_samples = len(self.results)
        errors = sum(1 for r in self.results if r.get('error'))
        successful = total_samples - errors

        # Handle both baseline and ensemble result formats
        confidences = []
        for r in self.results:
            if r.get('error'):
                continue
            if 'prediction' in r:
                confidences.append(r['prediction'].get('confidence_score', 0.0))
            elif 'final_prediction' in r:
                confidences.append(r['final_prediction'].get('confidence_score', 0.0))

        avg_confidence = np.mean(confidences) if confidences else 0.0

        return {
            'total_samples': total_samples,
            'successful': successful,
            'errors': errors,
            'avg_confidence': float(avg_confidence)
        }

    def print_metrics(self, metrics: Dict[str, Any]):
        """Print formatted metrics report"""
        print("\n" + "="*60)
        print("IDS UNCERTAINTY ESTIMATION - METRICS REPORT")
        print("="*60)

        # Summary
        print("\nSUMMARY")
        print("-" * 60)
        summary = metrics['summary']
        print(f"Total samples: {summary['total_samples']}")
        print(f"Successful: {summary['successful']}")
        print(f"Errors: {summary['errors']}")
        print(f"Average confidence: {summary['avg_confidence']:.3f}")

        # Classification
        print("\nCLASSIFICATION METRICS")
        print("-" * 60)
        cm = metrics['classification']
        print(f"Accuracy: {cm.accuracy:.3f}")
        print(f"Precision: {cm.precision:.3f}")
        print(f"Recall: {cm.recall:.3f}")
        print(f"F1 Score: {cm.f1_score:.3f}")
        print(f"False Positive Rate: {cm.false_positive_rate:.3f}")
        print(f"False Negative Rate: {cm.false_negative_rate:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {cm.true_positives}, TN: {cm.true_negatives}")
        print(f"  FP: {cm.false_positives}, FN: {cm.false_negatives}")

        # Calibration
        print("\nCALIBRATION METRICS")
        print("-" * 60)
        cal = metrics['calibration']
        print(f"Expected Calibration Error (ECE): {cal.expected_calibration_error:.3f}")
        print(f"Maximum Calibration Error (MCE): {cal.maximum_calibration_error:.3f}")
        print(f"Brier Score: {cal.brier_score:.3f}")

        # Correlation
        print("\nCORRELATION METRICS")
        print("-" * 60)
        corr = metrics['correlation']
        print(f"Confidence-Accuracy Correlation: {corr.confidence_accuracy_correlation:.3f}")
        print(f"Correlation p-value: {corr.correlation_p_value:.4f}")
        print("\nSelective Risk (Accuracy at Coverage):")
        for coverage, acc in corr.selective_risk.items():
            print(f"  {coverage}: {acc:.3f}")

        # Routing
        print("\nROUTING METRICS")
        print("-" * 60)
        rt = metrics['routing']
        print(f"Automation Rate: {rt.automation_rate:.1%}")
        print(f"Escalation Rate: {rt.escalation_rate:.1%}")
        print(f"Rejection Rate: {rt.rejection_rate:.1%}")
        print(f"Routing Accuracy: {rt.routing_accuracy:.3f}")
        print(f"Accepted Samples Accuracy: {rt.accepted_accuracy:.3f}")
        print(f"Escalated Samples Accuracy: {rt.escalated_accuracy:.3f}")
        print(f"\nRouting Distribution: {rt.routing_distribution}")

        print("\n" + "="*60)


def main():
    """Main entry point for metrics calculation"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Calculate metrics from baseline results")
    parser.add_argument("results_file", help="Path to JSONL results file")
    parser.add_argument("--output", help="Output JSON file for metrics")
    parser.add_argument("--bins", type=int, default=10, help="Number of bins for calibration")

    args = parser.parse_args()

    # Calculate metrics
    calculator = MetricsCalculator(args.results_file, n_bins=args.bins)

    # CRITICAL: Validate minimum sample size for statistical analysis
    # Count valid (non-error) samples
    valid_samples = [r for r in calculator.results if not r.get('error')]

    # EARLY FAILURE: Exit immediately if insufficient valid data
    if len(valid_samples) < 2:
        print(f"\n❌ EXPERIMENT FAILED: Only {len(valid_samples)}/{len(calculator.results)} valid samples")
        print(f"   Minimum 2 valid samples required for statistical analysis")
        print(f"   This indicates API failures or data pipeline issues")

        # Log failure reason for debugging
        error_count = sum(1 for r in calculator.results if r.get('error'))
        print(f"   Error samples: {error_count}")

        # Exit with error code to signal orchestrator failure
        sys.exit(1)

    metrics = calculator.calculate_all_metrics()

    # Print report
    calculator.print_metrics(metrics)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dicts for JSON serialization
        metrics_dict = {
            'classification': metrics['classification'].__dict__,
            'calibration': metrics['calibration'].__dict__,
            'correlation': metrics['correlation'].__dict__,
            'routing': metrics['routing'].__dict__,
            'summary': metrics['summary']
        }

        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
