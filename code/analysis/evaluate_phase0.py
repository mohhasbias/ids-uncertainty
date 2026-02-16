#!/usr/bin/env python3
"""
Phase 0 Evaluation Script

Evaluates Phase 0 experiment results against success criteria and provides
a research-appropriate go/no-go decision for Phase 1.

Phase 0 Success Criteria:
- Baseline accuracy > 70% (demonstrates task is learnable)
- ECE < 0.35 for baseline (relaxed threshold for small sample)
- Infrastructure stability (no critical failures)

Research Goal: At least 1-2 models pass all criteria for GO decision.
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from metrics_calculator import MetricsCalculator


@dataclass
class Phase0Criteria:
    """Phase 0 success criteria thresholds."""

    baseline_accuracy_threshold: float = 0.70
    baseline_ece_threshold: float = 0.35
    infrastructure_stability: float = 0.99  # <1% failures
    min_passing_models: int = 1  # Research: need 1-2 models


@dataclass
class CriteriaResult:
    """Result of checking a single criterion."""

    criterion: str
    passed: bool
    value: float
    threshold: float
    message: str


@dataclass
class ExperimentResult:
    """Result of evaluating a single experiment."""

    model: str
    dataset: str
    method: str
    run_dir: Path
    sample_count: int
    error_count: int
    criteria_results: Dict[str, CriteriaResult]
    passes_all_criteria: bool
    status: str  # "COMPLETE", "IN_PROGRESS", "INFRASTRUCTURE_ISSUE"


class ResultScanner:
    """Scans scenario_results/ to find Phase 0 experiments."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.scenario_results_dir = base_dir / "scenario_results"

    def find_phase0_results(self, method: str = "baseline") -> List[Tuple[Path, str, str]]:
        """
        Find all Phase 0 experiment results (50-sample runs).

        Returns:
            List of (run_dir, model_name, dataset_name) tuples
        """
        results = []
        method_dir = self.scenario_results_dir / method

        if not method_dir.exists():
            return results

        # Scan all run directories
        for run_dir in method_dir.glob("run_*"):
            result_file = run_dir / f"{method}_results.jsonl"

            if not result_file.exists():
                continue

            # Extract model and dataset from results
            model, dataset, sample_count, error_count = self._analyze_result_file(result_file)

            if model and dataset:
                # Phase 0 marker: exactly 50 samples (including errors)
                total_samples = sample_count + error_count
                if total_samples == 50:
                    results.append((run_dir, model, dataset, sample_count, error_count))

        # Deduplicate - prefer complete (0-error) runs, then latest timestamp
        latest_runs = {}
        for run_dir, model, dataset, sample_count, error_count in results:
            # Extract timestamp from run_dir name: run_YYYYMMDD_HHMMSS
            # Example: run_20260122_011527 -> "20260122_011527"
            timestamp = run_dir.name.replace("run_", "", 1) if run_dir.name.startswith("run_") else ""

            key = (model, dataset)

            existing = latest_runs.get(key)
            if existing is None:
                latest_runs[key] = (run_dir, timestamp, sample_count, error_count)
            else:
                existing_has_errors = existing[3] > 0
                new_has_errors = error_count > 0
                if existing_has_errors and not new_has_errors:
                    # Prefer complete run over run with errors
                    latest_runs[key] = (run_dir, timestamp, sample_count, error_count)
                elif existing_has_errors == new_has_errors and timestamp > existing[1]:
                    # Same error status — prefer latest timestamp
                    latest_runs[key] = (run_dir, timestamp, sample_count, error_count)

        # Return deduplicated list (remove timestamp from tuple)
        deduplicated_results = [
            (run_dir, model, dataset, sample_count, error_count)
            for (model, dataset), (run_dir, timestamp, sample_count, error_count) in latest_runs.items()
        ]

        return deduplicated_results

    def _analyze_result_file(self, result_file: Path) -> Tuple[Optional[str], Optional[str], int, int]:
        """
        Analyze a result file to extract model, dataset, and sample counts.

        Counts unique sample_ids to handle retry entries correctly.
        A sample is "successful" if ANY entry for that sample_id has no error.

        Returns:
            (model_name, dataset_name, successful_sample_count, error_count)
        """
        model_name = None
        dataset_name = None
        # Track per-sample success: sample_id -> bool (True if any entry succeeded)
        sample_success = {}

        try:
            with open(result_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())

                        # Extract model from first entry
                        if model_name is None and "model_name" in entry:
                            model_name = self._normalize_model_name(entry["model_name"])

                        # Extract dataset from sample_id
                        if dataset_name is None and "sample_id" in entry:
                            dataset_name = self._extract_dataset_from_sample_id(entry["sample_id"])

                        # Track success per unique sample_id
                        sample_id = entry.get("sample_id")
                        if sample_id is not None:
                            is_success = not entry.get("error")
                            # A sample is successful if ANY entry for it succeeded
                            sample_success[sample_id] = sample_success.get(sample_id, False) or is_success

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            warnings.warn(f"Error analyzing {result_file}: {e}")

        sample_count = sum(1 for success in sample_success.values() if success)
        error_count = sum(1 for success in sample_success.values() if not success)

        return model_name, dataset_name, sample_count, error_count

    def _extract_dataset_from_sample_id(self, sample_id: str) -> str:
        """
        Extract dataset name from sample_id.

        Examples:
            cicids2018_1 → cicids2018
            cicids_1 → cicids
            hard_1 → hard
            "1" → hard (backward compatibility)
        """
        if isinstance(sample_id, str):
            # Split on underscore and take all but last part
            parts = sample_id.rsplit("_", 1)
            if len(parts) == 2:
                dataset = parts[0]
                # Map to config.yaml dataset keys
                if dataset == "cicids":
                    return "cicids"
                elif dataset == "cicids2018":
                    return "cicids2018"
                else:
                    return dataset
            else:
                # No underscore, assume hard mock data
                return "hard"
        else:
            # Numeric sample_id, backward compatibility
            return "hard"

    def _normalize_model_name(self, model_name: str) -> str:
        """
        Normalize model names to match config.yaml conventions.

        Handles legacy names (e.g., nvidia-kimi → groq-kimi).
        """
        # Legacy model name mappings
        legacy_map = {
            "nvidia-kimi": "groq-kimi",
            "groq-llama4-scout": "nvidia-llama4-scout",
            "groq-llama-70b": "nvidia-llama-70b",
            "groq-qwen": "qwen3-next-80b",
        }

        return legacy_map.get(model_name, model_name)


class MetricsAggregator:
    """Loads or calculates metrics for experiments."""

    def __init__(self, calculate_missing: bool = False, n_bins: int = 10):
        self.calculate_missing = calculate_missing
        self.n_bins = n_bins

    def get_metrics(self, run_dir: Path, method: str) -> Optional[Dict]:
        """
        Get metrics for an experiment run.

        Args:
            run_dir: Path to run directory
            method: Experiment method (e.g., "baseline")

        Returns:
            Metrics dict or None if unavailable
        """
        metrics_path = run_dir / "metrics.json"
        result_file = run_dir / f"{method}_results.jsonl"

        # Check for existing metrics
        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                warnings.warn(f"Error loading {metrics_path}: {e}")

        # Calculate if requested and result file exists
        if self.calculate_missing and result_file.exists():
            try:
                return self._calculate_metrics(result_file)
            except Exception as e:
                warnings.warn(f"Error calculating metrics for {result_file}: {e}")
                return None

        return None

    def _calculate_metrics(self, result_file: Path) -> Dict:
        """
        Calculate metrics using MetricsCalculator.

        Returns:
            Metrics dict with classification, calibration, correlation, routing
        """
        calculator = MetricsCalculator(str(result_file), n_bins=self.n_bins)
        metrics = calculator.calculate_all_metrics()

        # Convert dataclasses to dict for JSON serialization
        metrics_dict = {
            "classification": asdict(metrics["classification"]),
            "calibration": asdict(metrics["calibration"]),
            "correlation": asdict(metrics["correlation"]),
            "routing": asdict(metrics["routing"]),
            "summary": metrics.get("summary", {}),
        }

        return metrics_dict


class CriteriaChecker:
    """Evaluates experiments against Phase 0 success criteria."""

    def __init__(self, criteria: Phase0Criteria):
        self.criteria = criteria

    def evaluate_experiment(
        self, model: str, dataset: str, method: str, run_dir: Path, sample_count: int, error_count: int, metrics: Optional[Dict]
    ) -> ExperimentResult:
        """
        Evaluate a single experiment against Phase 0 criteria.

        Returns:
            ExperimentResult with detailed evaluation
        """
        total_samples = sample_count + error_count
        error_rate = error_count / total_samples if total_samples > 0 else 0

        # Determine experiment status
        if error_rate > 0.05:  # >5% failure rate
            status = "INFRASTRUCTURE_ISSUE"
        elif sample_count < 50:
            status = "IN_PROGRESS"
        else:
            status = "COMPLETE"

        criteria_results = {}
        passes_all = False

        # Only evaluate criteria if metrics available and status is COMPLETE
        if metrics and status == "COMPLETE":
            criteria_results = {
                "accuracy": self._check_accuracy(metrics),
                "ece": self._check_ece(metrics),
                "infrastructure": self._check_infrastructure(sample_count, error_count),
            }
            passes_all = all(r.passed for r in criteria_results.values())
        elif status == "INFRASTRUCTURE_ISSUE":
            # Add infrastructure failure criterion
            criteria_results["infrastructure"] = self._check_infrastructure(sample_count, error_count)

        return ExperimentResult(
            model=model,
            dataset=dataset,
            method=method,
            run_dir=run_dir,
            sample_count=sample_count,
            error_count=error_count,
            criteria_results=criteria_results,
            passes_all_criteria=passes_all,
            status=status,
        )

    def _check_accuracy(self, metrics: Dict) -> CriteriaResult:
        """Check baseline accuracy criterion."""
        accuracy = metrics["classification"]["accuracy"]
        threshold = self.criteria.baseline_accuracy_threshold

        return CriteriaResult(
            criterion="Baseline Accuracy",
            passed=accuracy >= threshold,
            value=accuracy,
            threshold=threshold,
            message=f"{accuracy:.1%} (threshold: >{threshold:.0%})",
        )

    def _check_ece(self, metrics: Dict) -> CriteriaResult:
        """Check ECE criterion."""
        # Handle both key formats: "ece" and "expected_calibration_error"
        calibration = metrics.get("calibration", {})
        ece = calibration.get("ece") or calibration.get("expected_calibration_error", 0.0)
        threshold = self.criteria.baseline_ece_threshold

        return CriteriaResult(
            criterion="ECE",
            passed=ece < threshold,
            value=ece,
            threshold=threshold,
            message=f"{ece:.3f} (threshold: <{threshold:.2f})",
        )

    def _check_infrastructure(self, sample_count: int, error_count: int) -> CriteriaResult:
        """Check infrastructure stability criterion."""
        total = sample_count + error_count
        error_rate = error_count / total if total > 0 else 0
        success_rate = 1 - error_rate
        threshold = self.criteria.infrastructure_stability

        return CriteriaResult(
            criterion="Infrastructure Stability",
            passed=success_rate >= threshold,
            value=success_rate,
            threshold=threshold,
            message=f"{sample_count}/{total} samples, {error_count} errors ({error_rate:.1%} failure rate)",
        )

    def determine_go_no_go(self, experiments: List[ExperimentResult]) -> Tuple[bool, str, List[str], List[str]]:
        """
        Determine go/no-go decision based on experiment results.

        Research Goal: At least 1-2 models pass all criteria.

        Returns:
            (is_go, reason, passing_models, failing_models)
        """
        # Group experiments by model
        model_performance = defaultdict(list)
        for exp in experiments:
            if exp.status == "COMPLETE":  # Only consider complete experiments
                model_performance[exp.model].append(exp.passes_all_criteria)

        # Model passes if it passes criteria on at least 1 dataset
        passing_models = [model for model, results in model_performance.items() if any(results)]

        failing_models = [model for model, results in model_performance.items() if not any(results)]

        num_passing = len(passing_models)

        if num_passing >= self.criteria.min_passing_models:
            return (
                True,
                f"{num_passing} model(s) pass all criteria ({', '.join(passing_models)})",
                passing_models,
                failing_models,
            )
        else:
            return (
                False,
                f"Only {num_passing} model(s) pass criteria (need {self.criteria.min_passing_models})",
                passing_models,
                failing_models,
            )


def _build_dataset_mapping(config: Dict) -> Dict[str, str]:
    """
    Build tracker_key → display_name mapping from config.yaml.

    Mirrors progress_tracker.py's transformation logic.
    Only includes datasets that have a display name configured (production datasets).

    Returns:
        Dict mapping tracker keys (cli_param) to display names
        Example: {"cicids": "CIC-IDS2017 (50 Samples)", "cicids2018": "CIC-IDS2018"}
    """
    dataset_mapping = {}

    # Get display name mappings (display_name → config_key)
    display_names = config.get('display_names', {})
    dataset_display_names = display_names.get('datasets', {})

    # Iterate through datasets to build reverse mapping
    for dataset_key, dataset_config in config.get('datasets', {}).items():
        # Get tracker key (cli_param)
        tracker_key = dataset_config.get('cli_param', dataset_key)

        # Find display name by reverse lookup
        display_name = None
        for disp_name, disp_key in dataset_display_names.items():
            if disp_key == dataset_key:
                display_name = disp_name
                break

        # Only include datasets that have a display name (production datasets)
        if display_name:
            dataset_mapping[tracker_key] = display_name

    return dataset_mapping


def _build_model_mapping(config: Dict) -> Dict[str, str]:
    """
    Build internal_model_key → display_name mapping from config.yaml.

    Returns:
        Dict mapping internal keys to display names
        Example: {"nvidia-llama4-scout": "Llama4S", "groq-kimi": "Kimi"}
    """
    model_mapping = {}

    # Get display name mappings (display_name → internal_key)
    display_names = config.get('display_names', {})
    model_display_names = display_names.get('models', {})

    # Reverse the mapping: internal_key → display_name
    for disp_name, internal_key in model_display_names.items():
        model_mapping[internal_key] = disp_name

    return model_mapping


class ReportGenerator:
    """Generates evaluation reports in various formats."""

    def __init__(self, config: Dict):
        self.config = config
        self.display_names = config.get("display_names", {})
        self.dataset_mapping = _build_dataset_mapping(config)
        self.model_mapping = _build_model_mapping(config)

    def generate_console_report(
        self, experiments: List[ExperimentResult], is_go: bool, reason: str, passing_models: List[str], failing_models: List[str]
    ) -> str:
        """Generate human-readable console report."""
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("PHASE 0 EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Overall status
        status_icon = "✅" if is_go else "❌"
        status_text = "GO" if is_go else "NO-GO"
        lines.append(f"Overall Status: {status_icon} {status_text}")
        # Transform model names in reason string
        reason_display = self._transform_model_names_in_text(reason)
        lines.append(f"Reason: {reason_display}")
        lines.append("")

        # Summary statistics
        complete = sum(1 for e in experiments if e.status == "COMPLETE")
        in_progress = sum(1 for e in experiments if e.status == "IN_PROGRESS")
        infra_issues = sum(1 for e in experiments if e.status == "INFRASTRUCTURE_ISSUE")

        lines.append(f"Experiments Analyzed: {complete} complete, {in_progress} in progress, {infra_issues} infrastructure issues")
        lines.append(f"Total Experiments: {len(experiments)}")
        lines.append("")

        # Results by model/dataset
        lines.append("=== Results by Model/Dataset ===")
        lines.append("")

        for exp in sorted(experiments, key=lambda e: (self._get_display_name("models", e.model), self._get_display_name("datasets", e.dataset))):
            model_display = self._get_display_name("models", exp.model)
            dataset_display = self._get_display_name("datasets", exp.dataset)

            # Status line
            if exp.status == "COMPLETE" and exp.passes_all_criteria:
                status_line = f"{model_display} + {dataset_display}: ✅ PASS ALL CRITERIA"
            elif exp.status == "COMPLETE":
                status_line = f"{model_display} + {dataset_display}: ❌ FAIL CRITERIA"
            elif exp.status == "IN_PROGRESS":
                status_line = f"{model_display} + {dataset_display}: 🔄 IN PROGRESS"
            else:
                status_line = f"{model_display} + {dataset_display}: ⚠️  INFRASTRUCTURE ISSUE"

            lines.append(status_line)

            # Criteria details
            for criterion_name, result in exp.criteria_results.items():
                icon = "✅" if result.passed else "❌"
                lines.append(f"  {icon} {result.criterion}: {result.message}")

            lines.append("")

        # Summary
        lines.append("=== Summary ===")
        lines.append("")

        if passing_models:
            lines.append(f"Models Passing (all criteria): {len(passing_models)}")
            for model in passing_models:
                model_display = self._get_display_name("models", model)
                # Count datasets passed
                datasets_passed = sum(
                    1 for e in experiments if e.model == model and e.passes_all_criteria and e.status == "COMPLETE"
                )
                datasets_total = sum(1 for e in experiments if e.model == model and e.status == "COMPLETE")
                lines.append(f"  ✅ {model_display} ({datasets_passed}/{datasets_total} datasets pass)")
            lines.append("")

        if failing_models:
            lines.append(f"Models Failing: {len(failing_models)}")
            failing_display = [self._get_display_name("models", m) for m in failing_models]
            lines.append(f"  ❌ {', '.join(failing_display)}")
            lines.append("")

        # Recommendation
        lines.append("=== Recommendation ===")
        lines.append("")
        if is_go:
            lines.append("✅ PROCEED TO PHASE 1")
            lines.append("")
            lines.append("Next Steps:")
            lines.append("1. Scale up to 300 samples per dataset")
            lines.append("2. Run baseline + ensemble methods")
            lines.append("3. Validate Phase 1 success criteria (ECE < 0.25)")
        else:
            lines.append("❌ DO NOT PROCEED TO PHASE 1")
            lines.append("")
            lines.append("Required Actions:")
            lines.append("1. Investigate failing models")
            lines.append("2. Check prompt engineering and configuration")
            lines.append("3. Re-run Phase 0 with improvements")

        return "\n".join(lines)

    def generate_markdown_report(
        self, experiments: List[ExperimentResult], is_go: bool, reason: str, passing_models: List[str], failing_models: List[str]
    ) -> str:
        """Generate detailed markdown report."""
        lines = []

        # Header
        lines.append("# Phase 0 Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {self._get_timestamp()}")
        lines.append("")

        # Overall status
        status_icon = "✅" if is_go else "❌"
        status_text = "GO" if is_go else "NO-GO"
        lines.append(f"## Overall Status: {status_icon} {status_text}")
        lines.append("")
        # Transform model names in reason string
        reason_display = self._transform_model_names_in_text(reason)
        lines.append(f"**Reason:** {reason_display}")
        lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric                  | Value |")
        lines.append("|:------------------------|------:|")

        complete = sum(1 for e in experiments if e.status == "COMPLETE")
        in_progress = sum(1 for e in experiments if e.status == "IN_PROGRESS")
        infra_issues = sum(1 for e in experiments if e.status == "INFRASTRUCTURE_ISSUE")

        lines.append(f"| Total Experiments       | {len(experiments):>5} |")
        lines.append(f"| Complete                | {complete:>5} |")
        lines.append(f"| In Progress             | {in_progress:>5} |")
        lines.append(f"| Infrastructure Issues   | {infra_issues:>5} |")
        lines.append(f"| Passing Models          | {len(passing_models):>5} |")
        lines.append(f"| Failing Models          | {len(failing_models):>5} |")
        lines.append("")

        # Detailed results table
        lines.append("## Detailed Results")
        lines.append("")
        lines.append("| Model               | Dataset    | Status            | Accuracy | ECE   | Samples | Errors |")
        lines.append("|:--------------------|:-----------|:------------------|:--------:|:-----:|--------:|-------:|")

        for exp in sorted(experiments, key=lambda e: (self._get_display_name("models", e.model), self._get_display_name("datasets", e.dataset))):
            model_display = self._get_display_name("models", exp.model)
            dataset_display = self._get_display_name("datasets", exp.dataset)

            if exp.status == "COMPLETE":
                acc_result = exp.criteria_results.get("accuracy")
                ece_result = exp.criteria_results.get("ece")
                accuracy = f"{acc_result.value:.1%}" if acc_result else "N/A"
                ece = f"{ece_result.value:.3f}" if ece_result else "N/A"

                if exp.passes_all_criteria:
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
            elif exp.status == "IN_PROGRESS":
                status = "🔄 IN PROGRESS"
                accuracy = "N/A"
                ece = "N/A"
            else:
                status = "⚠️  INFRA ISSUE"
                accuracy = "N/A"
                ece = "N/A"

            lines.append(
                f"| {model_display:<19} | {dataset_display:<10} | {status:<17} | {accuracy:>8} | {ece:>5} | {exp.sample_count:>7} | {exp.error_count:>6} |"
            )

        lines.append("")

        # Passing models
        if passing_models:
            lines.append("## Passing Models")
            lines.append("")
            for model in passing_models:
                model_display = self._get_display_name("models", model)
                lines.append(f"### {model_display}")
                lines.append("")

                # Find experiments for this model
                model_exps = [e for e in experiments if e.model == model and e.status == "COMPLETE"]

                for exp in model_exps:
                    dataset_display = self._get_display_name("datasets", exp.dataset)
                    if exp.passes_all_criteria:
                        lines.append(f"**{dataset_display}:** ✅ PASS")
                    else:
                        lines.append(f"**{dataset_display}:** ❌ FAIL")

                    for criterion_name, result in exp.criteria_results.items():
                        icon = "✅" if result.passed else "❌"
                        lines.append(f"- {icon} {result.criterion}: {result.message}")

                    lines.append("")

        # Failing models
        if failing_models:
            lines.append("## Failing Models")
            lines.append("")
            for model in failing_models:
                model_display = self._get_display_name("models", model)
                lines.append(f"### {model_display}")
                lines.append("")

                # Find experiments for this model
                model_exps = [e for e in experiments if e.model == model and e.status == "COMPLETE"]

                for exp in model_exps:
                    dataset_display = self._get_display_name("datasets", exp.dataset)
                    lines.append(f"**{dataset_display}:** ❌ FAIL")

                    for criterion_name, result in exp.criteria_results.items():
                        icon = "✅" if result.passed else "❌"
                        lines.append(f"- {icon} {result.criterion}: {result.message}")

                    lines.append("")

        # Recommendation
        lines.append("## Recommendation")
        lines.append("")
        if is_go:
            lines.append("### ✅ PROCEED TO PHASE 1")
            lines.append("")
            lines.append("**Next Steps:**")
            lines.append("1. Scale up to 300 samples per dataset")
            lines.append("2. Run baseline + ensemble methods")
            lines.append("3. Validate Phase 1 success criteria:")
            lines.append("   - ECE < 0.25 for baseline")
            lines.append("   - Ensemble shows ≥5% ECE improvement")
            lines.append("   - Confidence-accuracy correlation ≥ 0.2")
        else:
            lines.append("### ❌ DO NOT PROCEED TO PHASE 1")
            lines.append("")
            lines.append("**Required Actions:**")
            lines.append("1. Investigate failing models")
            lines.append("2. Review prompt engineering and configuration")
            lines.append("3. Check dataset quality and preprocessing")
            lines.append("4. Re-run Phase 0 with improvements")

        return "\n".join(lines)

    def generate_json_report(
        self, experiments: List[ExperimentResult], is_go: bool, reason: str, passing_models: List[str], failing_models: List[str]
    ) -> Dict:
        """Generate JSON report for automation."""
        return {
            "overall_status": "GO" if is_go else "NO_GO",
            "reason": reason,
            "passing_models": passing_models,
            "failing_models": failing_models,
            "summary": {
                "total_experiments": len(experiments),
                "complete": sum(1 for e in experiments if e.status == "COMPLETE"),
                "in_progress": sum(1 for e in experiments if e.status == "IN_PROGRESS"),
                "infrastructure_issues": sum(1 for e in experiments if e.status == "INFRASTRUCTURE_ISSUE"),
                "passing_models_count": len(passing_models),
                "failing_models_count": len(failing_models),
            },
            "experiments": [
                {
                    "model": exp.model,
                    "dataset": exp.dataset,
                    "method": exp.method,
                    "status": exp.status,
                    "sample_count": exp.sample_count,
                    "error_count": exp.error_count,
                    "passes_all_criteria": exp.passes_all_criteria,
                    "criteria_results": {name: asdict(result) for name, result in exp.criteria_results.items()},
                }
                for exp in experiments
            ],
        }

    def _get_display_name(self, category: str, key: str) -> str:
        """Get display name from config.yaml."""
        if category == "datasets":
            # Use pre-built tracker_key → display_name mapping
            return self.dataset_mapping.get(key, key)
        elif category == "models":
            # Use pre-built internal_key → display_name mapping
            return self.model_mapping.get(key, key)
        else:
            # Fallback for other categories
            display_map = self.display_names.get(category, {})
            return display_map.get(key, key)

    def _transform_model_names_in_text(self, text: str) -> str:
        """Transform internal model keys to display names in text."""
        result = text
        for internal_key, display_name in self.model_mapping.items():
            result = result.replace(internal_key, display_name)
        return result

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_config() -> Dict:
    """Load config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 0 experiment results against success criteria",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (console output)
  uv run python evaluate_phase0.py

  # Generate markdown report
  uv run python evaluate_phase0.py --output markdown

  # Auto-calculate missing metrics
  uv run python evaluate_phase0.py --calculate-metrics

  # Custom criteria thresholds
  uv run python evaluate_phase0.py --criteria-accuracy 0.80 --criteria-ece 0.30

  # Save JSON for automation
  uv run python evaluate_phase0.py --output json > phase0_status.json
        """,
    )

    parser.add_argument(
        "--method", default="baseline", help="Method to evaluate (default: baseline)"
    )

    parser.add_argument(
        "--output",
        choices=["console", "markdown", "json"],
        default="console",
        help="Output format (default: console)",
    )

    parser.add_argument(
        "--calculate-metrics",
        action="store_true",
        help="Auto-calculate missing metrics",
    )

    parser.add_argument(
        "--criteria-accuracy",
        type=float,
        default=0.70,
        help="Baseline accuracy threshold (default: 0.70)",
    )

    parser.add_argument(
        "--criteria-ece", type=float, default=0.35, help="ECE threshold (default: 0.35)"
    )

    parser.add_argument(
        "--min-passing-models",
        type=int,
        default=1,
        help="Minimum models for GO decision (default: 1)",
    )

    parser.add_argument("--verbose", action="store_true", help="Show detailed diagnostics")

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Initialize components
    base_dir = Path(__file__).parent
    scanner = ResultScanner(base_dir)
    aggregator = MetricsAggregator(calculate_missing=args.calculate_metrics)
    criteria = Phase0Criteria(
        baseline_accuracy_threshold=args.criteria_accuracy,
        baseline_ece_threshold=args.criteria_ece,
        min_passing_models=args.min_passing_models,
    )
    checker = CriteriaChecker(criteria)
    reporter = ReportGenerator(config)

    # Find Phase 0 results
    if args.verbose:
        print("Scanning for Phase 0 experiments...")

    phase0_results = scanner.find_phase0_results(method=args.method)

    if not phase0_results:
        print(f"No Phase 0 experiments found for method: {args.method}")
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(phase0_results)} unique Phase 0 experiments (deduplicated by latest run per model+dataset)")

    # Filter to only production datasets (those with display names)
    dataset_mapping = _build_dataset_mapping(config)
    production_datasets = set(dataset_mapping.keys())

    phase0_results_filtered = [
        (run_dir, model, dataset, sample_count, error_count)
        for run_dir, model, dataset, sample_count, error_count in phase0_results
        if dataset in production_datasets
    ]

    if args.verbose:
        excluded_count = len(phase0_results) - len(phase0_results_filtered)
        if excluded_count > 0:
            print(f"Filtered out {excluded_count} experiments for non-production datasets")
            excluded_datasets = set(
                dataset
                for _, _, dataset, _, _ in phase0_results
                if dataset not in production_datasets
            )
            print(f"Excluded datasets: {', '.join(sorted(excluded_datasets))}")

    phase0_results = phase0_results_filtered

    if not phase0_results:
        print("No Phase 0 experiments found for production datasets")
        sys.exit(1)

    # Evaluate experiments
    experiments = []
    missing_metrics_count = 0

    for run_dir, model, dataset, sample_count, error_count in phase0_results:
        metrics = aggregator.get_metrics(run_dir, args.method)

        if metrics is None and sample_count >= 50:
            missing_metrics_count += 1

        experiment = checker.evaluate_experiment(model, dataset, args.method, run_dir, sample_count, error_count, metrics)
        experiments.append(experiment)

    # Warn about missing metrics
    if missing_metrics_count > 0 and not args.calculate_metrics:
        warnings.warn(
            f"{missing_metrics_count}/{len(phase0_results)} experiments missing metrics.json\n"
            "Use --calculate-metrics to auto-calculate"
        )

    # Determine go/no-go
    is_go, reason, passing_models, failing_models = checker.determine_go_no_go(experiments)

    # Generate report
    if args.output == "console":
        report = reporter.generate_console_report(experiments, is_go, reason, passing_models, failing_models)
        print(report)

    elif args.output == "markdown":
        report = reporter.generate_markdown_report(experiments, is_go, reason, passing_models, failing_models)
        output_path = base_dir / "PHASE_0_EVALUATION.md"
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Markdown report written to: {output_path}")

    elif args.output == "json":
        report = reporter.generate_json_report(experiments, is_go, reason, passing_models, failing_models)
        print(json.dumps(report, indent=2))

    # Exit with appropriate code
    sys.exit(0 if is_go else 1)


if __name__ == "__main__":
    main()
