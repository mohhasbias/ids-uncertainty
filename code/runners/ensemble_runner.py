#!/usr/bin/env python3
"""
Ensemble Voting Uncertainty Method Runner
Method 2: Ensemble with Prompt Variants

This method creates K different agents with variant prompts (different framings
of the IDS task) and measures disagreement as an uncertainty signal. Combines
self-reported confidence with ensemble agreement for improved calibration.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import Counter

from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.openai import OpenAIChat
from agno.models.nvidia import Nvidia

from ids_models import (
    IDSStructuredResponse,
    GroundTruth,
    UncertaintyMetrics,
    AttackType,
    ThreatLevel
)
from dataset_loader import IDSDatasetLoader, IDSSample
from retry_utils import retry_with_intelligent_backoff, RateLimitError
from config_loader import ConfigLoader


@dataclass
class EnsembleTestResult:
    """Single test result for ensemble method"""
    sample_id: str
    model_name: str
    iteration: int
    timestamp: str

    # Input
    description: str
    ground_truth: Dict[str, Any]

    # Individual predictions from K variants
    variant_predictions: List[Dict[str, Any]]

    # Aggregated prediction
    final_prediction: Dict[str, Any]

    # Evaluation
    is_correct: bool
    uncertainty_metrics: Dict[str, Any]
    routing_decision: str

    # Ensemble-specific metrics
    agreement_score: float
    prediction_variance: float

    # Metadata
    execution_time: float
    error: Optional[str] = None


class EnsembleRunner:
    """
    Runner for ensemble-based uncertainty estimation
    Uses K different prompt variants to measure disagreement
    """

    # Define three distinct prompt variants
    PROMPT_VARIANTS = {
        "network_security": {
            "role": "network security analyst",
            "instructions": [
                "You are an expert network security analyst specializing in traffic analysis and intrusion detection.",
                "Your task is to analyze network flow descriptions and classify them as benign or malicious.",
                "Focus on network-level indicators:",
                "- Protocol analysis (TCP/UDP/ICMP patterns)",
                "- Traffic volume and packet characteristics",
                "- Port usage and connection patterns",
                "- Network timing and flow statistics",
                "",
                "Provide a structured response with:",
                "1. Classification decision (malicious/benign, threat level, attack type)",
                "2. Confidence score (0.0 to 1.0) based on clarity of network signatures",
                "3. Clear reasoning focusing on network-level features",
                "4. Key network indicators that influenced your decision",
                "",
                "Be conservative with confidence when:",
                "- Traffic patterns are ambiguous or could be normal spikes",
                "- Protocol usage is unusual but not definitively malicious",
                "- Volume-based indicators could have legitimate causes",
                "",
                "Higher confidence when:",
                "- Clear attack signatures are present (port scans, SYN floods)",
                "- Protocol violations or anomalies are detected",
                "- Traffic patterns match known attack fingerprints"
            ]
        },
        "soc_analyst": {
            "role": "SOC analyst",
            "instructions": [
                "You are a Security Operations Center (SOC) analyst responsible for triaging security alerts.",
                "Your task is to investigate network activity and determine if incidents require escalation.",
                "Focus on operational security priorities:",
                "- Threat actor behavior and tactics",
                "- Alert severity and business impact",
                "- False positive likelihood based on context",
                "- Incident response priorities",
                "",
                "Provide a structured response with:",
                "1. Classification decision (malicious/benign, threat level, attack type)",
                "2. Confidence score (0.0 to 1.0) reflecting investigation certainty",
                "3. Clear reasoning from an incident response perspective",
                "4. Key indicators of compromise (IOCs) or benign indicators",
                "",
                "Be cautious with confidence when:",
                "- Activity could be legitimate security testing or maintenance",
                "- Insufficient context about business operations",
                "- Mixed signals requiring additional investigation",
                "",
                "Higher confidence when:",
                "- Clear indicators of compromise are present",
                "- Attack patterns match known threat intelligence",
                "- Activity has no plausible legitimate explanation"
            ]
        },
        "forensics": {
            "role": "digital forensics investigator",
            "instructions": [
                "You are a digital forensics investigator analyzing network evidence for security incidents.",
                "Your task is to examine network activity with forensic rigor and determine malicious intent.",
                "Focus on evidence-based analysis:",
                "- Observable artifacts and their forensic significance",
                "- Chain of events and temporal relationships",
                "- Attribution indicators and attack sophistication",
                "- Evidence quality and completeness",
                "",
                "Provide a structured response with:",
                "1. Classification decision (malicious/benign, threat level, attack type)",
                "2. Confidence score (0.0 to 1.0) based on evidence strength",
                "3. Clear reasoning citing specific evidence",
                "4. Key forensic artifacts supporting your conclusion",
                "",
                "Be measured with confidence when:",
                "- Evidence is circumstantial or incomplete",
                "- Multiple interpretations are forensically valid",
                "- Attack sophistication suggests possible evasion techniques",
                "",
                "Higher confidence when:",
                "- Multiple independent artifacts corroborate the finding",
                "- Clear forensic signatures of attack techniques",
                "- Strong evidence chain with consistent indicators"
            ]
        }
    }

    def __init__(
        self,
        model_name: str = "groq-llama-70b",
        dataset_config: Optional[Dict[str, Any]] = None,
        results_dir: str = "scenario_results/ensemble",
        simulation_mode: bool = False,
        routing_thresholds: Optional[Dict[str, float]] = None,
        num_variants: int = 3,
        resume_from: Optional[str] = None
    ):
        """
        Initialize ensemble runner

        Args:
            model_name: Name of LLM model to use
            dataset_config: Dataset configuration from config.yaml
            results_dir: Directory to save results
            simulation_mode: If True, use mock predictions without API calls
            routing_thresholds: Confidence thresholds for routing decisions
            num_variants: Number of prompt variants to use (max 3)
            resume_from: Path to existing run directory to resume from
        """
        self.model_name = model_name
        self.dataset_config = dataset_config
        self.simulation_mode = simulation_mode
        self.num_variants = min(num_variants, 3)

        # Load configuration
        self.config_loader = ConfigLoader()

        # Set up routing thresholds
        self.routing_thresholds = routing_thresholds or {
            "accept_threshold": 0.85,
            "escalate_threshold": 0.60
        }

        # Create or resume results directory
        if resume_from:
            self.results_dir = Path(resume_from)
            self.timestamp = self.results_dir.name.replace("run_", "")
            print(f"📂 Resuming from existing run: {self.results_dir}")
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = Path(results_dir) / f"run_{self.timestamp}"
            self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.results_dir / "ensemble_results.jsonl"

        # Initialize ensemble of agents (unless in simulation mode)
        self.agents = None
        if not simulation_mode:
            self.agents = self._create_ensemble_agents(model_name, self.num_variants)

    def _load_existing_results(self, results_file: Path) -> tuple[List[EnsembleTestResult], set, set]:
        """
        Load existing results from previous run for resume capability

        Returns:
            Tuple of (all_results, completed_sample_ids, error_sample_ids)
        """
        if not results_file.exists():
            return [], set(), set()

        results = []
        completed_ids = set()
        error_ids = set()

        try:
            with open(results_file, 'r') as f:
                for line in f:
                    try:
                        result_dict = json.loads(line)
                        # Convert dict back to EnsembleTestResult
                        result = EnsembleTestResult(**result_dict)
                        results.append(result)

                        sample_id = result.sample_id

                        # Check if sample completed successfully
                        if result.error is None or result.error == "null":
                            completed_ids.add(sample_id)
                        else:
                            error_ids.add(sample_id)
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

        except FileNotFoundError:
            pass  # File doesn't exist yet

        return results, completed_ids, error_ids

    def _create_ensemble_agents(self, model_name: str, num_variants: int) -> List[Agent]:
        """Create ensemble of agents with different prompt variants"""
        agents = []
        variant_names = list(self.PROMPT_VARIANTS.keys())[:num_variants]

        # Get model configuration from config.yaml
        model_config = self._get_model_config(model_name)
        model_id = model_config["model"]
        provider = model_config["provider"]
        temperature = model_config.get("temperature", 0.0)

        for variant_name in variant_names:
            variant = self.PROMPT_VARIANTS[variant_name]

            # Create provider-specific LLM instance
            if provider == "groq":
                llm_model = Groq(
                    id=model_id,
                    temperature=temperature,
                    base_url="https://api.groq.com"
                )
            elif provider == "nvidia":
                llm_model = Nvidia(
                    id=model_id,
                    temperature=temperature
                )
            elif provider == "openai":
                llm_model = OpenAIChat(id=model_id, temperature=temperature)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Create agent with variant-specific instructions
            agent = Agent(
                model=llm_model,
                output_schema=IDSStructuredResponse,
                structured_outputs=True,
                instructions=variant["instructions"]
            )

            agents.append(agent)
            print(f"  Created agent {len(agents)}/{num_variants}: {variant['role']}")

        return agents

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get model configuration from config.yaml

        Args:
            model_name: Model CLI parameter (e.g., 'groq-llama4-scout', 'nvidia-qwen')

        Returns:
            Model configuration dictionary with 'model', 'provider', 'temperature' fields

        Raises:
            ValueError: If model not found in configuration
        """
        model_config = self.config_loader.get_model_by_cli_param(model_name)

        if not model_config:
            available_models = list(self.config_loader.list_available_models().values())
            raise ValueError(
                f"Unknown model: {model_name}\n"
                f"Available models: {', '.join(available_models)}"
            )

        return model_config

    def run_test(
        self,
        samples: List[IDSSample],
        iterations: int = 1,
        delay: float = 0.5
    ) -> List[EnsembleTestResult]:
        """
        Run ensemble uncertainty test on dataset samples
        Supports resume from previous runs - skips completed samples, retries errors

        Args:
            samples: List of IDS samples to test
            iterations: Number of iterations per sample
            delay: Delay between API calls (seconds)

        Returns:
            List of test results
        """
        # Load existing results if resuming
        existing_results, completed_ids, error_ids = self._load_existing_results(self.results_file)

        results = []
        total_tests = len(samples) * iterations

        print(f"\n{'='*60}")
        print(f"ENSEMBLE UNCERTAINTY TEST")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Variants: {self.num_variants}")
        print(f"Samples: {len(samples)}")
        print(f"Iterations: {iterations}")
        print(f"Total tests: {total_tests} (× {self.num_variants} variants = {total_tests * self.num_variants} API calls)")
        print(f"Simulation mode: {self.simulation_mode}")
        print(f"Results directory: {self.results_dir}")

        if existing_results:
            print(f"\n🔄 RESUME MODE:")
            print(f"  ✅ Completed: {len(completed_ids)} samples")
            print(f"  ❌ Errors: {len(error_ids)} samples (will retry)")
            print(f"  ⏭️  Will skip: {len(completed_ids)} successful samples")

        print(f"{'='*60}\n")

        test_count = 0
        skipped_count = 0
        retry_count = 0

        for sample in samples:
            for iteration in range(iterations):
                test_count += 1
                sample_id = sample.id

                # Skip if already completed successfully
                if sample_id in completed_ids:
                    skipped_count += 1
                    print(f"\n[{test_count}/{total_tests}] ⏭️  Skipping sample {sample_id} (already completed)")
                    continue

                # Check if retrying an error
                is_retry = sample_id in error_ids
                if is_retry:
                    retry_count += 1
                    print(f"\n[{test_count}/{total_tests}] 🔄 Retrying sample {sample_id} (iteration {iteration + 1}/{iterations}) - previous error")
                else:
                    print(f"\n[{test_count}/{total_tests}] Testing sample {sample_id} (iteration {iteration + 1}/{iterations})")

                # Run single test
                result = self._run_single_test(sample, iteration)
                results.append(result)

                # Save result immediately
                self._save_result(result)

                # Progress update
                if result.error:
                    print(f"  Error occurred: {result.error}")
                else:
                    print(f"  Final classification: {result.final_prediction['attack_type']} (confidence: {result.final_prediction['confidence_score']:.2f})")
                    print(f"  Agreement: {result.agreement_score:.2f}, Variance: {result.prediction_variance:.3f}")
                    print(f"  Correct: {result.is_correct}, Routing: {result.routing_decision}")

                # Rate limiting (delay applies to entire ensemble)
                if not self.simulation_mode and delay > 0:
                    time.sleep(delay)

        print(f"\n{'='*60}")
        print(f"Test completed: {len(results)} new results")
        if skipped_count > 0:
            print(f"  Skipped: {skipped_count} already completed samples")
        if retry_count > 0:
            print(f"  Retried: {retry_count} error samples")
        print(f"  Total in file: {len(existing_results) + len(results)} results")
        print(f"  Saved to: {self.results_file}")
        print(f"{'='*60}\n")

        return results

    def _run_single_test(self, sample: IDSSample, iteration: int) -> EnsembleTestResult:
        """Run single ensemble test on one sample"""
        start_time = time.time()

        try:
            # Get predictions from all variants
            if self.simulation_mode:
                variant_predictions = [
                    self._get_simulated_prediction(sample) for _ in range(self.num_variants)
                ]
            else:
                variant_predictions = self._get_ensemble_predictions(sample)

            # Calculate agreement and variance
            agreement_score = self._calculate_agreement(variant_predictions)
            prediction_variance = self._calculate_variance(variant_predictions)

            # Aggregate predictions
            final_prediction = self._aggregate_predictions(variant_predictions, agreement_score)

            # Evaluate prediction
            is_correct = self._check_correctness(final_prediction, sample.ground_truth)

            # Calculate composite uncertainty metrics
            avg_self_report_confidence = np.mean([p.confidence_score for p in variant_predictions])
            composite_confidence = self._calculate_composite_confidence(
                avg_self_report_confidence, agreement_score, prediction_variance
            )

            uncertainty_metrics = UncertaintyMetrics(
                confidence=composite_confidence,
                entropy=None,
                variance=prediction_variance,
                consistency=agreement_score,
                composite_score=composite_confidence
            )

            # Routing decision based on composite confidence
            routing_decision = self._make_routing_decision(composite_confidence)

            execution_time = time.time() - start_time

            return EnsembleTestResult(
                sample_id=sample.id,
                model_name=self.model_name,
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                description=sample.description,
                ground_truth=sample.ground_truth.model_dump() if hasattr(sample.ground_truth, 'model_dump') else sample.ground_truth.dict(),
                variant_predictions=[
                    {
                        "is_malicious": p.is_malicious,
                        "threat_level": p.threat_level,
                        "attack_type": p.attack_type,
                        "confidence_score": p.confidence_score,
                        "reasoning": p.reasoning[:200] + "..." if len(p.reasoning) > 200 else p.reasoning,  # Truncate for storage
                        "key_features": p.key_features
                    }
                    for p in variant_predictions
                ],
                final_prediction={
                    "is_malicious": final_prediction.is_malicious,
                    "threat_level": final_prediction.threat_level,
                    "attack_type": final_prediction.attack_type,
                    "confidence_score": final_prediction.confidence_score,
                    "reasoning": final_prediction.reasoning,
                    "key_features": final_prediction.key_features
                },
                is_correct=is_correct,
                uncertainty_metrics=uncertainty_metrics.model_dump() if hasattr(uncertainty_metrics, 'model_dump') else uncertainty_metrics.dict(),
                routing_decision=routing_decision,
                agreement_score=agreement_score,
                prediction_variance=prediction_variance,
                execution_time=execution_time,
                error=None
            )

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  ERROR: {str(e)}")

            return EnsembleTestResult(
                sample_id=sample.id,
                model_name=self.model_name,
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                description=sample.description,
                ground_truth=sample.ground_truth.model_dump() if hasattr(sample.ground_truth, 'model_dump') else sample.ground_truth.dict(),
                variant_predictions=[],
                final_prediction={},
                is_correct=False,
                uncertainty_metrics={},
                routing_decision="error",
                agreement_score=0.0,
                prediction_variance=0.0,
                execution_time=execution_time,
                error=str(e)
            )

    @retry_with_intelligent_backoff(
        max_attempts=3,
        initial_delay=2.0,
        max_delay=60.0,
        rate_limit_threshold=3600.0  # Exit if wait > 1 hour
    )
    def _call_agent_with_retry(self, agent: Agent, prompt: str) -> Any:
        """Call agent with intelligent retry handling"""
        return agent.run(prompt)

    def _get_ensemble_predictions(self, sample: IDSSample) -> List[IDSStructuredResponse]:
        """Get predictions from all ensemble agents"""
        predictions = []

        prompt = f"""Analyze this network flow and classify it as benign or malicious:

{sample.description}

Provide your classification, confidence level, and reasoning."""

        for i, agent in enumerate(self.agents):
            try:
                response = self._call_agent_with_retry(agent, prompt)

                # Extract structured output (handle Agno 2.1.8 response structure)
                if isinstance(response, IDSStructuredResponse):
                    prediction = response
                elif hasattr(response, 'content') and isinstance(response.content, IDSStructuredResponse):
                    prediction = response.content
                else:
                    raise ValueError(f"Cannot extract structured output from variant {i+1}")

                predictions.append(prediction)
                print(f"    Variant {i+1}: {prediction.attack_type} (conf: {prediction.confidence_score:.2f})")

            except RateLimitError as e:
                # Rate limit - should exit and resume later
                print(f"    Variant {i+1} RATE LIMITED: Exiting gracefully")
                raise  # Propagate to main loop
            except Exception as e:
                print(f"    Variant {i+1} FAILED: {str(e)}")
                raise

        return predictions

    def _get_simulated_prediction(self, sample: IDSSample) -> IDSStructuredResponse:
        """Generate simulated prediction for testing without API calls"""
        import random

        # Simulate prediction with some noise
        is_correct = random.random() > 0.15  # 85% accuracy

        if is_correct:
            is_malicious = sample.ground_truth.is_malicious
            attack_type = sample.ground_truth.attack_type
            confidence = random.uniform(0.75, 0.95)
        else:
            is_malicious = not sample.ground_truth.is_malicious
            attack_type = AttackType.UNKNOWN if is_malicious else AttackType.BENIGN
            confidence = random.uniform(0.50, 0.75)

        threat_level = ThreatLevel.HIGH if is_malicious else ThreatLevel.BENIGN

        return IDSStructuredResponse(
            is_malicious=is_malicious,
            threat_level=threat_level,
            attack_type=attack_type,
            confidence_score=confidence,
            reasoning=f"[SIMULATED] Classification based on pattern analysis",
            key_features=["simulated_feature"]
        )

    def _calculate_agreement(self, predictions: List[IDSStructuredResponse]) -> float:
        """
        Calculate agreement score across ensemble predictions
        Returns: 0.0 (complete disagreement) to 1.0 (complete agreement)
        """
        if not predictions:
            return 0.0

        # Primary agreement: malicious vs benign classification
        malicious_votes = sum(1 for p in predictions if p.is_malicious)
        malicious_ratio = malicious_votes / len(predictions)

        # Agreement is highest when all agree (1.0 or 0.0), lowest at 0.5 (split)
        primary_agreement = 1.0 - 2.0 * abs(malicious_ratio - 0.5)

        # Secondary agreement: attack type consistency (among those who agree on is_malicious)
        if malicious_votes > 0:
            malicious_predictions = [p for p in predictions if p.is_malicious]
            attack_types = [p.attack_type for p in malicious_predictions]
            if attack_types:
                most_common_attack = Counter(attack_types).most_common(1)[0]
                attack_agreement = most_common_attack[1] / len(attack_types)
            else:
                attack_agreement = 1.0
        else:
            attack_agreement = 1.0  # All agree it's benign

        # Combine primary and secondary agreement (weight primary more heavily)
        overall_agreement = 0.7 * primary_agreement + 0.3 * attack_agreement

        return overall_agreement

    def _calculate_variance(self, predictions: List[IDSStructuredResponse]) -> float:
        """
        Calculate variance in predictions
        Returns: Normalized variance score (higher = more disagreement)
        """
        if not predictions or len(predictions) < 2:
            return 0.0

        # Confidence variance
        confidences = [p.confidence_score for p in predictions]
        confidence_var = float(np.var(confidences))

        # Classification variance (binary entropy)
        malicious_ratio = sum(1 for p in predictions if p.is_malicious) / len(predictions)
        if malicious_ratio == 0 or malicious_ratio == 1:
            classification_entropy = 0.0
        else:
            classification_entropy = -(
                malicious_ratio * np.log2(malicious_ratio) +
                (1 - malicious_ratio) * np.log2(1 - malicious_ratio)
            )

        # Combine variance signals (normalize entropy to 0-1 range)
        normalized_entropy = classification_entropy / 1.0  # Max entropy is 1.0 for binary
        combined_variance = 0.5 * confidence_var + 0.5 * normalized_entropy

        return float(combined_variance)

    def _aggregate_predictions(
        self,
        predictions: List[IDSStructuredResponse],
        agreement_score: float
    ) -> IDSStructuredResponse:
        """
        Aggregate ensemble predictions into final prediction
        Uses majority voting with confidence weighting
        """
        if not predictions:
            raise ValueError("No predictions to aggregate")

        # Majority vote for is_malicious
        malicious_votes = sum(1 for p in predictions if p.is_malicious)
        final_is_malicious = malicious_votes > len(predictions) / 2

        # For attack type, use majority among predictions that agree on is_malicious
        if final_is_malicious:
            attack_types = [p.attack_type for p in predictions if p.is_malicious]
        else:
            attack_types = [AttackType.BENIGN]

        if attack_types:
            final_attack_type = Counter(attack_types).most_common(1)[0][0]
        else:
            final_attack_type = AttackType.UNKNOWN

        # Threat level from majority
        threat_levels = [p.threat_level for p in predictions]
        final_threat_level = Counter(threat_levels).most_common(1)[0][0]

        # Average confidence (will be adjusted by agreement score later)
        avg_confidence = np.mean([p.confidence_score for p in predictions])

        # Aggregate reasoning
        reasoning_parts = [f"Variant {i+1}: {p.reasoning[:100]}" for i, p in enumerate(predictions)]
        final_reasoning = f"Ensemble decision (agreement: {agreement_score:.2f}). " + " | ".join(reasoning_parts[:2])

        # Aggregate key features (union of all features)
        all_features = []
        for p in predictions:
            all_features.extend(p.key_features)
        final_features = list(set(all_features))[:10]  # Top 10 unique features

        return IDSStructuredResponse(
            is_malicious=final_is_malicious,
            threat_level=final_threat_level,
            attack_type=final_attack_type,
            confidence_score=avg_confidence,  # Will be adjusted in composite calculation
            reasoning=final_reasoning,
            key_features=final_features
        )

    def _calculate_composite_confidence(
        self,
        avg_self_report: float,
        agreement_score: float,
        variance: float
    ) -> float:
        """
        Calculate composite confidence combining self-report and ensemble signals

        Formula: composite = w1 * self_report + w2 * agreement - w3 * variance
        Where weights sum to 1.0
        """
        # Weights (tuned for IDS: prioritize self-report, use agreement as modifier)
        w_self_report = 0.6
        w_agreement = 0.3
        w_variance = 0.1

        composite = (
            w_self_report * avg_self_report +
            w_agreement * agreement_score -
            w_variance * variance
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, composite))

    def _check_correctness(self, prediction: IDSStructuredResponse, ground_truth: GroundTruth) -> bool:
        """Check if prediction matches ground truth"""
        return prediction.is_malicious == ground_truth.is_malicious

    def _make_routing_decision(self, confidence: float) -> str:
        """Make routing decision based on confidence threshold"""
        if confidence >= self.routing_thresholds["accept_threshold"]:
            return "accept"
        elif confidence >= self.routing_thresholds["escalate_threshold"]:
            return "escalate"
        else:
            return "reject"

    def _save_result(self, result: EnsembleTestResult):
        """Save single result to JSONL file"""
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + "\n")


def main():
    """Main entry point for ensemble testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Run ensemble IDS uncertainty test")
    parser.add_argument("--model", default="groq-llama-70b", help="Model name")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--iterations", type=int, default=1, help="Iterations per sample")
    parser.add_argument("--variants", type=int, default=3, help="Number of prompt variants (max 3)")
    parser.add_argument("--simulation", action="store_true", help="Simulation mode (no API calls)")
    parser.add_argument(
        "--dataset",
        default="hard",
        choices=["easy", "hard", "cicids", "cicids2018"],
        help="Dataset: 'easy' (100%% acc mock), 'hard' (60-80%% acc mock), 'cicids' (CIC-IDS2017), or 'cicids2018' (CIC-IDS2018)"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from existing run directory (e.g., scenario_results/ensemble/run_20251022_120050)"
    )

    args = parser.parse_args()

    # Load configuration
    config_loader = ConfigLoader()

    # Validate and get dataset configuration
    is_valid, message = config_loader.validate_dataset(args.dataset)
    if not is_valid:
        print(f"❌ {message}")
        return

    print(f"✅ {message}")

    # Get dataset config from config.yaml
    dataset_config = config_loader.get_dataset_by_cli_param(args.dataset)
    if not dataset_config:
        print(f"Error: Could not load dataset config for: {args.dataset}")
        return

    # Load dataset
    print(f"Loading {args.samples} samples from {dataset_config['file']}...")
    loader = IDSDatasetLoader(dataset_config)
    samples = loader.load(limit=args.samples)

    if not samples:
        print(f"No samples loaded from {dataset_config['file']}")
        return

    # Run ensemble test
    runner = EnsembleRunner(
        model_name=args.model,
        dataset_config=dataset_config,
        simulation_mode=args.simulation,
        num_variants=args.variants,
        resume_from=args.resume_from
    )

    results = runner.run_test(
        samples=samples,
        iterations=args.iterations,
        delay=1.0 if not args.simulation else 0  # Longer delay for K API calls
    )

    # Print summary (only if we have new results)
    if results:
        print("\nEnsemble Test Summary:")
        print(f"  Total tests: {len(results)}")
        correct = sum(1 for r in results if r.is_correct)
        print(f"  Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

        avg_confidence = sum(r.final_prediction.get('confidence_score', 0) for r in results) / len(results)
        print(f"  Average composite confidence: {avg_confidence:.3f}")

        avg_agreement = sum(r.agreement_score for r in results) / len(results)
        print(f"  Average agreement score: {avg_agreement:.3f}")

        avg_variance = sum(r.prediction_variance for r in results) / len(results)
        print(f"  Average prediction variance: {avg_variance:.3f}")

        routing_counts = {}
        for r in results:
            routing_counts[r.routing_decision] = routing_counts.get(r.routing_decision, 0) + 1
        print(f"  Routing decisions: {routing_counts}")
    else:
        print("\nNo new results (all samples were skipped or already completed)")


if __name__ == "__main__":
    main()
