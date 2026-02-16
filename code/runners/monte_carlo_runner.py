#!/usr/bin/env python3
"""
Monte Carlo Prompting Runner for IDS Uncertainty Estimation

Implements Monte Carlo sampling with M=5 stochastic predictions per input using temperature=0.8.
Outputs results to scenario_results/monte_carlo/run_YYYYMMDD_HHMMSS/monte_carlo_results.jsonl

Usage:
    uv run python monte_carlo_runner.py --model groq-qwen --dataset cicids --samples 50 --mc-samples 5
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.openai import OpenAIChat
from agno.models.nvidia import Nvidia

from ids_models import (
    IDSStructuredResponse,
    GroundTruth,
    MonteCarloTestResult,
    UncertaintyMetrics,
    AttackType,
    ThreatLevel,
)
from dataset_loader import IDSDatasetLoader, IDSSample, generate_mock_dataset
from retry_utils import retry_with_intelligent_backoff, RateLimitError
from config_loader import ConfigLoader


# Base prompt for IDS classification
BASE_PROMPT = """You are an expert network security analyst tasked with detecting intrusion attempts.

Analyze the following network flow and classify it as either benign or malicious.
Provide your confidence score, threat level, attack type, and reasoning.

Network Flow Data:
{description}

Classify this network activity and explain your reasoning."""


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Monte Carlo Prompting Runner for IDS Uncertainty Estimation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="groq-llama-70b",
        help="LLM model to use for predictions (e.g., groq-qwen, groq-llama-70b)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hard",
        choices=["easy", "hard", "cicids", "cicids2018"],
        help="Dataset selection (easy=100%% acc, hard=60-80%% acc, cicids=CIC-IDS2017, cicids2018=CIC-IDS2018)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=5,
        help="Monte Carlo samples per test input (M parameter)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations per sample",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from existing run directory",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Simulation mode (no API calls, generates mock predictions)",
    )
    return parser.parse_args()


def initialize_agent(model_name: str, temperature: float = 0.8) -> Agent:
    """
    Initialize Agno agent with specified LLM provider at specified temperature.

    Args:
        model_name: Model CLI parameter (e.g., 'groq-qwen', 'nvidia-llama4-scout')
        temperature: Sampling temperature (0.8 for stochastic behavior)

    Returns:
        Configured Agent instance
    """
    # Load model configuration from config.yaml (single source of truth)
    config_loader = ConfigLoader()
    model_config = config_loader.get_model_by_cli_param(model_name)

    if not model_config:
        available_models = list(config_loader.list_available_models().values())
        raise ValueError(
            f"Unknown model: {model_name}\n"
            f"Available models: {', '.join(available_models)}"
        )

    # Create provider-specific LLM instance
    model_id = model_config["model"]
    provider = model_config["provider"]

    if provider == "groq":
        # Fix Agno 2.1.8 Groq URL duplication bug - use base domain only
        model = Groq(
            id=model_id,
            temperature=temperature,
            base_url="https://api.groq.com"
        )
    elif provider == "nvidia":
        model = Nvidia(
            id=model_id,
            temperature=temperature
        )
    elif provider == "openai":
        model = OpenAIChat(id=model_id, temperature=temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    agent = Agent(
        model=model,
        instructions=BASE_PROMPT,
        output_schema=IDSStructuredResponse,
        structured_outputs=True,
    )

    return agent


def load_dataset(dataset_name: str, num_samples: int) -> List[IDSSample]:
    """
    Load dataset samples.

    Args:
        dataset_name: Dataset identifier (easy/hard/cicids)
        num_samples: Number of samples to load

    Returns:
        List of IDSSample objects
    """
    if dataset_name == "easy":
        # Generate easy mock dataset (100% accuracy baseline)
        return generate_mock_dataset(n_samples=num_samples)
    elif dataset_name == "hard":
        # Generate hard mock dataset (60-80% accuracy baseline)
        from dataset_loader import generate_hard_mock_dataset
        samples = generate_hard_mock_dataset()
        return samples[:num_samples]
    elif dataset_name == "cicids" or dataset_name == "cicids2018":
        # Load CIC-IDS dataset using config
        config_loader = ConfigLoader()
        dataset_config = config_loader.get_dataset_by_cli_param(dataset_name)
        if not dataset_config:
            raise ValueError(f"Could not load dataset config for: {dataset_name}")
        loader = IDSDatasetLoader(dataset_config)
        return loader.load(limit=num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


@retry_with_intelligent_backoff(max_attempts=3)
def run_single_prediction(
    agent: Agent,
    description: str,
    sample_id: str,
    simulation: bool = False,
) -> IDSStructuredResponse:
    """
    Run a single Monte Carlo prediction.

    Args:
        agent: Configured Agent instance
        description: Network flow description
        sample_id: Sample identifier (for logging)
        simulation: If True, generate mock prediction without API call

    Returns:
        IDSStructuredResponse prediction
    """
    if simulation:
        # Generate mock prediction for testing
        import random
        return IDSStructuredResponse(
            is_malicious=random.choice([True, False]),
            threat_level=random.choice(list(ThreatLevel)),
            attack_type=random.choice(list(AttackType)),
            confidence_score=random.uniform(0.6, 0.95),
            reasoning="Simulated prediction for testing",
            key_features=["simulated_feature_1", "simulated_feature_2"],
        )

    # Real API call
    response = agent.run(description)
    return response.content


def aggregate_predictions(
    predictions: List[IDSStructuredResponse],
) -> IDSStructuredResponse:
    """
    Aggregate multiple predictions via majority voting.

    Args:
        predictions: List of M predictions

    Returns:
        Aggregated final prediction (mode for is_malicious, mean for confidence)
    """
    # Majority vote on is_malicious
    malicious_votes = [p.is_malicious for p in predictions]
    final_is_malicious = Counter(malicious_votes).most_common(1)[0][0]

    # Filter predictions that match majority
    majority_predictions = [p for p in predictions if p.is_malicious == final_is_malicious]

    # Mode for attack_type among majority
    attack_types = [p.attack_type for p in majority_predictions]
    final_attack_type = Counter(attack_types).most_common(1)[0][0]

    # Mode for threat_level among majority
    threat_levels = [p.threat_level for p in majority_predictions]
    final_threat_level = Counter(threat_levels).most_common(1)[0][0]

    # Mean confidence
    final_confidence = float(np.mean([p.confidence_score for p in predictions]))

    # Use reasoning from most confident prediction
    most_confident = max(predictions, key=lambda p: p.confidence_score)

    return IDSStructuredResponse(
        is_malicious=final_is_malicious,
        threat_level=final_threat_level,
        attack_type=final_attack_type,
        confidence_score=final_confidence,
        reasoning=most_confident.reasoning,
        key_features=most_confident.key_features,
    )


def calculate_consistency_score(
    predictions: List[IDSStructuredResponse],
    final_prediction: IDSStructuredResponse,
) -> float:
    """
    Calculate consistency score as agreement rate with majority vote.

    Args:
        predictions: List of M predictions
        final_prediction: Aggregated prediction

    Returns:
        Consistency score in [0.0, 1.0]
    """
    matches = sum(1 for p in predictions if p.is_malicious == final_prediction.is_malicious)
    return matches / len(predictions)


def calculate_prediction_variance(predictions: List[IDSStructuredResponse]) -> float:
    """
    Calculate variance of confidence scores.

    Args:
        predictions: List of M predictions

    Returns:
        Variance (>= 0.0)
    """
    confidence_scores = [p.confidence_score for p in predictions]
    return float(np.var(confidence_scores))


def calculate_routing_decision(
    confidence: float,
    is_correct: bool,
    threshold_accept: float = 0.85,
    threshold_reject: float = 0.60,
) -> str:
    """
    Calculate routing decision based on confidence thresholds.

    Args:
        confidence: Final confidence score
        is_correct: Whether prediction is correct
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


def run_monte_carlo_sample(
    agent: Agent,
    sample: IDSSample,
    mc_samples: int,
    iteration: int,
    model_name: str,
    simulation: bool = False,
) -> MonteCarloTestResult:
    """
    Run Monte Carlo estimation for a single sample.

    Args:
        agent: Configured Agent instance
        sample: IDSSample to test
        mc_samples: Number of MC samples (M parameter)
        iteration: Iteration number
        model_name: Model identifier
        simulation: If True, use simulation mode

    Returns:
        MonteCarloTestResult
    """
    start_time = time.time()

    try:
        # Run M stochastic predictions
        mc_predictions = []
        for i in range(mc_samples):
            prediction = run_single_prediction(
                agent=agent,
                description=sample.description,
                sample_id=sample.id,
                simulation=simulation,
            )
            mc_predictions.append(prediction)

        # Aggregate via majority voting
        final_prediction = aggregate_predictions(mc_predictions)

        # Calculate metrics
        consistency_score = calculate_consistency_score(mc_predictions, final_prediction)
        prediction_variance = calculate_prediction_variance(mc_predictions)

        # Check correctness
        is_correct = (
            final_prediction.is_malicious == sample.ground_truth.is_malicious
            and final_prediction.attack_type == sample.ground_truth.attack_type
        )

        # Uncertainty metrics
        uncertainty_metrics = UncertaintyMetrics(
            confidence=final_prediction.confidence_score,
            variance=prediction_variance,
            consistency=consistency_score,
            composite_score=final_prediction.confidence_score,  # No composite yet
        )

        # Routing decision
        routing_decision = calculate_routing_decision(
            confidence=final_prediction.confidence_score,
            is_correct=is_correct,
        )

        execution_time = time.time() - start_time

        return MonteCarloTestResult(
            sample_id=sample.id,
            model_name=model_name,
            iteration=iteration,
            timestamp=datetime.utcnow().isoformat() + "Z",
            description=sample.description,
            ground_truth=sample.ground_truth,
            mc_predictions=mc_predictions,
            final_prediction=final_prediction,
            consistency_score=consistency_score,
            prediction_variance=prediction_variance,
            is_correct=is_correct,
            uncertainty_metrics=uncertainty_metrics,
            routing_decision=routing_decision,
            execution_time=execution_time,
            error=None,
        )

    except Exception as e:
        execution_time = time.time() - start_time

        # Return error result
        return MonteCarloTestResult(
            sample_id=sample.id,
            model_name=model_name,
            iteration=iteration,
            timestamp=datetime.utcnow().isoformat() + "Z",
            description=sample.description,
            ground_truth=sample.ground_truth,
            mc_predictions=[],
            final_prediction=IDSStructuredResponse(
                is_malicious=False,
                threat_level=ThreatLevel.BENIGN,
                attack_type=AttackType.UNKNOWN,
                confidence_score=0.0,
                reasoning="Error occurred",
                key_features=[],
            ),
            consistency_score=0.0,
            prediction_variance=0.0,
            is_correct=False,
            uncertainty_metrics=UncertaintyMetrics(
                confidence=0.0,
                composite_score=0.0,
            ),
            routing_decision="reject",
            execution_time=execution_time,
            error=str(e),
        )


def setup_output_directory(resume_from: Optional[str] = None) -> Path:
    """
    Setup output directory with timestamp.

    Args:
        resume_from: Existing run directory to resume from

    Returns:
        Path to output directory
    """
    if resume_from:
        output_dir = Path(resume_from)
        if not output_dir.exists():
            raise ValueError(f"Resume directory does not exist: {resume_from}")
        print(f"📂 Resuming from: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"scenario_results/monte_carlo/run_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Created output directory: {output_dir}")

    return output_dir


def get_completed_samples(output_file: Path) -> set:
    """
    Get set of completed sample IDs from existing JSONL file.

    Args:
        output_file: Path to JSONL output file

    Returns:
        Set of completed sample IDs
    """
    completed = set()

    if not output_file.exists():
        return completed

    with open(output_file, "r") as f:
        for line in f:
            try:
                result = json.loads(line)
                if result.get("error") is None:
                    completed.add(result["sample_id"])
            except json.JSONDecodeError:
                continue

    return completed


def main():
    """Main execution function"""
    args = parse_arguments()

    print("=" * 80)
    print("Monte Carlo Prompting Runner")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.samples}")
    print(f"MC Samples (M): {args.mc_samples}")
    print(f"Iterations: {args.iterations}")
    print(f"Simulation: {args.simulation}")
    print(f"Temperature: 0.8 (stochastic sampling)")
    print("=" * 80)

    # Setup output directory
    output_dir = setup_output_directory(args.resume_from)
    output_file = output_dir / "monte_carlo_results.jsonl"

    # Get completed samples for resume
    completed_samples = get_completed_samples(output_file)
    if completed_samples:
        print(f"✅ Found {len(completed_samples)} completed samples (resuming)")

    # Initialize agent with temperature=0.8
    agent = initialize_agent(args.model, temperature=0.8)

    # Load dataset
    print(f"\n📊 Loading dataset: {args.dataset}")
    samples = load_dataset(args.dataset, args.samples)
    print(f"✅ Loaded {len(samples)} samples")

    # Open output file in append mode
    with open(output_file, "a") as f:
        total_processed = 0
        total_correct = 0
        total_errors = 0

        for idx, sample in enumerate(samples, 1):
            # Skip completed samples
            if sample.id in completed_samples:
                print(f"⏭️  [{idx}/{len(samples)}] Skipping completed sample: {sample.id}")
                continue

            for iteration in range(args.iterations):
                print(f"\n[{idx}/{len(samples)}] Testing sample {sample.id} (iteration {iteration + 1}/{args.iterations})")

                # Run Monte Carlo estimation
                result = run_monte_carlo_sample(
                    agent=agent,
                    sample=sample,
                    mc_samples=args.mc_samples,
                    iteration=iteration,
                    model_name=args.model,
                    simulation=args.simulation,
                )

                # Write result to JSONL
                f.write(json.dumps(result.model_dump()) + "\n")
                f.flush()

                # Update statistics
                total_processed += 1
                if result.error:
                    total_errors += 1
                    print(f"❌ Error: {result.error}")
                else:
                    if result.is_correct:
                        total_correct += 1

                    print(f"Classification: {result.final_prediction.attack_type} "
                          f"(confidence: {result.final_prediction.confidence_score:.2f})")
                    print(f"Consistency: {result.consistency_score:.2f}, "
                          f"Variance: {result.prediction_variance:.4f}")
                    print(f"Correct: {result.is_correct}, Routing: {result.routing_decision}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total samples processed: {total_processed}")
    if total_processed > 0:
        accuracy = total_correct / total_processed
        print(f"Accuracy: {total_correct}/{total_processed} ({accuracy:.1%})")
        print(f"Errors: {total_errors}")
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
