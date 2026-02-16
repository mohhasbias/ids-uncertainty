#!/usr/bin/env python3
"""
Baseline Uncertainty Method Runner
Method 1: Self-Report Confidence (Direct LLM uncertainty)

This is the simplest method where we ask the LLM to report its confidence
directly in the structured response. This serves as the baseline for comparison.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.openai import OpenAIChat
from agno.models.nvidia import Nvidia

from ids_models import (
    IDSStructuredResponse,
    GroundTruth,
    EvaluationResult,
    UncertaintyMetrics,
    AttackType
)
from dataset_loader import IDSDatasetLoader, IDSSample, generate_mock_dataset
from retry_utils import retry_with_intelligent_backoff, RateLimitError
from config_loader import ConfigLoader


@dataclass
class BaselineTestResult:
    """Single test result for baseline method"""
    sample_id: str
    model_name: str
    iteration: int
    timestamp: str

    # Input
    description: str
    ground_truth: Dict[str, Any]

    # LLM Response
    prediction: Dict[str, Any]

    # Evaluation
    is_correct: bool
    uncertainty_metrics: Dict[str, Any]
    routing_decision: str

    # Metadata
    execution_time: float
    error: Optional[str] = None


class BaselineRunner:
    """
    Runner for baseline uncertainty estimation method
    Uses direct self-reported confidence from LLM
    """

    def __init__(
        self,
        model_name: str = "groq-llama-70b",
        dataset_config: Optional[Dict[str, Any]] = None,
        results_dir: str = "scenario_results/baseline",
        simulation_mode: bool = False,
        routing_thresholds: Optional[Dict[str, float]] = None,
        resume_from: Optional[str] = None
    ):
        """
        Initialize baseline runner

        Args:
            model_name: Name of LLM model to use
            dataset_config: Dataset configuration from config.yaml
            results_dir: Directory to save results
            simulation_mode: If True, use mock predictions without API calls
            routing_thresholds: Confidence thresholds for routing decisions
            resume_from: Path to existing run directory to resume from (e.g., "scenario_results/baseline/run_20251022_120050")
        """
        self.model_name = model_name
        self.dataset_config = dataset_config
        self.simulation_mode = simulation_mode

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

        self.results_file = self.results_dir / "baseline_results.jsonl"

        # Initialize LLM agent (unless in simulation mode)
        self.agent = None
        if not simulation_mode:
            self.agent = self._create_agent(model_name)

    def _load_existing_results(self, results_file: Path) -> tuple[List[BaselineTestResult], set, set]:
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
                        # Convert dict back to BaselineTestResult
                        result = BaselineTestResult(**result_dict)
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

    def _create_agent(self, model_name: str) -> Agent:
        """Create Agno agent for LLM interaction"""

        # Get model configuration from config.yaml
        model_config = self._get_model_config(model_name)
        model_id = model_config["model"]
        provider = model_config["provider"]
        temperature = model_config.get("temperature", 0.0)

        # Create provider-specific LLM instance
        if provider == "groq":
            # Fix Agno 2.1.8 Groq URL duplication bug - use base domain only
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

        # Create agent with structured response
        # Note: Use output_schema (Pydantic model), not output_model (Agno Model)
        agent = Agent(
            model=llm_model,
            output_schema=IDSStructuredResponse,
            structured_outputs=True,
            instructions=[
                "You are an expert network security analyst specializing in intrusion detection.",
                "Your task is to analyze network flow descriptions and classify them as benign or malicious.",
                "Provide a structured response with:",
                "1. Classification decision (malicious/benign, threat level, attack type)",
                "2. Confidence score (0.0 to 1.0) reflecting your certainty in the classification",
                "3. Clear reasoning explaining your decision and confidence level",
                "4. Key network features that influenced your decision",
                "",
                "Be honest about uncertainty. Lower confidence when:",
                "- Network behavior is ambiguous or has mixed signals",
                "- Attack patterns are subtle or sophisticated",
                "- Insufficient context or unusual scenarios",
                "",
                "Higher confidence when:",
                "- Clear attack signatures are present (e.g., port scans, brute force)",
                "- Normal business operations are obvious",
                "- Strong evidence supports the classification"
            ]
        )

        return agent

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
    ) -> List[BaselineTestResult]:
        """
        Run baseline uncertainty test on dataset samples
        Supports resume from previous runs - skips completed samples, retries errors

        Args:
            samples: List of IDS samples to test
            iterations: Number of iterations per sample (for consistency testing)
            delay: Delay between API calls (seconds)

        Returns:
            List of test results
        """
        # Load existing results if resuming
        existing_results, completed_ids, error_ids = self._load_existing_results(self.results_file)

        results = []
        total_tests = len(samples) * iterations

        print(f"\n{'='*60}")
        print(f"BASELINE UNCERTAINTY TEST")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Samples: {len(samples)}")
        print(f"Iterations: {iterations}")
        print(f"Total tests: {total_tests}")
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
                    print(f"  Classification: {result.prediction['attack_type']} (confidence: {result.prediction['confidence_score']:.2f})")
                    print(f"  Correct: {result.is_correct}, Routing: {result.routing_decision}")

                # Rate limiting
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

    def _run_single_test(self, sample: IDSSample, iteration: int) -> BaselineTestResult:
        """Run single test on one sample"""
        start_time = time.time()

        try:
            # Get prediction from LLM
            if self.simulation_mode:
                prediction = self._get_simulated_prediction(sample)
            else:
                prediction = self._get_llm_prediction(sample)

            # Evaluate prediction
            is_correct = self._check_correctness(prediction, sample.ground_truth)

            # Calculate uncertainty metrics
            uncertainty_metrics = UncertaintyMetrics(
                confidence=prediction.confidence_score,
                entropy=None,  # Not applicable for baseline
                variance=None,  # Not applicable for single prediction
                consistency=None,  # Would need multiple iterations
                composite_score=prediction.confidence_score  # Same as confidence for baseline
            )

            # Routing decision based on confidence
            routing_decision = self._make_routing_decision(prediction.confidence_score)

            execution_time = time.time() - start_time

            return BaselineTestResult(
                sample_id=sample.id,
                model_name=self.model_name,
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                description=sample.description,
                ground_truth=sample.ground_truth.model_dump() if hasattr(sample.ground_truth, 'model_dump') else sample.ground_truth.dict(),
                prediction={
                    "is_malicious": prediction.is_malicious,
                    "threat_level": prediction.threat_level,
                    "attack_type": prediction.attack_type,
                    "confidence_score": prediction.confidence_score,
                    "reasoning": prediction.reasoning,
                    "key_features": prediction.key_features
                },
                is_correct=is_correct,
                uncertainty_metrics=uncertainty_metrics.model_dump() if hasattr(uncertainty_metrics, 'model_dump') else uncertainty_metrics.dict(),
                routing_decision=routing_decision,
                execution_time=execution_time,
                error=None
            )

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  ERROR: {str(e)}")

            return BaselineTestResult(
                sample_id=sample.id,
                model_name=self.model_name,
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                description=sample.description,
                ground_truth=sample.ground_truth.model_dump() if hasattr(sample.ground_truth, 'model_dump') else sample.ground_truth.dict(),
                prediction={},
                is_correct=False,
                uncertainty_metrics={},
                routing_decision="error",
                execution_time=execution_time,
                error=str(e)
            )

    @retry_with_intelligent_backoff(
        max_attempts=3,
        initial_delay=2.0,
        max_delay=60.0,
        rate_limit_threshold=3600.0  # Exit if wait > 1 hour
    )
    def _call_agent_with_retry(self, prompt: str) -> Any:
        """Call agent with intelligent retry handling"""
        return self.agent.run(prompt)

    def _get_llm_prediction(self, sample: IDSSample) -> IDSStructuredResponse:
        """Get prediction from LLM using Agno"""
        prompt = f"""Analyze this network flow and classify it as benign or malicious:

{sample.description}

Provide your classification, confidence level, and reasoning."""

        try:
            response = self._call_agent_with_retry(prompt)

            # Debug: Print response details
            print(f"  DEBUG: Response type: {type(response)}")
            print(f"  DEBUG: Response attributes: {[a for a in dir(response) if not a.startswith('_')]}")

            # Try different ways to extract the structured output
            if isinstance(response, IDSStructuredResponse):
                return response
            elif hasattr(response, 'content') and isinstance(response.content, IDSStructuredResponse):
                return response.content
            elif hasattr(response, 'data') and isinstance(response.data, IDSStructuredResponse):
                return response.data
            elif hasattr(response, 'content'):
                content = response.content
                print(f"  DEBUG: Content type: {type(content)}")
                print(f"  DEBUG: Content value: {content}")
                raise ValueError(f"response.content exists but is {type(content)}, not IDSStructuredResponse")
            else:
                available = [a for a in dir(response) if not a.startswith('_')]
                raise ValueError(f"Cannot find structured output. Response type: {type(response)}, available attributes: {available}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"LLM prediction failed: {type(e).__name__}: {str(e)}")

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

        from ids_models import ThreatLevel
        threat_level = ThreatLevel.HIGH if is_malicious else ThreatLevel.BENIGN

        return IDSStructuredResponse(
            is_malicious=is_malicious,
            threat_level=threat_level,
            attack_type=attack_type,
            confidence_score=confidence,
            reasoning=f"[SIMULATED] Classification based on pattern analysis",
            key_features=["simulated_feature"]
        )

    def _check_correctness(self, prediction: IDSStructuredResponse, ground_truth: GroundTruth) -> bool:
        """Check if prediction matches ground truth"""
        # Primary check: malicious vs benign
        return prediction.is_malicious == ground_truth.is_malicious

    def _make_routing_decision(self, confidence: float) -> str:
        """
        Make routing decision based on confidence threshold

        Returns:
            'accept', 'escalate', or 'reject'
        """
        if confidence >= self.routing_thresholds["accept_threshold"]:
            return "accept"
        elif confidence >= self.routing_thresholds["escalate_threshold"]:
            return "escalate"
        else:
            return "reject"

    def _save_result(self, result: BaselineTestResult):
        """Save single result to JSONL file"""
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + "\n")


def main():
    """Main entry point for baseline testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline IDS uncertainty test")
    parser.add_argument("--model", default="groq-llama-70b", help="Model name")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--iterations", type=int, default=1, help="Iterations per sample")
    parser.add_argument("--simulation", action="store_true", help="Simulation mode (no API calls)")
    parser.add_argument("--generate-data", action="store_true", help="Generate mock dataset")
    parser.add_argument(
        "--dataset",
        default="easy",
        choices=["easy", "hard", "cicids", "cicids2018"],
        help="Dataset: 'easy' (100%% acc mock), 'hard' (60-80%% acc mock), 'cicids' (CIC-IDS2017), or 'cicids2018' (CIC-IDS2018)"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from existing run directory (e.g., scenario_results/baseline/run_20251022_120050)"
    )

    args = parser.parse_args()

    # Generate mock dataset if requested
    if args.generate_data:
        print("Generating mock dataset...")
        generate_mock_dataset(n_samples=args.samples)

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
        print("No samples loaded. Run with --generate-data to create mock dataset.")
        return

    # Run baseline test
    runner = BaselineRunner(
        model_name=args.model,
        dataset_config=dataset_config,
        simulation_mode=args.simulation,
        resume_from=args.resume_from
    )

    results = runner.run_test(
        samples=samples,
        iterations=args.iterations,
        delay=0.5 if not args.simulation else 0
    )

    # Print summary (only if we have new results)
    if results:
        print("\nTest Summary:")
        print(f"  Total tests: {len(results)}")
        correct = sum(1 for r in results if r.is_correct)
        print(f"  Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

        avg_confidence = sum(r.prediction.get('confidence_score', 0) for r in results) / len(results)
        print(f"  Average confidence: {avg_confidence:.3f}")

        routing_counts = {}
        for r in results:
            routing_counts[r.routing_decision] = routing_counts.get(r.routing_decision, 0) + 1
        print(f"  Routing decisions: {routing_counts}")
    else:
        print("\nNo new results (all samples were skipped or already completed)")


if __name__ == "__main__":
    main()
