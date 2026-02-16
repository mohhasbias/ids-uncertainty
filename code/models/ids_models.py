#!/usr/bin/env python3
"""
Pydantic models for IDS uncertainty estimation
Defines structured response schemas for LLM-based intrusion detection
"""

from pydantic import BaseModel, Field, BeforeValidator
from typing import Optional, List, Dict, Annotated
from enum import Enum
from enum_loader import DynamicEnumLoader

# Initialize dynamic enum loader
loader = DynamicEnumLoader()

def validate_threat_level(v: str) -> str:
    """
    Validate and auto-register threat levels.
    
    Args:
        v (str): The threat level string to validate.
        
    Returns:
        str: The normalized threat level.
    """
    if not v or not isinstance(v, str):
        return v
    v_norm = v.lower().strip()
    if v_norm not in loader.get_threat_levels():
        # Auto-register new threat level
        loader.update_enums(new_threat_levels=loader.get_threat_levels() + [v_norm])
    return v_norm

def validate_attack_type(v: str) -> str:
    """
    Validate and auto-register attack types.
    
    Args:
        v (str): The attack type string to validate.
        
    Returns:
        str: The normalized and resolved attack type.
    """
    if not v or not isinstance(v, str):
        return v
    # Normalize: lowercase, strip, replace spaces/hyphens with underscores
    v_norm = v.lower().strip().replace(' ', '_').replace('-', '_')
    
    # Resolve alias
    v_resolved = loader.resolve_alias(v_norm)
    
    if v_resolved not in loader.get_attack_types():
        # Auto-register new attack type
        loader.add_attack_type(v_resolved)
    return v_resolved

# Dynamic types for Pydantic validation
DynamicThreatLevel = Annotated[str, BeforeValidator(validate_threat_level)]
DynamicAttackType = Annotated[str, BeforeValidator(validate_attack_type)]


class ThreatLevel(str, Enum):
    """
    Threat classification levels.
    Note: Used for constants/reference. Validation uses DynamicThreatLevel.
    """
    BENIGN = "benign"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(str, Enum):
    """
    Comprehensive attack types covering CIC-IDS2017/2018 taxonomy and common variations.
    Note: Used for constants/reference. Validation uses DynamicAttackType.
    """
    # Benign traffic
    BENIGN = "benign"

    # Denial of Service attacks
    DOS = "dos"
    DDOS = "ddos"
    SLOWLORIS = "slowloris"
    HULK = "hulk"
    GOLDEN_EYE = "goldeneye"

    # Reconnaissance and scanning
    PORT_SCAN = "port_scan"
    RECONNAISSANCE = "reconnaissance"

    # Brute force attacks
    BRUTE_FORCE = "brute_force"
    FTP_PATATOR = "ftp_patator"
    SSH_PATATOR = "ssh_patator"

    # Web-based attacks
    WEB_ATTACK = "web_attack"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"

    # Advanced threats and data exfiltration
    INFILTRATION = "infiltration"
    BOTNET = "botnet"
    EXFILTRATION = "exfiltration"
    HEARTBLEED = "heartbleed"

    # Unknown/uncategorized
    UNKNOWN = "unknown"


class IDSStructuredResponse(BaseModel):
    """
    Structured response schema for IDS classification with uncertainty

    This is the core model that all LLM responses must conform to.
    It captures both the classification decision and uncertainty estimates.
    """

    # Classification decision
    is_malicious: bool = Field(
        description="Binary classification: True if attack detected, False if benign"
    )

    threat_level: DynamicThreatLevel = Field(
        description="Severity classification of the detected threat"
    )

    attack_type: DynamicAttackType = Field(
        description="Specific type of attack detected (or benign)"
    )

    # Uncertainty quantification
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported confidence from 0.0 (no confidence) to 1.0 (full confidence)"
    )

    # Reasoning and explanation
    reasoning: str = Field(
        description="Explanation of classification decision and confidence assessment"
    )

    key_features: List[str] = Field(
        default_factory=list,
        description="List of network features that influenced the decision (e.g., 'high packet rate', 'unusual port')"
    )

    # Optional metadata
    uncertainty_factors: Optional[str] = Field(
        default=None,
        description="Explanation of what contributes to uncertainty in this prediction"
    )


class IDSEnsembleResponse(BaseModel):
    """
    Response model for ensemble-based uncertainty estimation
    Used in Method 2: Ensemble Sampling
    """

    predictions: List[IDSStructuredResponse] = Field(
        description="List of predictions from ensemble members"
    )

    ensemble_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Aggregate confidence based on ensemble agreement"
    )

    variance: float = Field(
        ge=0.0,
        description="Variance in predictions across ensemble members"
    )

    final_prediction: IDSStructuredResponse = Field(
        description="Aggregated final prediction from ensemble"
    )


class IDSMonteCarloResponse(BaseModel):
    """
    Response model for Monte Carlo Dropout uncertainty estimation
    Used in Method 3: Monte Carlo Dropout (simulated via repeated sampling)
    """

    samples: List[IDSStructuredResponse] = Field(
        description="Multiple samples from the same model with stochastic behavior"
    )

    mean_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean confidence across all Monte Carlo samples"
    )

    predictive_variance: float = Field(
        ge=0.0,
        description="Variance in predictions indicating epistemic uncertainty"
    )

    final_prediction: IDSStructuredResponse = Field(
        description="Final prediction based on mode or mean of samples"
    )


class UncertaintyMetrics(BaseModel):
    """
    Container for various uncertainty metrics
    Used for evaluation and comparison across methods
    """

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Primary confidence/certainty score"
    )

    entropy: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Shannon entropy of prediction distribution"
    )

    variance: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Variance across multiple predictions"
    )

    consistency: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Consistency score from repeated predictions"
    )

    composite_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Weighted combination of multiple uncertainty signals"
    )


class GroundTruth(BaseModel):
    """
    Ground truth labels from CIC-IDS dataset
    Used for evaluation and calibration

    Uses dynamic types to support novel attack types discovered in datasets
    or generated by LLMs during experiments.
    """

    is_malicious: bool = Field(
        description="True label: attack or benign"
    )

    attack_type: DynamicAttackType = Field(
        description="True attack type (uses dynamic enum system)"
    )

    threat_level: Optional[DynamicThreatLevel] = Field(
        default=None,
        description="Optional severity level if available in dataset"
    )


class EvaluationResult(BaseModel):
    """
    Evaluation metrics for a single prediction
    Compares prediction against ground truth
    """

    sample_id: str = Field(
        description="Unique identifier for the network sample"
    )

    prediction: IDSStructuredResponse = Field(
        description="LLM prediction with uncertainty"
    )

    ground_truth: GroundTruth = Field(
        description="True label from dataset"
    )

    is_correct: bool = Field(
        description="Whether the classification was correct"
    )

    uncertainty_metrics: UncertaintyMetrics = Field(
        description="Uncertainty quantification metrics"
    )

    routing_decision: str = Field(
        description="Automated routing decision: 'accept', 'escalate', or 'reject'"
    )


class MonteCarloTestResult(BaseModel):
    """
    Result entity for Monte Carlo uncertainty estimation (M=5 stochastic predictions)
    Stored in scenario_results/monte_carlo/run_YYYYMMDD_HHMMSS/monte_carlo_results.jsonl
    """
    sample_id: str = Field(description="Unique sample identifier")
    model_name: str = Field(description="LLM model identifier (e.g., 'groq-qwen')")
    iteration: int = Field(ge=0, description="Iteration number for multi-iteration runs")
    timestamp: str = Field(description="ISO 8601 timestamp")
    description: str = Field(description="Network flow description (input text)")
    ground_truth: GroundTruth = Field(description="Ground truth labels")
    mc_predictions: List[IDSStructuredResponse] = Field(
        description="Array of M stochastic predictions (default M=5)"
    )
    final_prediction: IDSStructuredResponse = Field(
        description="Aggregated prediction via majority voting"
    )
    consistency_score: float = Field(
        ge=0.0, le=1.0, description="Agreement rate with majority vote"
    )
    prediction_variance: float = Field(
        ge=0.0, description="Variance of confidence scores across predictions"
    )
    is_correct: bool = Field(
        description="Whether final prediction matches ground truth"
    )
    uncertainty_metrics: UncertaintyMetrics = Field(
        description="Uncertainty quantification metrics"
    )
    routing_decision: str = Field(
        description="Routing decision: 'accept', 'escalate', or 'reject'"
    )
    execution_time: float = Field(
        ge=0.0, description="Total execution time in seconds"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )


class HybridTestResult(BaseModel):
    """
    Result entity for Hybrid uncertainty estimation (K=3 variants × M=5 samples = 15 predictions)
    Stored in scenario_results/hybrid/run_YYYYMMDD_HHMMSS/hybrid_results.jsonl
    """
    sample_id: str = Field(description="Unique sample identifier")
    model_name: str = Field(description="LLM model identifier")
    iteration: int = Field(ge=0, description="Iteration number")
    timestamp: str = Field(description="ISO 8601 timestamp")
    description: str = Field(description="Network flow description")
    ground_truth: GroundTruth = Field(description="Ground truth labels")
    all_predictions: List[IDSStructuredResponse] = Field(
        description="Array of K×M predictions (default 3×5=15)"
    )
    final_prediction: IDSStructuredResponse = Field(
        description="Aggregated prediction via majority voting"
    )
    agreement_score: float = Field(
        ge=0.0, le=1.0, description="Voting consistency across all predictions"
    )
    prediction_variance: float = Field(
        ge=0.0, description="Variance of confidence scores"
    )
    is_correct: bool = Field(
        description="Whether final prediction matches ground truth"
    )
    uncertainty_metrics: UncertaintyMetrics = Field(
        description="Uncertainty quantification metrics"
    )
    routing_decision: str = Field(
        description="Routing decision: 'accept', 'escalate', or 'reject'"
    )
    execution_time: float = Field(
        ge=0.0, description="Total execution time in seconds"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )


class CompositeTestResult(BaseModel):
    """
    Result entity for Composite confidence calculation (post-processes Hybrid results)
    Stored in scenario_results/composite/run_YYYYMMDD_HHMMSS/composite_results.jsonl
    Uses penalty-based formula: C_composite = C_self - β·(1 - C_conf_consistency) - γ·(1 - C_resp_consistency)
    """
    sample_id: str = Field(description="Unique sample identifier (matches hybrid)")
    model_name: str = Field(description="LLM model (copied from hybrid)")
    timestamp: str = Field(description="Processing timestamp (NOT API call time)")
    ground_truth: GroundTruth = Field(
        description="Ground truth labels (copied from hybrid)"
    )
    composite_confidence: float = Field(
        ge=0.0, le=1.0, description="Penalty-based composite confidence (clamped to [0,1])"
    )
    composite_components: Dict[str, float] = Field(
        description="Breakdown: c_self, c_conf_consistency, c_resp_consistency"
    )
    final_prediction: IDSStructuredResponse = Field(
        description="Copied from hybrid (is_malicious, threat_level, attack_type)"
    )
    hybrid_run_reference: str = Field(
        description="Path to source hybrid JSONL file"
    )
    is_correct: bool = Field(
        description="Copied from hybrid"
    )
    routing_decision: str = Field(
        description="Recalculated with composite confidence"
    )
    execution_time: float = Field(
        ge=0.0, description="Processing time (computation only, no API)"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )
