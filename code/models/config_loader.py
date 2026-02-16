#!/usr/bin/env python3
"""
Configuration loader for IDS uncertainty estimation experiments.
Provides single source of truth for dataset and model configurations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Loads and manages experiment configuration from config.yaml
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize config loader

        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                "Please ensure config.yaml exists in the current directory."
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config.yaml: {e}")

    def get_dataset_config(self, display_name: str) -> Optional[Dict[str, Any]]:
        """
        Get dataset configuration by display name

        Args:
            display_name: Human-readable name (e.g., 'Mock IDS Hard', 'CIC-IDS2017')

        Returns:
            Dataset configuration dictionary or None if not found
        """
        config_key = self.config.get('display_names', {}).get('datasets', {}).get(display_name)
        if not config_key:
            return None
        return self.config.get('datasets', {}).get(config_key)

    def get_dataset_by_cli_param(self, cli_param: str) -> Optional[Dict[str, Any]]:
        """
        Get dataset configuration by CLI parameter

        Args:
            cli_param: CLI parameter value (e.g., 'easy', 'hard', 'cicids')

        Returns:
            Dataset configuration dictionary or None if not found
        """
        for dataset_config in self.config.get('datasets', {}).values():
            if dataset_config.get('cli_param') == cli_param:
                return dataset_config
        return None

    def get_model_config(self, display_name: str) -> Optional[Dict[str, Any]]:
        """
        Get model configuration by display name

        Args:
            display_name: Human-readable name (e.g., 'Qwen', 'Llama4S')

        Returns:
            Model configuration dictionary or None if not found
        """
        config_key = self.config.get('display_names', {}).get('models', {}).get(display_name)
        if not config_key:
            return None
        return self.config.get('llms', {}).get(config_key)

    def get_model_by_cli_param(self, cli_param: str) -> Optional[Dict[str, Any]]:
        """
        Get model configuration by CLI parameter

        Args:
            cli_param: CLI parameter value (e.g., 'groq-qwen', 'groq-llama4-scout')

        Returns:
            Model configuration dictionary or None if not found
        """
        return self.config.get('llms', {}).get(cli_param)

    def dataset_file_exists(self, display_name: str) -> bool:
        """
        Check if dataset file exists

        Args:
            display_name: Human-readable dataset name

        Returns:
            True if file exists, False otherwise
        """
        dataset_config = self.get_dataset_config(display_name)
        if not dataset_config:
            return False

        file_path = self.config_path.parent / dataset_config['file']
        return file_path.exists()

    def get_experiment_type_mapping(self, display_name: str) -> Optional[str]:
        """
        Get experimental code term from manuscript/display name

        Args:
            display_name: Manuscript term (e.g., 'Multi-Prompt Voting', 'Baseline')

        Returns:
            Experimental code term (e.g., 'ensemble', 'baseline') or None
        """
        return self.config.get('display_names', {}).get('experiment_types', {}).get(display_name)

    def list_available_datasets(self) -> Dict[str, str]:
        """
        List all available datasets with their CLI parameters

        Returns:
            Dictionary mapping display names to CLI parameters
        """
        result = {}
        display_names = self.config.get('display_names', {}).get('datasets', {})
        for display_name, config_key in display_names.items():
            dataset_config = self.config.get('datasets', {}).get(config_key)
            if dataset_config:
                result[display_name] = dataset_config.get('cli_param', 'unknown')
        return result

    def list_available_models(self) -> Dict[str, str]:
        """
        List all available models with their CLI parameters

        Returns:
            Dictionary mapping display names to CLI parameters
        """
        result = {}
        display_names = self.config.get('display_names', {}).get('models', {})
        for display_name, config_key in display_names.items():
            model_config = self.config.get('llms', {}).get(config_key)
            if model_config:
                result[display_name] = model_config.get('cli_param', config_key)
        return result

    def validate_dataset(self, cli_param: str) -> tuple[bool, str]:
        """
        Validate dataset CLI parameter

        Args:
            cli_param: Dataset CLI parameter

        Returns:
            Tuple of (is_valid, message)
        """
        dataset_config = self.get_dataset_by_cli_param(cli_param)
        if not dataset_config:
            available = list(self.list_available_datasets().values())
            return False, f"Unknown dataset: {cli_param}. Available: {', '.join(available)}"

        file_path = self.config_path.parent / dataset_config['file']
        if not file_path.exists():
            return False, f"Dataset file not found: {dataset_config['file']}"

        return True, f"Dataset ready: {dataset_config['file']}"

    def validate_model(self, cli_param: str) -> tuple[bool, str]:
        """
        Validate model CLI parameter

        Args:
            cli_param: Model CLI parameter

        Returns:
            Tuple of (is_valid, message)
        """
        model_config = self.get_model_by_cli_param(cli_param)
        if not model_config:
            available = list(self.list_available_models().values())
            return False, f"Unknown model: {cli_param}. Available: {', '.join(available)}"

        return True, f"Model ready: {model_config['model']}"
