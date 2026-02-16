import yaml
import shutil
import os
import re
from pathlib import Path
from typing import List, Set, Optional, Dict
import threading
import logging

# Configure logging
logger = logging.getLogger(__name__)

class DynamicEnumLoader:
    """
    Singleton class to load and manage dynamic enums for LLM outputs.
    Handles loading from YAML, validation, and atomic updates.
    """
    _instance = None
    _lock = threading.Lock()
    
    MANDATED_ATTACK_TYPES = {
        "benign", "dos", "ddos", "slowloris", "hulk", "goldeneye",
        "port_scan", "reconnaissance", "brute_force", "ftp_patator",
        "ssh_patator", "web_attack", "sql_injection", "xss",
        "infiltration", "botnet", "exfiltration", "heartbleed", "unknown"
    }

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DynamicEnumLoader, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
            
        self.config_path: Path = Path(__file__).parent / "config" / "enum_definitions.yaml"
        self._threat_levels: List[str] = []
        self._attack_types: List[str] = []
        self._aliases: Dict[str, str] = {}
        self.load_enums()
        self._initialized = True

    def load_enums(self) -> None:
        """
        Load enums from YAML configuration file.
        
        Handles file not found and parsing errors by falling back to defaults.
        Ensures mandated attack types are always present in memory.
        """
        if not self.config_path.exists():
            logger.critical(f"Config file not found at {self.config_path}. Using default fallback.")
            self._use_fallback_defaults()
            return
            
        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
                
            self._threat_levels = data.get('threat_levels', [])
            self._attack_types = data.get('attack_types', [])
            self._aliases = data.get('aliases', {})
            
            # Ensure mandated types are present in memory even if missing in file
            missing_mandated = self.MANDATED_ATTACK_TYPES - set(self._attack_types)
            if missing_mandated:
                logger.warning(f"Mandated attack types missing from config: {missing_mandated}. Adding them to memory.")
                self._attack_types.extend(list(missing_mandated))
                
        except Exception as e:
            logger.critical(f"Failed to load enums from {self.config_path}: {e}. Using default fallback.")
            self._use_fallback_defaults()

    def _use_fallback_defaults(self) -> None:
        """Set default values if loading fails."""
        self._threat_levels = ["benign", "low", "medium", "high", "critical"]
        self._attack_types = list(self.MANDATED_ATTACK_TYPES)
        self._aliases = {}

    def get_threat_levels(self) -> List[str]:
        """
        Return a copy of the current threat levels.
        
        Returns:
            List[str]: List of valid threat level strings.
        """
        return self._threat_levels.copy()

    def get_attack_types(self) -> List[str]:
        """
        Return a copy of the current attack types.
        
        Returns:
            List[str]: List of valid attack type strings.
        """
        return self._attack_types.copy()

    def resolve_alias(self, value: str) -> str:
        """
        Resolve value against configured aliases.
        
        Args:
            value (str): The value to resolve.
            
        Returns:
            str: The resolved value if an alias exists, otherwise the original value.
        """
        if not value:
            return value
        return self._aliases.get(value, value)

    def update_enums(self, new_attack_types: Optional[List[str]] = None, new_threat_levels: Optional[List[str]] = None) -> None:
        """
        Update enums with new values.
        Preserves mandated attack types.
        Uses atomic write strategy.
        
        Args:
            new_attack_types (Optional[List[str]]): New list of attack types.
            new_threat_levels (Optional[List[str]]): New list of threat levels.
            
        Raises:
            ValueError: If attempting to remove mandated attack types.
        """
        # Read current file content to preserve other fields if any
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    current_data = yaml.safe_load(f) or {}
            else:
                current_data = {}
        except Exception:
            current_data = {}

        # Update data
        if new_threat_levels is not None:
            current_data['threat_levels'] = new_threat_levels
            self._threat_levels = new_threat_levels
            
        if new_attack_types is not None:
            # Validation: Ensure mandated types are present
            provided_types = set(new_attack_types)
            if not self.MANDATED_ATTACK_TYPES.issubset(provided_types):
                missing = self.MANDATED_ATTACK_TYPES - provided_types
                raise ValueError(f"Cannot remove mandated attack types: {missing}")
            
            # Validation: Check format of new types (FR-012)
            # We only validate types that are NOT in the mandated list (just in case mandated ones violate rules, though they shouldn't)
            for at in new_attack_types:
                if at not in self.MANDATED_ATTACK_TYPES and not self._validate_enum_value(at):
                     logger.warning(f"Skipping invalid enum value: {at}")
                     continue
            
            current_data['attack_types'] = new_attack_types
            self._attack_types = new_attack_types

        # Atomic write
        temp_path = self.config_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                yaml.dump(current_data, f, default_flow_style=False, sort_keys=False)
            
            shutil.move(temp_path, self.config_path)
            logger.info(f"Successfully updated enums in {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to write enums to file: {e}")
            if temp_path.exists():
                os.remove(temp_path)
            raise

    def _validate_enum_value(self, value: str) -> bool:
        """
        Validate enum value format (FR-012).
        Alphanumeric plus underscores only, 1-100 character length.
        
        Args:
            value (str): The value to validate.
            
        Returns:
            bool: True if valid, False otherwise.
        """
        if not value or not isinstance(value, str):
            return False
        if not (1 <= len(value) <= 100):
            return False
        if not re.match(r'^[a-zA-Z0-9_]+$', value):
            return False
        return True

    def add_attack_type(self, attack_type: str) -> bool:
        """
        Add a single attack type if it doesn't exist.
        
        Args:
            attack_type (str): The new attack type to add.
            
        Returns:
            bool: True if added, False if already exists or invalid.
        """
        if not attack_type:
            return False
            
        if attack_type in self._attack_types:
            return False
            
        if not self._validate_enum_value(attack_type):
            logger.warning(f"Invalid attack type format: {attack_type}")
            return False
            
        new_list = self._attack_types + [attack_type]
        self.update_enums(new_attack_types=new_list)
        return True
