"""
Configuration loader for Hybrid Multimodal Lab Assistant system.

Provides centralized configuration management with validation and defaults.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Return repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


class Config:
    """Configuration container with dot-notation access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Configuration parameters as nested dictionary
        """
        self._config = config_dict
        
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'detection.confidence_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config.get('detection.confidence_threshold')
            0.25
            >>> config.get('llm.model_name')
            'gpt-4o'
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._config
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file. If None, searches in:
                    1. Current directory
                    2. Parent directory
                    3. THESIS_CONFIG_DIR environment variable
                    
    Returns:
        Config object with loaded parameters
        
    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file has invalid YAML syntax
        
    Examples:
        >>> config = load_config('config.yaml')
        >>> config.get('detection.confidence_threshold')
        0.25
    """
    # Load .env once up front while keeping environment variable precedence intact.
    load_dotenv(dotenv_path=_project_root() / '.env', override=False)

    if config_path is None:
        # Search for config.yaml in common locations
        search_paths = [
            Path.cwd() / "config.yaml",
            Path.cwd().parent / "config.yaml",
            _project_root() / "config.yaml",
        ]
        
        # Check environment variable
        env_config_dir = os.getenv('THESIS_CONFIG_DIR')
        if env_config_dir:
            search_paths.insert(0, Path(env_config_dir) / "config.yaml")
        
        # Find first existing config
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                logger.info(f"Found config at: {config_path}")
                break
        else:
            raise FileNotFoundError(
                f"config.yaml not found in: {[str(p) for p in search_paths]}\n"
                "Provide config_path or set THESIS_CONFIG_DIR environment variable."
            )
    
    # Load YAML file
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")
    
    # Validate required sections
    required_sections = ['models', 'detection', 'classification', 'rag', 'llm', 'pipeline']
    missing = [section for section in required_sections if section not in config_dict]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")
    
    # Expand environment variables in paths
    config_dict = _expand_env_vars(config_dict)
    
    logger.info("Configuration loaded successfully")
    return Config(config_dict)


def _expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively expand environment variables in config values.
    
    Replaces ${VAR_NAME} with environment variable value.
    """
    if isinstance(config, dict):
        return {k: _expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Replace ${VAR} with environment variable
        if config.startswith('${') and config.endswith('}'):
            var_name = config[2:-1]
            return os.getenv(var_name, config)
        return config
    else:
        return config


def validate_config(config: Config) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration values are invalid
    """
    # Validate detection parameters
    conf_threshold = config.get('detection.confidence_threshold')
    if not (0.0 <= conf_threshold <= 1.0):
        raise ValueError(f"detection.confidence_threshold must be in [0.0, 1.0], got {conf_threshold}")
    
    # Validate classification parameters
    mc_passes = config.get('classification.mc_dropout_passes')
    if mc_passes < 1:
        raise ValueError(f"classification.mc_dropout_passes must be >= 1, got {mc_passes}")
    
    # Validate uncertainty thresholds
    low_conf = config.get('classification.uncertainty.low.min_confidence')
    med_conf = config.get('classification.uncertainty.medium.min_confidence')
    if low_conf <= med_conf:
        raise ValueError(
            f"LOW confidence threshold ({low_conf}) must be > MEDIUM threshold ({med_conf})"
        )
    
    # Validate RAG parameters
    top_k = config.get('rag.retrieval.top_k')
    if top_k < 1:
        raise ValueError(f"rag.retrieval.top_k must be >= 1, got {top_k}")
    
    # Validate LLM parameters
    temperature = config.get('llm.temperature')
    if not (0.0 <= temperature <= 2.0):
        raise ValueError(f"llm.temperature must be in [0.0, 2.0], got {temperature}")
    
    logger.info("Configuration validation passed")


def get_model_path(config: Config, model_name: str) -> Path:
    """
    Get absolute path for model file.
    
    Args:
        config: Configuration object
        model_name: Model identifier ('yolo_detection' or 'efficientnet_classification')
        
    Returns:
        Absolute path to model file
        
    Raises:
        ValueError: If model_name is invalid
    """
    model_path = config.get(f'models.{model_name}')
    if model_path is None:
        raise ValueError(f"Unknown model: {model_name}")

    configured = Path(model_path)
    root = _project_root()

    candidates = []

    # 1) Explicit configured path (absolute or repo-relative)
    candidates.append(configured if configured.is_absolute() else root / configured)

    # 2) Environment override directory by basename
    env_models_dir = os.getenv('THESIS_MODELS_DIR')
    if env_models_dir:
        candidates.append(Path(env_models_dir) / configured.name)

    # 3) Conventional repo model directory fallback by basename
    candidates.append(root / 'models' / configured.name)

    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Resolved model '{model_name}' to: {candidate}")
            return candidate

    searched = [str(c) for c in candidates]
    raise FileNotFoundError(
        f"Model file for '{model_name}' not found. Checked: {searched}. "
        "Set THESIS_MODELS_DIR or update config.yaml models.* paths."
    )
