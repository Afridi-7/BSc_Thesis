"""
Logging configuration for Hybrid Multimodal Lab Assistant system.

Provides consistent logging across all modules with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, creates timestamped file in logs/
        console: Whether to log to console
        log_format: Custom log format string. If None, uses default
        
    Returns:
        Configured root logger
        
    Examples:
        >>> logger = setup_logging(log_level="INFO")
        >>> logger.info("Pipeline started")
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
    else:
        # Create logs directory
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"pipeline_{timestamp}.log"
    
    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    root_logger.info(f"Logging initialized: level={log_level}, file={log_path}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Logger instance
        
    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


def set_module_log_level(module_name: str, level: str) -> None:
    """
    Set log level for a specific module.
    
    Args:
        module_name: Module name (e.g., 'src.detection')
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Examples:
        >>> set_module_log_level('src.detection', 'DEBUG')
    """
    logger = logging.getLogger(module_name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
