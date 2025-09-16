"""
Shared logging utilities for the Squid Control System.

This module provides a centralized logging setup function that can be imported
by all components of the system, ensuring consistent logging configuration
across the entire application.
"""

import inspect
import logging
import logging.handlers
import os
from typing import Optional


def setup_logging(log_file: Optional[str] = None, max_bytes: int = 100000, backup_count: int = 3) -> logging.Logger:
    """
    Set up logging with both file and console handlers.

    Args:
        log_file: Path to the log file. If None, only console logging is used.
                 If relative path, it will be created in the logs/ directory.
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance

    Example:
        # Use default log file in logs/ directory
        logger = setup_logging()

        # Use custom log file
        logger = setup_logging("my_app.log")

        # Console-only logging
        logger = setup_logging(log_file=None)
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get logger for the calling module
    frame = inspect.currentframe().f_back
    module_name = frame.f_globals.get('__name__', 'unknown')
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Try to create file handler if log_file is specified
    if log_file is not None:
        try:
            # If log_file doesn't contain a path separator, put it in logs/ directory
            if os.sep not in log_file and '/' not in log_file:
                log_file = f"logs/{log_file}"

            # Ensure logs directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except (PermissionError, OSError) as e:
            print(f"Warning: Could not create log file {log_file}: {e}")  # noqa: T201
            print("Falling back to console-only logging")  # noqa: T201

    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
