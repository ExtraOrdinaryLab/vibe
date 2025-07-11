"""
Logging utility module that provides functions for configuring and obtaining loggers.
"""
import sys
import logging
from pathlib import Path
from typing import Optional, Union, TextIO


def get_logger(
    name: Optional[str] = None,
    level: Union[int, str] = logging.INFO,
    fmt: Optional[str] = None,
    date_fmt: Optional[str] = "%Y-%m-%d %H:%M:%S",
    fpath: Optional[Union[str, Path]] = None,
    stream: Optional[TextIO] = sys.stdout,
    use_console: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger.
    
    Parameters:
        name: Logger name, defaults to the calling module's name
        level: Log level, can be a string or integer (e.g., "INFO", "DEBUG" or logging.INFO)
        fmt: Log format string
        date_fmt: Date format string
        fpath: Log file path, adds a file handler if provided
        stream: Console output stream, defaults to sys.stdout
        use_console: Whether to add a console handler
    
    Returns:
        Configured logging.Logger instance
    """
    # Set default log format
    if fmt is None:
        fmt = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"
    
    # Get the logger
    logger = logging.getLogger(name if name else __name__)
    
    # Set log level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(fmt, datefmt=date_fmt)
    
    # Add console handler
    if use_console:
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler (if file path is provided)
    if fpath is not None:
        # Ensure directory exists
        if isinstance(fpath, str):
            fpath = Path(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(fpath)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_log_level(logger: logging.Logger, level: Union[int, str]) -> None:
    """
    Dynamically set the logger's level.
    
    Parameters:
        logger: The logger to set the level for
        level: Log level, can be a string or integer (e.g., "INFO", "DEBUG" or logging.INFO)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)