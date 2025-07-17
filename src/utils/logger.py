# src/utils/logger.py
import logging

def configure_logging(verbose: bool = False):
    """
    Configure root logger with console output and appropriate verbosity level.
    
    Args:
        verbose (bool): If True, set logging level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(console_handler)
    
    return root_logger

def get_logger(name: str = None):
    """
    Get a module-specific logger instance.
    
    Args:
        name (str): Logger name, typically __name__ from calling module.
    
    Returns:
        logging.Logger: Logger instance for the specified module.
    """
    return logging.getLogger(name)
