# src/utils/__init__.py

# Import the functions from paths.py to make them available at the package level
from .paths import get_app_data_dir, get_cache_dir

# Import logging utilities
from .logger import configure_logging, get_logger

# Define what's exported when someone does "from utils import *"
__all__ = [
    'get_app_data_dir', 
    'get_cache_dir',
    'configure_logging',
    'get_logger'
]
