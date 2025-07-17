import os
import platform

def get_app_data_dir(app_name: str = "wav2scape") -> str:
    """
    Get the platform-appropriate application data directory.
    
    Args:
        app_name (str): Name of the application for directory naming.
    
    Returns:
        str: Path to the application data directory, created if it doesn't exist.
    """
    if platform.system() == "Darwin":  # macOS
        data_dir = os.path.expanduser(f"~/Library/Application Support/{app_name}")
    elif platform.system() == "Windows":
        data_dir = os.path.join(os.environ["APPDATA"], app_name)
    else:  # Linux and others
        data_dir = os.path.expanduser(f"~/.local/share/{app_name}")
    
    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    return data_dir
    
def get_cache_dir() -> str:
    """
    Get the cache directory within the application data directory.
    
    Returns:
        str: Path to the cache directory for storing model files, created if it doesn't exist.
    """
    cache_dir = os.path.join(get_app_data_dir(), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
