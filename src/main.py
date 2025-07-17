import tkinter as tk
import os, sys
import multiprocessing
import shutil

from gui import MainWindow, InstallationDialog
from controller import ExperimentController
from utils import *
import argparse

def setup_environment() -> dict:
    """
    Set up application directories and environment configuration.
    
    Creates platform-appropriate directories for cache, outputs, images, and logs.
    Resets output directories to ensure clean experiment runs.
    
    Returns:
        dict: Dictionary containing application directories:
            - cache_dir (str): Directory for caching model files
            - outputs_dir (str): Directory for experiment outputs
            - images_dir (str): Directory for generated images
            - logs_dir (str): Directory for log files
    """
    cache_dir = get_cache_dir()
    
    # Create other necessary directories
    outputs_dir = os.path.join(get_app_data_dir(), "outputs") #e.g. MACOS: f"~/Library/Application Support/{app_name}/outputs"
    images_dir = os.path.join(outputs_dir, "images")
    logs_dir = os.path.join(outputs_dir, "logs")

    def reset_directory(dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)  # Delete the directory and all its contents
        os.makedirs(dir_path, exist_ok=True)  # Recreate the empty directory
    
    reset_directory(outputs_dir)
    reset_directory(images_dir)
    reset_directory(logs_dir)
    
    return {
        "cache_dir": cache_dir,
        "outputs_dir": outputs_dir,
        "images_dir": images_dir,
        "logs_dir": logs_dir
    }

def main():
    """
    Main application entry point.
    
    Handles command-line arguments, initializes logging, sets up the application environment,
    displays the installation dialog for model loading, and launches the main GUI.
    """
    parser = argparse.ArgumentParser(description='wav2scape: From Raw Audio to Distances')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Configure logging based on verbose flag
    configure_logging(args.verbose)
    
    # Get a logger for the main module
    logger = get_logger(__name__)
    
    # Your application logic here
    logger.info("Application starting")
    if args.verbose:
        logger.debug("Verbose mode is enabled")

    # dirs for each platform
    dirs = setup_environment()

    # Create controller (without initializing audio model yet)
    controller = ExperimentController(dirs)
    
    # Show installation dialog for model loading
    logger.info("Showing installation dialog...")
    installation_dialog = InstallationDialog()
    installation_dialog.start_installation(controller.initialize_audio_model)
    
    success = installation_dialog.show()
    
    if not success:
        logger.error("Installation failed or was cancelled")
        sys.exit(1)  # Exit with error code
    
    logger.info("Installation completed successfully")
    
    # Create main application window
    root = tk.Tk()
    root.title("wav2scape")
    root.geometry("900x700")
    
    # Initialize main window with controller
    app = MainWindow(root, controller)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    
    # pyinstaller fix
    multiprocessing.freeze_support()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nInstallation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)
