import tkinter as tk
from tkinter import ttk
import threading
import os
import sys
from utils import get_logger

logger = get_logger(__name__)

class InstallationDialog:
    """
    Dialog window that displays installation progress for wav2scape components.
    
    Shows a progress bar and status updates while downloading and initializing
    the Wav2Vec2 model during first-time setup or when models are not cached.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the installation dialog.
        
        Args:
            parent (tkinter.Widget, optional): Parent window for the dialog.
        """
        # Create the dialog window
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("wav2scape Initialization")
        self.root.geometry("500x175")
        self.root.resizable(False, False)
        
        # Center the window
        self.root.transient(parent) if parent else None
        self.root.grab_set() if parent else None
        
        # Make it stay on top
        self.root.lift()
        self.root.attributes('-topmost', True)
        
        # Variables
        self.installation_complete = False
        self.installation_error = None
        self.user_cancelled = False
        
        # Set up window close protocol
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        self._setup_ui()
        self._center_window()
        
    def _setup_ui(self):
        """Set up the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="wav2scape Initialization", 
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(
            main_frame,
            text="Initializing wav2scape...",
            font=("Arial", 10)
        )
        self.status_label.pack(pady=(0, 15))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=400
        )
        self.progress_bar.pack(pady=(0, 15))
        
        # Info label
        self.info_label = ttk.Label(
            main_frame,
            text="On first run, this may take a while as model files need to be downloaded (~1.3GB).\nSubsequent starts will be much faster.",
            font=("Arial", 9),
            foreground="gray",
            justify=tk.CENTER
        )
        self.info_label.pack(pady=(0, 10))
        
    def _center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
    def start_installation(self, installation_func, *args, **kwargs):
        """
        Start the installation process in a separate thread.
        
        Args:
            installation_func (callable): Function to execute for installation.
            *args: Positional arguments for installation function.
            **kwargs: Keyword arguments for installation function.
        """
        self.progress_bar.start(10)  # Start the indeterminate progress bar
        
        # Start installation in separate thread
        self.installation_thread = threading.Thread(
            target=self._run_installation,
            args=(installation_func, args, kwargs),
            daemon=True
        )
        self.installation_thread.start()
        
        # Check installation status periodically
        self._check_installation_status()
        
    def _run_installation(self, installation_func, args, kwargs):
        """Run the installation function"""
        try:
            logger.info("Starting model download...")
            self.root.after(0, lambda: self.status_label.config(text="Loading AI model components..."))
            
            # Run the installation function
            result = installation_func(*args, **kwargs)
            
            self.installation_complete = True
            logger.info("Model download completed successfully")
            
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            self.installation_error = str(e)
            
    def _check_installation_status(self):
        """Check if installation is complete"""
        if self.installation_complete:
            self._on_installation_complete()
        elif self.installation_error:
            self._on_installation_error()
        else:
            # Check again in 100ms
            self.root.after(100, self._check_installation_status)
            
    def _on_installation_complete(self):
        """Handle successful installation completion"""
        self.progress_bar.stop()
        self.status_label.config(text="wav2scape initialized!")
        self.info_label.config(text="wav2scape is ready to use. Opening main window...", foreground="gray")
        
        # Close dialog after a short delay
        self.root.after(2000, self._close_dialog)
        
    def _on_installation_error(self):
        """Handle installation error"""
        self.progress_bar.stop()
        self.status_label.config(text="Download failed!")
        self.info_label.config(
            text=f"Error: {self.installation_error}\nPlease check your internet connection and try again.",
            foreground="gray"
        )
        
    def _on_window_close(self):
        """Handle window close event (red X button)"""
        self._handle_cancellation()
        
    def _handle_cancellation(self):
        """Handle cancellation from any source"""
        logger.info("Installation cancelled by user")
        self.user_cancelled = True
        self.progress_bar.stop()
        
        # Force quit the application
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
        
        # Force exit the entire application
        os._exit(0)
        
    def _close_dialog(self):
        """Close the dialog"""
        try:
            self.root.quit()
        except:
            pass
        
    def show(self):
        """Show the dialog and wait for completion"""
        try:
            self.root.mainloop()
        except:
            pass
        finally:
            try:
                if hasattr(self, 'root'):
                    self.root.destroy()
            except:
                pass
                
        # If user cancelled, exit the application
        if self.user_cancelled:
            os._exit(0)
            
        return self.installation_complete and not self.installation_error
