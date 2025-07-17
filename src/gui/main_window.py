import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import os
import sys
import threading

class TextRedirector:
    """
    Redirects stdout to a tkinter Text widget for real-time console output display.
    
    This class allows the GUI to show console output from the ML pipeline
    in a scrollable text area within the application window.
    """
    
    def __init__(self, text_widget):
        """
        Initialize the text redirector.
        
        Args:
            text_widget (tkinter.Text): The Text widget to redirect output to.
        """
        self.text_widget = text_widget
        self.buffer = ""
        self.original_stdout = sys.stdout

    def write(self, string: str):
        """
        Write string to both console and GUI text widget.
        
        Args:
            string (str): Text to write to outputs.
        """
        self.original_stdout.write(string)  # Still write to console
        self.buffer += string
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)  # Auto-scroll to the end
        self.text_widget.update_idletasks()  # Force update to show text immediately

    def flush(self):
        """Flush the output streams."""
        self.original_stdout.flush()
        pass

class MainWindow:
    """
    Main application window for wav2scape.
    
    Provides the GUI interface for selecting audio files, running experiments,
    and displaying results. Handles file validation, experiment execution,
    and result saving workflows.
    """
    
    def __init__(self, root, controller):
        """
        Initialize the main window.
        
        Args:
            root (tkinter.Tk): The root tkinter window.
            controller (ExperimentController): Controller for experiment execution.
        """
        self.root = root
        self.controller = controller
        self.audio_files = []
        self.audio_extensions = ('.wav', '.mp3', '.ogg', '.flac')
        
        self._create_widgets()
        self._create_context_menu()
        
    def _create_widgets(self):
        # App title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(pady=20)
        
        title_label = ttk.Label(title_frame, text="wav2scape", font=("Arial", 35, "bold"))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="From Raw Audio to Distances", font=("Arial", 20, "bold"))
        subtitle_label.pack()
        
        # File selection options
        self.selection_frame = ttk.LabelFrame(self.root, text="Select Audio Files")
        self.selection_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Center the folder button by using a container frame
        button_container = ttk.Frame(self.selection_frame)
        button_container.pack(fill=tk.X, pady=5)
        
        self.folder_button = ttk.Button(button_container, text="Select Folder", command=self._browse_folder)
        self.folder_button.pack(padx=5, pady=5, anchor=tk.CENTER)
        
        # Files list
        self.files_frame = ttk.LabelFrame(self.root, text="Selected Audio Files")
        self.files_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.tree = ttk.Treeview(self.files_frame, columns=("Filename", "CategoryA", "CategoryB"), show="headings")
        self.tree.heading("Filename", text="Filename")
        self.tree.heading("CategoryA", text="CategoryA")
        self.tree.heading("CategoryB", text="CategoryB")
        self.tree.column("Filename", width=370)
        self.tree.column("CategoryA", width=100)
        self.tree.column("CategoryB", width=100)
        
        scrollbar = ttk.Scrollbar(self.files_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add right-click binding to the treeview - use multiple bindings for Mac compatibility
        self.tree.bind("<Button-2>", self._show_context_menu)  # macOS right-click or two-finger click
        self.tree.bind("<Button-3>", self._show_context_menu)  # Standard right-click
        self.tree.bind("<Control-Button-1>", self._show_context_menu)  # Control+click on Mac
        
        # Add keyboard shortcut for deletion
        self.tree.bind("<Delete>", self._remove_file)
        self.tree.bind("<BackSpace>", self._remove_file)
        # Common Mac deletion shortcut
        self.tree.bind("<Command-BackSpace>", self._remove_file)

        # Run button
        self.run_button = ttk.Button(
            self.root, 
            text="Run Experiment", 
            command=self._run_experiment,
            state=tk.DISABLED  # Initially disabled
        )
        self.run_button.pack(pady=20)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="")
        self.status_label.pack(pady=5)
        
        # Terminal output area
        terminal_frame = ttk.LabelFrame(self.root, text="Terminal Output")
        terminal_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.terminal = scrolledtext.ScrolledText(terminal_frame, wrap=tk.WORD, height=10, 
                                                 background="black", foreground="white")
        self.terminal.pack(fill=tk.BOTH, expand=True)
        
        # Redirect stdout to our terminal widget
        self.redirector = TextRedirector(self.terminal)
        sys.stdout = self.redirector
    
    def _create_context_menu(self):
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Delete File", command=self._remove_file)
    
    def _show_context_menu(self, event):
        try:
            # Get the item at the current mouse position
            clicked_item = self.tree.identify_row(event.y)
            
            if not clicked_item:
                return
                
            # If there are already selected items and the clicked item is not among them
            # add the new item to the selection without clearing the previous selection
            if self.tree.selection() and clicked_item not in self.tree.selection():
                # Keep previous selection and add the clicked item
                current_selection = list(self.tree.selection())
                current_selection.append(clicked_item)
                for item in current_selection:
                    self.tree.selection_add(item)
            elif not self.tree.selection():
                # If nothing is selected, select the clicked item
                self.tree.selection_set(clicked_item)
            
            # Count selected items
            selected_count = len(self.tree.selection())
            
            # Update context menu label based on selection count
            if selected_count > 1:
                self.context_menu.entryconfigure(0, label=f"Delete Files ({selected_count})")
            else:
                self.context_menu.entryconfigure(0, label="Delete File")
                
            # Show the popup menu
            self.context_menu.post(event.x_root, event.y_root)
        finally:
            # Make sure to release the grab
            self.context_menu.grab_release()
    
    def _remove_file(self, event=None):
        """Remove selected file(s) from the list"""
        selection = self.tree.selection()
        if not selection:
            return
        
        # Store selected items and their values
        to_remove = []
        for item in selection:
            # Get filename to match with audio_files list
            filename = self.tree.item(item, "values")[0]  # First column is the filename
            path = None
            
            # Find the full path in audio_files by matching basename
            for file_path in self.audio_files:
                if os.path.basename(file_path) == filename:
                    path = file_path
                    break
            
            if path:
                to_remove.append((item, path))
        
        # Remove items from tree and list
        for item, path in to_remove:
            self.tree.delete(item)
            self.audio_files.remove(path)
        
        self._update_run_button()
    
    def _find_audio_files(self, folder: str) -> list[str]:
        """
        Recursively find all audio files in folder and subfolders.
        
        Args:
            folder (str): Path to the folder to search.
        
        Returns:
            list[str]: List of paths to found audio files.
        """
        found_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(self.audio_extensions):
                    found_files.append(os.path.join(root, file))
        return found_files
    
    def _browse_folder(self):
        """
        Open folder dialog to select the folder containing audio files.
        
        Updates the file list with all audio files found in the selected folder
        and its subfolders.
        """
        folder = filedialog.askdirectory(title="Select Folder Containing Audio Files")
        
        if folder:
            print(f"\nLoading audio files from folder: {folder}")
            found_files = self._find_audio_files(folder)
            print(f"... Found {len(found_files)} audio files in folder and subfolders")  # Uncommented line
            self._update_files_list(new_files=found_files)
            
    def _extract_categoryB_id(self, file_path: str) -> str:
        """
        Extract Category B identifier from audio filename.
        
        Args:
            file_path (str): Path to the audio file.
        
        Returns:
            str: Category B identifier (last part of filename before extension).
        """
        filename = os.path.basename(file_path)
        parts = filename.split(".")[0].split("_")
        return parts[-1]

    def _extract_categoryA_id(self, file_path: str) -> str:
        """
        Extract Category A identifier from audio filename.
        
        Args:
            file_path (str): Path to the audio file.
        
        Returns:
            str: Category A identifier (second to last part of filename before extension).
        """
        filename = os.path.basename(file_path)
        parts = filename.split(".")[0].split("_")
        return parts[-2]

    def _update_files_list(self, new_files=None):
        if new_files:
            self.audio_files.extend(new_files)
        
        # Clear existing items and repopulate
        self.tree.delete(*self.tree.get_children())
        for audio_file in self.audio_files:
            categoryA_id = self._extract_categoryA_id(audio_file)
            categoryB_id = self._extract_categoryB_id(audio_file)
            self.tree.insert("", tk.END, values=(os.path.basename(audio_file), categoryA_id, categoryB_id))
        
        self._update_run_button()

    def _update_run_button(self):
        """Update the run button state and status message based on selected audio files."""
        category_a_set = set()
        category_b_set = set()
        identifier_set = set()
        
        for audio_file in self.audio_files:
            categoryA_id = self._extract_categoryA_id(audio_file)
            categoryB_id = self._extract_categoryB_id(audio_file)
            identifier = f"{categoryA_id}_{categoryB_id}"  # unique combination
            
            category_a_set.add(categoryA_id)
            category_b_set.add(categoryB_id)
            identifier_set.add(identifier)
        
        num_files = len(self.audio_files)
        num_category_a = len(category_a_set)
        num_category_b = len(category_b_set)
        num_identifiers = len(identifier_set)
        
        if num_files > 0 and num_identifiers >= 2:  # we need at least 2 identifiers
            self.run_button.config(state=tk.NORMAL)
            self.status_label.config(
                text=f"Found {num_files} audio files: "
                     f"{num_category_a} CategoryA types, {num_category_b} CategoryB types and "
                     f"{num_identifiers} unique combinations (CategoryA_CategoryB)."
            )
        else:
            self.run_button.config(state=tk.DISABLED)
            if num_files == 0:
                self.status_label.config(text="Please select a folder containing audio files")
            else:
                self.status_label.config(
                    text=f"Need at least 2 unique combinations (CategoryA_CategoryB) to run experiment. "
                         f"Currently found {num_identifiers} combination{'s' if num_identifiers != 1 else ''}."
                )
    
    def _run_experiment(self):
        """Run the experiment with selected audio files"""
        if not self.audio_files:
            messagebox.showwarning("No Files", "No audio files selected.")
            return
        
        # Show processing indicator
        self.status_label.config(text="Processing... Please wait.")
        self.root.update()
        
        print("\n--- Starting experiment ---")
        
        # Run the experiment in a separate thread to keep the GUI responsive
        def run_experiment_thread():
            # Run the experiment through the controller
            success = self.controller.run_experiment(self.audio_files)
            
            # Update GUI from the main thread
            self.root.after(0, lambda: self._handle_experiment_result(success))
        
        thread = threading.Thread(target=run_experiment_thread)
        thread.daemon = True
        thread.start()
    
    def _handle_experiment_result(self, success):
        """Handle the experiment result (called from the main thread)"""
        if success:
            outputs_dir = self.controller.dirs["outputs_dir"]
            print(f"Experiment completed successfully!")
            print(f"Results saved to: {outputs_dir}")
            
            # 1. Notify user experiment is done and wait for acknowledgment
            messagebox.showinfo("Success", "Experiment completed successfully!")
            
            # 2. Ask user for experiment name
            experiment_name = simpledialog.askstring("Experiment Name", 
                                                   "Please enter a name for this experiment:")
            
            if not experiment_name:
                print("\nClosing wav2scape...")
                self._show_closing_progress_bar()
                return
                
            # 3. Ask user where to store the results
            destination = filedialog.askdirectory(title="Select folder to save experiment results")
            if destination:
                try:
                    # Create a folder with the experiment name
                    experiment_folder = os.path.join(destination, experiment_name)
                    os.makedirs(experiment_folder, exist_ok=True)
                    
                    # Copy all files from outputs_dir to the experiment folder
                    print(f"Saving results to: {experiment_folder}")
                    for item in os.listdir(outputs_dir):
                        source_path = os.path.join(outputs_dir, item)
                        if os.path.isdir(source_path):
                            # For directories like 'images', copy the whole directory
                            dest_dir = os.path.join(experiment_folder, item)
                            import shutil
                            shutil.copytree(source_path, dest_dir, dirs_exist_ok=True)
                        else:
                            # For individual files
                            shutil.copy2(source_path, experiment_folder)
                    
                    print(f"Results saved successfully to: {experiment_folder}")
                    self.status_label.config(text=f"Experiment completed. Results saved to {experiment_folder}")
                    messagebox.showinfo("Success", f"Experiment saved to {experiment_folder}")
                    
                    self._open_folder(experiment_folder)
                    
                    # Auto-close with progress bar
                    self._show_closing_progress_bar()
                    
                except Exception as e:
                    error_msg = f"Failed to save results: {e}"
                    print(error_msg)
                    messagebox.showerror("Error", error_msg)
                    self.status_label.config(text=f"Experiment completed. Results in original location: {outputs_dir}")
                    self._show_closing_progress_bar()
            else:
                # User canceled the folder selection
                self.status_label.config(text=f"Experiment completed. Results in original location: {outputs_dir}")
                # Auto-close with progress bar
                self._show_closing_progress_bar()
        else:
            error_msg = "Experiment failed. Please check the files and try again."
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg)
            
    def _show_closing_progress_bar(self):
        """Show a progress bar and close the application after it completes"""
        # Create a top-level window for the progress bar
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Closing...")
        progress_window.geometry("300x100")
        progress_window.resizable(False, False)
        
        # Make it stay on top and remove decorations
        progress_window.attributes("-topmost", True)
        
        # Center the progress window relative to the main window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 150
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 50
        progress_window.geometry(f"+{x}+{y}")
        
        # Add a label
        ttk.Label(progress_window, text="Closing wav2scape...", 
                 font=("Arial", 12)).pack(pady=(15, 5))
        
        # Add progress bar
        progress = ttk.Progressbar(progress_window, orient=tk.HORIZONTAL, 
                                  length=250, mode='determinate')
        progress.pack(pady=5, padx=25)
        
        close_delay_ms = 1250  # 1 second delay
        
        def update_progress(current_ms=0):
            if current_ms <= close_delay_ms:
                # Calculate percentage and update progress bar
                progress['value'] = (current_ms / close_delay_ms) * 100
                progress_window.update()
                
                # Schedule the next update after 50ms
                progress_window.after(50, lambda: update_progress(current_ms + 50))
            else:
                # When progress is complete, close the application
                print("Closing wav2scape...")
                self.root.destroy()
        
        # Start the progress update
        update_progress()
    
    def _open_folder(self, path):
        """Open a folder in the system's file explorer"""
        if not os.path.exists(path):
            print(f"Cannot open folder: {path} does not exist")
            return
            
        try:
            import platform
            system = platform.system()
            
            if system == 'Darwin':  # macOS
                import subprocess
                subprocess.Popen(['open', path])
            elif system == 'Windows':  # Windows
                import subprocess
                subprocess.Popen(['explorer', path])
            elif system == 'Linux':  # Linux
                import subprocess
                subprocess.Popen(['xdg-open', path])
            else:
                print(f"Unsupported operating system: {system}")
        except Exception as e:
            print(f"Failed to open folder: {e}")

