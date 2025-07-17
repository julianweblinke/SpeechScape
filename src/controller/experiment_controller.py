from ml import (
    AudioProcessor,
    CodebookAnalyzer, 
    SimilarityMatrix, 
    MLPipeline
)
from filehandler import ExperimentSaver
from utils import get_logger

# Create module-specific logger
logger = get_logger(__name__)  

class ExperimentController:
    """Controller that connects the GUI with the ML pipeline"""
    
    def __init__(self, dirs: dict):
        """
        Initialize the experiment controller with ML components.
        
        Args:
            dirs (dict): Dictionary containing application directories:
                - cache_dir: Directory for caching model files
                - outputs_dir: Directory for experiment outputs
                - images_dir: Directory for generated images
                - logs_dir: Directory for log files
        """
        self.dirs = dirs
        # Create AudioProcessor without loading model initially
        logger.debug(f"... (ExperimentController.__init__): cache_dir: {dirs['cache_dir']}")
        self.audio_processor = AudioProcessor(cache_dir=dirs["cache_dir"], skip_model_loading=True)
        self.codebook_analyzer = CodebookAnalyzer()
        self.similarity_matrix = SimilarityMatrix()
        self.experiment_saver = ExperimentSaver()
        
        # Create ML pipeline
        self.ml_pipeline = MLPipeline(
            self.audio_processor,
            self.codebook_analyzer,
            self.similarity_matrix
        )
    
    def initialize_audio_model(self):
        """
        Initialize the Wav2Vec2 audio model for processing.
        
        This method downloads and loads the pre-trained Wav2Vec2 model if not already cached.
        Called from the installation dialog during application startup.
        """
        self.audio_processor.initialize_model()
    
    def run_experiment(self, audio_files: list[str]) -> bool:
        """
        Run the complete speech analysis experiment on provided audio files.
        
        Args:
            audio_files (list[str]): List of paths to audio files to analyze.
                                   Files should follow naming convention: ``*_categoryA_categoryB.ext``
        
        Returns:
            bool: True if experiment completed successfully, False otherwise.
        """
        if not self._validate_files(audio_files):
            return False, "Invalid files. Please check file naming and provide files for at least 2 categoryA_ids and 2 categoryB_ids."
        
        # Process the files through the ML pipeline
        results = self.ml_pipeline.process_files(audio_files)

        # save the experiment results
        self.experiment_saver.process_results(results, self.dirs)
            
        return True
    
    def _validate_files(self, audio_files: list[str]) -> bool:
        """
        Validate that audio files meet format requirements.
        
        Args:
            audio_files (list[str]): List of audio file paths to validate.
        
        Returns:
            bool: True if all files have valid extensions, False otherwise.
        """
        # Check file extensions
        valid_extensions = ('.wav', '.mp3', '.ogg', '.flac')
        if not all(file.lower().endswith(valid_extensions) for file in audio_files):
            return False
        return True
