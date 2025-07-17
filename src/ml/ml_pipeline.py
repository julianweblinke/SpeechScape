import os
import torch
from tqdm import tqdm
from utils import get_logger
import numpy as np
import time

# Create module-specific logger
logger = get_logger(__name__)  # Will be 'src.ml.ml_pipeline'

class MLPipeline:
    """
    Complete machine learning pipeline for speech analysis experiment.
    
    Coordinates audio processing, codebook analysis, and similarity matrix generation
    to produce comprehensive speech analysis results.
    """
    
    def __init__(self, audio_processor, codebook_analyzer, similarity_matrix):
        """
        Initialize the ML pipeline with processing components.
        
        Args:
            audio_processor (AudioProcessor): Handles audio file processing and feature extraction.
            codebook_analyzer (CodebookAnalyzer): Analyzes codebook usage patterns.
            similarity_matrix (SimilarityMatrix): Generates similarity matrices between categories.
        """
        self.audio_processor = audio_processor
        self.codebook_analyzer = codebook_analyzer
        self.similarity_matrix = similarity_matrix
        
    def process_files(self, audio_files: list[str]) -> dict:
        """
        Process a list of audio files through the complete analysis pipeline.
        
        Args:
            audio_files (list[str]): List of paths to audio files to process.
                                   Files should follow naming convention: ``*_categoryA_categoryB.ext``
        
        Returns:
            dict: Complete experiment results containing:
                - processed_files (list[str]): List of processed file paths
                - categoryA_ids (list[str]): Unique Category A identifiers found
                - categoryB_ids (list[str]): Unique Category B identifiers found
                - identifiers (list[str]): Unique identifier combinations (categoryA_categoryB)
                - data (dict): Analysis results with:
                    - identifier2CodebookUsagePerTime (dict): Raw codebook usage per file
                    - identifier2AverageCodebookUsage (dict): Average codebook usage per identifier
                    - similarity_matrix (np.ndarray): Similarity matrix between identifiers
                    - similarity_matrix_PCA (np.ndarray): PCA-reduced similarity matrix
                - log (dict): Analysis logs with:
                    - stats_identifier2AverageCodebookUsage (dict): Statistical summaries
                    - pca_info (dict): PCA analysis information
        """
        results = {
            "processed_files": [],
            "categoryA_ids": [],
            "categoryB_ids": [],
            "identifiers": [],
            "data": {
                "identifier2CodebookUsagePerTime": {},
                "identifier2AverageCodebookUsage": {},
                "similarity_matrix": {},
                "similarity_matrix_PCA": {},
            },
            "log": {
                "stats_identifier2AverageCodebookUsage": {},
                "pca_info": {}
            },
        }
        print(f"\nProcessing {len(audio_files)} audio files:")
        
        # identifier2CodebookUsagePerTime
        number_files = len(audio_files)
        start_time = time.time()
        for idx, audio_file in enumerate(audio_files): # calculate codebook usage of each audio_file
            current_time = time.time()
            elapsed = current_time - start_time
            avg_time_per_file = elapsed / (idx + 1)
            remaining_files = number_files - (idx + 1)
            est_remaining = avg_time_per_file * remaining_files

            # Format estimated remaining time as H:MM:SS
            est_remaining_str = time.strftime('%H:%M:%S', time.gmtime(est_remaining))

            print(
                f"... Process audio_file {idx+1}/{number_files} "
                f"({np.round((idx+1)/number_files*100,2)}%; ETA: {est_remaining_str}) "
                f"{audio_file}"
            )

            # Load and preprocess audio
            logger.debug(f"Start processing with audio_processor...")
            codebook_indices, file_info = self.audio_processor.process(audio_file)
            categoryA_id = file_info["categoryA_id"]
            categoryB_id = file_info["categoryB_id"]
            identifier = f"{categoryA_id}_{categoryB_id}" # this specifies category dots (e.g. 005M_RS vs 005M_CS)
            logger.debug(f"Collect file_infos (identifier: {identifier}) ...")

            # save results
            results["processed_files"].append(file_info["path"])
            if categoryA_id not in results["categoryA_ids"]:  results["categoryA_ids"].append(categoryA_id)
            if categoryB_id not in results["categoryB_ids"]:  results["categoryB_ids"].append(categoryB_id)
            if identifier not in results["data"]["identifier2CodebookUsagePerTime"]:
                results["data"]["identifier2CodebookUsagePerTime"][identifier] = []
            results["data"]["identifier2CodebookUsagePerTime"][identifier].append(codebook_indices)
        
        # identifier2AverageCodebookUsage
        print(f"\nCodebook Usage:")
        for identifier in results["data"]["identifier2CodebookUsagePerTime"].keys(): # statistics for each identifier (categoryB wrt categoryA)

            print(f"... Calculate average codebook usage for identifier: {identifier}")
            codebook_indices_average, category_usage = self.codebook_analyzer.analyze_category_usage(results["data"]["identifier2CodebookUsagePerTime"][identifier])

            # save results
            results["log"]["stats_identifier2AverageCodebookUsage"][identifier] = category_usage
            if identifier not in results["data"]["identifier2AverageCodebookUsage"]:
                results["data"]["identifier2AverageCodebookUsage"][identifier] = []
            results["data"]["identifier2AverageCodebookUsage"][identifier].append(codebook_indices_average)
        
        # Generate similarity matrix
        print("\nGenerate similarity matrix:")
        matrix, matrix_PCA, identifiers, pca_info = self.similarity_matrix.generate_matrix(results["data"]["identifier2AverageCodebookUsage"])
        results["data"]["similarity_matrix"] = matrix
        results["data"]["similarity_matrix_PCA"] = matrix_PCA
        results["identifiers"] = identifiers 
        results["log"]["pca_info"] = pca_info 

        return results
