import numpy as np
from collections import defaultdict, Counter
import os
import json

class CodebookAnalyzer:
    """
    Analyzes codebook usage patterns from Wav2Vec2 quantized representations.
    
    This class processes lists of codebook indices to generate statistics
    about how different discrete speech units are used across audio files.
    """
    
    def analyze_category_usage(self, ListUsedCodebookIndicesPerFile: list) -> tuple[np.ndarray, dict]:
        """
        Analyze codebook usage patterns for a category.
        
        Args:
            ListUsedCodebookIndicesPerFile (list): List of PyTorch tensors containing codebook indices
                                                 from multiple audio files of the same category.
        
        Returns:
            tuple: (codebook_indices_average, category_usage)
                - codebook_indices_average (np.ndarray): Normalized usage distribution over 102400 codebook entries
                - category_usage (dict): Statistical analysis containing:
                    - used_entries (list[int]): Sorted list of codebook indices that were used
                    - number_used_entries (list[int]): Count of unique codebook entries used
                    - most_used_entries (list[tuple]): Codebook entries sorted by usage frequency
                    - codebook_usage_vector (list[int]): Raw usage counts for all 102400 entries
                    - normalized_codebook_usage_vector (list[float]): Normalized usage distribution
        """
        # Convert tensors to numpy and flatten
        flattened_indices = np.concatenate(
            [tensor.cpu().numpy().ravel() for tensor in ListUsedCodebookIndicesPerFile]
        ).astype(int)

        # 1. Create usage vector
        usage_vector = np.zeros(102400, dtype=int)
        np.add.at(usage_vector, flattened_indices, 1)

        # 2. Calculate normalized version
        total_usage = np.sum(usage_vector)
        codebook_indices_average = usage_vector / total_usage if total_usage > 0 else usage_vector
        # print(np.sum(codebook_indices_average)) # sums to 1!

        # 3. Generate statistics
        non_zero_indices = np.nonzero(usage_vector)[0]
        usage_counts = usage_vector[non_zero_indices]
        
        category_usage = {
            "used_entries": sorted(non_zero_indices.tolist()),
            "number_used_entries": [len(non_zero_indices.tolist())],
            "most_used_entries": sorted(
                zip(non_zero_indices.tolist(), usage_counts.tolist()),
                key=lambda x: (-x[1], x[0])  # Sort by count descending, then index ascending
            ),
            "codebook_usage_vector": usage_vector.tolist(),
            "normalized_codebook_usage_vector": codebook_indices_average.tolist()
        }

        return codebook_indices_average, category_usage