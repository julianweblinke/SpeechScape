import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import OrderedDict

class ExperimentSaver:
    """
    Handles saving and visualization of speech analysis experiment results.
    
    This class creates various plots, saves matrices, and organizes experiment
    outputs including similarity matrices, PCA visualizations, and statistical logs.
    """
    
    def __init__(self):
        """Initialize the ExperimentSaver with default plotting parameters."""
        self.FS = 10
        self.FIGSIZE = (10,8)

    def _plot_save_similarity_matrix(self, similarity_matrix: np.ndarray, 
                                     fixed_colorbar: bool = False, 
                                     filename: str = 'similarity_matrix.png'):
        """
        Create and save a heatmap visualization of the similarity matrix.
        
        Args:
            similarity_matrix (np.ndarray): NxN similarity matrix between identifiers.
            fixed_colorbar (bool): If True, use fixed colorbar range [0,1]; otherwise auto-scale.
            filename (str): Name of the output image file.
        """
        images_dir = self.output_dirs['images_dir']
        os.makedirs(images_dir, exist_ok=True)
        
        N = similarity_matrix.shape[0]
        print(f"... similarity_matrix.shape: {similarity_matrix.shape}")
        font_size = max(self.FS, self.FS * 10 / N)
        print(f"... font_size: {font_size}")

        plt.figure(figsize=self.FIGSIZE)
        if fixed_colorbar:
            sns.heatmap(similarity_matrix, 
                        annot=True, 
                        cmap='Reds', 
                        vmin=0, 
                        vmax=1, 
                        fmt='.2f',
                        xticklabels=self.identifiers, 
                        yticklabels=self.identifiers,
                        annot_kws={"size": font_size})
        else:
            sns.heatmap(similarity_matrix, 
                        annot=True, 
                        cmap='Reds', 
                        fmt='.2f',
                        xticklabels=self.identifiers, 
                        yticklabels=self.identifiers,
                        annot_kws={"size": font_size})
        plt.xticks(fontsize=font_size, rotation=90)
        plt.yticks(fontsize=font_size, rotation=0)
        plt.tight_layout()
        save_path = os.path.join(images_dir, filename)
        print(f"... Save {save_path}")
        plt.savefig(save_path)
        plt.close()
        
    def _plot_save_PCA_scatter(self, PCA_matrix: np.ndarray):
        """
        Create and save PCA scatter plots with category annotations.
        
        Args:
            PCA_matrix (np.ndarray): NxM matrix with PCA-reduced dimensions (M=2 or 3).
        """
        images_dir = self.output_dirs['images_dir']
        os.makedirs(images_dir, exist_ok=True)

        N = PCA_matrix.shape[1]
        print(f"... PCA_matrix.shape: {PCA_matrix.shape}")
        MS = 750 # markersize
        font_size = self.FS * 2
        print(f"... font_size: {font_size}")

        # Extract identifiers and create color mapping
        self.categoryA_ids = [id.split("_")[1] for id in self.identifiers]
        self.categoryB_ids = [id.split("_")[0] for id in self.identifiers]

        def _create_plotAB(x_idx, y_idx):
            """Helper function to create scatter plots colored by Category A."""
            x_label, y_label = f'PCA{x_idx+1}', f'PCA{y_idx+1}'
            plt.figure(figsize=self.FIGSIZE)

            unique_cats = list(set(self.categoryA_ids))
            color_map = {cat: col for cat, col in zip(unique_cats, sns.color_palette("husl", len(unique_cats)))}
            
            # Create scatter plot with annotations
            for i, (x, y) in enumerate(PCA_matrix[:, [x_idx, y_idx]]):
                plt.scatter(x, y, color=color_map[self.categoryA_ids[i]], alpha=1, s=MS, label=self.categoryA_ids[i])
                plt.text(x, y, 
                        self.categoryB_ids[i], 
                        fontsize=font_size, 
                        horizontalalignment='center', 
                        verticalalignment='center')

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles)) 
            plt.legend(by_label.values(), by_label.keys(), fontsize=font_size)
            plt.axhline(0, c='gray', ls='-', lw=1)
            plt.axvline(0, c='gray', ls='-', lw=1)
            plt.xlabel(x_label, fontsize=font_size)
            plt.ylabel(y_label, fontsize=font_size)
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.grid()
            plt.tight_layout()
            save_path = os.path.join(images_dir, f'AB_{x_label}_vs_{y_label}_scatter.png')
            print(f"... Save {save_path}")
            plt.savefig(save_path)
            plt.close()

        def _create_plotBA(x_idx, y_idx):
            """Helper function to create scatter plots colored by Category B."""
            x_label, y_label = f'PCA{x_idx+1}', f'PCA{y_idx+1}'
            plt.figure(figsize=self.FIGSIZE)

            unique_cats = list(set(self.categoryB_ids))
            color_map = {cat: col for cat, col in zip(unique_cats, sns.color_palette("husl", len(unique_cats)))}
            
            # Create scatter plot with annotations
            for i, (x, y) in enumerate(PCA_matrix[:, [x_idx, y_idx]]):
                plt.scatter(x, y, color=color_map[self.categoryB_ids[i]], alpha=1, s=MS, label=self.categoryB_ids[i])
                plt.text(x, y, 
                        self.categoryA_ids[i], 
                        fontsize=font_size, 
                        horizontalalignment='center', 
                        verticalalignment='center')

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles)) 
            plt.legend(by_label.values(), by_label.keys(), fontsize=font_size)
            plt.axhline(0, c='gray', ls='-', lw=1)
            plt.axvline(0, c='gray', ls='-', lw=1)
            plt.xlabel(x_label, fontsize=font_size)
            plt.ylabel(y_label, fontsize=font_size)
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.grid()
            plt.tight_layout()
            save_path = os.path.join(images_dir, f'BA_{x_label}_vs_{y_label}_scatter.png')
            print(f"... Save {save_path}")
            plt.savefig(save_path)
            plt.close()

        # Generate all three plot combinations
        pairs = [(0, 1), (0, 2), (1, 2)] if N >= 3 else [(0, 1)] # for two identifiers only PCA1_PCA2 plot
        for pair in pairs:
            _create_plotAB(*pair)
            _create_plotBA(*pair)

    def _save_PCA_matrix_csv(self, PCA_matrix: np.ndarray):
        """
        Save PCA matrix to CSV file for further analysis.
        
        Args:
            PCA_matrix (np.ndarray): PCA-reduced matrix to save.
        """
        images_dir = self.output_dirs['images_dir']
        os.makedirs(images_dir, exist_ok=True)
        PCA_matrix_file_path = os.path.join(images_dir, "PCA_matrix.csv")
        np.savetxt(PCA_matrix_file_path, PCA_matrix, delimiter=',')
        print(f"Similarity matrix saved to {PCA_matrix_file_path}")

    def _save_PCA_info_txt(self, PCA_info: dict):
        """
        Save PCA analysis information to text files.
        
        Args:
            PCA_info (dict): Dictionary containing PCA analysis results.
        """
        images_dir = self.output_dirs['images_dir']
        os.makedirs(images_dir, exist_ok=True)
        for key, list_stats in PCA_info.items():
            PCA_info_file_path = f"{os.path.join(images_dir, key)}.txt"
            print(f"... Save {PCA_info_file_path}")
            with open(PCA_info_file_path, "w") as fout:
                for row in list_stats:
                    fout.write(f"{row}\n")
                fout.close()

    def _write_log_identifier2AverageCodebookUsage(self, stats: dict):
        """
        Write detailed codebook usage statistics to log files.
        
        Args:
            stats (dict): Dictionary mapping identifiers to their usage statistics.
        """
        logs_dir = self.output_dirs['logs_dir']
        for identifier, dict_stats in stats.items():
            logs_identifier_dir = os.path.join(logs_dir, identifier)
            print(f"... Create dir {logs_identifier_dir}")
            os.makedirs(logs_identifier_dir, exist_ok=True)
            for key, list_stats in dict_stats.items():
                logs_identifier_file_path = f"{os.path.join(logs_identifier_dir, key)}.txt"
                print(f"... Save {logs_identifier_file_path}")
                with open(logs_identifier_file_path, "w") as fout:
                    for row in list_stats:
                        fout.write(f"{row}\n")
                    fout.close()

    def process_results(self, results: dict, output_dirs: dict):
        """
        Process and save all experiment results including plots and logs.
        
        Args:
            results (dict): Complete experiment results from MLPipeline.
            output_dirs (dict): Dictionary of output directories for saving results.
        """
        self.output_dirs = output_dirs
        self.identifiers = results["identifiers"]
        print(f"\nSave results to {output_dirs['outputs_dir']}")
        
        # plots
        similarity_matrix = results["data"]['similarity_matrix']
        self._plot_save_similarity_matrix(similarity_matrix, fixed_colorbar=False, filename='similarity_matrix_autoscaled.png')
        self._plot_save_similarity_matrix(similarity_matrix, fixed_colorbar=True, filename='similarity_matrix_normalized.png')
        PCA_matrix = np.array(results["data"]['similarity_matrix_PCA'])
        self._plot_save_PCA_scatter(PCA_matrix)
        self._save_PCA_matrix_csv(PCA_matrix)
        PCA_info = results["log"]["pca_info"]
        self._save_PCA_info_txt(PCA_info)

        # logs
        stats_identifier2AverageCodebookUsage = results["log"]["stats_identifier2AverageCodebookUsage"]
        self._write_log_identifier2AverageCodebookUsage(stats_identifier2AverageCodebookUsage)

