import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.stats import entropy
from numpy.linalg import norm


class SimilarityMatrix:
    """
    Generates similarity matrices and PCA visualizations for speech categories.
    
    This class computes Jensen-Shannon Divergence (JSD) similarity between
    codebook usage distributions and provides PCA dimensionality reduction.
    """

    def JSD(self, P: np.ndarray, Q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon Divergence similarity between two distributions.
        
        Args:
            P (np.ndarray): First probability distribution.
            Q (np.ndarray): Second probability distribution.
        
        Returns:
            float: JSD similarity score (0-1, where 1 is most similar).
        """
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return 1 - (0.5 * (entropy(_P, _M) + entropy(_Q, _M)))

    def JSD_similarity(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise JSD similarity matrix for all distributions.
        
        Args:
            X (np.ndarray): Matrix where each row is a probability distribution.
        
        Returns:
            np.ndarray: Symmetric similarity matrix of shape (n_samples, n_samples).
        """
        sim = np.zeros((np.shape(X)[0], np.shape(X)[0]), dtype=np.float32)
        
        print('... Calculate JSD similarity matrix of X: {} ...'.format(X.shape))
        for row, x in enumerate(X):
            for col, _ in enumerate(sim):
                #print('calculate JSD similarity of features ({}, {})'.format(row, col))
                sim[row, col] = self.JSD(x, X[col,:])

        return sim

    def generate_matrix(self, identifier2AverageCodebookUsage: dict) -> tuple[np.ndarray, np.ndarray, list, dict]:
        """
        Generate similarity matrix and PCA visualization between all identifiers.
        
        Args:
            identifier2AverageCodebookUsage (dict): Dictionary mapping identifier strings to 
                                                  their average codebook usage distributions.
        
        Returns:
            tuple: (matrix, matrix_PCA, identifiers, pca_info)
                - matrix (np.ndarray): NxN JSD similarity matrix between identifiers
                - matrix_PCA (np.ndarray): Nx3 matrix with PCA-reduced dimensions for visualization
                - identifiers (list[str]): Sorted list of identifiers corresponding to matrix rows/columns
                - pca_info (dict): PCA analysis information containing:
                    - PCA_variances (list[tuple]): Variance explained by each PC as (name, percentage)
                    - PCA_eigenvectors (list[list]): Principal component eigenvectors
                    - PCA_eigenvalues (list[float]): Principal component eigenvalues
        """
        # Extract identifiers and their corresponding average codebook usage
        identifiers = sorted(identifier2AverageCodebookUsage.keys())
        data = np.array([np.mean(identifier2AverageCodebookUsage[identifier], axis=0) for identifier in identifiers])

        # Compute similarity matrix
        matrix = self.JSD_similarity(data)

        # Apply PCA to reduce dimensionality to 3D
        print('... Calculate PCA of similarity matrix')
        n_comps = 3 if matrix.shape[0] > 2 else 2
        pca = PCA(n_components=n_comps)
        matrix_PCA = pca.fit_transform(matrix)

        # Prepare PCA info dictionary, including eigenvalues
        # pca.explained_variance_ratio_ = pca.explained_variance_ / pca.explained_variance_.sum()
        pca_variance = [(f"PCA{i+1}", float(round(var * 100, 2))) for i, var in enumerate(pca.explained_variance_ratio_)]
        pca_eigenvectors = pca.components_.tolist()
        pca_eigenvalues = pca.explained_variance_.tolist()  # eigenvalues as list of floats

        pca_info = {
            "PCA_variances": pca_variance,
            "PCA_eigenvectors": pca_eigenvectors,
            "PCA_eigenvalues": pca_eigenvalues
        }

        return matrix, matrix_PCA, identifiers, pca_info
