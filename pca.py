import numpy as np
from numpy import ndarray


class PCA:
    def __init__(self) -> None:
        self.x_mean = None
        self.eigen_vectors = None
        self.sorted_eigen_indices = None
        self.ok_to_transform = False

    def fit(self, x: ndarray):
        """
        x : ndarray
            vectors to fit with dimension => [amount_of_vectors, vector_dim]
        """

        self.x_mean = np.mean(x, axis=0, keepdims=True)
        x_zero_mean = x - self.x_mean

        # find eigenvecoters of covariance_matrix_x on complex field
        cov_x = np.cov(x_zero_mean.T)
        eig_values, eig_vectors = np.linalg.eigh(cov_x)  # eig_vectors => [vector_dim, eigen_value_amount]

        # sort eigenvetors with eigenvalues and just select an amount of pca_vector_dim eigenvectors
        sorted_indices = np.argsort(-eig_values)  # to get indices in descending order

        # store eigenvectors to transform datas
        self.sorted_eigen_indices = sorted_indices
        self.eigen_vectors = eig_vectors
        self.ok_to_transform = True

    def transform(self, x: ndarray, pca_vector_dim: int):  # x => [amount, vector_dim]
        """
        x : ndarray
            vectors to be transformed with dimension => [amount, vector_dim]

        pca_vector_dim : int
            vector dimension after pca transform
        """
        assert pca_vector_dim <= np.shape(x)[-1]  # make sure pca_vector_dim is less than vector_dim
        if not self.ok_to_transform:
            raise Exception("Need to fit training data before transform")

        x_zero_mean = x - self.x_mean
        # x_zero_mean = x - np.mean(x, axis=1, keepdims=True)

        sorted_indices = self.sorted_eigen_indices[:pca_vector_dim]
        sorted_eig_vectors = self.eigen_vectors[:, sorted_indices]  # [vector_dim, pca_vector_dim]

        x_reduced = np.dot(x_zero_mean, sorted_eig_vectors)  # x_reduced => [amount, pca_vector_dim]
        return x_reduced
