import sys

sys.path.append("..")

import numpy as np
import pandas as pd
import torch
from typing import Optional
from sklearn.model_selection import train_test_split


from .solvers import Result, alternating_optimizer, nmf_alt_minimizer
from .helpers import make_regularized_pca_loss


class GLRM:
    """Generalised low-rank model implementation.

    Overall assumptions:
    - Numerical data only.
    """

    def __init__(self, max_iterations: int, seed: int, lambd: float):
        """Sets up the model.

        Args:
            max_iterations: Maximum number of alternating optimization steps.
            seed: Random seed to use for initialization.
            lambd: Regularisation hyperparameter.
        """
        self.max_iterations = max_iterations
        self.lambd = lambd
        self.seed = seed

    def _preprocess(self, data: np.ndarray):
        self.offset = np.mean(data, axis=0)
        self.scale = np.std(data, axis=0)
        # Handle columns with zero variance.
        self.scale = np.where(self.scale == 0, np.ones_like(self.scale), self.scale)
        return (data - self.offset) / self.scale

    def _postprocess(self, data: np.ndarray):
        return data

    def pca(self, data: np.ndarray, rank: int) -> Result:
        """Runs quadratically-regularized PCA on data."""
        result = alternating_optimizer(
            self._preprocess(data),
            rank=rank,
            seed=self.seed,
            objective=make_regularized_pca_loss(self.lambd, norm=2),
            lr=1e-2,
            max_iterations=self.max_iterations,
            use_svd_init=False,
        )
        result.X = self._postprocess(result.X)
        return result

    def sparse_PCA(self, data: np.ndarray, rank: int) -> Result:
        """Runs sparsity-regularized PCA on data."""
        result = alternating_optimizer(
            self._preprocess(data),
            rank=rank,
            seed=self.seed,
            objective=make_regularized_pca_loss(
                self.lambd, norm=1
            ),  # Quadratically loss regularised with l1 norm for sparsity
            lr=1e-2,
            max_iterations=self.max_iterations,
            use_svd_init=False,
        )

        result.X = self._postprocess(result.X)
        return result

    def nmf(self, data: np.ndarray, rank: int) -> Result:
        """Non-negative matrix factorization."""
        result = nmf_alt_minimizer(
            data,  # No preprocessing.
            rank=rank,
            seed=self.seed,
            lr=1e-2,
            lambd=self.lambd,
            max_iterations=self.max_iterations,
            use_svd_init=False,
        )

        result.X = self._postprocess(result.X)
        return result

    def simple_linear_regression(self, data: np.ndarray, use_svd: bool = True):
        """Runs simple linear regression on data.

        Assumes that the final column of data is the regression target.
        """
        data = self._preprocess(data)

        # Assume that the final column is the target.
        A = data[:, :-1]
        b = data[:, -1]

        # Partition the dataset.
        Atrain, Atest, btrain, btest = train_test_split(
            A, b, test_size=0.3, random_state=1
        )

        Atrain = torch.from_numpy(Atrain.astype(np.float32))
        btrain = torch.from_numpy(btrain.astype(np.float32))
        Atest = torch.from_numpy(Atest.astype(np.float32))
        btest = torch.from_numpy(btest.astype(np.float32))

        # Note: no regularisation.
        if use_svd:
            U, S, V_t = torch.linalg.svd(Atrain, full_matrices=False)
            U_t = U.t()
            Ut_b = torch.matmul(U_t, btrain)
            y = torch.div(Ut_b, S)
            x = torch.matmul(y, (V_t.t()))
            predictions = torch.matmul(Atest, x)
        else:
            coeffs = torch.linalg.lstsq(Atrain, btrain).solution
            predictions = torch.matmul(Atest, coeffs)

        return predictions.numpy(), btest.numpy()

    def learn(
        self,
        dataset: np.ndarray,
        rank: Optional[int] = None,
        supervised: bool = True,
    ) -> ...:
        """General learning function.

        In the unsupervised case, runs all three of {NMF, sparse-PCA, PCA} and returns the best fit.
        """

        if supervised:
            return self.simple_linear_regression(dataset)

        n, m = dataset.shape

        models = {
            self.pca,
            self.sparse_PCA,
            self.nmf,
        }
        results = [model(dataset, rank) for model in models]

        best_loss = np.inf
        best_model_idx = None
        for idx, result in enumerate(results):
            reconstruction_loss = np.linalg.norm(
                dataset - result.X @ result.Y, ord="fro"
            ) ** 2 / (n * m)
            if reconstruction_loss < best_loss:
                best_loss = reconstruction_loss
                best_model_idx = idx

        return results[best_model_idx]
