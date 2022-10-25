
import math

import numpy as np
import torch

def zero_mean_unit_variance(data: np.ndarray) -> np.ndarray:
    """Normalizes to zero mean and unit variance."""
    offset = np.mean(data, axis=0)
    scale = np.std(data, axis=0)
    # Handle columns with zero variance.
    scale = np.where(scale == 0, np.ones_like(scale), scale)
    return (data - offset) / scale


def normalize(data: np.ndarray) -> np.ndarray:
    """Normalizes to range [0, 1]."""
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    return (data - data_min) / (data_max - data_min)


def make_regularized_pca_loss(lambd: float, *, norm: int = 2):
    """Sets up the objective function

    Args:
        lambd (float): shrinkage penalty for regularization
        norm (int, optional): Shrinkage penalty:Lasso = 1, Ridge = 2. Defaults to 2.
    """

    def loss(X, Y, A):
        """squared frobenius norm loss funnction."""

        n, m = A.shape
        n_x, k = X.shape
        k, m_y = Y.shape
        assert n_x == n
        assert m_y == m

        mse = torch.norm(A - X @ Y) ** 2 / (n * m)
        x_regulariser = torch.norm(X, norm) ** norm / (n * k)
        y_regulariser = torch.norm(Y, norm) ** norm / (m * k)

        return mse + lambd * (x_regulariser + y_regulariser)

    return loss


def SVD_initialization(A: torch.Tensor, rank: int):
    """Args: Transformed Data Matrix A

    Initializes X = U ̃ * (Sigma ̃)^1/2, and Y = (Sigma ̃)^1/2 * V ̃T diag(sigma), with
    offset row initialized with the means.
    """

    # SVD to get initial point.
    A = A.cpu().detach().numpy()
    stdev = A.std(0)
    u, s, v = np.linalg.svd(A, full_matrices=False)
    u = u[:, :rank]
    s = np.diag(np.sqrt(s[:rank]))
    v = v[:rank, :]

    X_init, Y_init = np.asarray(u.dot(s)), np.asarray(s.dot(v)) * np.asarray(stdev)

    X_init, Y_init = torch.from_numpy(X_init), torch.from_numpy(Y_init)
    X_init.requires_grad = True
    Y_init.requires_grad = True

    return X_init, Y_init
