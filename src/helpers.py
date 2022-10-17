import numpy as np
import cvxpy as cp
import os
import sys
import torch
import math

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def make_regularized_pca_loss(lambd: float, *, norm: int = 2):
    """_summary_

    Args:
        lambd (float): _description_
        norm (int, optional): Shrinkage penalty:Lasso = 1, Ridge = 2. Defaults to 2.
    """

    def loss(X, Y, A):

        n, m = A.shape
        n_x, k = X.shape
        k, m_y = Y.shape
        assert n_x == n
        assert m_y == m

        return (
            torch.norm(A - X @ Y) ** 2 / (n * m)
            + lambd * torch.norm(X, norm) ** norm / (n * k)
            + lambd * torch.norm(Y, norm) ** norm / (m * k)
        )

    return loss


def SVD_initialization(A, rank):
    """Args: Transformed Data Matrix A

    Initializes X = U ̃ * (Sigma ̃)^1/2, and Y = (Sigma ̃)^1/2 * V ̃T diag(sigma), with
    offset row initialized with the means.
    """

    # SVD to get initial point
    A = A.cpu().detach().numpy()
    stdev = A.std(0)
    C = math.sqrt(1e-2 / rank)
    u, s, v = np.linalg.svd(A, full_matrices=False)
    u = u[:, :rank]
    s = np.diag(np.sqrt(s[:rank]))
    v = v[:rank, :]

    X_init, Y_init = np.asarray(u.dot(s)), np.asarray(s.dot(v)) * np.asarray(stdev)

    X_init, Y_init = torch.from_numpy(X_init), torch.from_numpy(Y_init)
    X_init.requires_grad = True
    Y_init.requires_grad = True

    return X_init, Y_init
