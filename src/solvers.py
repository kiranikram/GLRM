import pdb
import functools
import numpy as np
import sys
import os
import dataclasses
from typing import List
import torch


from helpers import make_regularized_pca_loss, SVD_initialization


EARLY_STOPPING_TOL = 1e-5


@dataclasses.dataclass
class Result:
    metrics: list
    X: np.ndarray
    Y: np.ndarray
    rank: int


def nmf_alt_minimizer(
    A: np.ndarray,
    *,
    rank: int,
    seed: int,
    objective,
    lr: float = 1e-2,
    max_iterations: int = 1_000,
    use_svd_init: bool = False,
) -> Result:
    """Applies constrained optimization using alternating minimization for non-negative matrix factorization"""
    A = torch.tensor(A, dtype=torch.float)
    n, m = A.shape

    X = torch.rand((n, rank), requires_grad=True)
    Y = torch.rand((rank, m), requires_grad=True)
    Y_ = torch.rand(rank, m)
    X_ = torch.rand(n, rank)

    opt_X = torch.optim.SGD([X], lr=lr)
    opt_Y = torch.optim.SGD([Y], lr=lr)

    all_metrics = []
    prev_loss = np.inf
    for iteration in range(max_iterations):

        # copy the values into Y_ from Y
        Y_.copy_(Y)
        loss_X = torch.norm(A - X @ Y_) ** 2 / (n * m)

        opt_X.zero_grad()
        loss_X.backward()
        opt_X.step()

        with torch.no_grad():
            X = X.clamp_(min=0)
            assert torch.all(X >= 0)

        loss = loss_X

        # copy the values into X_ from X
        X_.copy_(X)
        loss_Y = torch.norm(A - X_ @ Y) ** 2 / (n * m)

        opt_Y.zero_grad()
        loss_Y.backward()
        opt_Y.step()

        with torch.no_grad():
            Y = Y.clamp_(min=0)
            assert torch.all(Y >= 0)
        loss = loss_Y

        current_loss = loss.detach().numpy().item()
        if np.abs(current_loss - prev_loss) < EARLY_STOPPING_TOL:
            print("Terminating early.")
            break
        prev_loss = current_loss
        metrics = {
            "loss": loss.detach().numpy().item(),
            "iteration": iteration,
        }
        if iteration % 100 == 0:
            print(metrics)
        all_metrics.append(metrics)

    return Result(
        X=X.detach().numpy(),
        Y=Y.detach().numpy(),
        rank=rank,
        metrics=all_metrics,
    )


def alternating_optimizer(
    A: np.ndarray,
    *,
    rank: int,
    seed: int,
    objective,
    lr: float = 1e-2,
    max_iterations: int = 1_000,
    use_svd_init: bool = False,
) -> Result:
    """Alternating minimization"""

    A = torch.tensor(A, dtype=torch.float)
    n, m = A.shape

    if use_svd_init:
        X, Y = SVD_initialization(A, rank)
    else:
        X = torch.rand(
            n,
            rank,
            requires_grad=True,
        )  # for updates
        Y = torch.rand(rank, m, requires_grad=True)

    optimizer_x = torch.optim.Adam([X], lr=lr)
    optimizer_y = torch.optim.Adam([Y], lr=lr)

    all_metrics = []

    prev_loss = np.inf
    for iteration in range(max_iterations):

        if iteration % 2 == 0:
            X.requires_grad = True
            Y.requires_grad = False
        else:
            X.requires_grad = False
            Y.requires_grad = True

        loss = objective(X, Y, A)

        if iteration % 2 == 0:
            optimizer_x.zero_grad()
            loss.backward()
            optimizer_x.step()
        else:
            optimizer_y.zero_grad()
            loss.backward()
            optimizer_y.step()

        current_loss = loss.detach().numpy().item()
        if np.abs(current_loss - prev_loss) < EARLY_STOPPING_TOL:
            print("Terminating early.")
            break
        prev_loss = current_loss
        metrics = {
            "loss": current_loss,
            "iteration": iteration,
        }

        if iteration % 100 == 0:
            print(metrics)
        all_metrics.append(metrics)

    return Result(
        X=X.detach().numpy(),
        Y=Y.detach().numpy(),
        rank=rank,
        metrics=all_metrics,
    )