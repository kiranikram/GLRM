import dataclasses

import numpy as np
import torch


from helpers import SVD_initialization


EARLY_STOPPING_TOL = 1e-8


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
    lr: float = 1e-3,
    lambd: float,
    max_iterations: int = 1_000,
    use_svd_init: bool = False,
) -> Result:
    """Applies constrained optimization using alternating minimization for non-negative matrix factorization"""
    A = torch.tensor(A, dtype=torch.float)
    n, m = A.shape

    torch.manual_seed(seed)
    X = torch.rand((n, rank), requires_grad=True)
    Y = torch.rand((rank, m), requires_grad=True)
    X_ = torch.zeros_like(X)
    Y_ = torch.zeros_like(Y)

    opt_X = torch.optim.Adam([X], lr=lr)
    opt_Y = torch.optim.Adam([Y], lr=lr)

    all_metrics = []
    prev_loss = np.inf
    for iteration in range(max_iterations):

        # copy the values into Y_ from Y
        Y_.copy_(Y)
        frob_norm = torch.norm(A - X @ Y_) ** 2 / (n * m)
        regularizer = torch.linalg.matrix_norm(X, ord=2) / (n * rank)
        loss_X = frob_norm + lambd * regularizer

        opt_X.zero_grad()
        loss_X.backward()
        opt_X.step()

        with torch.no_grad():
            X = X.clamp_(min=0)
            assert torch.all(X >= 0)

        loss = loss_X

        # copy the values into X_ from X
        X_.copy_(X)
        frob_norm = torch.norm(A - X_ @ Y) ** 2 / (n * m)
        regularizer = torch.linalg.matrix_norm(Y, ord=2) / (m * rank)
        loss_Y = frob_norm + lambd * regularizer

        opt_Y.zero_grad()
        loss_Y.backward()
        opt_Y.step()

        with torch.no_grad():
            Y = Y.clamp_(min=0)
            assert torch.all(Y >= 0)
        loss = loss_Y

        current_loss = loss.detach().numpy().item()
        if np.abs(current_loss - prev_loss) < EARLY_STOPPING_TOL:
            print(f"Terminating early at iteration {iteration}/{max_iterations}.")
            break
        prev_loss = current_loss
        metrics = {
            "loss": loss.detach().numpy().item(),
            "frob_norm": frob_norm.detach().numpy().item(),
            "iteration": iteration,
        }
        if iteration % 500 == 0:
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
        torch.manual_seed(seed)
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
            "frob_norm": (torch.norm(A - X @ Y) ** 2).detach().numpy().item() / (n * m),
        }

        if iteration % 500 == 0:
            print(metrics)
        all_metrics.append(metrics)

    return Result(
        X=X.detach().numpy(),
        Y=Y.detach().numpy(),
        rank=rank,
        metrics=all_metrics,
    )
