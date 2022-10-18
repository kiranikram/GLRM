import csv
import pandas as pd
import sys
import os
import dataclasses
from typing import List

import numpy as np

sys.path.append("..")

from src.glrm import GLRM


def run_unsupervised_experiment(dataset: np.ndarray, name: str):
    """ Sweeps over a range of ranks and a range of values for lambda, 
    runs PCA, Sparse PCA and NMF on dataset. 
    Writes results to csv"""

    n, m = dataset.shape

    results = []
    seed = 1
    fracs = [0.01, 0.1, 0.5, 1.0]
    for lambd in [0.0, 0.3, 1.0]:
        my_glrm = GLRM(max_iterations=2000, seed=seed, lambd=lambd)
        models = {
            "pca": my_glrm.pca,
            "sparse_pca": my_glrm.sparse_PCA,
            "nmf": my_glrm.nmf,
        }
        for model_name, model in models.items():
            for frac in fracs:
                rank = max(int(frac * m), 1)
                print(f"Running GLRM for model {model_name} rank: {rank}")
                result = model(dataset, rank=rank)
                # Note: assuming numerical PCA.
                reconstruction_loss = np.linalg.norm(
                    dataset - result.X @ result.Y, ord="fro"
                ) ** 2 / (n * m)
                for metrics in result.metrics:
                    results.append(
                        {
                            **metrics,
                            "reconstruction_loss": reconstruction_loss,
                            "model": model_name,
                            "rank": rank,
                            "dim_reduction_fraction": 1 - frac,
                            "lambd": lambd,
                            "seed": seed,
                            "num_features": m,
                            "num_data": n,
                            "name": name,
                        }
                    )

    results_df = pd.DataFrame(results)

    with open(f"../results/reg/results-{name}.csv", "w") as f:
        results_df.to_csv(f)


def make_synthetic_data(
    num_data: int,
    num_features: int,
    rank: int,
    non_negative: bool = False,
) -> np.ndarray:
    """As described in section 7.5, synthetic dataset is created and tested with PCA, Sparse PCA and NMF"""

    # Make some synthetic data that has a low-rank factorisation.
    X_true = np.random.randn(num_data, rank)
    Y_true = np.random.randn(rank, num_features)

    if non_negative:
        X_true = np.clip(X_true, a_min=0, a_max=None)
        Y_true = np.clip(Y_true, a_min=0, a_max=None)

    return X_true @ Y_true

df = pd.read_csv("/Users/ikram/Desktop/GLRM/data/statlog.csv")
run_unsupervised_experiment(df.values, 'statlog')