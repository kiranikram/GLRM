import pandas as pd
import sys
import numpy as np

sys.path.append("..")


from src.glrm import GLRM
from src.helpers import normalize


def run_unsupervised_experiment(dataset: np.ndarray, name: str):
    """Sweeps over a range of ranks and a range of values for lambda,
    runs PCA, Sparse PCA and NMF on dataset.
    Writes results to csv"""

    n, m = dataset.shape

    results = []
    seed = 1
    rank_fracs = [0.01, 0.1, 0.5, 1.0]
    for lambd in [0.0, 1, 3, 10]:
        my_glrm = GLRM(
            max_iterations=2_000,
            seed=seed,
            lambd=lambd,
        )
        models = {
            "pca": my_glrm.pca,
            "sparse_pca": my_glrm.sparse_PCA,
            "nmf": my_glrm.nmf,
        }
        for model_name, model in models.items():
            for frac in rank_fracs:
                rank = max(int(frac * m), 1)
                print(
                    f"Running GLRM for model={model_name}, rank={rank}, lambda={lambd}"
                )
                result = model(dataset, rank=rank)
                for metrics in result.metrics:
                    results.append(
                        {
                            **metrics,
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

    with open(f"../results/results-{name}.csv", "w") as f:
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


# df = pd.read_csv("/Users/ikram/Desktop/GLRM/data/credit_card.csv").sample(frac=0.01)
# normalized_df = normalize(df.values)
# run_unsupervised_experiment(normalized_df, "credit_card")

# df = pd.read_csv("/Users/ikram/Desktop/GLRM/data/statlog.csv")
# normalized_df = normalize(df.values)
# run_unsupervised_experiment(normalized_df, "statlog")

# data = make_synthetic_data(200, 200, rank=2, non_negative=False)
# run_unsupervised_experiment(data, "synthetic_regular")

# data = make_synthetic_data(200, 200, rank=2, non_negative=True)
# run_unsupervised_experiment(data, "synthetic_nonnegative")


#learn
# GLRM(20, seed=1, lambd=0).learn(normalized_df, rank=3, supervised=False)