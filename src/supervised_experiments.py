import sys

import pandas as pd
import numpy as np
import torch


from src.glrm import GLRM
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


def test_sklearn_regr(data):

    # Preprocessing.
    offset = np.mean(data, axis=0)
    scale = np.std(data, axis=0)
    data = (data - offset) / scale

    A = data[:, :-1]
    b = data[:, -1]

    Atrain, Atest, btrain, btest = train_test_split(A, b, test_size=0.3, random_state=1)
    regr = linear_model.LinearRegression()

    btrain = btrain[..., None]
    regr.fit(
        Atrain,
        btrain,
    )

    y_pred = regr.predict(Atest)

    return y_pred, btest


def supervised_experiment(path):
    data = pd.read_csv(path).to_numpy()
    glrm = GLRM(max_iterations=2000, seed=1, lambd=0)

    algs = {
        "glrm": glrm.simple_linear_regression,
        "sklearn": test_sklearn_regr,
    }

    all_metrics = []
    for alg_name, alg in algs.items():
        print(f"Running supervised learning ({alg_name})")
        preds, truth = alg(data)

        all_metrics.append(
            {
                "algorithm": alg_name,
                "mse": np.mean((preds - truth) ** 2),
                "r_squared": r2_score(truth, preds),
            }
        )
    df = pd.DataFrame(all_metrics)

    with open(f"../results/results-supervised.csv", "w") as f:
        df.to_csv(f)



