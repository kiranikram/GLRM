# GLRM

This repo implements components from "Generalized Low Rank Models" (https://web.stanford.edu/~boyd/papers/pdf/glrm.pdf). The codebase provides implementation of PCA, Sparse PCA, Non Negative Matrix Factorization and Least Squares. 

# Dataset

To test the low rank models two datsets were utilized (1) the multivariate Statlog Dataset (https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29) which comprises 14 features and (2) Default of Credit Card Clients Dataset (https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) which comprises 24 features for analysis. 

# Usage

The experiments.py folder in src can be used to run the low rank models, which can be found in glrm.py

# Analyis

Plots of ranks as well as shrinkage hyperparameters can be found in the analysis notebook. 
