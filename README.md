# GLRM

This repo implements components from "Generalized Low Rank Models" (https://web.stanford.edu/~boyd/papers/pdf/glrm.pdf). The codebase provides implementation of PCA, Sparse PCA, Non Negative Matrix Factorization and Least Squares. 

# Dataset

To test the low rank models two datsets were utilized (1) the multivariate Statlog Dataset (https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29) which comprises 14 features and (2) Default of Credit Card Clients Dataset (https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) which comprises 24 features for analysis. 

Data is available here: 
https://drive.google.com/drive/folders/1Y1_NC_HSQ88CPiOLiv_DboKyAYWHsCx9?usp=sharing

# Analyis

Plots of ranks as well as shrinkage hyperparameters can be found in the analysis notebook. 

# Usage

The user should specify a dataset and supervised vs unsupervised to call learn from GLRM. glrm.py contains the class GLRM. The experiments folders were used to conduct the experiments as described in analysis.py


# Future Work
- Apply NMF to more apt dataset, eg image or text

- Adding efficiency to the optimizers (1) parallelization (2) inner iterations for X and Y updates

- Loss functions for abstract data types

- Interpretability for each new archetype

- Other forms of optimization 

- Larger heterogenuous dataset

- Implicit offset/scaling in loss functions for non-savvy users


