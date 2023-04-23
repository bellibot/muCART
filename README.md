## muCART - Measure Inducing Classification and Regression Trees

`muCART` is a Python package that implements Measure Inducing Classification and Regression Trees for Functional Data.

The estimators are implemented with the familiar `fit`/`predict`/`score` interface, and also support multiple predictors of pissibly different lengths (as list of np.ndarray matrices). The following tasks are supported, based on loss function inside each node of the tree:

- Regression (mse, mae)
- Binary and Multiclass Classification (gini, misclassification error, entropy)

A custom `cross-validation` object is provided in order to perform grid search hyperparameter tuning (with any splitter from `scikit-learn`), and uses `multiprocessing` for parallelization (default `n_jobs = -1`).

## Installation

The package can be installed from terminal with the command `pip install muCART`. Inside each node of the tree, the optimization problems (quadratic with equality and/or inequality constraints) are formulated using `Pyomo`, which in turn needs a `solver` to interface with. All the code was tested on Ubuntu using the solver [Ipopt](https://doi.org/10.1007/s10107-004-0559-y). You just need to download the [executable binary](https://ampl.com/products/solvers/open-source/#ipopt), and then add the folder that contains it to your path.


## Usage

The following lines show how to fit an estimator with its own parameters and grid search object, by using a `StratifiedKFold` splitter:

```sh
import numpy as np
import muCART.grid_search as gs
from muCART.mu_cart import muCARTClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_wine
from sklearn.metrics import balanced_accuracy_score

X, Y = load_wine(return_X_y = True)
# wrap the single predictor in a List
X = [X]
train_index = [i for i in range(100)]
test_index = [i for i in range(100, len(X))]

lambda_list = np.logspace(-5, 5, 10, base = 2)
solver_options = {'solver':'ipopt',
                  'max_iter':500}

estimator = muCARTClassifier(solver_options)
parameters = {'lambda':lambda_list}
cv = StratifiedKFold(n_splits = 2,
                     random_state = 46,
                     shuffle = True)
grid_search = gs.GridSearchCV(estimator,
                              parameters,
                              cv,
                              scoring = balanced_accuracy_score,
                              verbose = False,
                              n_jobs = -1)
# extract train samples for each predictor
X_train = [X[i][train_index] for i in range(len(X))]
grid_search.fit(X_train,
                Y[train_index])
# extract test samples for each predictor
X_test = [X[i][test_index] for i in range(len(X))]
score = grid_search.score(X_test,
                          Y[test_index])
```
The test folder in the `github` repo contains two sample scripts that show how to use the estimator in both classification and regression tasks. Regarding the `scoring`, both estimators and the grid search class use `accuracy`/`R^2` as default scores (when the argument `scoring = None`), but you can provide any `Callable` scoring function found in `sklearn.metrics`. Beware that higher is better, and therefore when scoring with errors like `sklearn.metrics.mean_squared_error`, you need to wrap that in a custom function that changes its sign.

## Citing

The code published in this package has been used in the case studies of [this](https://doi.org/10.1002/sam.11569) paper.
