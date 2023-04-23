import itertools, joblib

import numpy as np
import muCART.grid_search as gs

from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from muCART.mu_cart import muCARTClassifier


if __name__ == "__main__":
    random_state = 46
    X, Y = make_classification(n_samples = 90,
                               n_features = 120,
                               n_redundant = 50,
                               n_repeated = 0,
                               n_informative = 50,
                               n_classes = 3,
                               random_state = random_state)
    #muCART supports ragged arrays as multiple predictors
    X = [X, -X[:,:50]]
    scoring = None
    criterion = 'gini'
    min_samples_leaf_list = [i for i in range(1,5)]
    n_splits = 2
    n_splits_grid_search = 2
    begin = -1
    end = 5
    n_lambda = 4
    lambda_list = np.logspace(begin,
                              end,
                              num=n_lambda,
                              base=2).tolist()
    print_tree_flag = False
    class_weight = 'balanced'
    solver_options = {'solver':'ipopt',
                      'max_iter':500}

    mucart_score_list = []    
          
    ### Train/Test splitter ###
    cv = StratifiedKFold(n_splits=n_splits,
                         random_state=random_state,
                         shuffle=True)
                   
    for counter,(train_index,test_index) in enumerate(cv.split(X[0],Y)):
        ### Train/Validation splitter (for Grid Search) ###
        _cv = StratifiedKFold(n_splits=n_splits_grid_search, 
                              random_state=random_state,
                              shuffle=True)

        ### muCART ###
        estimator = muCARTClassifier(solver_options,
                                     criterion=criterion,
                                     class_weight=class_weight,
                                     print_tree_flag=print_tree_flag)
        parameters = {'min_samples_leaf':min_samples_leaf_list,
                      'lambda':lambda_list,
                      'max_depth': [None]}
        grid_search = gs.GridSearchCV(estimator,
                                      parameters,
                                      _cv,
                                      scoring = scoring)
        #muCART supports ragged arrays as covariates
        X_train = [X[i][train_index] for i in range(len(X))]
        grid_search.fit(X_train,
                        Y[train_index])
        #muCART supports ragged arrays as covariates
        X_test = [X[i][test_index] for i in range(len(X))]
        score = grid_search.score(X_test,
                                  Y[test_index])
        mucart_score_list.append(score)

        print()
        print('MU CART: After Validating on Grid Search Splits')        
        print(f'lambd == {grid_search.best_estimator.lambd}')
        print(f'min_samples_leaf == {grid_search.best_estimator.min_samples_leaf}')
        print(f'depth = {grid_search.best_estimator.depth}')
        print(f'n_inner_nodes = {grid_search.best_estimator.n_inner_nodes}')
        print(f'n_leaves = {grid_search.best_estimator.n_leaves}')
        print(f'n_nodes by input covariate = {grid_search.best_estimator.n_nodes_by_input}')
        print(f'mean_pos {grid_search.best_estimator.n_nodes_mean_pos}')
        print(f'mean_neg {grid_search.best_estimator.n_nodes_mean_neg}')
        print(f'mean_sgn {grid_search.best_estimator.n_nodes_mean_sgn}')
        print(f'mean_uni {grid_search.best_estimator.n_nodes_mean_uni}')
        print(f'var_pos {grid_search.best_estimator.n_nodes_var_pos}')
        print(f'var_neg {grid_search.best_estimator.n_nodes_var_neg}')
        print(f'var_sgn {grid_search.best_estimator.n_nodes_var_sgn}')
        print(f'var_uni {grid_search.best_estimator.n_nodes_var_uni}')
        print(f'cosine_pos {grid_search.best_estimator.n_nodes_cosine_pos}')
        print(f'cosine_neg {grid_search.best_estimator.n_nodes_cosine_neg}')
        print(f'cosine_uni {grid_search.best_estimator.n_nodes_cosine_uni}')
        print(f'class_cosine_pos {grid_search.best_estimator.n_nodes_class_cosine_pos}')
        print(f'class_cosine_neg {grid_search.best_estimator.n_nodes_class_cosine_neg}')
        print(f'class_cosine_uni {grid_search.best_estimator.n_nodes_class_cosine_uni}')
        print()
    print()
    print(f'muCART TEST SCORE:          {np.mean(mucart_score_list)}')

