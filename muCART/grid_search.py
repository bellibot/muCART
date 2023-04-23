import numpy as np
import itertools, copy, joblib

from sklearn.metrics import accuracy_score, r2_score
from typing import Dict, Callable, List
from muCART.mu_cart import muCARTBase


class GridSearchCV:
    def __init__(self,
                 estimator: muCARTBase,
                 parameters: Dict,
                 cv: 'ScikitCVObj',
                 scoring: Callable = None,
                 verbose: bool = False,
                 n_jobs: int = -1):
        self.parameters_packed = parameters
        flattened = [[(key, val) for val in values] for key, values in self.parameters_packed.items()]
        self.parameters_unpacked = [dict(items) for items in itertools.product(*flattened)]
        self.estimator = estimator
        self.cv = cv
        if scoring == None:
            if estimator.estimator_type=='classifier':
                self.scoring = accuracy_score 
            elif estimator.estimator_type=='regressor':
                self.scoring = r2_score                     
        elif callable(scoring):
            self.scoring = scoring
        else:
            raise ValueError(f'Illegal scoring {self.scoring}')
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.fold_estimators = []
        self.fold_best_estimators = []
        self.fold_train_scores = []
        self.fold_test_scores = []
        self.mean_fold_train_scores = []
        self.mean_fold_test_scores = []
        self.std_fold_train_scores = []
        self.std_fold_test_scores = []
        self.best_estimator = None
        self.best_parameters = None
        self.best_estimator_train_score = None 


    def fit(self, 
            X: List[np.ndarray], 
            Y: np.ndarray):
        if len(X)>1:
            shapes_list = [x.shape[0] for x in X]
            if not(shapes_list[1:]==shapes_list[:-1]):
                raise ValueError('Error: all covariates must have the same sample size')
        for counter,(train_index, test_index) in enumerate(self.cv.split(X[0],Y)):
            estimators = []
            for param_dict in self.parameters_unpacked:
                estimator = copy.deepcopy(self.estimator)
                estimator.set_parameters(param_dict)
                estimators.append(estimator)
            X_train = [X[i][train_index] for i in range(len(X))]
            estimators = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(estimators[i].fit)(X_train,Y[train_index]) for i in range(len(self.parameters_unpacked)))
            inner_train_scores = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(estimators[i].score)(X_train,Y[train_index],self.scoring) for i in range(len(self.parameters_unpacked)))
            X_test = [X[i][test_index] for i in range(len(X))]
            inner_test_scores = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(estimators[i].score)(X_test,Y[test_index],self.scoring) for i in range(len(self.parameters_unpacked)))
            self.fold_estimators.append(estimators)
            self.fold_train_scores.append(inner_train_scores)
            self.fold_test_scores.append(inner_test_scores)
            best_index = inner_test_scores.index(max(inner_test_scores))
            self.fold_best_estimators.append(estimators[best_index])
            if self.verbose:
                print()
                print(f' Grid Search Iternum: {counter}')
                print()
                for p,s in zip(self.parameters_unpacked,inner_test_scores):   
                    print(f' Inner Parameters: {p}')
                    print(f' Inner Score: {s}')
                    print()
                print()
                if len(set(inner_test_scores))==1:
                    print(' All the parameter tuples have the same score in this fold')  
        self.mean_fold_train_scores = np.mean(self.fold_train_scores, axis=0)
        self.mean_fold_test_scores = np.mean(self.fold_test_scores, axis=0)
        self.std_fold_train_scores = np.std(self.fold_train_scores, axis=0)
        self.std_fold_test_scores = np.std(self.fold_test_scores, axis=0)
        best_index = np.argmax(self.mean_fold_test_scores)
        self.best_parameters = self.parameters_unpacked[best_index]
        if self.verbose:
            print()
            print(f' Best Parameters: {self.best_parameters}')
            print()     
        estimator = copy.deepcopy(self.estimator)
        estimator.set_parameters(self.best_parameters)           
        estimator.fit(X,Y)
        self.best_estimator = estimator
        self.best_estimator_train_score = estimator.score(X,Y,self.scoring)
        

    def score(self,
              X: List[np.ndarray],
              Y: np.ndarray) -> np.ndarray:
        return self.best_estimator.score(X, Y)

