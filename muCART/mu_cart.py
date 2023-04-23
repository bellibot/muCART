import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Callable, List
from sklearn.metrics import accuracy_score, r2_score
from muCART.mu_node import muNodeBase, muNodeClassifier, muNodeRegressor


class muCARTBase(ABC):
    def set_parameters(self,
                       parameters: Dict):
        self.parameters = parameters
        self.max_depth = parameters['max_depth']
        self.min_samples_leaf = parameters['min_samples_leaf']
        self.lambd = parameters['lambda']
        self.root.min_samples_leaf = self.min_samples_leaf
        self.root.lambd = self.lambd
        if self.lambd <= 0:
            raise ValueError('Lambda <= 0')
        elif (self.max_depth is not None) and (self.max_depth < 1):
            raise ValueError(f'Illegal max_depth {max_depth}')
        elif self.min_samples_leaf < 1:
            raise ValueError(f'Illegal min_samples_leaf {min_samples_leaf}')


    def fit(self,
            X: List[np.ndarray],
            Y: np.ndarray):
        if len(X)>1:
            shapes_list = [x.shape[0] for x in X]
            if not(shapes_list[1:]==shapes_list[:-1]):
                raise ValueError('Error: all covariates must have the same sample size')
        if self.root.is_not_pure(Y):
            self.root.run(X,Y)
            self.root.n_elements = len(self.root.node_indexes)
            if self.max_depth==None:
                self.max_depth = self.root.n_elements
            
            if len(self.root.R1_indexes)==0 or len(self.root.R2_indexes)==0:
                self._finalize_node(self.root,Y)   
            else:    
                self.root.l_child = self._make_new_node(self.root,
                                                        1,
                                                        'left')         
                if self.root.l_child.is_not_pure(Y) and \
                   (len(self.root.R1_indexes) >= 2*self.min_samples_leaf) and (self.max_depth>1):
                    self._split_node(self.root.l_child,X,Y)
                else:
                    self._finalize_node(self.root.l_child,Y)       
                self.root.r_child = self._make_new_node(self.root,
                                                        1,
                                                        'right')
                if self.root.r_child.is_not_pure(Y) and \
                   (len(self.root.R2_indexes) >= 2*self.min_samples_leaf) and (self.max_depth>1):
                    self._split_node(self.root.r_child,X,Y)
                else:
                    self._finalize_node(self.root.r_child,Y)
            self._validate_tree(self.root)
            if not(self._valid_tree):
                raise ValueError('Critical Error: invalid tree')
            if self.print_tree_flag:
                self._print_tree(self.root,X)
        else:
            print('Same response Y for each X, no need to fit the tree')
        return self


    def predict(self,
                X: List[np.ndarray]) -> np.ndarray:
        pred_Y = []
        for n in range(len(X[0])):
            x = [X[i][n] for i in range(len(X))]
            pred_Y.append(self._traversal(self.root,x))
        return np.array(pred_Y)


    def score(self,
              X: List[np.ndarray],
              Y: np.ndarray,
              scoring: Callable = None) -> np.ndarray:
        pred_Y = self.predict(X) 
        if scoring == None:
            score = self.default_scoring(Y, pred_Y) 
        elif callable(scoring):
            score = scoring(Y, pred_Y)
        else:
            raise ValueError(f'Illegal scoring {scoring}')          
        return score


    @abstractmethod
    def _make_new_node():
        raise NotImplementedError()


    def _split_node(self,
                    node: muNodeBase,
                    X: List[np.ndarray],
                    Y: np.ndarray):
        if node.is_not_pure(Y):
            node.run(X,Y)
            if len(node.R1_indexes)==0 or len(node.R2_indexes)==0:
                self._finalize_node(node,Y)
            else:
                node.l_child = self._make_new_node(node,
                                                   node.depth+1,
                                                   'left')
                if node.l_child.is_not_pure(Y) and \
                   (len(node.R1_indexes) >= 2*self.min_samples_leaf) and (node.depth<self.max_depth):
                    self._split_node(node.l_child,X,Y)
                else:
                    self._finalize_node(node.l_child,Y)
                node.r_child = self._make_new_node(node,
                                                node.depth+1,
                                                'right')
                if node.r_child.is_not_pure(Y) and \
                   (len(node.R2_indexes) >= 2*self.min_samples_leaf) and (node.depth<self.max_depth):
                    self._split_node(node.r_child,X,Y)
                else:
                    self._finalize_node(node.r_child,Y)
        else:
            self._finalize_node(node,Y)


    def _finalize_node(self,
                       node: muNodeBase,
                       Y: np.ndarray):
        node.R1_indexes = None
        node.R2_indexes = None
        if node.node_id=='left' or node.node_id=='right':
            node.best_covariate = node.parent.best_covariate
        else:
            node.best_covariate = 0
        node.n_elements = len(node.node_indexes)
        node.split_error, node.y = node._compute_split_error(Y,node.node_indexes,[])
        node.is_leaf = True


    def _traversal(self,
                   node: muNodeBase,
                   x: np.ndarray):
        if node.is_leaf:
            return node.y
        else:
            _x = x[node.best_covariate]
            if 'mean' in node.split_feature:
                if node.scale*np.dot(node.w,_x) <= node.split_value:
                    return self._traversal(node.l_child,x)
                else:
                    return self._traversal(node.r_child,x)
            elif 'var' in node.split_feature:
                if node.scale*np.dot((_x-np.mean(node.w*_x))**2,node.w) <= node.split_value:
                    return self._traversal(node.l_child,x)
                else:
                    return self._traversal(node.r_child,x)
            elif 'cosine' in node.split_feature:
                denom = (node.sqrt_mean_local_X*np.sqrt(node.scale*np.dot(_x**2,node.w)))
                if denom == 0:
                    denom = self.tol
                if (node.scale*np.dot(node.mean_local_X,_x))/denom<= node.split_value:
                    return self._traversal(node.l_child,x)
                else:
                    return self._traversal(node.r_child,x)


    def _validate_tree(self,
                       node: muNodeBase):
        if self.verbose_validation:
            print()
            print(f'id == {node.node_id}')
            if node.node_id != 'root':
                print(f'n_parent == {node.parent.n_elements}')
                print(f'n_elements == {node.n_elements}')
                if node.n_elements==0:
                    print(f'depth == {node.depth}')
                    print(f'error == {node.split_error}')
                    print(f'value == {node.split_value}')
            else:
                print(f'n_elements == {node.n_elements}')
            if node.is_leaf:
                print('leaf')       
            print() 
        if node.depth > self.depth:
            self.depth = node.depth
        if node.is_leaf:
            self.n_leaves += 1
            if node.l_child or node.r_child or node.R1_indexes or node.R2_indexes:
                self._valid_tree = False
        else:
            if node.best_covariate in self.n_nodes_by_input:
                self.n_nodes_by_input[node.best_covariate] +=1
            else:
                self.n_nodes_by_input[node.best_covariate] = 1
            self.n_inner_nodes += 1
            if node.split_feature=='mean_pos':
                self.n_nodes_mean_pos += 1
            elif node.split_feature=='mean_neg':
                self.n_nodes_mean_neg += 1
            elif node.split_feature=='mean_sgn':
                self.n_nodes_mean_sgn += 1 
            elif node.split_feature=='mean_uni':
                self.n_nodes_mean_uni += 1        
            elif node.split_feature=='var_pos':
                self.n_nodes_var_pos += 1
            elif node.split_feature=='var_neg':
                self.n_nodes_var_neg += 1 
            elif node.split_feature=='var_sgn':
                self.n_nodes_var_sgn += 1 
            elif node.split_feature=='var_uni':
                self.n_nodes_var_uni += 1          
            elif node.split_feature=='cosine_pos':
                self.n_nodes_cosine_pos += 1
            elif node.split_feature=='cosine_neg':
                self.n_nodes_cosine_neg += 1
            elif node.split_feature=='cosine_uni':
                self.n_nodes_cosine_neg += 1      
            elif node.split_feature=='class_cosine_pos':
                self.n_nodes_class_cosine_pos += 1
            elif node.split_feature=='class_cosine_neg':
                self.n_nodes_class_cosine_neg += 1
            elif node.split_feature=='class_cosine_uni':
                self.n_nodes_class_cosine_uni += 1
            else:
                print(f'''Invalid split_feature == {node.split_feature}
                          inside node {node.node_id} with depth {node.depth}''')
                self._valid_tree = False
            if node.l_child:                 
                self._validate_tree(node.l_child)
            if node.r_child:        
                self._validate_tree(node.r_child)                   

    
    def _print_tree(self,
                    node: muNodeBase,
                    X: List[np.ndarray]):
        node.print_curves_in_node(X[node.best_covariate])
        if not(node.is_leaf):
            node.print_w_curve()
        if node.l_child:
            self._print_tree(node.l_child,X)
        if node.r_child:    
            self._print_tree(node.r_child,X)


class muCARTClassifier(muCARTBase):
    def __init__(self,
                 solver_options: Dict,
                 criterion: str = 'gini',
                 print_tree_flag = False,
                 print_path: str = '',
                 class_weight = 'balanced',
                 verbose_validation: bool = False,
                 tol: float = 10**-9):
            self.estimator_type = 'classifier'
            self.default_scoring = accuracy_score
            self.classification_criteria = ['gini',
                                            'misclass',
                                            'entropy']
            if criterion not in self.classification_criteria:
                raise ValueError(f'Illegal criterion == {criterion}')
            self.solver_options = solver_options
            self.criterion = criterion
            self.print_tree_flag = print_tree_flag
            self.print_path = print_path
            self.class_weight = class_weight
            self.verbose_validation = verbose_validation
            self.tol = tol
            self._valid_tree = True
            self.depth = 0
            self.n_inner_nodes = 0
            self.n_leaves = 0
            self.n_nodes_mean_pos = 0
            self.n_nodes_mean_neg = 0
            self.n_nodes_mean_sgn = 0
            self.n_nodes_mean_uni = 0
            self.n_nodes_var_pos = 0
            self.n_nodes_var_neg = 0
            self.n_nodes_var_sgn = 0
            self.n_nodes_var_uni = 0
            self.n_nodes_cosine_pos = 0
            self.n_nodes_cosine_neg = 0
            self.n_nodes_cosine_uni = 0
            self.n_nodes_class_cosine_pos = 0
            self.n_nodes_class_cosine_neg = 0
            self.n_nodes_class_cosine_uni = 0
            self.n_nodes_by_input = {}
            self.parameters = {}
            self.lambd = None
            self.min_samples_leaf = 1
            self.max_depth = None
            self.root = muNodeClassifier(None,
                                         0,
                                         'root',
                                         self.criterion,
                                         self.solver_options,
                                         self.min_samples_leaf,
                                         self.lambd,
                                         self.class_weight,
                                         self.print_tree_flag,
                                         self.print_path,
                                         self.tol)


    def _make_new_node(self,
                       parent: muNodeClassifier,
                       depth: int,
                       node_id: str) -> muNodeClassifier:
        new_node = muNodeClassifier(parent,
                                    depth,
                                    node_id,
                                    self.criterion,
                                    self.solver_options,
                                    self.min_samples_leaf,
                                    self.lambd,
                                    self.class_weight,
                                    self.print_tree_flag,
                                    self.print_path,
                                    self.tol)
        return new_node


class muCARTRegressor(muCARTBase):
    def __init__(self,
                 solver_options: Dict,
                 criterion: str = 'mse',
                 print_tree_flag = False,
                 print_path: str = '',
                 verbose_validation: bool = False,
                 tol: float = 10**-9):
            self.estimator_type = 'regressor'
            self.default_scoring = r2_score
            self.regression_criteria = ['mse',
                                        'mae']
            if criterion not in self.regression_criteria:
                raise ValueError(f'Illegal criterion == {criterion}')
            self.solver_options = solver_options    
            self.criterion = criterion
            self.print_tree_flag = print_tree_flag
            self.print_path = print_path
            self.tol = tol
            self.verbose_validation = verbose_validation
            self._valid_tree = True
            self.depth = 0
            self.n_inner_nodes = 0
            self.n_leaves = 0  
            self.n_nodes_mean_pos = 0
            self.n_nodes_mean_neg = 0
            self.n_nodes_mean_sgn = 0
            self.n_nodes_mean_uni = 0
            self.n_nodes_var_pos = 0
            self.n_nodes_var_neg = 0
            self.n_nodes_var_sgn = 0
            self.n_nodes_var_uni = 0
            self.n_nodes_cosine_pos = 0
            self.n_nodes_cosine_neg = 0
            self.n_nodes_cosine_uni = 0
            self.n_nodes_class_cosine_pos = 0
            self.n_nodes_class_cosine_neg = 0
            self.n_nodes_class_cosine_uni = 0
            self.n_nodes_by_input = {}
            self.parameters = {}
            self.lambd = None
            self.min_samples_leaf = 1
            self.max_depth = None
            self.root = muNodeRegressor(None,
                                        0,
                                        'root',
                                        self.criterion,
                                        self.solver_options,
                                        self.min_samples_leaf,
                                        self.lambd,
                                        self.print_tree_flag,
                                        self.print_path,
                                        self.tol) 


    def _make_new_node(self,
                       parent: muNodeRegressor,
                       depth: int,
                       node_id: str) -> muNodeRegressor:
        new_node = muNodeRegressor(parent,
                                   depth,
                                   node_id,
                                   self.criterion,
                                   self.solver_options,
                                   self.min_samples_leaf,
                                   self.lambd,
                                   self.print_tree_flag,
                                   self.print_path,
                                   self.tol)
        return new_node

