import logging

import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo

from abc import ABC
from pathlib import Path
from typing import Dict, List
from pyomo.opt import SolverFactory
from sklearn.preprocessing import minmax_scale, scale
from sklearn.utils.class_weight import compute_class_weight

logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class muNodeBase(ABC):
    def print_w_curve(self):
        t = [i for i in range(len(self.w))]           
        plt.plot(t,
                 self.w,
                 linewidth=0.8,
                 label=str(self.lambd),
                 color='black',
                 linestyle='-')
        filename = f'w | depth_{self.depth} | id_{self.node_id} | y_{self.y} | spliterror_{round(self.split_error,2)}.pdf'
        filename = Path(self.print_path,
                        filename)
        plt.savefig(filename,
                    format='pdf')
        plt.clf()


    def print_curves_in_node(self,X):
        if not(self.is_leaf):
            t = [i for i in range(X.shape[1])]
            for x in X[self.R1_indexes,:]:
                plt.plot(t,
                         x,
                         linewidth=0.15,
                         color='firebrick')
            for x in X[self.R2_indexes,:]:
                plt.plot(t,
                         x,
                         linewidth=0.15,
                         color='royalblue')
            feature_range = (np.amin(X[self.R1_indexes+self.R2_indexes]),
                             np.amax(X[self.R1_indexes+self.R2_indexes]))                    
            plt.plot(t,
                     minmax_scale(self.w,feature_range=feature_range),
                     color='black',
                     linestyle='-',
                     linewidth=0.9,
                     label='weight')          
        else:
            t = [i for i in range(X.shape[1])]
            for x in X[self.node_indexes,:]:
                plt.plot(t,
                         x,
                         linewidth=0.3,
                         color='forestgreen')
            if self.node_id=='root':   
                plt.plot(t,
                         minmax_scale(self.w,feature_range=(np.amin(X),np.amax(X))),
                         color='black',
                         linestyle='-',
                         linewidth=0.9,
                         label='weight')
        topdown_path = ''
        if self.node_id=='left':
            current_id = 'L'
        elif self.node_id=='right':
            current_id = 'R'
        else:
            current_id = 'tooR'
        topdown_path += current_id
        parent = self.parent
        while parent:
            if parent.node_id=='left':
                current_id = 'L'
            elif parent.node_id=='right':
                current_id = 'R'
            else:
                current_id = 'tooR'
            topdown_path += current_id
            parent = parent.parent 
        topdown_path = topdown_path[::-1]
        if self.is_leaf:
            topdown_path += '(leaf)'
        filename = f'node | depth_{self.depth} | path_{topdown_path} | y_{self.y} | spliterror_{round(self.split_error,2)} | splitfeat_{self.split_feature} | input_{self.best_covariate}.pdf'
        filename = Path(self.print_path,
                        filename)
        plt.savefig(filename,
                    format='pdf')
        plt.clf()


    def is_not_pure(self,
                    Y: np.ndarray):
        if self.node_id == 'right':
            self.node_indexes = self.parent.R2_indexes
        elif self.node_id == 'left':
            self.node_indexes = self.parent.R1_indexes
        else:
            self.node_indexes = [i for i in range(len(Y))]            
        is_not_pure = True  
        if len(np.unique(Y[self.node_indexes]))==1:
            if self.node_id == 'left':
                self.R1_indexes = self.parent.R1_indexes
            elif self.node_id == 'right':
                self.R1_indexes = self.parent.R2_indexes
            else:
                self.R1_indexes = [i for i in range(Y.shape[0])]
            self.is_leaf = True
            self.y = Y[self.node_indexes][0] 
            self.split_error = 0
            is_not_pure = False
        return is_not_pure


    def _search_best_split(self,s_dict,Y):
        initialized = False  
        for j in s_dict:
            R1_indexes = []
            R2_indexes = []
            for i in self.node_indexes:
                if s_dict[i] <= s_dict[j]:
                    R1_indexes.append(i)
                else:    
                    R2_indexes.append(i)
            len_R1_indexes = len(R1_indexes)
            len_R2_indexes = len(R2_indexes)        
            if (len_R1_indexes==0 or len_R2_indexes==0) or \
               (len_R1_indexes>=self.min_samples_leaf and len_R2_indexes>=self.min_samples_leaf):
                error, leaf_value = self._compute_split_error(Y,
                                                              R1_indexes,
                                                              R2_indexes)
                if initialized:
                    if error<best_split_error: 
                        best_split_error = error
                        best_split_y = leaf_value
                        best_split_value = s_dict[j] 
                        best_R1_indexes = R1_indexes
                        best_R2_indexes = R2_indexes
                else:
                    initialized = True       
                    best_split_error = error
                    best_split_y = leaf_value
                    best_split_value = s_dict[j] 
                    best_R1_indexes = R1_indexes
                    best_R2_indexes = R2_indexes
        return best_R1_indexes, best_R2_indexes, best_split_error, best_split_value, best_split_y


    def _split_mean(self,w,X,Y):
        scale = 1/w.shape[0]                            
        mean_dict = {i:scale*np.dot(X[i],w) for i in self.node_indexes}
        R1_indexes, R2_indexes, split_error, split_value, split_y = self._search_best_split(mean_dict,Y)
        return R1_indexes, R2_indexes, split_error, split_value, split_y     


    def _split_var(self,w,X,Y):
        scale = 1/w.shape[0]                      
        mean_dict = {i:scale*np.dot(X[i],w) for i in self.node_indexes}
        var_dict = {i:scale*np.dot((X[i]-mean_dict[i])**2,w) for i in self.node_indexes}
        R1_indexes, R2_indexes, split_error, split_value, split_y = self._search_best_split(var_dict,Y)
        return R1_indexes, R2_indexes, split_error, split_value, split_y     


    def _split_cosine_distance(self,w,X,Y):
        scale = 1/w.shape[0]              
        mean_local_X = np.mean(X[self.node_indexes],axis=0)
        sqrt_mean_local_X = np.sqrt(scale*np.dot(mean_local_X**2,w))
        cosine_dict = {}
        for i in self.node_indexes:
            sqrt_local_X = np.sqrt(scale*np.dot(X[i]**2,w))
            denom = sqrt_local_X*sqrt_mean_local_X
            if denom == 0:
                denom = self.tol
            cosine_dict[i] = (scale*np.dot(X[i]*mean_local_X,w))/denom
        R1_indexes, R2_indexes, split_error, split_value, split_y = self._search_best_split(cosine_dict,Y)
        return R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X     


    def _run_mean(self,w,X,Y):
        p_R1_indexes, p_R2_indexes, p_split_error, p_split_value, p_split_y = self._split_mean(w[0],X,Y)
        n_R1_indexes, n_R2_indexes, n_split_error, n_split_value, n_split_y = self._split_mean(w[1],X,Y)
        s_R1_indexes, s_R2_indexes, s_split_error, s_split_value, s_split_y = self._split_mean(w[2],X,Y)
        u_R1_indexes, u_R2_indexes, u_split_error, u_split_value, u_split_y = self._split_mean(w[3],X,Y)
        errors_list = [p_split_error,n_split_error,s_split_error,u_split_error]
        best_index = np.argmin(errors_list)
        if best_index==0:
            R1_indexes = p_R1_indexes 
            R2_indexes = p_R2_indexes
            split_error = p_split_error
            split_value = p_split_value
            split_y = p_split_y
            split_feature = 'mean_pos'
        elif best_index==1:
            R1_indexes = n_R1_indexes 
            R2_indexes = n_R2_indexes
            split_error = n_split_error
            split_value = n_split_value
            split_y = n_split_y
            split_feature = 'mean_neg'
        elif best_index==2:
            R1_indexes = s_R1_indexes 
            R2_indexes = s_R2_indexes
            split_error = s_split_error
            split_value = s_split_value
            split_y = s_split_y
            split_feature = 'mean_sgn'
        else:
            R1_indexes = u_R1_indexes 
            R2_indexes = u_R2_indexes
            split_error = u_split_error
            split_value = u_split_value
            split_y = u_split_y
            split_feature = 'mean_uni'          
        mean_local_X = None
        sqrt_mean_local_X = None
        return R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature


    def _run_mean_multiple_inputs(self,w_list,X,Y):
        best_covariate = 0
        R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature = self._run_mean(w_list[0],X[0],Y)
        for i in range(1,len(X)):
            tmp_R1_indexes, tmp_R2_indexes, tmp_split_error, tmp_split_value, tmp_split_y, tmp_mean_local_X, tmp_sqrt_mean_local_X, tmp_split_feature = self._run_mean(w_list[i],X[i],Y)
            if tmp_split_error<=split_error:
                best_covariate = i
                R1_indexes = tmp_R1_indexes
                R2_indexes = tmp_R2_indexes
                split_error = tmp_split_error
                split_value = tmp_split_value
                split_y = tmp_split_y
                mean_local_X = tmp_mean_local_X
                sqrt_mean_local_X = tmp_sqrt_mean_local_X
                split_feature = tmp_split_feature
        return best_covariate, R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature      


    def _run_var(self,w,X,Y):
        p_R1_indexes, p_R2_indexes, p_split_error, p_split_value, p_split_y = self._split_var(w[0],X,Y)
        n_R1_indexes, n_R2_indexes, n_split_error, n_split_value, n_split_y = self._split_var(w[1],X,Y)
        s_R1_indexes, s_R2_indexes, s_split_error, s_split_value, s_split_y = self._split_var(w[2],X,Y)
        u_R1_indexes, u_R2_indexes, u_split_error, u_split_value, u_split_y = self._split_var(w[3],X,Y)
        errors_list = [p_split_error,n_split_error,s_split_error,u_split_error]
        best_index = np.argmin(errors_list)
        if best_index==0:
            R1_indexes = p_R1_indexes 
            R2_indexes = p_R2_indexes
            split_error = p_split_error
            split_value = p_split_value
            split_y = p_split_y
            split_feature = 'var_pos'
        elif best_index==1:
            R1_indexes = n_R1_indexes 
            R2_indexes = n_R2_indexes
            split_error = n_split_error
            split_value = n_split_value
            split_y = n_split_y
            split_feature = 'var_neg' 
        elif best_index==2:
            R1_indexes = s_R1_indexes 
            R2_indexes = s_R2_indexes
            split_error = s_split_error
            split_value = s_split_value
            split_y = s_split_y
            split_feature = 'var_sgn'
        else:
            R1_indexes = u_R1_indexes 
            R2_indexes = u_R2_indexes
            split_error = u_split_error
            split_value = u_split_value
            split_y = u_split_y
            split_feature = 'var_uni'
        mean_local_X = None
        sqrt_mean_local_X = None
        return R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature


    def _run_var_multiple_inputs(self,w_list,X,Y):
        best_covariate = 0
        R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature = self._run_var(w_list[0],X[0],Y)
        for i in range(1,len(X)):
            tmp_R1_indexes, tmp_R2_indexes, tmp_split_error, tmp_split_value, tmp_split_y, tmp_mean_local_X, tmp_sqrt_mean_local_X, tmp_split_feature = self._run_var(w_list[i],X[i],Y)
            if tmp_split_error<=split_error:
                best_covariate = i
                R1_indexes = tmp_R1_indexes
                R2_indexes = tmp_R2_indexes
                split_error = tmp_split_error
                split_value = tmp_split_value
                split_y = tmp_split_y
                mean_local_X = tmp_mean_local_X
                sqrt_mean_local_X = tmp_sqrt_mean_local_X
                split_feature = tmp_split_feature
        return best_covariate, R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature


    def _run_cosine_distance(self,w,X,Y):
        p_R1_indexes, p_R2_indexes, p_split_error, p_split_value, p_split_y, p_mean_local_X, p_sqrt_mean_local_X = self._split_cosine_distance(w[0],X,Y)
        n_R1_indexes, n_R2_indexes, n_split_error, n_split_value, n_split_y, n_mean_local_X, n_sqrt_mean_local_X = self._split_cosine_distance(w[1],X,Y)
        u_R1_indexes, u_R2_indexes, u_split_error, u_split_value, u_split_y, u_mean_local_X, u_sqrt_mean_local_X = self._split_cosine_distance(w[3],X,Y)
        errors_list = [p_split_error,n_split_error,u_split_error]
        best_index = np.argmin(errors_list)
        if best_index==0:
            R1_indexes = p_R1_indexes 
            R2_indexes = p_R2_indexes
            split_error = p_split_error
            split_value = p_split_value
            split_y = p_split_y
            mean_local_X = p_mean_local_X
            sqrt_mean_local_X = p_sqrt_mean_local_X
            split_feature = 'cosine_pos'
        elif best_index==1:
            R1_indexes = n_R1_indexes 
            R2_indexes = n_R2_indexes
            split_error = n_split_error
            split_value = n_split_value
            split_y = n_split_y
            mean_local_X = n_mean_local_X
            sqrt_mean_local_X = n_sqrt_mean_local_X
            split_feature = 'cosine_neg'
        else:
            R1_indexes = u_R1_indexes 
            R2_indexes = u_R2_indexes
            split_error = u_split_error
            split_value = u_split_value
            split_y = u_split_y
            mean_local_X = u_mean_local_X
            sqrt_mean_local_X = u_sqrt_mean_local_X
            split_feature = 'cosine_uni'
        return R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature        


    def _run_cosine_distance_multiple_inputs(self,w_list,X,Y):
        best_covariate = 0
        R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature = self._run_cosine_distance(w_list[0],X[0],Y)
        for i in range(1,len(X)):
            tmp_R1_indexes, tmp_R2_indexes, tmp_split_error, tmp_split_value, tmp_split_y, tmp_mean_local_X, tmp_sqrt_mean_local_X, tmp_split_feature = self._run_cosine_distance(w_list[i],X[i],Y)
            if tmp_split_error<=split_error:
                best_covariate = i
                R1_indexes = tmp_R1_indexes
                R2_indexes = tmp_R2_indexes
                split_error = tmp_split_error
                split_value = tmp_split_value
                split_y = tmp_split_y
                mean_local_X = tmp_mean_local_X
                sqrt_mean_local_X = tmp_sqrt_mean_local_X
                split_feature = tmp_split_feature
        return best_covariate, R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature


class muNodeClassifier(muNodeBase):
    def __init__(self,
                 parent,
                 depth,
                 node_id,
                 criterion,
                 solver_options,
                 min_samples_leaf,
                 lambd,
                 class_weight,
                 print_tree_flag,
                 print_path,
                 tol):
        self.parent = parent
        self.depth = depth
        self.node_id = node_id
        self.criterion = criterion
        self.solver_options = solver_options
        self.min_samples_leaf = min_samples_leaf
        self.lambd = lambd
        self.class_weight = class_weight
        self.print_tree_flag = print_tree_flag
        self.print_path = print_path
        self.tol = tol
        self.n_elements = None
        self.l_child = None
        self.r_child = None
        self.node_indexes = None
        self.R1_indexes = []
        self.R2_indexes = []
        self.w = None
        self.scale = None
        self.split_error = None
        self.split_value = None
        self.split_feature = None
        self.best_covariate = None
        self.y = None
        self.mean_local_X = None
        self.sqrt_mean_local_X = None
        self.is_leaf = False
        self.ovr_classes = None
        self.ovr_classes_relabeled = None


    def run(self,
            X: List[np.ndarray],
            Y: np.ndarray):
        w_list = self._compute_w(X,Y)
        if len(X)==1:
            self.best_covariate = 0
            mean_R1_indexes, mean_R2_indexes, mean_split_error, mean_split_value, mean_split_y, mean_mean_local_X, mean_sqrt_mean_local_X, mean_split_feature = self._run_mean(w_list[0],X[0],Y)
            var_R1_indexes, var_R2_indexes, var_split_error, var_split_value, var_split_y, var_mean_local_X, var_sqrt_mean_local_X, var_split_feature = self._run_var(w_list[0],X[0],Y)
            cosine_R1_indexes, cosine_R2_indexes, cosine_split_error, cosine_split_value, cosine_split_y, cosine_mean_local_X, cosine_sqrt_mean_local_X, cosine_split_feature = self._run_cosine_distance(w_list[0],X[0],Y)
            classcos_R1_indexes, classcos_R2_indexes, classcos_split_error, classcos_split_value, classcos_split_y, classcos_mean_local_X, classcos_sqrt_mean_local_X, classcos_split_feature = self._run_class_cosine_distance(w_list[0],X[0],Y)
        else:
            mean_best_covariate, mean_R1_indexes, mean_R2_indexes, mean_split_error, mean_split_value, mean_split_y, mean_mean_local_X, mean_sqrt_mean_local_X, mean_split_feature = self._run_mean_multiple_inputs(w_list,X,Y)
            var_best_covariate, var_R1_indexes, var_R2_indexes, var_split_error, var_split_value, var_split_y, var_mean_local_X, var_sqrt_mean_local_X, var_split_feature = self._run_var_multiple_inputs(w_list,X,Y)
            cosine_best_covariate, cosine_R1_indexes, cosine_R2_indexes, cosine_split_error, cosine_split_value, cosine_split_y, cosine_mean_local_X, cosine_sqrt_mean_local_X, cosine_split_feature = self._run_cosine_distance_multiple_inputs(w_list,X,Y)
            classcos_best_covariate, classcos_R1_indexes, classcos_R2_indexes, classcos_split_error, classcos_split_value, classcos_split_y, classcos_mean_local_X, classcos_sqrt_mean_local_X, classcos_split_feature = self._run_class_cosine_distance_multiple_inputs(w_list,X,Y)
        errors_list = [mean_split_error,var_split_error,cosine_split_error, classcos_split_error]
        min_error_index = errors_list.index(min(errors_list))    
        if min_error_index==0:
            self.R1_indexes = mean_R1_indexes
            self.R2_indexes = mean_R2_indexes
            self.split_error = mean_split_error
            self.split_value = mean_split_value
            self.y = mean_split_y
            self.mean_local_X = mean_mean_local_X  
            self.sqrt_mean_local_X = mean_sqrt_mean_local_X
            self.split_feature = mean_split_feature 
            if len(X)>1:
                self.best_covariate = mean_best_covariate
        elif min_error_index==1:    
            self.R1_indexes = var_R1_indexes
            self.R2_indexes = var_R2_indexes
            self.split_error = var_split_error
            self.split_value = var_split_value
            self.y = var_split_y
            self.mean_local_X = var_mean_local_X  
            self.sqrt_mean_local_X = var_sqrt_mean_local_X
            self.split_feature = var_split_feature
            if len(X)>1:
                self.best_covariate = var_best_covariate
        elif min_error_index==2:
            self.R1_indexes = cosine_R1_indexes
            self.R2_indexes = cosine_R2_indexes
            self.split_error = cosine_split_error
            self.split_value = cosine_split_value
            self.y = cosine_split_y
            self.mean_local_X = cosine_mean_local_X  
            self.sqrt_mean_local_X = cosine_sqrt_mean_local_X
            self.split_feature = cosine_split_feature 
            if len(X)>1:
                self.best_covariate = cosine_best_covariate
        else:
            self.R1_indexes = classcos_R1_indexes
            self.R2_indexes = classcos_R2_indexes
            self.split_error = classcos_split_error
            self.split_value = classcos_split_value
            self.y = classcos_split_y
            self.mean_local_X = classcos_mean_local_X  
            self.sqrt_mean_local_X = classcos_sqrt_mean_local_X
            self.split_feature = classcos_split_feature 
            if len(X)>1:
                self.best_covariate = classcos_best_covariate        
        if 'pos' in self.split_feature:    
            self.w = w_list[self.best_covariate][0]
        elif 'neg' in self.split_feature:
            self.w = w_list[self.best_covariate][1]
        elif 'sgn' in self.split_feature:
            self.w = w_list[self.best_covariate][2]
        else:
            self.w = w_list[self.best_covariate][3]
        self.scale = 1/self.w.shape[0]


    def _compute_split_error(self,Y,R1_indexes,R2_indexes):
        R1_error = self._compute_prospect_node_error(Y[R1_indexes])
        R2_error = self._compute_prospect_node_error(Y[R2_indexes])
        unique_labels = np.unique(Y[R1_indexes+R2_indexes])     
        counts_dict = {label:0 for label in unique_labels}
        for label in Y[R1_indexes+R2_indexes]:
                counts_dict[label] += 1
        max_count = 0
        max_count_label = None    
        for label in unique_labels:
            if counts_dict[label] >= max_count:
                max_count = counts_dict[label]
                max_count_label = label
        leaf_value = max_count_label
        split_error = (len(R1_indexes)/(len(R1_indexes)+len(R2_indexes)))*R1_error + \
                      (len(R2_indexes)/(len(R1_indexes)+len(R2_indexes)))*R2_error
        return split_error, leaf_value


    def _compute_prospect_node_error(self,
                                     this_node_Y: np.ndarray):
        error = 0
        if len(this_node_Y)>0:
            unique_labels = np.unique(this_node_Y)
            counts_dict = {label:0 for label in unique_labels}          
            for label in this_node_Y:
                counts_dict[label] += 1
            if self.class_weight=='balanced':
                weights = compute_class_weight('balanced',
                                               classes=unique_labels,
                                               y=this_node_Y)
                weights_dict = {label:weight for label,weight in zip(unique_labels,weights)}
                for label in unique_labels: 
                    counts_dict[label] = counts_dict[label]*weights_dict[label]
            max_count = 0
            max_count_label = None    
            for label in unique_labels:
                if counts_dict[label] >= max_count:
                    max_count = counts_dict[label]
                    max_count_label = label
            if self.criterion == 'misclass':        
                error = 1 - (max_count/len(this_node_Y))
            elif self.criterion == 'gini':
                for label in unique_labels:
                    error += (counts_dict[label]/len(this_node_Y))*(1 - (counts_dict[label]/len(this_node_Y)))
            else:
                for label in unique_labels:
                    if counts_dict[label]>0:
                        error -= (counts_dict[label]/len(this_node_Y))*np.log((counts_dict[label]/len(this_node_Y)))
        return error


    def _split_class_cosine_distance(self,w,X,Y): 
        scale = 1/w.shape[0]              
        mean_local_X_list = []
        sqrt_mean_local_X_list = []
        cosine_dict_list = []
        R1_indexes_list = []
        R2_indexes_list = []
        split_error_list = []
        split_value_list = []
        split_y_list = []
        unique_labels = sorted(np.unique(Y[self.node_indexes]))
        for label in unique_labels:
            class_indexes = [i for i in self.node_indexes if Y[i]==label]
            other_classes_indexes = [i for i in self.node_indexes if Y[i]!=label]
            class_mean_local_X = np.mean(X[class_indexes],axis=0)
            sqrt_class_mean_local_X = np.sqrt(scale*np.dot(class_mean_local_X**2,w))
            cosine_dict = {}
            for i in self.node_indexes:
                sqrt_local_X = np.sqrt(scale*np.dot(X[i]**2,w))
                denom = sqrt_local_X*sqrt_class_mean_local_X
                if denom == 0:
                    denom = self.tol
                cosine_dict[i] = (scale*np.dot(X[i]*class_mean_local_X,w))/denom
            R1_indexes, R2_indexes, split_error, split_value, split_y = self._search_best_split(cosine_dict,Y)
            mean_local_X_list.append(class_mean_local_X)
            sqrt_mean_local_X_list.append(sqrt_class_mean_local_X)
            cosine_dict_list.append(cosine_dict)
            R1_indexes_list.append(R1_indexes)
            R2_indexes_list.append(R2_indexes)
            split_error_list.append(split_error)
            split_value_list.append(split_value)
            split_y_list.append(split_y)
        min_error_index = split_error_list.index(min(split_error_list))
        mean_local_X = mean_local_X_list[min_error_index]
        sqrt_mean_local_X = sqrt_mean_local_X_list[min_error_index]
        R1_indexes = R1_indexes_list[min_error_index]
        R2_indexes = R2_indexes_list[min_error_index]
        split_error = split_error_list[min_error_index]
        split_value = split_value_list[min_error_index]
        split_y = split_y_list[min_error_index]
        return R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X


    def _run_class_cosine_distance(self,w,X,Y):
        p_R1_indexes, p_R2_indexes, p_split_error, p_split_value, p_split_y, p_mean_local_X, p_sqrt_mean_local_X = self._split_class_cosine_distance(w[0],X,Y)
        n_R1_indexes, n_R2_indexes, n_split_error, n_split_value, n_split_y, n_mean_local_X, n_sqrt_mean_local_X = self._split_class_cosine_distance(w[1],X,Y)
        u_R1_indexes, u_R2_indexes, u_split_error, u_split_value, u_split_y, u_mean_local_X, u_sqrt_mean_local_X = self._split_class_cosine_distance(w[3],X,Y)
        errors_list = [p_split_error,n_split_error,u_split_error]
        best_index = np.argmin(errors_list)
        if best_index==0:
            R1_indexes = p_R1_indexes 
            R2_indexes = p_R2_indexes
            split_error = p_split_error
            split_value = p_split_value
            split_y = p_split_y
            mean_local_X = p_mean_local_X
            sqrt_mean_local_X = p_sqrt_mean_local_X
            split_feature = 'class_cosine_pos'
        elif best_index==1:
            R1_indexes = n_R1_indexes 
            R2_indexes = n_R2_indexes
            split_error = n_split_error
            split_value = n_split_value
            split_y = n_split_y
            mean_local_X = n_mean_local_X
            sqrt_mean_local_X = n_sqrt_mean_local_X
            split_feature = 'class_cosine_neg'
        else:
            R1_indexes = u_R1_indexes 
            R2_indexes = u_R2_indexes
            split_error = u_split_error
            split_value = u_split_value
            split_y = u_split_y
            mean_local_X = u_mean_local_X
            sqrt_mean_local_X = u_sqrt_mean_local_X
            split_feature = 'class_cosine_uni'
        return R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature      


    def _run_class_cosine_distance_multiple_inputs(self,w_list,X,Y):
        best_covariate = 0
        R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature = self._run_class_cosine_distance(w_list[0],X[0],Y)
        for i in range(1,len(X)):
            tmp_R1_indexes, tmp_R2_indexes, tmp_split_error, tmp_split_value, tmp_split_y, tmp_mean_local_X, tmp_sqrt_mean_local_X, tmp_split_feature = self._run_class_cosine_distance(w_list[i],X[i],Y)
            if tmp_split_error<=split_error:
                best_covariate = i
                R1_indexes = tmp_R1_indexes
                R2_indexes = tmp_R2_indexes
                split_error = tmp_split_error
                split_value = tmp_split_value
                split_y = tmp_split_y
                mean_local_X = tmp_mean_local_X
                sqrt_mean_local_X = tmp_sqrt_mean_local_X
                split_feature = tmp_split_feature
        return best_covariate, R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature


    def _compute_w(self,X,Y):
        class_counts_list = [(y,list(Y[self.node_indexes]).count(y)) for y in np.unique(Y[self.node_indexes])]
        max_count = 0
        modal_class = None
        for tupl in class_counts_list:
            if tupl[1]>max_count:
                max_count = tupl[1] 
                modal_class = tupl[0]
        self.ovr_classes = [[modal_class,max_count], [tupl for tupl in class_counts_list if tupl[0]!=modal_class]]
        _Y = np.zeros(shape=Y[self.node_indexes].shape,dtype=int)
        _Y[np.where(Y[self.node_indexes]!=modal_class)] = 1
        self.ovr_classes_relabeled = [(y,list(_Y).count(y)) for y in np.unique(_Y)]
        w_list = []
        for i in range(len(X)):
            _X = scale(X[i])
            w_pos = self._solve_optimization_problem(_X[self.node_indexes],_Y,'pos')
            w_neg = self._solve_optimization_problem(_X[self.node_indexes],_Y,'neg')
            w_sgn = self._solve_optimization_problem(_X[self.node_indexes],_Y,'sgn')
            w_uni = np.ones(_X.shape[1])/_X.shape[1]
            w_list.append((w_pos,w_neg,w_sgn,w_uni))
        return w_list


    def _solve_optimization_problem(self,X,Y,measure_type):
        P = X.shape[1]
        N = X.shape[0]
        if self.class_weight=='balanced':
            unique_weights = N/(len(np.unique(Y))*np.bincount(Y))
            sample_weights = np.zeros(N)
            for i in np.unique(Y):
                sample_weights[np.where(Y==i)] = unique_weights[i]
        else:
            sample_weights = np.ones(N)
        model = pyo.ConcreteModel()
        model.P = pyo.RangeSet(0,P-1)
        model.intercept = pyo.Var(domain=pyo.Reals)
        if measure_type=='pos':
            model.w = pyo.Var(model.P,
                              domain=pyo.NonNegativeReals)
            for p in range(P):
                model.w[p]=1/P
        elif measure_type=='neg':
            model.w = pyo.Var(model.P,
                              domain=pyo.NonPositiveReals)
            for p in range(P):
                model.w[p]=-1/P
        else:
            model.w = pyo.Var(model.P,
                              domain=pyo.Reals)
            for p in range(P):
                model.w[p]=0
        
        def obj_expression(model):
            obj = -sum(sample_weights[j]*(Y[j]*((1/P)*sum(X[j,p]*model.w[p] for p in range(P))+model.intercept) - pyo.log(1+pyo.exp(model.intercept+(1/P)*sum(X[j,p]*model.w[p] for p in range(P)))))  for j in range(N))
            if self.lambd>0:
                obj = obj + self.lambd*(1/P)*sum(model.w[p]**2 for p in range(0,P))
            return obj
        model.obj = pyo.Objective(rule=obj_expression,
                                  sense=pyo.minimize)

        def w_constraint_rule_sum_I(model):
            if measure_type=='pos' or measure_type=='sgn':
                I = 1
            else:
                I = -1
            return (1/P)*sum(model.w[p] for p in model.P) == I
        model.w_constraint = pyo.Constraint(rule=w_constraint_rule_sum_I)

        opt = SolverFactory(self.solver_options['solver'],
                            options={'max_iter': self.solver_options['max_iter']}) 
        opt.solve(model)
        w = []
        for p in range(P):
            if measure_type=='neg':
                w.append(abs(pyo.value(model.w[p])))
            else:
                w.append(pyo.value(model.w[p]))
        return np.array(w)


class muNodeRegressor(muNodeBase):
    def __init__(self,
                 parent,
                 depth,
                 node_id,
                 criterion,
                 solver_options,
                 min_samples_leaf,
                 lambd,
                 print_tree_flag,
                 print_path,
                 tol):
        self.parent = parent
        self.depth = depth
        self.node_id = node_id
        self.criterion = criterion
        self.solver_options = solver_options
        self.min_samples_leaf = min_samples_leaf
        self.lambd = lambd
        self.print_tree_flag = print_tree_flag
        self.print_path = print_path
        self.tol = tol
        self.n_elements = None
        self.l_child = None
        self.r_child = None
        self.node_indexes = None
        self.R1_indexes = []
        self.R2_indexes = []
        self.w = None
        self.scale = None
        self.split_error = None
        self.split_value = None
        self.split_feature = None
        self.best_covariate = None
        self.y = None
        self.mean_local_X = None
        self.sqrt_mean_local_X = None
        self.is_leaf = False


    def run(self,
            X: List[np.ndarray],
            Y: np.ndarray):
        w_list = self._compute_w(X,Y)
        if len(X)==1:
            self.best_covariate = 0
            mean_R1_indexes, mean_R2_indexes, mean_split_error, mean_split_value, mean_split_y, mean_mean_local_X, mean_sqrt_mean_local_X, mean_split_feature = self._run_mean(w_list[0],X[0],Y)
            var_R1_indexes, var_R2_indexes, var_split_error, var_split_value, var_split_y, var_mean_local_X, var_sqrt_mean_local_X, var_split_feature = self._run_var(w_list[0],X[0],Y)
            cosine_R1_indexes, cosine_R2_indexes, cosine_split_error, cosine_split_value, cosine_split_y, cosine_mean_local_X, cosine_sqrt_mean_local_X, cosine_split_feature = self._run_cosine_distance(w_list[0],X[0],Y)
        else:
            mean_best_covariate, mean_R1_indexes, mean_R2_indexes, mean_split_error, mean_split_value, mean_split_y, mean_mean_local_X, mean_sqrt_mean_local_X, mean_split_feature = self._run_mean_multiple_inputs(w_list,X,Y)
            var_best_covariate, var_R1_indexes, var_R2_indexes, var_split_error, var_split_value, var_split_y, var_mean_local_X, var_sqrt_mean_local_X, var_split_feature = self._run_var_multiple_inputs(w_list,X,Y)
            cosine_best_covariate, cosine_R1_indexes, cosine_R2_indexes, cosine_split_error, cosine_split_value, cosine_split_y, cosine_mean_local_X, cosine_sqrt_mean_local_X, cosine_split_feature = self._run_cosine_distance_multiple_inputs(w_list,X,Y)
        
        errors_list = [mean_split_error,var_split_error,cosine_split_error]
        min_error_index = errors_list.index(min(errors_list))    
        if min_error_index==0:
            self.R1_indexes = mean_R1_indexes
            self.R2_indexes = mean_R2_indexes
            self.split_error = mean_split_error
            self.split_value = mean_split_value
            self.y = mean_split_y
            self.mean_local_X = mean_mean_local_X  
            self.sqrt_mean_local_X = mean_sqrt_mean_local_X
            self.split_feature = mean_split_feature 
            if len(X)>1:
                self.best_covariate = mean_best_covariate
        elif min_error_index==1:    
            self.R1_indexes = var_R1_indexes
            self.R2_indexes = var_R2_indexes
            self.split_error = var_split_error
            self.split_value = var_split_value
            self.y = var_split_y
            self.mean_local_X = var_mean_local_X  
            self.sqrt_mean_local_X = var_sqrt_mean_local_X
            self.split_feature = var_split_feature
            if len(X)>1:
                self.best_covariate = var_best_covariate
        else:
            self.R1_indexes = cosine_R1_indexes
            self.R2_indexes = cosine_R2_indexes
            self.split_error = cosine_split_error
            self.split_value = cosine_split_value
            self.y = cosine_split_y
            self.mean_local_X = cosine_mean_local_X  
            self.sqrt_mean_local_X = cosine_sqrt_mean_local_X
            self.split_feature = cosine_split_feature 
            if len(X)>1:
                self.best_covariate = cosine_best_covariate
        if 'pos' in self.split_feature:    
            self.w = w_list[self.best_covariate][0]
        elif 'neg' in self.split_feature:
            self.w = w_list[self.best_covariate][1]
        elif 'sgn' in self.split_feature:
            self.w = w_list[self.best_covariate][2]
        else:
            self.w = w_list[self.best_covariate][3]
        self.scale = 1/self.w.shape[0]


    def _compute_split_error(self,Y,R1_indexes,R2_indexes):
        R1_error = self._compute_prospect_node_error(Y[R1_indexes])
        R2_error = self._compute_prospect_node_error(Y[R2_indexes])
        leaf_value = np.mean(Y[R1_indexes+R2_indexes])
        split_error = (len(R1_indexes)/(len(R1_indexes)+len(R2_indexes)))*R1_error + \
                      (len(R2_indexes)/(len(R1_indexes)+len(R2_indexes)))*R2_error
        return split_error, leaf_value


    def _compute_prospect_node_error(self,
                                     this_node_Y: np.ndarray):
        error = 0
        if len(this_node_Y)>0:
            if self.criterion == 'mse':
                error = (1/len(this_node_Y))*((this_node_Y-np.mean(this_node_Y))**2).sum()
            else:
                error = (1/len(this_node_Y))*np.abs(this_node_Y-np.median(this_node_Y)).sum()          
        return error


    def _compute_w(self,X,Y):
        _Y = Y[self.node_indexes]  
        w_list = []
        for i in range(len(X)):
            _X = scale(X[i])
            w_pos = self._solve_optimization_problem(_X[self.node_indexes],_Y,'pos')
            w_neg = self._solve_optimization_problem(_X[self.node_indexes],_Y,'neg')
            w_sgn = self._solve_optimization_problem(_X[self.node_indexes],_Y,'sgn')
            w_uni = np.ones(_X.shape[1])/_X.shape[1]
            w_list.append((w_pos,w_neg,w_sgn,w_uni))
        return w_list


    def _solve_optimization_problem(self,X,Y,measure_type):
        P = X.shape[1]
        N = X.shape[0]
        model = pyo.ConcreteModel()
        model.P = pyo.RangeSet(0,P-1)
        model.intercept = pyo.Var(domain=pyo.Reals)
        if measure_type=='pos':
            model.w = pyo.Var(model.P,
                              domain=pyo.NonNegativeReals)
            for p in range(P):
                model.w[p]=1/P
        elif measure_type=='neg':
            model.w = pyo.Var(model.P,
                              domain=pyo.NonPositiveReals)
            for p in range(P):
                model.w[p]=-1/P
        else:
            model.w = pyo.Var(model.P,
                              domain=pyo.Reals)
            for p in range(P):
                model.w[p]=0

        def obj_expression(model):
            obj = sum((Y[j]-(1/P)*sum(X[j,p]*model.w[p] for p in range(P))-model.intercept)**2 for j in range(N))
            if self.lambd>0:
                obj = obj + self.lambd*(1/P)*sum(model.w[p]**2 for p in range(0,P))
            return obj
        model.obj = pyo.Objective(rule=obj_expression,
                                  sense=pyo.minimize)

        def w_constraint_rule_sum_I(model):
            if measure_type=='pos' or measure_type=='sgn':
                I = 1
            else:
                I = -1
            return (1/P)*sum(model.w[p] for p in model.P) == I
        model.w_constraint = pyo.Constraint(rule=w_constraint_rule_sum_I)

        opt = SolverFactory(self.solver_options['solver'],
                            options={'max_iter':self.solver_options['max_iter']})
        opt.solve(model)

        w = []
        for p in range(P):
            if measure_type=='neg':
                w.append(abs(pyo.value(model.w[p])))
            else:
                w.append(pyo.value(model.w[p]))
        return np.array(w)

