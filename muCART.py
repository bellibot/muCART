import sys, os, math, itertools, time, joblib
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from sklearn.preprocessing import minmax_scale, scale
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)  
    
        
class mu_node():
    def __init__(self,parent,depth,node_id,min_samples_leaf,lambd,class_weight,solver_options,print_flag,task_flag,loss_flag,print_path,eps):
        self.parent = parent
        self.depth = depth
        self.node_id = node_id
        self.n_elements = None 
        self.min_samples_leaf = min_samples_leaf
        self.lambd = lambd
        self.class_weight = class_weight
        self.solver_options = solver_options
        self.print_flag = print_flag
        self.task_flag = task_flag
        self.loss_flag = loss_flag
        self.print_path = print_path
        self.eps = eps
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
        self.best_input = None
        self.y = None
        self.mean_local_X = None
        self.sqrt_mean_local_X = None
        self.is_leaf = False
        self.ovr_classes = None
        self.ovr_classes_relabeled = None
        
            
    def print_w_curve(self):
        t = [i for i in range(len(self.w))]           
        plt.plot(t, self.w ,linewidth=0.8,label=str(self.lambd), color='black', linestyle='-')
        filename = os.path.join(self.print_path,'w | depth_{} | id_{} | y_{} | spliterror_{}.pdf'.format(self.depth,self.node_id,self.y,round(self.split_error,2)))
        plt.savefig(filename,format='pdf')
        plt.clf()
            
                            
    def print_curves_in_node(self,X):
        if not(self.is_leaf):
            t = [i for i in range(X.shape[1])]
            for x in X[self.R1_indexes,:]:
                plt.plot(t, x ,linewidth=0.15,color='firebrick')
            for x in X[self.R2_indexes,:]:
                plt.plot(t, x ,linewidth=0.15,color='royalblue')
            feature_range = (np.amin(X[self.R1_indexes+self.R2_indexes]),np.amax(X[self.R1_indexes+self.R2_indexes]))                    
            plt.plot(t,minmax_scale(self.w,feature_range=feature_range),color='black',linestyle='-',linewidth=0.9,label='weight')          
        else:
            t = [i for i in range(X.shape[1])]
            for x in X[self.node_indexes,:]:
                plt.plot(t, x ,linewidth=0.3,color='forestgreen')
            if self.node_id=='root':   
                plt.plot(t,minmax_scale(self.w,feature_range=(np.amin(X),np.amax(X))),color='black',linestyle='-',linewidth=0.9,label='weight')
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
        filename = os.path.join(self.print_path,'node | depth_{} | path_{} | y_{} | spliterror_{} | splitfeat_{} | input_{}.pdf'.format(self.depth,topdown_path,self.y,round(self.split_error,2),self.split_feature,self.best_input))
        plt.savefig(filename,format='pdf')
        plt.clf()


    def is_not_pure(self,Y):
        if self.node_id == 'right':
            self.node_indexes = self.parent.R2_indexes
        elif self.node_id == 'left':
            self.node_indexes = self.parent.R1_indexes
        else:
            self.node_indexes = [i for i in range(len(Y))]            
        is_not_pure = True
        if self.task_flag=='classification':                
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

    
    def _compute_split_error(self,Y,R1_indexes,R2_indexes):
        len_R1_indexes = len(R1_indexes)
        len_R2_indexes = len(R2_indexes)
        if self.task_flag=='classification':
            R1_error = 0
            if len_R1_indexes>0:
                unique_labels = np.unique(Y[R1_indexes])
                counts_dict = {label:0 for label in unique_labels}          
                for label in Y[R1_indexes]:
                    counts_dict[label] += 1
                if self.class_weight=='balanced':
                    weights = compute_class_weight('balanced',classes=unique_labels,y=Y[R1_indexes])
                    weights_dict = {label:weight for label,weight in zip(unique_labels,weights)}
                    for label in unique_labels: 
                        counts_dict[label] = counts_dict[label]*weights_dict[label]
                max_count = 0
                max_count_label = None    
                for label in unique_labels:
                    if counts_dict[label] >= max_count:
                        max_count = counts_dict[label]
                        max_count_label = label
                        
                if self.loss_flag == 'misclass':        
                    R1_error = 1 - (max_count/len_R1_indexes)           
                elif self.loss_flag == 'gini':
                    for label in unique_labels:
                        R1_error += (counts_dict[label]/len_R1_indexes)*(1 - (counts_dict[label]/len_R1_indexes))
                else:
                    for label in unique_labels:
                        if counts_dict[label]>0:
                            R1_error -= (counts_dict[label]/len_R1_indexes)*np.log((counts_dict[label]/len_R1_indexes))
                                                        
            R2_error = 0
            if len_R2_indexes>0:
                unique_labels = np.unique(Y[R2_indexes])        
                counts_dict = {label:0 for label in unique_labels}   
                for label in Y[R2_indexes]:
                    counts_dict[label] += 1
                if self.class_weight=='balanced':
                    weights = compute_class_weight('balanced',classes=unique_labels,y=Y[R2_indexes])
                    weights_dict = {label:weight for label,weight in zip(unique_labels,weights)}
                    for label in unique_labels: 
                        counts_dict[label] = counts_dict[label]*weights_dict[label]    
                max_count = 0
                max_count_label = None    
                for label in unique_labels:
                    if counts_dict[label] >= max_count:
                        max_count = counts_dict[label]
                        max_count_label = label
                        
                if self.loss_flag == 'misclass':        
                    R2_error = 1 - (max_count/len_R2_indexes)    
                elif self.loss_flag == 'gini':
                    for label in unique_labels:
                        R2_error += (counts_dict[label]/len_R2_indexes)*(1 - (counts_dict[label]/len_R2_indexes))
                else:
                    for label in unique_labels:
                        if counts_dict[label]>0:
                            R2_error -= (counts_dict[label]/len_R2_indexes)*np.log((counts_dict[label]/len_R2_indexes))
            
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
        else:
            R1_error = 0
            if len_R1_indexes>0:
                if self.loss_flag == 'mse':
                    R1_error = (1/len_R1_indexes)*((Y[R1_indexes]-np.mean(Y[R1_indexes]))**2).sum()
                else:   
                    R1_error = (1/len_R1_indexes)*np.abs(Y[R1_indexes]-np.median(Y[R1_indexes])).sum()
                                    
            R2_error = 0
            if len_R2_indexes>0: 
                if self.loss_flag == 'mse':
                    R2_error = (1/len_R2_indexes)*((Y[R2_indexes]-np.mean(Y[R2_indexes]))**2).sum()
                else:   
                    R2_error = (1/len_R2_indexes)*np.abs(Y[R2_indexes]-np.median(Y[R2_indexes])).sum()
      
            leaf_value = np.mean(Y[R1_indexes+R2_indexes])
            
        split_error = (len_R1_indexes/(len_R1_indexes+len_R2_indexes))*R1_error + (len_R2_indexes/(len_R1_indexes+len_R2_indexes))*R2_error                
        return split_error, leaf_value
                   
           
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
            if (len_R1_indexes==0 or len_R2_indexes==0) or (len_R1_indexes>=self.min_samples_leaf and len_R2_indexes>=self.min_samples_leaf):
                error, leaf_value = self._compute_split_error(Y,R1_indexes,R2_indexes)
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
                denom = self.eps
            cosine_dict[i] = (scale*np.dot(X[i]*mean_local_X,w))/denom
        R1_indexes, R2_indexes, split_error, split_value, split_y = self._search_best_split(cosine_dict,Y)
        return R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X     
    
    
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
                    denom = self.eps
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
        best_input = 0
        R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature = self._run_mean(w_list[0],X[0],Y)
        for i in range(1,len(X)):
            tmp_R1_indexes, tmp_R2_indexes, tmp_split_error, tmp_split_value, tmp_split_y, tmp_mean_local_X, tmp_sqrt_mean_local_X, tmp_split_feature = self._run_mean(w_list[i],X[i],Y)
            if tmp_split_error<=split_error:
                best_input = i
                R1_indexes = tmp_R1_indexes
                R2_indexes = tmp_R2_indexes
                split_error = tmp_split_error
                split_value = tmp_split_value
                split_y = tmp_split_y
                mean_local_X = tmp_mean_local_X
                sqrt_mean_local_X = tmp_sqrt_mean_local_X
                split_feature = tmp_split_feature
        return best_input, R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature      
             
             
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
        best_input = 0
        R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature = self._run_var(w_list[0],X[0],Y)
        for i in range(1,len(X)):
            tmp_R1_indexes, tmp_R2_indexes, tmp_split_error, tmp_split_value, tmp_split_y, tmp_mean_local_X, tmp_sqrt_mean_local_X, tmp_split_feature = self._run_var(w_list[i],X[i],Y)
            if tmp_split_error<=split_error:
                best_input = i
                R1_indexes = tmp_R1_indexes
                R2_indexes = tmp_R2_indexes
                split_error = tmp_split_error
                split_value = tmp_split_value
                split_y = tmp_split_y
                mean_local_X = tmp_mean_local_X
                sqrt_mean_local_X = tmp_sqrt_mean_local_X
                split_feature = tmp_split_feature
        return best_input, R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature
        
                            
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
        best_input = 0
        R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature = self._run_cosine_distance(w_list[0],X[0],Y)
        for i in range(1,len(X)):
            tmp_R1_indexes, tmp_R2_indexes, tmp_split_error, tmp_split_value, tmp_split_y, tmp_mean_local_X, tmp_sqrt_mean_local_X, tmp_split_feature = self._run_cosine_distance(w_list[i],X[i],Y)
            if tmp_split_error<=split_error:
                best_input = i
                R1_indexes = tmp_R1_indexes
                R2_indexes = tmp_R2_indexes
                split_error = tmp_split_error
                split_value = tmp_split_value
                split_y = tmp_split_y
                mean_local_X = tmp_mean_local_X
                sqrt_mean_local_X = tmp_sqrt_mean_local_X
                split_feature = tmp_split_feature
        return best_input, R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature
        
        
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
        best_input = 0
        R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature = self._run_class_cosine_distance(w_list[0],X[0],Y)
        for i in range(1,len(X)):
            tmp_R1_indexes, tmp_R2_indexes, tmp_split_error, tmp_split_value, tmp_split_y, tmp_mean_local_X, tmp_sqrt_mean_local_X, tmp_split_feature = self._run_class_cosine_distance(w_list[i],X[i],Y)
            if tmp_split_error<=split_error:
                best_input = i
                R1_indexes = tmp_R1_indexes
                R2_indexes = tmp_R2_indexes
                split_error = tmp_split_error
                split_value = tmp_split_value
                split_y = tmp_split_y
                mean_local_X = tmp_mean_local_X
                sqrt_mean_local_X = tmp_sqrt_mean_local_X
                split_feature = tmp_split_feature
        return best_input, R1_indexes, R2_indexes, split_error, split_value, split_y, mean_local_X, sqrt_mean_local_X, split_feature


    def run(self,X,Y):
        w_list = self._compute_w(X,Y)
        if len(X)==1:
            self.best_input = 0
            mean_R1_indexes, mean_R2_indexes, mean_split_error, mean_split_value, mean_split_y, mean_mean_local_X, mean_sqrt_mean_local_X, mean_split_feature = self._run_mean(w_list[0],X[0],Y)
            var_R1_indexes, var_R2_indexes, var_split_error, var_split_value, var_split_y, var_mean_local_X, var_sqrt_mean_local_X, var_split_feature = self._run_var(w_list[0],X[0],Y)
            cosine_R1_indexes, cosine_R2_indexes, cosine_split_error, cosine_split_value, cosine_split_y, cosine_mean_local_X, cosine_sqrt_mean_local_X, cosine_split_feature = self._run_cosine_distance(w_list[0],X[0],Y)
        else:
            mean_best_input, mean_R1_indexes, mean_R2_indexes, mean_split_error, mean_split_value, mean_split_y, mean_mean_local_X, mean_sqrt_mean_local_X, mean_split_feature = self._run_mean_multiple_inputs(w_list,X,Y)
            var_best_input, var_R1_indexes, var_R2_indexes, var_split_error, var_split_value, var_split_y, var_mean_local_X, var_sqrt_mean_local_X, var_split_feature = self._run_var_multiple_inputs(w_list,X,Y)
            cosine_best_input, cosine_R1_indexes, cosine_R2_indexes, cosine_split_error, cosine_split_value, cosine_split_y, cosine_mean_local_X, cosine_sqrt_mean_local_X, cosine_split_feature = self._run_cosine_distance_multiple_inputs(w_list,X,Y)
        
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
                self.best_input = mean_best_input
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
                self.best_input = var_best_input
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
                self.best_input = cosine_best_input
        if self.task_flag=='classification':
            if len(X)==1:
                self.best_input = 0
                classcos_R1_indexes, classcos_R2_indexes, classcos_split_error, classcos_split_value, classcos_split_y, classcos_mean_local_X, classcos_sqrt_mean_local_X, classcos_cosine_split_feature = self._run_class_cosine_distance(w_list[0],X[0],Y)
            else:
                classcos_best_input, classcos_R1_indexes, classcos_R2_indexes, classcos_split_error, classcos_split_value, classcos_split_y, classcos_mean_local_X, classcos_sqrt_mean_local_X, classcos_cosine_split_feature = self._run_class_cosine_distance_multiple_inputs(w_list,X,Y)
            if self.split_error>classcos_split_error:
                self.R1_indexes = classcos_R1_indexes
                self.R2_indexes = classcos_R2_indexes
                self.split_error = classcos_split_error
                self.split_value = classcos_split_value
                self.y = classcos_split_y
                self.mean_local_X = classcos_mean_local_X  
                self.sqrt_mean_local_X = classcos_sqrt_mean_local_X
                self.split_feature = classcos_cosine_split_feature 
                if len(X)>1:
                    self.best_input = classcos_best_input
        if 'pos' in self.split_feature:    
            self.w = w_list[self.best_input][0]
        elif 'neg' in self.split_feature:
            self.w = w_list[self.best_input][1]
        elif 'sgn' in self.split_feature:
            self.w = w_list[self.best_input][2]
        else:
            self.w = w_list[self.best_input][3]
        self.scale = 1/self.w.shape[0]
     
            
    def _compute_w(self,X,Y):
        if self.task_flag=='classification':
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
        else:
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
        if self.task_flag=='classification' and self.class_weight=='balanced':
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
            model.w = pyo.Var(model.P, domain=pyo.NonNegativeReals)
            for p in range(P):
                model.w[p]=1/P
        elif measure_type=='neg':
            model.w = pyo.Var(model.P, domain=pyo.NonPositiveReals)
            for p in range(P):
                model.w[p]=-1/P
        else:
            model.w = pyo.Var(model.P, domain=pyo.Reals)
            for p in range(P):
                model.w[p]=0
        
        def obj_expression(model):
            if self.task_flag=='classification':   
                obj = -sum(sample_weights[j]*(Y[j]*((1/P)*sum(X[j,p]*model.w[p] for p in range(P))+model.intercept) - pyo.log(1+pyo.exp(model.intercept+(1/P)*sum(X[j,p]*model.w[p] for p in range(P)))))  for j in range(N))
            else:
                obj = sum((Y[j]-(1/P)*sum(X[j,p]*model.w[p] for p in range(P))-model.intercept)**2 for j in range(N))
            if self.lambd>0:
                obj = obj + self.lambd*(1/P)*sum(model.w[p]**2 for p in range(0,P))
            return obj
        model.obj = pyo.Objective(rule=obj_expression, sense=pyo.minimize)
        
        def w_constraint_rule_sum_I(model):
            if measure_type=='pos' or measure_type=='sgn':
                I = 1
            else:
                I = -1
            return (1/P)*sum(model.w[p] for p in model.P) == I
        model.w_constraint = pyo.Constraint(rule=w_constraint_rule_sum_I)
         
        opt = SolverFactory(self.solver_options['solver'],options={'max_iter':self.solver_options['max_iter']}) 
        opt.solve(model)
        
        w = []
        for p in range(P):
            if measure_type=='neg':
                w.append(abs(pyo.value(model.w[p])))
            else: 
                w.append(pyo.value(model.w[p]))
        return np.array(w)
              
                                     
class tree():
    def __init__(self,lambd,solver_options,print_flag,loss_flag,print_path,min_samples_leaf=1,max_depth=None,class_weight='balanced',eps=10**(-16),verbose_validation=False):
            self.regression_losses = ['mse','mae']
            self.classification_losses = ['gini','misclass','entropy']
            if loss_flag in self.classification_losses:
                self.task_flag = 'classification'
            elif loss_flag in self.regression_losses:
                self.task_flag = 'regression'
            else:          
                print('Illegal Loss Flag == {}'.format(loss_flag))
                print('Terminating...')
                sys.exit()
            if lambd<0:
                print('Illegal (Negative) Value for Lambda')
                sys.exit()    
            self.lambd = lambd
            self.min_samples_leaf = min_samples_leaf
            self.max_depth = max_depth   
            self.class_weight = class_weight
            self.solver_options = solver_options
            self.print_flag = print_flag
            self.loss_flag = loss_flag
            self.print_path = print_path
            self.eps = eps
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
            self.root = mu_node(None,0,'root',self.min_samples_leaf,self.lambd,self.class_weight,self.solver_options,self.print_flag,self.task_flag,self.loss_flag,self.print_path,self.eps) 
    
    
    def predict(self,X):
        pred_Y = []
        for n in range(len(X[0])):
            x = [X[i][n] for i in range(len(X))]
            pred_Y.append(self._traversal(self.root,x))
        return np.array(pred_Y)
        
            
    def score(self,X,Y):
        _score = None            
        pred_Y = self.predict(X)
        if self.task_flag=='classification':
            _score = balanced_accuracy_score(Y,pred_Y)
        else:
            _score = np.mean((pred_Y-Y)**2)
        return _score
        
                
    def fit(self,X,Y):         
        if len(X)>1:
            shapes_list = [x.shape[0] for x in X]
            if not(shapes_list[1:]==shapes_list[:-1]):
                print('Error: all inputs must have the same sample size')
                sys.exit()  
        if self.root.is_not_pure(Y):        
            self.root.run(X,Y)
            self.root.n_elements = len(self.root.node_indexes)
            if self.max_depth==None:
                self.max_depth = self.root.n_elements
            
            if len(self.root.R1_indexes)==0 or len(self.root.R2_indexes)==0:
                self._finalize_node(self.root,Y)   
            else:    
                self.root.l_child = mu_node(self.root,1,'left',self.min_samples_leaf,self.lambd,self.class_weight,self.solver_options,self.print_flag,self.task_flag,self.loss_flag,self.print_path,self.eps)         
                if self.root.l_child.is_not_pure(Y) and (len(self.root.R1_indexes) >= 2*self.min_samples_leaf) and (self.max_depth>1):
                    self._split_node(self.root.l_child,X,Y)
                else:
                    self._finalize_node(self.root.l_child,Y)       
                self.root.r_child = mu_node(self.root,1,'right',self.min_samples_leaf,self.lambd,self.class_weight,self.solver_options,self.print_flag,self.task_flag,self.loss_flag,self.print_path,self.eps)    
                if self.root.r_child.is_not_pure(Y) and (len(self.root.R2_indexes) >= 2*self.min_samples_leaf) and (self.max_depth>1):    
                    self._split_node(self.root.r_child,X,Y)
                else:
                    self._finalize_node(self.root.r_child,Y)
        
            self._validate_tree(self.root)
            if not(self._valid_tree):
                print('Warning, invalid tree')
                sys.exit()      
         
            if self.print_flag:    
                self._print_tree(self.root,X)       
        else:
            print('Same response Y for each X, no need to fit the tree')   
        return self  
            
            
    def _split_node(self,node,X,Y):   
        if node.is_not_pure(Y):
            node.run(X,Y)         
            if len(node.R1_indexes)==0 or len(node.R2_indexes)==0:
                self._finalize_node(node,Y)    
            else:
                node.l_child = mu_node(node,node.depth+1,'left',self.min_samples_leaf,node.lambd,node.class_weight,node.solver_options,node.print_flag,node.task_flag,node.loss_flag,node.print_path,node.eps)
                if node.l_child.is_not_pure(Y) and (len(node.R1_indexes) >= 2*self.min_samples_leaf) and (node.depth<self.max_depth):
                    self._split_node(node.l_child,X,Y)
                else:
                    self._finalize_node(node.l_child,Y)
                                
                node.r_child = mu_node(node,node.depth+1,'right',self.min_samples_leaf,node.lambd,node.class_weight,node.solver_options,node.print_flag,node.task_flag,node.loss_flag,node.print_path,node.eps)    
                if node.r_child.is_not_pure(Y) and (len(node.R2_indexes) >= 2*self.min_samples_leaf) and (node.depth<self.max_depth):    
                    self._split_node(node.r_child,X,Y)
                else:
                    self._finalize_node(node.r_child,Y)                
        else:
            self._finalize_node(node,Y)


    def _finalize_node(self,node,Y): 
        node.R1_indexes = None
        node.R2_indexes = None
        if node.node_id=='left' or node.node_id=='right':
            node.best_input = node.parent.best_input
        else:
            node.best_input = 0
        node.n_elements = len(node.node_indexes)
        node.split_error, node.y = node._compute_split_error(Y,node.node_indexes,[])
        node.is_leaf = True
          
        
    def _traversal(self,node,x):
        if node.is_leaf:
            return node.y
        else:
            _x = x[node.best_input]
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
                    denom = self.eps
                if (node.scale*np.dot(node.mean_local_X,_x))/denom<= node.split_value:
                    return self._traversal(node.l_child,x)
                else:
                    return self._traversal(node.r_child,x)


    def _validate_tree(self,node):
        if self.verbose_validation:
            print()
            print('id == {}'.format(node.node_id))
            if node.node_id != 'root':
                print('n_parent == {}'.format(node.parent.n_elements))
                print('n_elements == {}'.format(node.n_elements))
                if node.n_elements==0:
                    print('depth == {}'.format(node.depth))
                    print('error == {}'.format(node.split_error))
                    print('value == {}'.format(node.split_value))
            else:
                print('n_elements == {}'.format(node.n_elements))
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
            if node.best_input in self.n_nodes_by_input:
                self.n_nodes_by_input[node.best_input] +=1
            else:
                self.n_nodes_by_input[node.best_input] = 1
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
                print('Invalid split_feature == {} inside node {} with depth {}'.format(node.split_feature,node.node_id,node.depth))
                self._valid_tree = False
            if node.l_child:                 
                self._validate_tree(node.l_child)
            if node.r_child:        
                self._validate_tree(node.r_child)                   

    
    def _print_tree(self,node,X):
        node.print_curves_in_node(X[node.best_input])
        if not(node.is_leaf):
            node.print_w_curve()
        if node.l_child:
            self._print_tree(node.l_child,X)
        if node.r_child:    
            self._print_tree(node.r_child,X)       
        

#TODO
class bagged_trees():
    def __init__(self,n_trees,lambd_list,min_samples_leaf_list,max_depth,balanced_bootstrap,solver_options,print_flag,loss_flag,print_path,random_state,n_jobs):
                self.regression_losses = ['mse','mae']
                self.classification_losses = ['gini','misclass','entropy']
                if loss_flag in self.classification_losses:
                    self.task_flag = 'classification'
                elif loss_flag in self.regression_losses:
                    self.task_flag = 'regression'
                else:          
                    print('Illegal Loss Flag == {}'.format(loss_flag))
                    print('Terminating...')
                    sys.exit()     
                self.n_trees = n_trees              
                self.lambd_list = lambd_list
                self.min_samples_leaf_list = min_samples_leaf_list
                self.max_depth = max_depth
                self.balanced_bootstrap = balanced_bootstrap      
                self.solver_options = solver_options
                self.random_state = random_state
                self.print_flag = print_flag
                self.loss_flag = loss_flag
                self.print_path = print_path
                self.n_jobs = n_jobs
                self.tree_list = []
                self.ib_samples = None
                self.oob_samples = None
      
      
    def predict(self,X):
        _prediction = None
        bagged_pred_Y = []
        for tree in self.tree_list:
            bagged_pred_Y.append(tree.predict(X))
        bagged_pred_Y = np.array(bagged_pred_Y)
        if self.task_flag=='classification':
            major_vote_Y = [] 
            for j in range(X[0].shape[0]):         
                unique_labels = np.unique(bagged_pred_Y[:,j])
                counts_dict = {label:0 for label in unique_labels}
                for label in bagged_pred_Y[:,j]:
                        counts_dict[label] += 1
                max_count = 0
                max_count_label = None    
                for label in unique_labels:
                    if counts_dict[label] >= max_count:
                        max_count = counts_dict[label]
                        max_count_label = label
                major_vote_Y.append(max_count_label)           
            _prediction = np.array(major_vote_Y)    
        else:
            mean_Y = [] 
            for j in range(X[0].shape[0]):
                mean_Y.append(np.mean(bagged_pred_Y[:,j]))
            _prediction = np.array(mean_Y) 
        return _prediction
        

    def score(self,X,Y):
        _score = None
        pred_Y = self.predict(X)
        if self.task_flag=='classification':
            _score = balanced_accuracy_score(Y,pred_Y)
        else:
            _score = np.mean((pred_Y-Y)**2)
        return _score        
                 
                 
    def fit(self,X,Y): 
        if self.task_flag=='classification':
            if self.balanced_bootstrap:
                self._bootstrap_balanced(Y.shape[0],Y)
            else:
                self._bootstrap(Y.shape[0])   
        else:
            self._bootstrap(Y.shape[0])  
        grid_search_oob_scores = []
        grid_search_estimators = []
        parameters = [p for p in itertools.product(self.lambd_list,self.min_samples_leaf_list)]
        for parameters_tuple in parameters:
            lambd = parameters_tuple[0]
            min_samples_leaf = parameters_tuple[1]
            estimators = []
            for i in range(self.n_trees):
                estimators.append(tree(lambd,min_samples_leaf,self.max_depth,None,self.solver_options,self.print_flag,self.loss_flag,self.print_path))                
            estimators = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(estimators[i].fit)([x[self.ib_samples[i]] for x in X],Y[self.ib_samples[i]]) for i in range(self.n_trees))
            oob_scores = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(estimators[i].score)([x[self.oob_samples[i]] for x in X],Y[self.oob_samples[i]]) for i in range(self.n_trees))
            
            grid_search_oob_scores.append(np.mean(oob_scores))
            grid_search_estimators.append(estimators)
        if self.task_flag=='classification':
            best_score_index = grid_search_oob_scores.index(max(grid_search_oob_scores))
        else:
            best_score_index = grid_search_oob_scores.index(min(grid_search_oob_scores))
        self.tree_list = grid_search_estimators[best_score_index]
        best_lambd = parameters[best_score_index][0]
        best_min_samples_leaf = parameters[best_score_index][1]
        print()
        print('muCART Ensemble: After Validating on OOB samples')
        print('Best lambd == {}'.format(best_lambd))
        print('Best min_samples_leaf == {}'.format(best_min_samples_leaf))
        print()
            
            
    def _bootstrap(self,n_samples):
        np.random.seed(self.random_state)
        self.ib_samples = np.random.choice(n_samples,(self.n_trees,n_samples))
        all_indexes = [i for i in range(n_samples)]
        self.oob_samples = []
        for j in range(self.ib_samples.shape[0]):
            self.oob_samples.append([all_indexes[i] for i in range(len(all_indexes)) if not(all_indexes[i] in self.ib_samples[j])])
        
        
    def _bootstrap_balanced(self,n_samples,Y):
        unique_classes = np.unique(Y)
        n_classes = len(unique_classes)
        n_samples_by_class = int(math.floor(n_samples/n_classes))
        indexes_by_class = {label:[i for i in range(len(Y)) if Y[i]==label] for label in unique_classes}
        ib_samples_by_class = {label:None for label in unique_classes}
        for key,val in indexes_by_class.items():
            np.random.seed(self.random_state)
            ib_samples_by_class[key] = np.random.choice(val,(self.n_trees,n_samples_by_class))
        self.ib_samples = []
        for i in range(self.n_trees):
            tree_ib_samples = []
            for key,val in ib_samples_by_class.items():
                tree_ib_samples.extend(val[i])
            self.ib_samples.append(tree_ib_samples)        
        all_indexes = [i for i in range(n_samples)]
        self.oob_samples = []
        for i in range(len(self.ib_samples)):
            ib = self.ib_samples[i]
            self.oob_samples.append([all_indexes[i] for i in range(len(all_indexes)) if not(all_indexes[i] in ib)])
        
        
                
     
                
