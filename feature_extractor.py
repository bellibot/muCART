import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sys

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,task_flag,multiple_inputs_begin_end_list=None,eps=10**(-16)):
        self.task_flag = task_flag
        self.multiple_inputs_begin_end_list = multiple_inputs_begin_end_list
        self.eps = eps
        self.mean_X_by_input = []
        self.sqrt_mean_X_by_input = []
        self.class_mean_X_dict_by_input = []
        self.sqrt_class_mean_X_dict_by_input = []
        
        
    def fit(self,X,Y):
        if self.multiple_inputs_begin_end_list:
            X_by_input = []
            for b_e in self.multiple_inputs_begin_end_list:
                begin = b_e[0]
                end = b_e[1]
                X_by_input.append(X[:,begin:end])    
        else:
            X_by_input = [X]
        for _X in X_by_input:
            mean_X = np.mean(_X,axis=0)
            self.mean_X_by_input.append(mean_X)
            self.sqrt_mean_X_by_input.append(np.sqrt(np.dot(mean_X,mean_X)))
            class_mean_X_dict = {}
            sqrt_class_mean_X_dict = {}
            if self.task_flag=='classification':
                unique_labels = sorted(list(np.unique(Y)))
                for label in unique_labels:
                    class_indexes = [i for i in range(len(Y)) if Y[i]==label]
                    class_mean_X_dict[label] = np.mean(_X[class_indexes],axis=0)
                    sqrt_class_mean_X_dict[label] = np.sqrt(np.dot(class_mean_X_dict[label],class_mean_X_dict[label]))
            self.class_mean_X_dict_by_input.append(class_mean_X_dict)
            self.sqrt_class_mean_X_dict_by_input.append(sqrt_class_mean_X_dict)
        return self
            
                
    def transform(self,X):
        if self.multiple_inputs_begin_end_list:
            X_by_input = []
            for b_e in self.multiple_inputs_begin_end_list:
                begin = b_e[0]
                end = b_e[1]
                X_by_input.append(X[:,begin:end])    
        else:
            X_by_input = [X]
        feature_X_list = []    
        for j,_X in enumerate(X_by_input):         
            cosine_distances = []
            sqrt_X = [np.sqrt(np.dot(_X[i],_X[i])) for i in range(_X.shape[0])] 
            num = [np.dot(_X[i],self.mean_X_by_input[j]) for i in range(_X.shape[0])]
            denom = [self.sqrt_mean_X_by_input[j]*sqrt_X[i] for i in range(_X.shape[0])] 
            denom = [val if val>0 else self.eps for val in denom]
            cosine_distances = np.array([num[i]/denom[i] for i in range(_X.shape[0])]).reshape(_X.shape[0],1)
            if self.task_flag=='classification':
                for label in self.class_mean_X_dict_by_input[j].keys():
                    num = [np.dot(_X[i],self.class_mean_X_dict_by_input[j][label]) for i in range(_X.shape[0])]
                    denom = [(sqrt_X[i]*self.sqrt_class_mean_X_dict_by_input[j][label]) for i in range(_X.shape[0])]
                    denom = [val if val>0 else self.eps for val in denom]
                    class_cosine_distances = np.array([num[i]/denom[i] for i in range(_X.shape[0])]).reshape(_X.shape[0],1)
                    cosine_distances = np.hstack((cosine_distances,class_cosine_distances))
            feature_X_list.append(np.hstack((np.mean(_X,axis=1).reshape(-1,1),np.var(_X,axis=1).reshape(-1,1),cosine_distances)))
        feature_X = feature_X_list[0]
        for j in range(1,len(feature_X_list)):
            feature_X = np.hstack((feature_X,feature_X_list[j]))
        return feature_X







