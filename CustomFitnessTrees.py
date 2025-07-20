# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 20:00:38 2024

@author: flavio
"""
# print('working ...')

import numpy as np

class TreeNode():
    def __init__(self, data, feature_idx, feature_val, prediction_probs, misfit_gain, output=None) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.misfit_gain = misfit_gain
        self.left = None
        self.right = None
        self.output = output

    def node_def(self) -> str:

        if (self.left or self.right):
            return f"NODE | Gain = {self.misfit_gain} | Split IF X[{self.feature_idx}] < {self.feature_val} THEN left O/W right"
        else:
            return f"LEAF | Output = {self.output}"


class DecisionTree():
    """
    Decision Tree Classifier
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """

    def __init__(self, max_depth=4, min_samples_leaf=1, 
                 min_misfit_gain=0.0, numb_of_features_splitting='sqrt') -> None:
        """
        Setting the class with hyperparameters
        max_depth: (int) -> max depth of the tree
        min_samples_leaf: (int) -> min # of samples required to be in a leaf to make the splitting possible
        min_information_gain: (float) -> min information gain required to make the splitting possible
        num_of_features_splitting: (str) ->  when splitting if sqrt then sqrt(# of features) features considered, 
                                                            if log then log(# of features) features considered
                                                            else all features are considered
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_misfit_gain = min_misfit_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.fitness = lambda data, target: np.linalg.norm(data[:,-1]-target)

    def _misfit(self, data: np.array, originaltarget: np.array) -> float:
        return  self.fitness(data, originaltarget)
    
   
    def _potentialmisfit(self, idxg1: list, idxg2: list, data: np.array, originaltarget: np.array) -> float:

        
        data = data.copy()
        
        data[idxg1,-1] = 0
        data[idxg2,-1] = 1
                    
        return self._misfit(data, originaltarget)
    
    
    
    def _split(self, data: np.array, idx: np.array, feature_idx: int, feature_val: float) -> tuple:
        
        mask_below_threshold = data[idx, feature_idx] < feature_val
        idxgroup1 = idx[np.where(mask_below_threshold)[0]]
        idxgroup2 = idx[np.where(~mask_below_threshold)[0]]

        return idxgroup1, idxgroup2
    
    def _select_features_to_use(self, data: np.array) -> list:
        """
        Randomly selects the features to use while splitting w.r.t. hyperparameter numb_of_features_splitting
        """
        feature_idx = list(range(data.shape[1]-1))

        if self.numb_of_features_splitting == "sqrt":
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.sqrt(len(feature_idx))))
        elif self.numb_of_features_splitting == "log":
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))))
        else:
            feature_idx_to_use = feature_idx

        return feature_idx_to_use
        
    def _find_best_split(self, data: np.array, idx: np.array, originaltarget: np.array) -> tuple:
        """
        Finds the best split (with the lowest entropy) given data
        Returns 2 splitted groups and split information
        """
        min_misfit = self._misfit(data, originaltarget)*(1+np.random.rand())
        feature_idx_to_use =  self._select_features_to_use(data)
        
        found = False

        for fid in feature_idx_to_use:
            
            vx = np.percentile(np.unique(data[idx, fid]), q=np.arange(25,100,25))                        
            feature_vals = np.percentile(data[idx, fid], q=np.linspace(0.001, 99.99, 5))
            
            feature_vals = np.unique(np.concatenate((feature_vals, vx*1.01,vx*0.99)))
            
            for feature_val in feature_vals:
                idxg1, idxg2, = self._split(data, idx, fid, feature_val)
                potentialmisfit = self._potentialmisfit(idxg1, idxg2, data, originaltarget)
                if potentialmisfit < min_misfit:
                    min_misfit = potentialmisfit
                    min_misfit_feature_idx = fid
                    min_misfit_feature_val = feature_val
                    idxg1_min, idxg2_min = idxg1, idxg2
                    found = True

        if found:
            return idxg1_min, idxg2_min, min_misfit_feature_idx, min_misfit_feature_val, min_misfit
        else:
            return None



    def _create_tree(self, data: np.array, idx: np.array, originaltarget: np.array, current_depth: int, output: int) -> TreeNode:
        """
        Recursive, depth first tree creation algorithm
        """

        # Check if the max depth has been reached (stopping criteria)
        if current_depth > self.max_depth:
            return None
        
        # Calculate the misfit
        node_misfit = self._misfit(data, originaltarget)
        
        # Find best split
        splitting = self._find_best_split(data, idx, originaltarget)
        
        if splitting is not None:
            
            idx_split_1_data, idx_split_2_data, split_feature_idx, split_feature_val, split_misfit = splitting
                        
            data[idx_split_1_data,-1] = 0
            data[idx_split_2_data,-1] = 1
            
            misfit_gain = node_misfit - split_misfit
            
            # Create node
            node = TreeNode(data, split_feature_idx, split_feature_val, node_misfit, misfit_gain, output)
    
            # Check if the min_samples_leaf has been satisfied (stopping criteria)
            if self.min_samples_leaf > len(idx_split_1_data) or self.min_samples_leaf > len(idx_split_2_data):
                return node
            # Check if the min_information_gain has been satisfied (stopping criteria)
            elif misfit_gain < self.min_misfit_gain:
                return node
    
            current_depth += 1
            node.left = self._create_tree(data, idx_split_1_data, originaltarget, current_depth, output=0)
            node.right = self._create_tree(data, idx_split_2_data, originaltarget, current_depth, output=1)
            
            return node
        else:
            
            return TreeNode(data, None, None, 1e9, 0, output)
            
    
    def _predict_one_sample(self, X: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        node = self.tree

        # Finds the leaf which X belongs
        while node:
            
            if node.feature_idx is None:
                out = node.output
                node = None
                
            elif X[node.feature_idx] < node.feature_val:
                out = node.output
                node = node.left
            else:
                out = node.output
                node = node.right

        return out

    def fit(self, X_train: np.array, Y_train: np.array) -> None:
        return self.train(X_train, Y_train)
         
    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Trains the model with given X and Y datasets
        """

        # Concat features and labels
        train_data = np.concatenate([X_train, np.reshape(np.random.randint(2, size=len(Y_train)),(-1,1))], axis=1)

        # Start creating the tree
        self.tree = self._create_tree(data=train_data, idx=np.arange(train_data.shape[0]), originaltarget=Y_train, current_depth=0, output=None)
               
        predictions = self.predict(X_train)
        
        self.train_misfit = self._misfit(np.concatenate([X_train, predictions[:,np.newaxis]], axis=1), Y_train)
        
        return self


    def predict(self, X_set: np.array) -> np.array:

        if len(X_set.shape)>1:
            preds = np.apply_along_axis(self._predict_one_sample, 1, X_set)
        else:
            preds = self._predict_one_sample(X_set)
        
        return preds    
        
    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)
        
        
        
class RandomForest():
    """
    Random Forest Classifier
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """

    def __init__(self, n_estimators=10, max_depth=5, min_samples_leaf=1, min_misfit_gain=0.0, \
                 numb_of_features_splitting=None, bootstrap_sample_fraction=None) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_misfit_gain = min_misfit_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.bootstrap_sample_fraction = bootstrap_sample_fraction

    def _create_bootstrap_samples(self, X, Y) -> tuple:
        """
        Creates bootstrap samples for each base learner
        """
        bootstrap_samples_X = []
        bootstrap_samples_Y = []
        bootstrap_feature_idx = []

        for i in range(self.n_estimators):
            
            if not self.bootstrap_sample_fraction:
                self.bootstrap_sample_fraction = 1.0
            
            bootstrap_sample_size = int(np.round(self.bootstrap_sample_fraction*X.shape[0]))
            feature_idx = np.sort(np.random.choice(X.shape[1], size=np.random.randint(1,high=X.shape[1]+1,size=1)[0], replace=False))
            
            sampled_idx = np.random.randint(X.shape[0]-bootstrap_sample_size+1) + np.arange(bootstrap_sample_size)
            bootstrap_samples_X.append(X[sampled_idx][:,feature_idx])
            bootstrap_samples_Y.append(Y[sampled_idx])
            bootstrap_feature_idx.append(feature_idx)

        return bootstrap_samples_X, bootstrap_samples_Y, bootstrap_feature_idx
    
    
    def create_tree(self):        
        return DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, \
                                    min_misfit_gain=self.min_misfit_gain, 
                                    numb_of_features_splitting=self.numb_of_features_splitting)

    def fit(self, X_train: np.array, Y_train: np.array) -> None:
        return self.train(X_train, Y_train)

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """Trains the model with given X and Y datasets"""
        bootstrap_samples_X, bootstrap_samples_Y, bootstrap_feature_idx = self._create_bootstrap_samples(X_train, Y_train)

        self.base_learner_list = []
        self.base_learner_feature_idx_list = []
        for base_learner_idx in range(self.n_estimators):
            base_learner = self.create_tree()
            
            base_learner.train(bootstrap_samples_X[base_learner_idx], bootstrap_samples_Y[base_learner_idx])
            self.base_learner_list.append(base_learner)
            self.base_learner_feature_idx_list.append(bootstrap_feature_idx[base_learner_idx])

        return self


    def _predict_one_sample(self, X: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        
        
        pred_list = []
        for base_learner, feature_idx in zip(self.base_learner_list, self.base_learner_feature_idx_list):
            pred_list.append(base_learner.predict(X[feature_idx]))

        freqdist = np.bincount([x for x in pred_list if x is not None])
        if (freqdist.size != 0) and not all(np.isnan(freqdist)):
            prediction = freqdist.argmax()
        else:
            prediction = -1
        
        return prediction


    def predict(self, X_set: np.array) -> np.array:
        """Returns the predicted labels for a given data set"""

        if len(X_set.shape)>1:
            preds = np.apply_along_axis(self._predict_one_sample, 1, X_set)
        else:
            preds = self._predict_one_sample(X_set)
           
        return preds
    
    
    def _predict_proba_one_sample(self, X: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        
        
        pred_list = []
        for base_learner, feature_idx in zip(self.base_learner_list, self.base_learner_feature_idx_list):
            pred_list.append(base_learner.predict(X[feature_idx]))

        x = [x for x in pred_list if x is not None] + [0, 1]
               
        freqdist = np.bincount(x)[-2:] 
        if (freqdist.size != 0) and not all(np.isnan(freqdist)):
            prediction = freqdist/np.sum(freqdist)
        else:
            prediction = [0.5, 0.5]
        
        return prediction[1]
    
    
    def predict_proba(self, X_set: np.array):
        # The predicted class probabilities of an input sample are computed as the fraction of samples of the same class in the trees.
        
        if len(X_set.shape)>1:
            preds = np.apply_along_axis(self._predict_proba_one_sample, 1, X_set)
        else:
            preds = self._predict_proba_one_sample(X_set)
           
        return preds

    
    
    def score(self, X: np.array, y: np.array) -> float:
        
        y_pred = self.predict(X)
        
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        
        return -self.base_learner_list[0]._misfit(y_pred, y)
        
        
    