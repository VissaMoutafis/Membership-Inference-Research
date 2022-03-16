"""
    Perturbation Mechanism for categorical data.
    Written by Vissarion Moutafis
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import random

from mia_v2.label_only import target_predict

class BinaryNaivePerturbator:
    """
        Naive Augmentation Mechanism that uses Brute Force to produce all possible perturbations. 
        Might not be the best option, since small seemingly insignificant changes in categorical features might result to a total label change. 
        Latter is not a perturbation, by definition, since we do not retain the features correlations.
    """
    def __init__(self, dataset, bool_cols, label_col=None):
        self.original_dataset = dataset
        self.bool_cols = bool_cols
        self.label_col = label_col 
        self.perturbations = None
    
    def perturbate_dataset_iter(self, n_perturbations=100, max_dist=1, sampling=True):
        """
            Return n_perturbations datasets. In each dataset:
            1. Choose one combination from C = combinations(#features choose #features-to-change) 
            2. Change the selected features, using the XOR 1 operation
            3. yield the results
            
            pretty naive huh?
            
        """
        
        if self.perturbations is None:
            # keep the first @n_perturbations of combs and apply them for each data point
            combs = list(c for i in range(1, max_dist+1) for c in combinations(self.bool_cols, i))
            self.perturbations = random.sample(combs, n_perturbations) if sampling else combs[:n_perturbations]
            
        for feat_idx in self.perturbations:
            # copy the original dataset, so that we do not mutate it
            perturbated_dataset = self.original_dataset.copy()
            # use the "XOR 1" operation to interchange 1 and 0
            perturbated_dataset.loc[:, feat_idx] = perturbated_dataset.loc[:, feat_idx] ^ 1  
            # return the perturbated dataset
            yield perturbated_dataset


perturbator = None

# create all perturbations' prediction vectors, which will be used to train the attack model
def augment_categorical_dataset(model, X, y_pred, **aug_gen_args):
    X_attack = None
    dt = pd.DataFrame(X)
    global perturbator   
    if perturbator is None:
        print('Initializing perturbator for the first time...')
        perturbator = BinaryNaivePerturbator(dt, dt.columns)
    else:
        perturbator.original_dataset = dt 
        perturbator.bool_cols = dt.columns

    for X_aug in perturbator.perturbate_dataset_iter(**aug_gen_args):
        # return query line
        y_aug = target_predict(model, X_aug.to_numpy(dtype=X.dtype))
        # transform the prediction column into a binary collumn where x_i = 1 when y_true == y_pred else 0
        X_attack_col = (y_pred == y_aug).astype(np.int8)

        if X_attack is None:
            X_attack = X_attack_col
        else:
            X_attack = np.concatenate((X_attack, X_attack_col), axis=1)
        del X_aug
    del dt
    return X_attack
