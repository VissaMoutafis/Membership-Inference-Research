import math
import numpy as np
import scipy.ndimage.interpolation as interpolation
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from sklearn.metrics import accuracy_score, log_loss
import torch
import gc 

"""
Divide a given dataset (X, y) according to the number of splits and the size of each D_i given.
@param n_splits: number of splits
@param dataset_size: number of instances in each generated datasets
@param X, y: dataset to split
"""
def divide_dataset(n_splits, dataset_size, X, y):
    D_splits = []
    for i in range(n_splits):
        sample_i = np.random.choice(
            range(X.shape[0]), dataset_size, replace=False)
        # sanity check for replace usage (might delete later)
        assert np.unique(sample_i).shape[0] == dataset_size
        D_splits.append((X[sample_i], y[sample_i]))
    return D_splits

"""
Returns a list of 'n_shadows' datasets
@param target_model: the model to attack (useless currently)
@param n_shadows: number of shadow models
@param: shadow_dataset_size: size of dataset for each D_shadow
@param n_classes: labels (useless currently)
@param attacker_X, attacker_y: attacker's provided dataset, to split among the shadows
"""
def generate_shadow_dataset(target_model, n_shadows, shadow_dataset_size, n_classes, attacker_X=None, attacker_y=None):
    # in case we give test data we will just divide those to train the shadow models
    if attacker_X is not None and attacker_y is not None:
        return divide_dataset(n_shadows, shadow_dataset_size, attacker_X, attacker_y)
    else:
        raise ValueError("X and y provided are None.")


# create all relative rotates for interpolation (returns 2*r + 1 translates)
def create_rotates(r):
    if r is None:
        return None

    rotates = np.linspace(-r, r, (r * 2 + 1))
    return rotates

# create all possible translates (returns 4*d+1 translates)
def create_translates(d):
    if d is None:
        return None

    def all_shifts(mshift):
        if mshift == 0:
            return [(0, 0, 0, 0)]

        all_pairs = []
        start = (0, mshift, 0, 0)
        end = (0, mshift, 0, 0)
        vdir = -1
        hdir = -1
        first_time = True
        while (start[1] != end[1] or start[2] != end[2]) or first_time:
            all_pairs.append(start)
            start = (0, start[1] + vdir, start[2] + hdir, 0)
            if abs(start[1]) == mshift:
                vdir *= -1
            if abs(start[2]) == mshift:
                hdir *= -1
            first_time = False
        all_pairs += [(0, 0, 0, 0)] #append no translate
        return all_pairs

    translates = all_shifts(d)
    return translates


def apply_augment(d, augment, type_):
    if type_ == 'd':
        d = interpolation.shift(d, augment, mode='constant')
    elif type_ == 'r':
        d = interpolation.rotate(d, augment, (1, 2), reshape=False)
    else:
        raise ValueError(f'Augmentation Type: \'{type_}\' doesn\'t exist. Try \'r\' or \'d\'')
    return d

"""
Method to augment a given dataset
@param model: model to query
@param X: input to perurb
@param y: labels of given input
@param r : rotations angles
@param d : translates
"""
def augment_dataset(X, y, r=2, d=1):
    # create perturbations
    rotates = create_rotates(r)
    translates = create_translates(d)
    X_aug = X.copy()
    y_aug = [y.copy() for _ in range(len(r+d))]

    for rot in rotates:
        #  create perturbed image
        X_perturbed = apply_augment(X, rot, 'r')
        if X_aug is None:
            X_aug = X_perturbed
        else:
            X_aug = np.concatenate((X_aug, X_perturbed))

    for tra in translates:
        #  create perturbed image
        X_perturbed = apply_augment(X, tra, 'd')
        if X_aug is None:
            X_aug = X_perturbed
        else:
            X_aug = np.concatenate((X_aug, X_perturbed))
            
    return X_aug, y_aug


def cifar_10_f_attack_builder(top_k=10):
    model = models.Sequential()
    model.add(layers.Dense(10, input_shape=(top_k+1, )))
    model.add(layers.LeakyReLU(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
        
    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    return model 


def evaluate_model_vulnerability(model, D_target, D_out, model_type, **evaluate_args):
    """ 
    Evaluate target model vulnerabilities given the target-model, the D_in (dataset in which membership we try to infer) and the dataset the attacker has at hand
    """
    if model_type == 'tf':
        # get the models loss and accuracy on target and attacker data
        loss_target, acc_target = model.evaluate(D_target[0], D_target[1], **evaluate_args)
        loss_out, acc_out = model.evaluate(D_out[0], D_out[1], **evaluate_args)
    elif model_type == 'sklearn':
        y_pred_target_proba = model.predict_proba(D_target[0])
        y_pred_target = np.argmax(y_pred_target_proba, axis=1)
        
        y_pred_out_proba = model.predict_proba(D_out[0])
        y_pred_out = np.argmax(y_pred_out_proba, axis=1)
        
        acc_target = accuracy_score(D_target[1], y_pred_target)
        acc_out = accuracy_score(D_out[1], y_pred_out)
        
        loss_target = log_loss(D_target[1], y_pred_target_proba)
        loss_out = log_loss(D_out[1], y_pred_out_proba)
        
    # define A and L
    a = acc_target / acc_out
    l = loss_out / loss_target
    
    # estimate Vulnerability metric (attacker's advantage)
    v = round(math.log(2 * (a*l) / (a+l) ), 2)
    
    return v


def free_gpu_cache():
    # gc.collect()
    torch.cuda.empty_cache()
