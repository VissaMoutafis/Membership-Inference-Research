# Small Framework to execute and evaluate Membership Inference Attacks
# in a black-box label only fashion
# ~Written by Vissarion Moutafis~ for Undergraduate Thesis
import numpy as np
from mia.attack_model import DefaultAttackModel
from mia.utilities import create_rotates, create_translates, apply_augment
from tensorflow.keras import layers

""" API of model to get predictions : returns labels only"""
def target_predict(model, X):
  prob = layers.Softmax()
  ret = prob(model.predict(X)).numpy()
  return np.apply_along_axis(np.argmax, 1, ret).reshape((-1, 1))

""" 
Returns a matrix of shape (n_samples, 2*r+4*d+2)
@param model: model to query
@param X: input to perurb
@param y_pred: predictions of the model for given input
@param r: rotates
@param d: translates
"""
def augmented_queries(model, X, y_pred, r=3, d=1):
    # create perturbations
    rotates = create_rotates(r)
    translates = create_translates(d)

    X_attack = None
    for rot in rotates:
        # skip the no rotation augment
        if rot == 0.0: continue
        #  create perturbed image
        X_perturbed = apply_augment(X, rot, 'r')
        # return query line
        y_perturbed = target_predict(model, X_perturbed)
        # transform the prediction column into a binary collumn where x_i = 1 when y_true == y_pred else 0
        X_attack_col = (y_pred == y_perturbed).astype(np.int8)

        if X_attack is None:
            X_attack = X_attack_col
        else:
            X_attack = np.concatenate((X_attack, X_attack_col), axis=1)

    for tra in translates:
        # skip the no translation augment
        if tra == (0, 0, 0, 0): continue
        X_perturbed = apply_augment(X, tra, 'd')
        # return query line
        y_perturbed = target_predict(model, X_perturbed)
        # transform the prediction column into a binary collumn where x_i = 1 when y_true == y_pred else 0
        X_attack_col = (y_pred == y_perturbed).astype(np.int8)
        # concate the col to the rest of x_attack feature vector
        if X_attack is None:
            X_attack = X_attack_col
        else:
            X_attack = np.concatenate((X_attack, X_attack_col), axis=1)
    return X_attack


class LabelOnlyAttackModel(DefaultAttackModel):
    """
    Default attack model for confidence vector attack
    @param shadow_batch: A ShadowModelBatch object with trained models that will be used as target imitators to execute attack
    @param n_classes: the number of classes in the classification models output
    @param X_attack_dim: the dimensions of the attack dataset's instances (i.e. in confidence vector we use (n_classes,))
    @param _optimizer: optimizer to use in attack-model fitting
    """
    def __init__(self, shadow_batch, n_classes, f_attack_builder):
        self.r = 2 # default values
        self.d = 1 #default values
        super(LabelOnlyAttackModel, self).__init__(shadow_batch, n_classes, f_attack_builder)

    """
    helper to prepare a batch of shadow data into a batch of attack data
    Can be called to create a test-dataset for the model, outside the model.
    Will apply the perturbation, given perturbation params in self.fit, else will go with defaults (r=2, d=1). 
    You can always change by setting up self.r, self.d
    @param model: the model to which we relate the data 
    @param X, y: instances and their true labels
    @param in_D: boolean-> True if data are in the model's training dataset, otherwise False
    
    """
    def prepare_batch(self, model, X, y, in_D):
        # decide membership
        y_member = np.ones(shape=(y.shape[0], 1)) if in_D else np.zeros(shape=(y.shape[0], 1))

        # get the y_pred 
        prob = layers.Softmax()
        ret = prob(model.predict(X)).numpy()
        y_pred = np.apply_along_axis(np.argmax, 1, ret).reshape((-1, 1))
        perturbed_queries_res = augmented_queries(model, X, y_pred, self.r, self.d)
        
        # return an instance <actual class, predicted class, perturbed_queries_res from shadow models, 'in'/'out' D_target membership> 
        return np.concatenate((y.reshape(-1, 1), y_pred, perturbed_queries_res, y_member), axis=1)
    
    """
    Fit the attack model, but first provide with perturbation params to generate the proper ammount of features per instance (given as X_inpt_dim in __init__)
    @param r: rotates
    @param d: translates
    """
    def fit(self, r=2, d=1, epochs=100):
        #set up r, d before fit
        self.r = r
        self.d = d
        super(LabelOnlyAttackModel, self).fit(epochs=epochs)
