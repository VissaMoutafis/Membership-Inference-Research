import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers, models, optimizers
from mia_v2.utilities import *
from mia_v2.attack_model import *

class TopKConfidenceMaskingModel(models.Sequential):
    def __init__(self, top_k):
        super(TopKConfidenceMaskingModel, self)
        self.top_k = top_k
    
    # prediction will return (confidence vector, classes) tuple 
    def predict(self, X):
        y_all = self(X)
        top_classes_idx = np.argsort(y_all, order='desc')[:, :self.top_k]
        return np.take_along_axis(y_all, top_classes_idx), top_classes_idx
    
    
class TopKConfidenceMaskingAttackModel(DefaultAttackModel):
    def prepare_batch(self, model, X, y, in_D):
        # decide membership
        y_member = np.ones(shape=(y.shape[0], 1)) if in_D else np.zeros(
            shape=(y.shape[0], 1))
        prob = layers.Softmax()
        y_conf, y_labels = model.predict(X)
        ret = prob(y_conf.astype(np.float32)).numpy().reshape((-1, self.n_classes))

        # return an instance <true label, top-k pred labels, confidence vector, 0/1 D_target membership>
        return np.concatenate((y.reshape((-1, 1)), y_labels.reshape(-1, 1), ret, y_member), axis=1)
