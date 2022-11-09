import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers, models, optimizers
from mia.utilities import *
from mia.attack_model import *

class TopKConfidenceMaskingModel():
    def __init__(self, top_k, model_builder, **model_builder_args):
        self.top_k = top_k
        self.model = model_builder(**model_builder_args)
    
    def fit(self, X_train, y_train, **tf_fit_args):
        return self.model.fit(X_train, y_train, **tf_fit_args)
        
    # prediction will return (confidence vector, classes) tuple 
    def predict(self, X):
        y_all = self.model(X).numpy()
        top_classes_idx = np.argsort(y_all)[:, :(-1-self.top_k):-1]
        return np.take_along_axis(y_all, top_classes_idx, axis=1), top_classes_idx
    
    def evaluate(self, **tf_eval_args):
        return self.model.evaluate(**tf_eval_args)
    
    
class TopKConfidenceMaskingAttackModel(DefaultAttackModel):
    def prepare_batch(self, model, X, y, in_D):
        # decide membership
        y_member = np.ones(shape=(y.shape[0], 1)) if in_D else np.zeros(
            shape=(y.shape[0], 1))
        prob = layers.Softmax()
        y_conf, y_labels = model.predict(X)
        ret = prob(y_conf.astype(np.float32)).numpy().reshape((-1, self.n_classes))

        # return an instance <true label, top-k pred labels, confidence vector, 0/1 D_target membership>
        return np.concatenate((y.reshape((-1, 1)), y_labels.reshape(-1, self.n_classes), ret, y_member), axis=1)
