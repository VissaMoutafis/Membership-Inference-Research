import numpy as np
from mia.attack_model import DefaultAttackModel
from mia.utilities import *
from mia.shadow_models import ShadowModelBatch

class ConfidenceVectorAttack():
    ATTACK_MODEL_OPTIMIZER = 'adam'
    ATTACK_MODEL_EPOCHS = 100
    def __init__(self, target_model, target_dataset, attacker_dataset, shadow_creator=None, n_shadows=1, D_shadow_size=1000, verbose=False):
        """
        param target_model: fitted target model
        param tartet_dataset: target's train dataset, in tuple format (X_train, y_train)
        param attacker_dataset: dataset the attacker uses to train shadows, in tuple format (X_train, y_train)
        param shadow_creator: function to return a compiled model used as shadow creator
        param n_shadows: # of shadow models
        param D_shadow_size: # of instances in each D_shadow_i
        param verbose: verbosity during attack phases 
        """
        
        # set up variables
        self.target_model = target_model
        self.target_dataset = target_dataset
        self.attack_model = None # don't create till attack
        self.attacker_dataset = attacker_dataset
        self.n_shadows = n_shadows 
        self.shadow_creator = shadow_creator
        self.D_shadow_size = D_shadow_size
        self.verbose = verbose 
        self.n_classes = len(np.unique(target_dataset[1]))
        self.D_shadows = None 
        self.trained = False 
        self.shadow_model_bundle = None 
        
    def create_shadows(self):
        shadow_models_batch = ShadowModelBatch(self.n_shadows, self.shadow_creator) # shadow model list
        shadow_models_batch.fit_all(self.D_shadows, epochs=25)
        return shadow_models_batch # return a list where every item is (model, acc), train-data, test-data

    def perform_attack(self):
        self.trained = True 
        # generate shadow datasets
        self.D_shadows = generate_shadow_dataset(self.target_model, self.n_shadows, self.D_shadow_size, self.n_classes, self.attacker_dataset[0], self.attacker_dataset[1])

        # create shadow models
        self.shadow_model_bundle = self.create_shadows()
        
        # create and train the attack model
        self.attack_model = DefaultAttackModel(self.shadow_model_bundle, self.n_classes, (self.n_classes+1, ), self.ATTACK_MODEL_OPTIMIZER)
        self.attack_model.fit(self.ATTACK_MODEL_EPOCHS)
        
    def evaluate_attack(self, D_in_sample_size=None, D_out_sample_size=None):
        if D_in_sample_size is None:
            D_in_sample_size = min(self.target_dataset[0].shape[0], self.attacker_dataset[0].shape[0])
        if D_out_sample_size is None:
            D_out_sample_size = min(self.target_dataset[0].shape[0], self.attacker_dataset[0].shape[0])

        D_in = self.attack_model.prepare_batch(self.target_model, self.target_dataset[0][:D_in_sample_size], self.target_dataset[1][:D_in_sample_size], True)
        D_out = self.attack_model.prepare_batch(self.target_model, self.attacker_dataset[0][:D_out_sample_size], self.attacker_dataset[1][:D_out_sample_size], False)
        self.attack_model.evaluate(np.concatenate((D_out[:, :-1], D_in[:, :-1])),  np.concatenate((D_out[:, -1], D_in[:, -1])), self.verbose)