import numpy as np
from mia_v2.attack_model import *
from mia_v2.label_only import LabelOnlyAttackModel
from mia_v2.utilities import *
from mia_v2.shadow_models import ShadowModelBatch

class MIAWrapper():
    ATTACK_MODEL_OPTIMIZER = 'adam'
    ATTACK_MODEL_EPOCHS = 100
    SHADOW_MODELS_EPOCHS = 100
    SHADOW_MODEL_TYPE = 'tf'
    ATTACK_MODEL_TYPE = 'tf'
    """
    Wrapper for MIA framework.
    @param target_model: the model to perform attack to.
    @param target_dataset: the target's training dataset
    @param attacker_dataset: the dataset that attacker has access to. Will be used in shadow models training
    @param attack_model_creator: function to return attack model architecture, compiled 
    @param shadow_creator: function to return a shadow model
    @param n_shadows: number of shadow models
    @param D_shadow_size: size of D_shadow_i for every shadow model
    @param verbose: verbosity meter
    
    GLOBAL:
    @param ATTACK_MODEL_OPTIMIZER: optimizer for attack model. Defaults to 'adam'
    @param ATTACK_MODEL_EPOCHS: epochs fo training for attack model. Defaults to 100
    """
    def __init__(self, target_model, target_dataset, attacker_dataset, attack_model_creator=None, shadow_creator=None, n_shadows=1, D_shadow_size=1000, verbose=False):
        DefaultAttackModel.VERBOSE = verbose
        ShadowModelBatch.VERBOSE = verbose 
        
        # set up variables
        self.target_model = target_model
        self.target_dataset = target_dataset
        self.attack_model_creator = attack_model_creator  
        self.attack_model = None 
        self.attacker_dataset = attacker_dataset
        self.n_shadows = n_shadows
        self.shadow_creator = shadow_creator
        self.D_shadow_size = D_shadow_size
        self.verbose = verbose
        self.n_classes = len(np.unique(target_dataset[1]))
        self.D_shadows = None
        self.trained = False
        self.shadow_model_bundle = None

    """
    Create shadow models batch. Fit all models.
    Warning: called inside the wrapper, not for outside usage. 
    """
    def create_shadows(self, epochs=100):
        shadow_models_batch = ShadowModelBatch(self.n_shadows, self.shadow_creator, model_type=self.SHADOW_MODEL_TYPE) # shadow model list
        shadow_models_batch.fit_all(self.D_shadows, epochs=epochs)
        return shadow_models_batch # return a list where every item is (model, acc), train-data, test-data

    """
    Attack evaluation based on given datasets
    @param D_in_sample_size: num of samples from D_in. Default is min (D_in_sample_size, D_out_sample_size)
    @param D_out_sample_size: num of samples from D_attacker (any instances out of D_in will do). Default is min (D_in_sample_size, D_out_sample_size)
    """
    def evaluate_attack(self, D_in_sample_size=None, D_out_sample_size=None):
        if D_in_sample_size is None:
            D_in_sample_size = min(self.target_dataset[0].shape[0], self.attacker_dataset[0].shape[0])
        if D_out_sample_size is None:
            D_out_sample_size = min(self.target_dataset[0].shape[0], self.attacker_dataset[0].shape[0])

        D_in = self.attack_model.prepare_batch(self.target_model, self.target_dataset[0][:D_in_sample_size], self.target_dataset[1][:D_in_sample_size], True)
        D_out = self.attack_model.prepare_batch(self.target_model, self.attacker_dataset[0][:D_out_sample_size], self.attacker_dataset[1][:D_out_sample_size], False)
        return self.attack_model.evaluate(np.concatenate((D_out[:, :-1], D_in[:, :-1])),  np.concatenate((D_out[:, -1], D_in[:, -1])), self.verbose)



class ConfidenceVectorAttack(MIAWrapper):
    """
    Wrapper for confidence vector MIA.
    @param target_model: the model to perform attack to.
    @param target_dataset: the target's training dataset
    @param attacker_dataset: the dataset that attacker has access to. Will be used in shadow models training
    @param shadow_creator: function to return a shadow model
    @param n_shadows: number of shadow models
    @param D_shadow_size: size of D_shadow_i for every shadow model
    @param verbose: verbosity meter
    """
    def __init__(self, target_model, target_dataset, attacker_dataset, attack_model_creator=None, shadow_creator=None, n_shadows=1, D_shadow_size=1000, verbose=False):
        super(ConfidenceVectorAttack, self).__init__(target_model, target_dataset, attacker_dataset, attack_model_creator, shadow_creator, n_shadows, D_shadow_size, verbose)

    """
    Generate shadow dataset, create and train shadows, generate attack model dataset, create and train attack model.
    """
    def perform_attack(self):
        self.trained = True 
        # generate shadow datasets
        self.D_shadows = generate_shadow_dataset(self.target_model, self.n_shadows, self.D_shadow_size, self.n_classes, self.attacker_dataset[0], self.attacker_dataset[1])

        # create shadow models
        self.shadow_model_bundle = self.create_shadows(
            epochs=self.SHADOW_MODELS_EPOCHS)
        
        # create and train the attack model
        self.attack_model = DefaultAttackModel(self.shadow_model_bundle, self.n_classes, self.attack_model_creator)
        self.attack_model.fit(self.ATTACK_MODEL_EPOCHS)
        
        
        
class LabelOnlyAttack(MIAWrapper):
    """
    Wrapper for confidence vector MIA.
    @param target_model: the model to perform attack to.
    @param target_dataset: the target's training dataset
    @param attacker_dataset: the dataset that attacker has access to. Will be used in shadow models training
    @param rotates: number of rotations (create 2*r rotates)
    @param translats: number of translates (create 4*d translates)
    @param shadow_creator: function to return a shadow model
    @param n_shadows: number of shadow models
    @param D_shadow_size: size of D_shadow_i for every shadow model
    @param verbose: verbosity meter
    """
    def __init__(self, target_model, target_dataset, attacker_dataset, rotates=3, translates=1, attack_model_creator=None, shadow_creator=None, n_shadows=1, D_shadow_size=1000, verbose=False):
        super(LabelOnlyAttack, self).__init__(target_model, target_dataset, attacker_dataset, attack_model_creator, shadow_creator, n_shadows, D_shadow_size, verbose)
        self.r = rotates
        self.d = translates 
    """
    Generate shadow dataset, create and train shadows, generate attack model dataset (instances of <y_true, shadow_i(x), shadow_i(aug(x))> ), create and train attack model.
    """
    def perform_attack(self):
        self.trained = True 
        # generate shadow datasets
        self.D_shadows = generate_shadow_dataset(self.target_model, self.n_shadows, self.D_shadow_size, self.n_classes, self.attacker_dataset[0], self.attacker_dataset[1])

        # create shadow models
        self.shadow_model_bundle = self.create_shadows(self.SHADOW_MODELS_EPOCHS)
        
        # create and train the attack model
        self.attack_model = LabelOnlyAttackModel(self.shadow_model_bundle, self.n_classes, self.attack_model_creator)
        self.attack_model.fit(self.r, self.d, self.ATTACK_MODEL_EPOCHS)
        
