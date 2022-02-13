"""
    Implementation of Shadow Models Enseble Bundle. This 'model' will return a [n samples, classes] vector of 
    - confidence vectors in normal attack settings (Default MIA Assumption)
    - labels (Label-Only attack settings)
    that we will use as train instances in the attack model. This will perform ensure that we capture the target's  
"""

from sklearn.model_selection import train_test_split
from mia_v2.shadow_models import ShadowModelBatch

class ShadowEnseble(ShadowModelBatch):
    """
    Shadow model Enseble wrapper
    @param shadow_models_init_list: list of shadow models, the enseble is consisted of
    @param model_types: list of model types for each of the models that consist the enseble
    """
    def __init__(self, shadow_models_init_list=[], model_types=[]):
        super(ShadowEnseble, self).__init__(
            n_shadows= len(shadow_models_init_list),
            shadow_creator=None,
            model_type='enseble'
        )
        self.model_types = model_types
        self.shadow_models = shadow_models_init_list
        assert len(self.model_types) == len(self.shadow_models), "Error: Model Types must be \
            of the same len with the models init list"    
    
    """
    Train all shadow models, given each one's dataset.
    @param D_shadows: list of D_shadow_i = (X, y)
    @param epochs: epochs of training for each shadow model
    """
    def fit_all(self, D_shadows, epochs=50):
        for i in range(self.n_shadows):
            # split the D_shadow in train, test sets
            X_train, X_test, y_train, y_test = train_test_split(D_shadows[i][0], D_shadows[i][1], shuffle=True, test_size=0.33)
            D_shadow_i = (X_train, y_train), (X_test, y_test)
            self.D_shadow.append(D_shadow_i)

            # fit the model
            if self.model_types[i] == 'tf':
                self.history.append(self.shadow_models[i].fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=self.VERBOSE))
            else:
                # just fit, no need for other arguments
                self.shadow_models[i].fit(X_train, y_train) 
