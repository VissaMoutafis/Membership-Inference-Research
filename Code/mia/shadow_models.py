# Shadow Models Batch Interpretation for MIA Attacks
# Written by VissaMoutafis

from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

class ShadowModelBatch():
    VERBOSE = False
    # param n_shadows: num of shadow models
    # param shadow_creator: function to create a shadow dataset
    def __init__(self, n_shadows, shadow_creator):
        self.shadow_models = [shadow_creator() for _ in range(n_shadows)]
        self.n_shadows = n_shadows
        # dataset in form of (X_train, y_train), (X_test, y_test)
        self.D_shadow = []
        self.history = []

    # param D_shadows: list of D_shadow_i = (X, y)
    def fit_all(self, D_shadows, epochs=50):

        for i in range(self.n_shadows):
            X_train, X_test, y_train, y_test = train_test_split(D_shadows[i][0], D_shadows[i][1], shuffle=True, test_size=0.33)
            self.history.append(self.shadow_models[i].fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=self.VERBOSE))
            D_shadow_i = (X_train, y_train), (X_test, y_test)
            self.D_shadow.append(D_shadow_i)

    # param i: idx of the i-th shadow model
    def __iter__(self):
        self.n = 0
        return self 

    def __next__(self):
        item = None
        if self.n < self.n_shadows:
            item = self.shadow_models[self.n], self.D_shadow[self.n]
            self.n += 1
        else:
            raise StopIteration
        
        return item 
