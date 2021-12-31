import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


class DefaultAttackModel():
    VERBOSE=False
    def __init__(self, shadow_batch, n_classes, X_attack_dim, _optimizer):
        # default structure as Shokri et al. suggested
        self.model = models.Sequential()
        self.model.add(layers.Dense(10, input_shape=X_attack_dim))
        self.model.add(layers.LeakyReLU(0.3))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        
        self.model.compile(optimizer=_optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        self.shadow_models_batch = shadow_batch 
        self.n_classes = n_classes 
        self.attack_dataset = []
        self.history = None
    
    # helper to prepare a batch of shadow data into a batch of attack data
    # param model: the model to which we relate the data 
    # param X, y: instances and their true labels
    # param in_D: boolean-> True if data are in the model's training dataset, otherwise False
    def prepare_batch(self, model, X, y, in_D):
        # decide membership
        y_member = np.ones(shape=(y.shape[0], 1)) if in_D else np.zeros(shape=(y.shape[0], 1))
        prob = layers.Softmax()
        ret = prob(model.predict(X)).numpy().reshape((-1,self.n_classes))
        
        # return an instance <true label, confidence vector, 0/1 D_target membership> 
        return np.concatenate((y.reshape((-1, 1)), ret, y_member), axis=1)

    # helper function to generate the attack dataset from the previously given shadow models batch
    def generate_attack_dataset(self):
        # input is a list where items are model, (X_train, y_train), (X_test, y_test)

        D_attack = None
        # D_attack_i format = <class, prob_vec, membership label (1 or 0)> 
        for shadow_model, ((X_train, y_train), (X_test, y_test)) in self.shadow_models_batch:
            s = min(X_train.shape[0], X_test.shape[0])
            print(f"Preparing shadow batch of size {2*s}")
            batch = np.concatenate((
                self.prepare_batch(shadow_model, X_train[:s], y_train[:s], True), # members of shadow dataset 
                self.prepare_batch(shadow_model, X_test[:s], y_test[:s], False)   # non members of shadow dataset
            ))   

            D_attack = np.concatenate((D_attack, batch)) if D_attack is not None else batch  
            print("Done!")
        return D_attack 

    def fit(self, epochs=100):
        # first generate the attack dataset
        self.attack_dataset = self.generate_attack_dataset()
        X, y = self.attack_dataset[:, :-1], self.attack_dataset[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)

        # fit the model
        self.history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=self.VERBOSE)
        
        return self.history
    
    def predict(self, X):
        return self.model.predict(X)
    

    def per_class_acc(self, X_attack, y_attack, n_classes):
      for c in range(n_classes):
        class_instances = X_attack[:, 0] == c # get same class samples
        test_loss, test_acc = self.model.evaluate(X_attack[class_instances, :], y_attack[class_instances], verbose=0)
        print(f"class-{c+1} acc: {test_acc}")

    def evaluate(self, X, y, verbose=0):
        """
        Print:
        - per class accuracy
        - classification report
        - ROC-Curve
        """

        self.per_class_acc(X, y, self.n_classes)

        y_pred_proba = self.predict(X)
        y_pred = y_pred_proba > 0.5
        
        print(classification_report(y.reshape(-1), y_pred.reshape(-1)))
        
        # ROC-Curve
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        plt.plot(fpr, tpr)
        print(f"AUC: {roc_auc_score(y, y_pred_proba)}")
