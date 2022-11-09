import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

class DefaultAttackModel():
    VERBOSE=False
    """
    Default attack model for confidence vector attack
    @param shadow_batch: A ShadowModelBatch object with trained models that will be used as target imitators to execute attack
    @param n_classes: the number of classes in the classification models output
    @param X_attack_dim: the dimensions of the attack dataset's instances (i.e. in confidence vector we use (n_classes,))
    @param _optimizer: optimizer to use in attack-model fitting
    """
    def __init__(self, shadow_batch, n_classes, f_attack_builder, f_atck_args={}):
        # default structure as Shokri et al. suggested
        self.model = f_attack_builder(**f_atck_args)
        
        self.shadow_models_batch = shadow_batch 
        self.n_classes = n_classes 
        self.attack_dataset = []
        self.history = None
    
    """
    helper to prepare a batch of shadow data into a batch of attack data
    Can be called to create a test-dataset for the model, outside the model.
    @param model: the model to which we relate the data 
    @param X, y: instances and their true labels
    @param in_D: boolean-> True if data are in the model's training dataset, otherwise False
    
    """
    def prepare_batch(self, model, X, y, in_D):
        # decide membership
        y_member = np.ones(shape=(y.shape[0], 1)) if in_D else np.zeros(shape=(y.shape[0], 1))
        prob = layers.Softmax()
        ret = prob(model.predict(X).astype(np.float32)).numpy().reshape((-1,self.n_classes))
        
        # return an instance <true label, confidence vector, 0/1 D_target membership> 
        return np.concatenate((y.reshape((-1, 1)), ret, y_member), axis=1)

    """ 
    helper function to generate the attack dataset from the previously given shadow models batch
    Warning: only called from inside the model.
    """
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


    """
    Classic fit loop that trains the model based on the shadow batch we provided at the constructor
    """
    def fit(self, **train_args):
        # first generate the attack dataset
        self.attack_dataset = self.generate_attack_dataset()
        X, y = self.attack_dataset[:, :-1], self.attack_dataset[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)

        # fit the model
        try:
            self.history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=self.VERBOSE, **train_args)
        except :
            print(f'Warning: Cannot fit with tf-like fit routine..trying sklearn.')
            self.model.fit(X_train, y_train)
            
        return self.history
    
    """Simple predict function. Returns logits"""
    def predict(self, X):
        return self.model.predict(X)
    
    """ 
    Only called from the self.evaluate function to evaluate per class accuracy of attack 
    """
    def per_class_acc(self, X_attack, y_attack, n_classes):
      for c in range(n_classes):
        class_instances = X_attack[:, 0] == c # get same class samples
        test_loss, test_acc = self.model.evaluate(X_attack[class_instances, :], y_attack[class_instances], verbose=0)
        print(f"class-{c+1} acc: {test_acc}")

    """
    Evaluate attack based on X, y (use preppare_batch to create them).
        Evaluation will print:
        - per class accuracy
        - classification report
        - ROC-Curve
        - AUC-Score
    """
    def evaluate(self, X, y, verbose=0):
        
        try:
            self.per_class_acc(X, y, self.n_classes)
        except:
            print("Warning: AttackModel has no attribute evaluate")
            
        y_pred_proba = self.predict(X)
        
        if (len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2):
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = y_pred_proba > 0.5
        
        _report = classification_report(y.reshape(-1).astype(np.int8), y_pred.reshape(-1).astype(np.int8), labels=[0,1], target_names=['Out', 'In'], output_dict=True)
        print(classification_report(y.reshape(-1).astype(np.int8),
              y_pred.reshape(-1).astype(np.int8), labels=[0, 1], target_names=['Out', 'In']))
        
        # ROC-Curve
        try:
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            plt.plot(fpr, tpr)
            auc_ = roc_auc_score(y, y_pred_proba)
            print(f"AUC: {auc_}")
        except:
            fpr, tpr, _ = roc_curve(y.reshape(-1), y_pred_proba[:,1])
            plt.plot(fpr, tpr)
            auc_ = roc_auc_score(y.reshape(-1), y_pred_proba[:, 1])
            print(f"AUC: {auc_}")

        # return classification report, false and true percentage, in case someone wants to use it
        return _report, auc_, fpr, tpr
