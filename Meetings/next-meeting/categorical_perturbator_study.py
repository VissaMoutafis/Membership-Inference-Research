from mia.categorical_perturbator import augment_categorical_dataset
from mia.wrappers import *
from mia.utilities import *
from mia.attack_model import *
from mia.shadow_models import *
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score

from tqdm.notebook import tqdm
import sys
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import datetime
import json 

N_CLASSES = 20

def f_target():
    global N_CLASSES
    model = models.Sequential([
        layers.Dense(128, activation='tanh'),
        layers.Dense(N_CLASSES)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def f_attack():
    global N_CLASSES
    model = models.Sequential(
        [
            layers.Dense(N_CLASSES+1),
            layers.LeakyReLU(0.3),
            layers.Dense(1, activation='sigmoid')
        ]
    )
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def label_only_attack_purchase_k(target_model, X_train, y_train, X_attacker, y_attacker, n_shadows, d_shadow_size, n_perturbations, max_dist, sampling):
    print(f'Label_only: Attacking with {X_attacker.shape[0]} datapoints...')
    attack = LabelOnlyAttack(target_model,
                             (X_train, y_train),
                             (X_attacker, y_attacker),
                             attack_model_creator=f_attack,
                             shadow_creator=f_target,
                             n_shadows=n_shadows,
                             D_shadow_size=d_shadow_size,
                             verbose=False)

    es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1e-3, patience=10)

    attack.perform_attack(
        augmentation_generator=augment_categorical_dataset,
        aug_gen_args={'n_perturbations': n_perturbations,
                      'max_dist': max_dist, 'sampling': sampling},
        shadow={'epochs': epochs, 'batch_size': 128, 'callbacks': [es]},
        attack={'epochs': 50, 'batch_size': 128})
    res = attack.evaluate_attack()

    return res

es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1e-3, patience=10)
epochs = 100
batch_size = 128

N_SHADOWS = 5
D_SHADOW_SIZE = 17000
N_PERTURBATIONS = [5, 10, 20, 50]
MAX_DIST = [1, 2, 3]



# get dataset
dataset_path = f'../../Datasets/purchase-dataset/purchase-datasets/purchase-{N_CLASSES}.csv' 
data_df = pd.read_csv(dataset_path, index_col=0)
y = data_df.pop('label').to_numpy(dtype=np.int8).reshape(-1)
X = data_df.to_numpy(dtype=np.int8) 

# divide to target and attack dataset
X_target, X_attacker, y_target, y_attacker = train_test_split(X, y, train_size=14926, test_size=3*10**4, shuffle=True, random_state=0)
# train-test split for the target
X_train, X_test, y_train, y_test = train_test_split(X_target, y_target, test_size=0.33, shuffle=True, random_state=0)
print(f'D_train_in size: {X_train.shape[0]}. Ready to train target model...')

# create and train target model
target_model = f_target()
history = target_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[es])

prec = []
rec = []
auc = []
config = {}
for n_perturbations in N_PERTURBATIONS:
    config['#augmentations']
    for max_dist in MAX_DIST:
        config['max_dist']
        # perform label only attack
        _res, _auc, _fpr, _tpr = label_only_attack_purchase_k(target_model, X_train, y_train, X_attacker, y_attacker)
        auc.append({**config, 'AUC Score': _auc
                    })
        rec.append({**config, 'Recall': _res['macro avg']['recall']
                    })
        prec.append({**config, 'Precision': _res['macro avg']['precision']
                     })

with open(f'categorical-perturbation-study-{datetime.today()}.json', 'w') as fp:
  json.dump({
      'prec': prec,
      'rec': rec,
      'auc': auc
  }, fp)
