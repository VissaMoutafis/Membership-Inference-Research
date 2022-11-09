# -*- coding: utf-8 -*-
"""mia-vs-confidence-masking.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1persX8fEsNjbfgHxmK3SAMAKWhKBlBcc

# MIA vs Overfitting

In this notebook we will study the effect of overfitting in MIA's performance, given a CNN model, CIFAR-10 dataset and a MIA framework that will perfrom the attacks for us.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import product 
from datetime import datetime

import math
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from tensorflow import keras

from mia.attack_model import *
from mia.label_only import *
from mia.shadow_models import *
from mia.utilities import *
from mia.confidence_masking import *
from mia.wrappers import ConfidenceVectorAttack, LabelOnlyAttack, TopKConfidenceMaskingAttack
from tqdm.notebook import tqdm
import sys

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def f_target():
  """
  Returns a trained target model, if test data are specified we will evaluate the model and print its accuracy
  """
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='tanh', input_shape=(32, 32, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='tanh'))
  model.add(layers.MaxPooling2D((2, 2)))


  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))

  model.add(layers.Dense(10))
  
  optimizer = keras.optimizers.Adam()

  model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  return model

def f_shadow():
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='tanh', input_shape=(32, 32, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='tanh'))
  model.add(layers.MaxPooling2D((2, 2)))


  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))

  model.add(layers.Dense(10))
  
  optimizer = keras.optimizers.Adam()

  model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  return model

def create_shadow_model(top_k=10, model_builder=f_shadow):
  return TopKConfidenceMaskingModel(top_k, model_builder)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = train_images/255
test_images = test_images/255

print(f"We have {len(train_images)} train instances and {len(test_images)} test instances.")

"""Let us give the attacker 10000 data points and let the rest of them be training-evaluation-testing datapoints for the target model."""

attacker_images, attacker_labels = (test_images, test_labels)

N_SHADOWS = [1, 5, 10, 15]
D_SHADOW= [5000, 7500]
D_TARGET = [2500, 7500, 10000]
TOP_K = [10, 8, 5, 3]
TEST_SET_SIZE = 0.3


attacks = {}
# train attacks for each case
for top_k in TOP_K:
  for attack_settings in product(N_SHADOWS, D_SHADOW):
    # set up settings
    n_shadows, d_shadow_size = attack_settings
    attack = TopKConfidenceMaskingAttack(None, (train_images, train_labels), 
                              (attacker_images, attacker_labels), top_k, 
                              shadow_creator=create_shadow_model, shd_crt_args={'top_k':top_k, 'model_builder':f_shadow},
                              attack_model_creator=cifar_10_f_attack_builder, atck_crt_args={'top_k':2*top_k},
                              n_shadows=n_shadows, D_shadow_size=d_shadow_size, verbose=False)
    attacks[(top_k, n_shadows, d_shadow_size)] = attack
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-4, patience=5)
    attack.perform_attack(shadow={'epochs':100, 'batch_size':128, 'callbacks':[es]}, attack={'epochs':100, 'batch_size':128})
    
prec = []
rec = []
acc = []
auc = []
model_vuln = []
config = {}
for top_k in TOP_K:
  config['|y|'] = top_k
  for d_target in D_TARGET:
    config['|D_target|'] = d_target
    d_total = int(d_target//TEST_SET_SIZE + 1)
    target_images, target_labels = train_images[:d_total], train_labels[:d_total]
    X_train, X_test, y_train, y_test = train_test_split(target_images, target_labels, test_size=TEST_SET_SIZE)
    
    # train target model 
    target_model = TopKConfidenceMaskingModel(top_k, f_target)
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-4, patience=5)
    history = target_model.fit(X_train, y_train, 
                        epochs=100, 
                        validation_data=(X_test, y_test),
                        callbacks=[es]
                        )

    for attack_settings in product(N_SHADOWS, D_SHADOW):
      # set up settings
      n_shadows, d_shadow_size = attack_settings
      config['#Shadow-Models'] = n_shadows
      config['|D_shadow_i|'] = d_shadow_size
      attack = attacks[(top_k, n_shadows, d_shadow_size)]
      attack.target_model = target_model 
      attack.target_dataset = (X_train, y_train)

      
      score_ = attack.evaluate_attack()

      
      auc.append({**config, 'AUC Score' : score_[1]
      })
      rec.append({**config, 'Recall' : score_[0]['macro avg']['recall']
      })
      prec.append({**config, 'Precision' : score_[0]['macro avg']['precision']
      })

with open(f'MIAvsConfMasking-{datetime.today()}.json', 'w') as fp:
  json.dump({
    'prec':prec,
    'rec':rec,
    'auc': auc
  }, fp)
