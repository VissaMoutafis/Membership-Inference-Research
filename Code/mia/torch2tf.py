from pickletools import optimize
from sre_constants import NOT_LITERAL_IGNORE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets
from tqdm.notebook import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from opacus.privacy_engine import forbid_accumulation_hook
from opacus.utils.batch_memory_manager import BatchMemoryManager
from mia.utilities import free_gpu_cache

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, X_data, y_data):
    self.X_data = X_data
    self.y_data = y_data

  def __getitem__(self, index):
    return self.X_data[index], self.y_data[index]

  def __len__(self):
    return len(self.X_data)


class TfIgniter():
    MAX_PHYSICAL_BATCH_SIZE = 32
    def __init__(self, torch_input_shape, tf_input_shape, model, device):
        self.model = model.double().to(device)
        self.torch_input_shape = torch_input_shape
        self.tf_input_shape = tf_input_shape
        self.device = device
        self.privacy_engine = None

    def trainning_step(self, epoch_i, optimizer, criterion, memory_safe_dataloader, eval_loader, history=None):
        # 1 step of backprop with train/test error estimation
        losses = []
        acc = []
        for i,(_X, _y) in enumerate(memory_safe_dataloader):
            # set the gradients to zero for new estimation
            optimizer.zero_grad()
            # move data to the same device as the model
            X, y = _X.to(self.device), _y.to(self.device)
            # forward pass
            y_pred = self.model(X)
            # compute loss
            loss = criterion(y_pred, y)
            # backpropagate error
            loss.backward()
            # adjust model's parameters
            optimizer.step()
            
            # calculate loss and accuracy for current batch
            losses.append(float(loss.item()))
            acc.append(float((torch.sum(y == y_pred.argmax(axis=1)).double()/len(_X)).item()))

        free_gpu_cache()
        
        train_loss = sum(losses)/len(memory_safe_dataloader)
        train_acc = sum(acc)/len(memory_safe_dataloader)
        losses = []
        acc = []

        if eval_loader is not None:
            with torch.no_grad():
                # iterate in test set and get stats for the model
                for _X, _y in eval_loader:
                    X, y = _X.to(self.device), _y.to(self.device)
                    y_pred = self.model(X)
                    loss = criterion(y_pred, y)
                    losses.append(loss.item())
                    acc.append((torch.sum(y == y_pred.argmax(axis=1)).double()/len(_X)).item())

                    free_gpu_cache()

        test_loss = sum(losses)/len(eval_loader)
        test_acc = sum(acc)/len(eval_loader)
        # if the user provides with a history dict the training step will save the current epoch's train-test loss
        if history is not None:
            history['train']['loss'].append(train_loss)
            history['train']['acc'].append(train_acc)
            history['test']['loss'].append(test_loss)
            history['test']['acc'].append(test_acc)

        return train_loss, train_acc, test_loss, test_acc

    def call_with_batches(self, X, y_true=None, batch_size=64, criterion=None, **kwargs):
        if y_true is not None:
            dataset = MyDataset(X.reshape(-1, *self.torch_input_shape), y_true)
        else:
            dataset = MyDataset(X.reshape(-1, *self.torch_input_shape), X.reshape(-1, *self.torch_input_shape))
            
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        losses = []
        acc = []
        y_pred_cat = None
        with torch.no_grad():
            for _X, _y in dataloader:
                X = _X.to(self.device)
                y_pred = self.model(X)
                y_pred_cat = torch.cat((y_pred_cat, y_pred), axis=0) if y_pred_cat is not None else y_pred
                if criterion:
                    y = _y.to(self.device)
                    loss = criterion(y_pred, y)
                    losses.append(loss.item())
                    acc.append((torch.sum(y == y_pred.argmax(axis=1)).double()/len(_X)).item())

            free_gpu_cache()
        
        return y_pred_cat, np.mean(losses), np.mean(acc) 
    
    def predict(self, X, **kwargs):
        self.model.eval()
        X_torch = torch.from_numpy(X.reshape(-1, *self.torch_input_shape))
        y_torch, _, _ = self.call_with_batches(X_torch, **kwargs)
        y = y_torch.cpu().detach().numpy()
        free_gpu_cache()

        return y

    def evaluate(self, X, y_true, loss_criterion=nn.CrossEntropyLoss(), **eval_args):
        _, loss, acc = self.call_with_batches(X, y_true=y_true, criterion=loss_criterion)
        return loss, acc
    
    def fit(self, X_train, y_train, validation_data=None, verbose=False, **train_args):
        trainset = MyDataset(X_train.reshape(-1, *self.torch_input_shape), y_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_args['batch_size'], shuffle=True)
        testset = MyDataset(validation_data[0].reshape(-1, *self.torch_input_shape), validation_data[1]) if validation_data is not None else None 
        testloader = torch.utils.data.DataLoader(testset, batch_size=train_args['batch_size'], shuffle=True) if testset is not None else None 
      
        history = {'train': {'loss':[], 'acc':[]}, 'test': {'loss':[], 'acc':[]}}
        
        pbar = tqdm(range(train_args['epochs']))
        if 'es' in train_args:
            train_args['es'].re_init()
            es = train_args['es']
        else:
            es = lambda obj : False
        
        optimizer = train_args['optimizer']['builder'](self.model.parameters(), **train_args['optimizer']['args'])
        criterion = train_args['criterion']()
        if 'privacy' in train_args and self.privacy_engine is None:
            history['privacy'] = {'epsilon':[]}
            self.privacy_engine = train_args['privacy']['engine']
            self.model, optimizer, trainloader = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=trainloader,
                **train_args['privacy']['args'],
            )
            print(f"Using sigma={optimizer.noise_multiplier} and C={train_args['privacy']['args']['max_grad_norm']}")

        
        if 'privacy' in train_args:
            with BatchMemoryManager(data_loader=trainloader, optimizer=optimizer, max_physical_batch_size=self.MAX_PHYSICAL_BATCH_SIZE) as opacus_loader:
                trainloader = opacus_loader

        self.model.train()
        for epoch in pbar:
            train_loss, train_acc, test_loss, test_acc = self.trainning_step(epoch, optimizer, criterion, trainloader, testloader, history)
            bar_msg = f'Epoch {epoch}: train loss: {train_loss}, train acc: {train_acc}, test_loss: {test_loss}, test acc: {test_acc}'
            
            if es(test_loss) :
                break
            
            if 'privacy' in train_args :
                epsilon = self.privacy_engine.accountant.get_epsilon(delta=train_args['privacy']['args']['target_delta'])
                history['privacy']['epsilon'].append(epsilon)
                bar_msg += f', epsilon-spent: {epsilon}'
        
            pbar.set_description(bar_msg, refresh=True)

        del trainset 
        del trainloader 
        del testset 
        del testloader
        
        return history


def MINIMIZE (delta):
    return delta < 0
def MAXIMIZE(delta):
    return delta > 0

class CustomEarlyStopping:
    
    def __init__(self, min_delta=1e-4, patience=0, optimization_type='MINIMIZE'):
        self.optimization_type = MINIMIZE if optimization_type=='MINIMIZE' else MAXIMIZE
        self.min_delta = min_delta 
        self.patience = patience
        self.n_triggers = 0
        self.last_obj = math.inf if optimization_type == 'MINIMIZE' else -math.inf
        self.update_tactic = min if optimization_type == 'MINIMIZE' else max 
        
    def re_init(self):
        self.n_triggers = 0
        
    def __call__(self, cur_obj):
        if self.last_obj is None:
            self.last_obj = cur_obj
            return False 
        
        # set delta
        delta = self.last_obj - cur_obj
        if self.optimization_type(delta - self.min_delta):
            self.n_triggers += 1
        else:
            self.n_triggers = 0
        
        # update last obj
        self.last_obj = self.update_tactic(cur_obj, self.last_obj)
        
        return self.n_triggers > self.patience
