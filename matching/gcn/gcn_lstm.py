#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:28:15 2017

Experiments with GCN + LSTM using seq2seq architecture.
No attention yet.

@author: vitorhadad
"""

import torch
from torch import nn, cuda
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np


def all_sum(x):
    return x.sum()
    

use_cuda = cuda.is_available()

def to_var(x, requires_grad = True):
    return Variable(torch.FloatTensor(np.array(x, dtype = np.float32)),
                    requires_grad = requires_grad)



class GraphConvNetLSTM(nn.Module):
    
    def __init__(self, 
                 feature_size,
                 gcn_size,
                 num_layers,
                 dropout_prob,
                 output_fn,
                 opt,
                 opt_params = dict(lr = 0.001),
                 loss_fn = nn.MSELoss,
                 seed = None):
    
        
        if seed : torch.manual_seed(seed)
        
        super(GraphConvNetLSTM, self).__init__()
        
        self.feature_size  = feature_size
        self.gcn_size      = gcn_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.bidirectional = True
        self.num_lstm_layers = 3
        self.num_directions = 1 + self.bidirectional
        self.lstm_hidden_size = 5
        self.batch_size = 1
        
        self.lstm_state_size = (self.num_lstm_layers * self.num_directions, 
                       self.batch_size,
                       self.lstm_hidden_size)
        
        self.output_fn = output_fn
        self.loss_fn = loss_fn()
        
        self.build_model()
    
        self.opt=opt(self.parameters(), **opt_params)
        
  
        
    def forward(self, A, X):
        
        # Prepare and embed
        A, X = self.make_variables(A, X)
        gcn_out = self.gcn(A, X)
        
        # Encode
        n = gcn_out.size()[0]
        hc = self.h0, self.c0
        for i in range(n):
            input = gcn_out[i].resize(1,1,self.gcn_size)
            out, hc = self.rnn(input, hc)   
        
        # Decode
        dec_outs = []
        dec_out = self.decode_token
        for j in range(n):
            dec_out, hc = self.rnn(dec_out, hc)
            p = self.output_layer(dec_out)
            dec_outs.append(p)
            
        probs = F.sigmoid(torch.stack(dec_outs).squeeze())
        return probs
    
    

    def build_model(self):
        
        self.gcn_layers = nn.ModuleList([nn.Linear(self.feature_size, self.gcn_size, bias = False)])
        self.dropouts = nn.ModuleList([nn.AlphaDropout(p = self.dropout_prob)])
        for p in range(self.num_layers - 1):
            layer = nn.Linear(self.gcn_size, self.gcn_size, bias = False)
            self.gcn_layers.append(layer)
            self.dropouts.append(nn.AlphaDropout(p = self.dropout_prob))
        
        self.rnn = nn.LSTM(input_size = self.gcn_size,
                           hidden_size = self.lstm_hidden_size,
                           num_layers = self.num_lstm_layers,
                           bias = True,
                           batch_first = False,
                           dropout = False,
                           bidirectional = self.bidirectional) # For now
    

        self.h0 = Variable(torch.randn(*self.lstm_state_size), requires_grad = True)
        self.c0 = Variable(torch.randn(*self.lstm_state_size), requires_grad = True)
        
        self.decode_token = Variable(torch.randn(1, self.batch_size, self.gcn_size), requires_grad = True)
        
        self.output_layer = nn.Linear(self.lstm_hidden_size * self.num_directions, 1)
        
    
    def make_variables(self, A, X):

        n = A.shape[0]
        
        outdegree = A.sum(1).flatten()
        with np.errstate(divide='ignore'):
            Dinvroot = np.sqrt(np.diag(np.where(outdegree > 0, 1 / outdegree, 0)))
        
        I = np.eye(n)
        newA = Dinvroot @ (I + A) @ Dinvroot
        vA = to_var(newA)
        vX = to_var(X)
        return vA, vX
            
    
    def gcn(self, A, X):
        H = X
        for layer, drop in zip(self.gcn_layers, self.dropouts):
            H = F.selu(layer(A @ H)) 
            H = drop(H)
        return H
    
    
    def run(self, A, X, y, train = True):
        
        if train:
            self.train()
        
        probs = self.forward(A, X)
        if train:
            
            loss = self.loss_fn(probs, to_var(y, False))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            return loss#.data.numpy()[0], probs.data.numpy().flatten()
        else:
            return probs.data.numpy().flatten()
    
    
    

#%%
if __name__ == "__main__":
    
    
    from make_data import merge_data
    import matplotlib.pyplot as plt    
    
    #%%
    
    data = merge_data("A", "X", "y",
                      path = "data/", 
                      horizon = 44, 
                      max_cycle_length = 2)
    
    #%%
    
    name = str(np.random.randint(1e8))
  
    
    net = GraphConvNetLSTM(feature_size = 10,
                       gcn_size = 10,
                       num_layers = 2,
                       dropout_prob = .2,
                       output_fn = nn.BCELoss,
                       opt = optim.Adam,
                       opt_params = dict(lr = 0.0001))
    
    training_losses = []
    
    #%%
    for training_epoch in range(1):
    
        for k, (A, X, y) in enumerate(zip(data["A"], data["X"], data["y"])):
            
            if y.sum() < 2: continue
            
            loss = net.run(A.toarray(), X, y)
            training_losses.append(loss.data.numpy()[0])
            print(loss)
            p = net.forward(A.toarray(), X)
        
    