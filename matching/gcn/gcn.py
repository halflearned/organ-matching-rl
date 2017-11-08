#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:28:15 2017

Experiments with GCN.
    Exactly the model in Kipf and Welling.
    Every node's y-value is the average of an initial y-value and
        its neighbors' y-values.
    Objective is to look at graph structure and predict y.
    The only regressor is a constant value.
    
Outcomes.
 1. None of these attempts worked with the formulation
        D^{-1} (I @ A)
    But more often that not they worked when we used
        D^{-1/2} (I @ A) D^{-1/2}

 2. Even for this very simple example, the algorithm's L2Loss
     dropped very little for the first 5000-10000 obs,
     but decreases fast after a certain point.
     
 3. Shallow works better: loss did not decrease when using 
     more than five layers.
     
 4. Both BCELoss and MSELoss worked, but results using L2-loss
     were farther from 0.5 (in the right direction) with 
     fewer observations.
            
    

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


class GraphConvNet(nn.Module):
    
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
        
        super(GraphConvNet, self).__init__()
        
        self.feature_size  = feature_size
        self.gcn_size      = gcn_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        
        self.output_fn = output_fn
        self.loss_fn = loss_fn()
        
        self.build_model()
    
        self.opt=opt(self.parameters(), **opt_params)
        
  
        
    def forward(self, A, X):
        A, X = self.make_variables(A, X)
        gcn_out = self.gcn(A, X).squeeze()
        return self.output_fn(gcn_out)
    

    def build_model(self):
        if self.num_layers > 1:
            self.gcn_layers = nn.ModuleList([nn.Linear(self.feature_size, self.gcn_size, bias = False)])
            self.dropouts = nn.ModuleList([nn.AlphaDropout(p = self.dropout_prob)])
            for p in range(self.num_layers - 2):
                layer = nn.Linear(self.gcn_size, self.gcn_size, bias = False)
                self.gcn_layers.append(layer)
                self.dropouts.append(nn.AlphaDropout(p = self.dropout_prob))
                
            self.gcn_layers.append(nn.Linear(self.gcn_size, 1))
        elif self.num_layers == 1:
            self.gcn_layers = nn.ModuleList([nn.Linear(self.feature_size, 1)])
            self.dropouts = []
        else:
            ValueError("Incorrect num_layers.")
            
    
    def make_variables(self, A, X):

        n = A.shape[0]
        
        outdegree = A.sum(1).flatten()
        with np.errstate(divide='ignore'):
            Dinvroot = np.sqrt(np.diag(np.where(outdegree > 0, 1 / outdegree, 0)))
        
        I = np.eye(n)
        newA = Dinvroot @ (I + A) @ Dinvroot
        assert np.all(np.isfinite(newA))
        vA = to_var(newA)
        vX = to_var(X)
        return vA, vX
            
    
    def gcn(self, A, X):
        H = X
        for layer, drop in zip(self.gcn_layers, self.dropouts):
            H = F.selu(layer(A @ H)) 
            H = drop(H)
        H = self.gcn_layers[-1](H)
        return H
    
    
    def run(self, A, X, y):
        
        self.train()
        
        out = self.forward(A, X)
        
        loss = self.loss_fn(out, to_var(y, False))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        if self.output_fn == F.log_softmax:
            out = torch.exp(out)
        
        return loss.data.numpy()[0], out.data.numpy()
    
    
    

#%%
if __name__ == "__main__":
    

    import numpy as np
    import matplotlib.pyplot as plt 
    import networkx as nx
    from tqdm import tqdm
    from pandas import Series

    def generate_naive_data(n, p):
        g = nx.erdos_renyi_graph(n, p).to_directed()
        y = np.zeros(n)
        for i in g.nodes():
            if g.out_degree(i) > n*p:
                g.node[i]["y"] = 1
                y[i] = 1
            else:
                g.node[i]["y"] = 0
        return g, y
    
    
    p = .2
    
    gcn = GraphConvNet( 
                 feature_size = 1,
                 gcn_size = 1,
                 num_layers = 2,
                 dropout_prob = .2,
                 output_fn = F.sigmoid,
                 loss_fn = nn.MSELoss,
                 opt = optim.Adam,
                 opt_params = dict(lr = 0.1))
    
    losses = []
    
    #%%
    for _ in tqdm(range(5000)):
        n = np.random.randint(50, 150)
        g, y = generate_naive_data(n, p)    
        A = np.array(nx.to_numpy_matrix(g))
        X = np.ones((n, 1))
        loss, _ = gcn.run(A, X, y)
        losses.append(loss)
        
    Series(losses).rolling(50).mean().plot()
    
    #%%
    test_losses = []
    gcn.eval()
    for _ in range(10):
        A = np.array(nx.to_numpy_matrix(g))
        X = np.ones((n, 1))
        loss, _ = gcn.run(A, X, y)
        test_losses.append(loss)
    
    print(np.mean(test_losses))
    print(gcn.forward(A, X)[:10])
    print(y[:10])
    
    
    
                