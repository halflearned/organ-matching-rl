#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:28:15 2017


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


class GCNet(nn.Module):
    
    def __init__(self, 
                 feature_size,
                 hidden_sizes = None,
                 output_size = 1,
                 dropout_prob = 0.2,
                 activation_fn = nn.SELU,
                 output_fn = nn.Sigmoid,
                 opt = optim.Adam,
                 opt_params = dict(lr = 0.001),
                 loss_fn = nn.MSELoss,
                 seed = None):
    
        
        if seed : torch.manual_seed(seed)
        
        super(GCNet, self).__init__()
        
        self.feature_size  = feature_size
        self.hidden_sizes = hidden_sizes or []
        self.output_size = output_size
        
        self.activation_fn = activation_fn
        self.dropout_prob = dropout_prob
        self.output_fn = output_fn()
        self.loss_fn = loss_fn()
        
        self.model = self.build_model()
    
        self.opt=opt(self.parameters(), **opt_params)
        
  
        
    def forward(self, A, X):
        A, X = self.make_variables(A, X)
        h = X
        for layer in self.model:
            h = layer(A @ X)
        return h
    

    def make_variables(self, A, X):

        with np.errstate(divide='ignore'):        
            outdegree = A.sum(1).flatten()
            Dinvroot = np.sqrt(np.diag(np.where(outdegree > 0, 1 / outdegree, 0)))

        n = A.shape[0]        
        I = np.eye(n)
        Atilde = to_var(Dinvroot @ (I + A) @ Dinvroot)
        X = to_var(X)
        
        return Atilde, X
            


    def build_model(self):
        
        layer = lambda inp, out: \
                nn.Sequential(
                    nn.Linear(inp, out),
                    self.activation_fn(),
                    nn.AlphaDropout(self.dropout_prob))
        
        sizes = [ self.feature_size,
                 *self.hidden_sizes,
                  self.output_size ]
        
        mlp = [layer(h0,h1) for h0, h1 in zip(sizes[:-1], sizes[1:])]
        
        return nn.Sequential(*mlp, self.output_fn)
            
    
    
    def run(self, X, y):
        
        self.train()
        
        out = self.forward(X)
        
        loss = self.loss_fn(out, to_var(y, False))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.data.numpy()[0], out.data.numpy()
    
    
    def __str__(self):
        
        if self.hidden_sizes:
            hs = "-".join([str(x) for x in self.hidden])
        else: 
            hs = "None"
        
        return "GCN_" + hs + \
            "_{:1d}".format(int(100*self.dropout_prob))
    


#%%    
    
if __name__ == "__main__":
    
    from sys import platform
    from itertools import cycle
    from random import choice
    from collections import deque
    
    from matching.utils.data_utils import balancing_weights
    
    #%%
    
    
    #%%

    if platform == "darwin":
        hidden = None
        dp = .2
    else:
        hidden = choice(None,
                        [10], [50], [100],
                        [10, 10], [50,50], [100, 100],
                        [50, 50, 50])
        dp = [0, .1, .2, .5]
        
    
    gcn = GCNet(10, hidden,
                 activation_fn = nn.SELU,
                 dropout_prob = dp,
                 loss_fn = nn.MSELoss,
                 opt_params = dict(lr = 0.005))

    #%%
    
    minibatch = 100
    iters = 50000
    training_losses = [] 
    training_accs = [] 
    recent_losses = deque(maxlen = 250)
    recent_accs = deque(maxlen = 250)
    name = str(gcn) + "_{}".format(str(np.random.randint(1000000)))
    
    #%%
    
    for i in cycle(range(10)):
        
        XX = np.load("data/policy_data_X_%d.npy" % i)
        y = np.load("data/policy_data_y_%d.npy" % i)
        n = XX.shape[0]
        ws = balancing_weights(XX, y)
        
        for i in range(iters):
        
            idx = np.random.choice(n, size = minibatch, p = ws)
            loss, out = mlp.run(XX[idx], y[idx])
            recent_losses.append(loss)
            acc = np.equal(out > .5, y[idx] > .5).mean()
            recent_accs.append(acc)
            
            if i % 100 == 0:
                training_losses.append(np.mean(recent_losses))
                training_accs.append(np.mean(recent_accs))
                print(i, out[0], training_losses[-1], training_accs[-1])
    
#%%
    filename = "results/{}.pkl".format(name)
    torch.save(mlp, filename)



    