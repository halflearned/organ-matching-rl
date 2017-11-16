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
            h = layer(A @ h)
        return h
    

    def make_variables(self, AA, XX):

        with np.errstate(divide='ignore'):  
            As = []
            for A in AA:
                outdeg = A.sum(1)
                D = np.diag(np.sqrt(safe_invert(outdeg)))
                I = np.eye(A.shape[0])
                Atilde = D @ (I + A) @ D
                As.append(Atilde)   
            As = np.stack(As)
            
        XX = to_var(XX)
        AA = to_var(AA)
        
        return AA, XX
            


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
            
    
    
    def run(self, A, X, y):
        
        lengths = X.any(2).sum(1)
        batch_size = X.shape[0]
        yhat = self.forward(A, X)
        ytruth = to_var(y, False)

        # Compute loss, leaving out padded bits      
        loss = 0
        for s,yh,yt in zip(lengths, yhat, ytruth):
            loss += self.loss_fn(yh[:s], yt[:s])
        loss /= batch_size
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.data.numpy()[0], yhat.data.numpy()
    
    
    
    def __str__(self):
        
        if self.hidden_sizes:
            hs = "-".join([str(x) for x in self.hidden_sizes])
        else: 
            hs = "None"
        
        return "GCN_" + hs + \
            "_{:1d}".format(int(100*self.dropout_prob))
    


def pad(A, X, y, size):
    
    if len(A.shape) > 2 or len(X.shape) > 2:
        raise ValueError("A and X must be 2D.")
        
    n = size - X.shape[0]
    y = y.reshape(-1, 1)
    
    if n > 0:
        A = np.pad(A.toarray(), ((0,n),(0,n)), mode = "constant", constant_values = 0) 
        X = np.pad(X, ((0,n),(0,0)), mode = "constant", constant_values = 0)
        y = np.pad(y, ((0,n),(0,0)), mode = "constant", constant_values = 0)    
        
    return A, X, y



safe_invert = lambda x: np.where(x > 0, 1/x, 0)

#%%    
    
if __name__ == "__main__":
    
    from sys import platform
    from random import choice
    from collections import deque
    import pickle
    
    #%%
    As = pickle.load(open("data/gcn_policy_data_A.npy", "rb"))
    Xs = pickle.load(open("data/gcn_policy_data_X.npy", "rb"))
    ys = pickle.load(open("data/gcn_policy_data_y.npy", "rb"))
    n = len(Xs)
    
    #%%

    if platform == "darwin":
        hidden = [50, 50]
        dp = .2
    else:
        hidden = choice([None,
                        [10], [50], [100],
                        [10, 10], [50,50], [100, 100],
                        [50, 50, 50]])
        dp = choice([0, .1, .2, .5])
        
    
    gcn = GCNet(10, hidden,
                 activation_fn = nn.SELU,
                 dropout_prob = dp,
                 loss_fn = nn.MSELoss,
                 opt_params = dict(lr = 0.005))

    #%%
    iters = 50000
    maxsize = 100
    training_losses = [] 
    training_accs = [] 
    recent_losses = deque(maxlen = 250)
    recent_accs = deque(maxlen = 250)
    name = str(gcn) + "_{}".format(str(np.random.randint(1000000)))
    minibatch = 128
    
    #%%
    ws = np.array([np.mean(y) for y in ys])
    ws /= np.sum(ws)
    
    #%%
    
    for k in range(iters):
        
        idx = np.random.choice(n, size = minibatch,  p = ws) 
        AA = []
        XX = []
        yy = []

        for i in idx:
            A,X,y = pad(As[i], Xs[i], ys[i], size = maxsize)
            AA.append(A)
            XX.append(X)
            yy.append(y)
        
        AA = np.stack(AA, 0)
        XX = np.stack(XX, 0)
        yy = np.stack(yy, 0)
        
        loss, out = gcn.run(AA, XX, yy)
        recent_losses.append(loss)
        acc = np.equal(out > .5, yy > .5).mean()
        recent_accs.append(acc)
        
        if k % 100 == 0:
            training_losses.append(np.mean(recent_losses))
            training_accs.append(np.mean(recent_accs))
            print(i, training_losses[-1], training_accs[-1])

        if k %  5000 == 0:
            filename = "results/{}.pkl".format(name)
            torch.save(gcn, filename)
            pickle.dump(training_accs, open("results/acc_{}.pkl".format(name), "wb"))
            pickle.dump(training_losses, open("results/losses_{}.pkl".format(name), "wb"))



    