#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:15:55 2017

@author: vitorhadad
"""

#!/usr/bin/env python3

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


class MLPNet(nn.Module):
    
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
        
        super(MLPNet, self).__init__()
        
        self.feature_size  = feature_size
        self.hidden_sizes = hidden_sizes or []
        self.output_size = output_size
        
        self.activation_fn = activation_fn
        self.dropout_prob = dropout_prob
        self.output_fn = output_fn()
        self.loss_fn = loss_fn()
        
        self.model = self.build_model()
    
        self.opt=opt(self.parameters(), **opt_params)
        
  
        
    def forward(self, XX):
        XX = to_var(XX, requires_grad=False)
        return self.model(XX)
    


    def build_model(self):
        
        layer = lambda inp, out: \
                nn.Sequential(
                    nn.Linear(inp, out),
                    self.activation_fn(),
                    nn.AlphaDropout(self.dropout_prob))
        
        sizes = [self.feature_size,
                 *self.hidden_sizes,
                 self.output_size]
        
        mlp = [layer(h0,h1) for h0,h1 in zip(sizes[:-1], sizes[1:])]
            
        return nn.Sequential(*mlp, self.output_fn)
            
    
    
    def run(self, X, y):
        
        self.train()
        
        out = self.forward(X)
        
        loss = self.loss_fn(out, to_var(y, False))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        if self.output_fn == F.log_softmax:
            out = torch.exp(out)
        
        return loss.data.numpy()[0], out.data.numpy()
    
    
    def __str__(self):
        
        if self.hidden_sizes:
            hs = "-".join([str(x) for x in self.hidden_sizes])
        else: 
            hs = "None"
        
        return "MLP_" + hs + \
            "_{:1d}".format(int(100*self.dropout_prob))
    
    
    
    
    
def balancing_weights(XX, y):
    yy = y.flatten()
    n1 = np.sum(yy)
    n0 = len(yy) - n1
    p = np.zeros(int(n0 + n1))
    p[yy == 0] = 1/n0
    p[yy == 1] = 1/n1
    p /= p.sum()
    return p
    
    

#%%    
    
if __name__ == "__main__":
    
    from sys import platform
    from itertools import cycle
    from random import choice
    from collections import deque
    
    
    #%%

    if platform == "darwin":
        hidden = None
        dp = .2
    else:
        hidden = choice([None,
                        [10], [50], [100],
                        [10, 10], [50,50], [100, 100],
                        [50, 50, 50]])
        dp = choice([0, .1, .2, .5])
        
    
    mlp = MLPNet(24, hidden,
                 activation_fn = nn.SELU,
                 dropout_prob = dp,
                 loss_fn = nn.MSELoss,
                 opt_params = dict(lr = 0.005))
    print(mlp)
    
    #%%
    
    minibatch = 100
    iters = 50000
    training_losses = [] 
    training_accs = [] 
    recent_losses = deque(maxlen = 250)
    recent_accs = deque(maxlen = 250)
    name = str(mlp) + \
            "_{}".format(str(np.random.randint(1000000)))
    
    #%%
    
    for k in cycle(range(10)):
        
        XX = np.load("data/policy_data_X_%d.npy" % k)
        y = np.load("data/policy_data_y_%d.npy" % k)
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



    