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

use_cuda = cuda.is_available()


class GraphSetNetwork(nn.Module):
    
    def __init__(self, 
                 feature_size,
                 gcn_size,
                 num_layers,
                 max_cycle_length,
                 max_cycles_per_period,
                 dropout_prob,
                 poly_order = 5):
    
        super(GraphSetNetwork, self).__init__()
        
        self.feature_size  = feature_size
        self.gcn_size      = gcn_size
        self.max_cycle_length       = max_cycle_length
        self.max_cycles_per_period  = max_cycles_per_period
        self.num_layers = num_layers
        self.poly_order = poly_order
        
        self.max_n = max_cycle_length * max_cycles_per_period
        
        self.poly = nn.ModuleList()
        for p in range(poly_order):
            layer = nn.Linear(feature_size, 
                              gcn_size,
                              bias = False)
            self.poly.append(layer)
        
        self.logit_layer = nn.Linear(gcn_size, 1)
        
        #self.init_weights()
    
        
    def init_weights(self):
        
        w0 = torch.FloatTensor([[0.1078], [-0.7220], [0.4832]])
        w1 = torch.FloatTensor([[-0.5870], [-0.8206], [-0.5438]])
        w2 = torch.FloatTensor([[-0.0282], [0.0507], [0.2431]])
        weights = [w0, w1, w2]

        w_logit = torch.FloatTensor([[0.0227, -0.5487, -0.4618]])
        
        for layer, w in zip(self.poly, weights):
            layer.weight.data = w
        
        self.logit_layer.weight.data = w_logit
        
        
    def forward(self, A, X):
        A, X, L = self.make_variables(A, X)
        X_poly = self.chebyshev(L, X)
        gcn_out = self.gcn(X_poly)
        logits = self.logit_layer(gcn_out).squeeze()
        return F.softmax(logits)
    
    
    
    def chebyshev(self, L, X):
        T = [X] # No transpose here?
        if self.poly_order > 1:
            T += [L @ X]
        for _ in range(2, self.poly_order):
            T += [2 * L @ T[-1] - T[-2]]
        return T
        
        
    
    def make_variables(self, A, X):

        #A = np.clip(A + A.transpose(0,2,1), a_min = 0, a_max = 1)
        
        D = np.stack([np.diag(a.sum(1)) for a in A])
        L = D - A

        R = np.linalg.eigvals(L)
        lmaxs = np.max(R, 1)
        for k, lmax in enumerate(lmaxs): 
            L[k] = 2/np.abs(lmax)*L[k] - np.eye(A.shape[1])
        # Taking abs just in case,
        # but for symmetric A, eig(L) should be real
        
        A = Variable(torch.FloatTensor(A))
        X = Variable(torch.FloatTensor(X))
        L = Variable(torch.FloatTensor(L))
        return A, X, L
            
        
    
    def gcn(self, X_poly):
        b, s, _ = X_poly[0].size()
        H = Variable(torch.zeros(b, s, self.gcn_size))
        for (layer, xx) in zip(self.poly, X_poly):
            H += layer(xx)        
        return F.selu(H)
    



#%%
if __name__ == "__main__":
    

    import numpy as np
    from environment  import KidneyExchange
    import matplotlib.pyplot as plt
    from os import listdir
    from itertools import cycle    
    from random import choice
    import pickle

    
    
    feature_size = 9 #choice([1, 9])
    gcn_size     = 10 #choice([10, 20, 50, 100])
    num_layers   = 5 #choice([1, 3, 5, 10])
    dropout_prob = .05 #choice([0.1, 0.2, 0.5])
    poly_order   = 20
    max_cycle_length      = 2
    max_cycles_per_period = 2
    
    ptr = GraphSetNetwork(feature_size          = feature_size, 
                          gcn_size              = gcn_size,
                          max_cycle_length      = max_cycle_length,
                          max_cycles_per_period = max_cycles_per_period,
                          num_layers            = num_layers,
                          dropout_prob          = dropout_prob,
                          poly_order            = poly_order)
    
    
    opt  = optim.RMSprop(ptr.parameters(), lr = 0.005)
    
    avg_rewards = []
    rewards = []
    solutions = []


#%%                
               
    data = []
    seeds = []
    files = listdir()
    for f in files:
        if f.startswith("y"):
            x = np.load(f)
            s = f.split("_")[2].split(".npy")[0]
            if x.shape == (400, 100):
                data.append(x)
                seeds.append(int(s))
    
    data = np.stack(data)[:, 100:, :]
    
    
   #%%
   
    epoch = 0
    k = 0
    rnd = np.random.randint(1e8)
    
    loss_function = nn.MSELoss()
    
#%%   
    for sol, seed in zip(data, seeds):
        
        env = KidneyExchange(batch_size  = 1, 
                             entry_rate  = 5, #data["entry_rate"],
                             death_rate  = .1, #data["death_rate"],
                             time_length = 300, #data["time_length"],
                             maxlen      = 100, #data["maxlen"],
                             seed        = seed) #data["seed"]) 
             
               
        episode_avg_rewards = []
         
        for t in range(env.time_length):
            
            A = env.A(t)
            X = env.X(t)[:,:,-feature_size:]
            
            out  = ptr.forward(A, X)
            
            loss = loss_function(out, 
                                 Variable(torch.FloatTensor(sol[t])))
            
            opt.zero_grad()
            
            loss.backward()
            
            opt.step()
            
            
            episode_avg_rewards.append(loss.data.numpy()[0])
            
            
            
        print("AVG MATCHES:", np.mean(rewards[-200:]))
            
        avg_rewards.append(np.mean(episode_avg_rewards))
        print("LOSS:", avg_rewards[-1])
        
        with open("multinomial_numlayers{}_featuresize{}_gcnsize{}_dropout_{}_{}.txt"\
                  .format(num_layers, feature_size, gcn_size, int(10*dropout_prob), rnd), "a") as f: 
            f.write(str(avg_rewards[-1]) + "\n")
            
        k += 1
            