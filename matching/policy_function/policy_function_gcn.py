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
                 output_size = 2,
                 dropout_prob = 0.2,
                 activation_fn = nn.ReLU,
                 output_fn = nn.Sigmoid,
                 opt = optim.Adam,
                 opt_params = dict(lr = 0.001),
                 loss_fn = None,
                 seed = None):
    
        
        if seed : torch.manual_seed(seed)
        
        super(GCNet, self).__init__()
        
        self.feature_size  = feature_size
        self.hidden_sizes = hidden_sizes or []
        self.output_size = output_size
        
        self.activation_fn = activation_fn
        self.dropout_prob = dropout_prob
        self.output_fn = output_fn()
        if loss_fn is None:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn
        
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
        
        if self.output_fn is not None:
            return nn.Sequential(*mlp, self.output_fn)
        else:
            return nn.Sequential(*mlp)
            
    
    
    def run(self, A, X, y, lengths = None):
        
        if lengths is None:
            lengths = X.any(2).sum(1)
            
        batch_size = X.shape[0]
        yhat = self.forward(A, X)
        ytruth = to_var(y, False).long()
        
        # Compute loss, leaving out padded bits      
        loss = 0
        for s,yh,yt in zip(lengths, yhat, ytruth):
            loss += self.loss_fn(yh[:s], yt[:s].view(-1)).mean()
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

    from sys import argv
    from matching.utils.data_utils import open_file, confusion

    batch_size = 32
    open_every = 100
    log_every = 10
    save_every = 500
 
    
    if len(argv) > 1:
        use_gn = bool(argv[1])
        hidden = [int(x) for x in argv[2:]]
    else:
        use_gn = False
        hidden = [100, 100]
    
        
    net = GCNet(280 + 14*use_gn, 
                hidden,
                dropout_prob = .2,
                loss_fn = loss_fn) 
    #%%

    for i in range(int(1e8)):
        if i % open_every == 0:
            if use_gn:
                A, X, GN, Y = open_file(open_A = True,
                                        open_GN = True)
                Z = np.concatenate([X, GN], 2)
            else:
                A, X, Y = open_file(open_A = True,
                                    open_GN = False)
                Z = X
            
        n = A.shape[0]
        b_idx = np.random.choice(n, size = batch_size)   
        inputs = (A[b_idx], Z[b_idx])
        ytrue = Y[b_idx]
        lens = inputs[1].any(2).sum(1)

        loss, yhat = net.run(*inputs, ytrue)
        tp, tn, fp, fn = confusion(yhat, ytrue, lens)
        tpr=tp/(tp+fn)
        tnr=tn/(tn+fp)
        
        if i % log_every == 0:
            msg = "{:1.4f},{:1.4f},{:1.4f},{:1.4f}"\
                .format(loss,
                        tpr, # True positive rate
                        tnr, # True negative rate
                        (tp + tn)/(tp+fp+tn+fn)) # Accuracy
            print(msg)
            with open("results/" + str(net) + ".txt", "a") as f:
                print(msg, file = f)
                
            if tpr > .8 and tnr < .5:
                net.loss_fn = nn.CrossEntropyLoss(reduce = False,
                            weight = torch.FloatTensor([1, 1]))
            elif tpr < .5 and tnr > .8:
                net.loss_fn = nn.CrossEntropyLoss(reduce = False,
                            weight = torch.FloatTensor([1, 10]))
            else:
                net.loss_fn = nn.CrossEntropyLoss(reduce = False,
                            weight = torch.FloatTensor([1, 5]))
        
    if i % save_every == 0:
        torch.save(net, "results/" + str(net))
 


    