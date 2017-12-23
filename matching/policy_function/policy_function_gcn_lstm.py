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


from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_lstm import RNN

def to_var(x, requires_grad = True):
    return Variable(torch.FloatTensor(np.array(x, dtype = np.float32)),
                    requires_grad = requires_grad)


class RGCNet(nn.Module):
    
    def __init__(self, 
                 input_size,
                 hidden_size,
                 num_layers):
        
        super(RGCNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 
        self.seq_len = 1
        self.gcn = GCNet(input_size,
                      [hidden_size]*num_layers)
        self.rnn = RNN(hidden_size,
                       hidden_size,
                       num_layers = 1)
        self.logit_layer = self.Linear(hidden_size, 2)
        self.count_layer = self.Linear(hidden_size, 1)
        
        self.opt = torch.optim.Adam(self.parameters())
        
        
    def forward(self, A, X, lens = None):
        if lens is None:
            lens = X.any(2).sum(1)
            
        A, X = self.gcn.make_variables(A, X)
        
        h = X
        for layer in self.gcn.model:
            h = layer(A @ h)
        
        logits, counts = self.rnn.forward(h, lens)
        return logits, counts
        
    
    def run(self, A, X, y, lengths):
        if lengths is None:
            lengths = X.any(2).sum(1)
            
        batch_size = X.shape[0]
        ylogits, ycount = self.forward(A, X)
        ytruth = to_var(y, False).long()
        
        # Compute loss, leaving out padded bits      
        logit_loss = 0
        for s,yh,yt in zip(lengths, ylogits, ytruth):
            try:
                logit_loss += self.logit_loss_fn(yh[:s], yt[:s].view(-1)).mean()
            except RuntimeError as e:
                print("RuntimeError", e)
        logit_loss /= batch_size
        
        count_loss = self.count_loss_fn(ycount, ytruth.sum(1).float())
    
        loss = logit_loss + count_loss
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return (logit_loss.data.numpy()[0], 
                count_loss.data.numpy()[0], 
                ylogits, ycount.data.numpy())
        
    
        
    def __str__(self):
        return "RGCNet-{}-{}"\
                .format(self.hidden_size,
                        self.num_layers)


#%%    
    
if __name__ == "__main__":

    from sys import argv, platform
    from matching.utils.data_utils import open_file, confusion

    batch_size = 32
    open_every = 100
    log_every = 10
    save_every = 500
 
    if platform == "darwin":
        argv = [None, "optn", "3", "100", "True", np.random.randint(1e8)]

    env_type = argv[1]
    hidden = int(argv[3])
    num_layers = int(argv[2])
    use_gn = bool(argv[4])
    s = str(argv[5])
    
    input_size = {"abo":10, "optn":280}
        
    net = GCNet(input_size[env_type] + 14*use_gn, 
                [hidden]*num_layers,
                dropout_prob = .2) 
    
    name = "{}-{}_{}".format(
            str(net),
            env_type,
            s)
    c = .5
    #%%

    for i in range(int(1e8)):
        if i % open_every == 0:
            print("new file!")
            if use_gn:
                A, X, GN, Y = open_file(env_type = env_type,
                                        open_A = True,
                                        open_GN = True)
                Z = np.concatenate([X, GN], 2)
            else:
                A, X, Y = open_file(env_type = env_type,
                                    open_A = True,
                                    open_GN = False)
                Z = X
            
        n = A.shape[0]
        idx = np.random.choice(n, size = batch_size)   
        inputs = (A[idx], Z[idx])
        ytrue = Y[idx]
        lens = inputs[1].any(2).sum(1)
        
        avg_ones = np.hstack([Y[k,:l,0] for k,l in zip(idx, lens)]).mean()
        if avg_ones > 0:
            w = c*1/avg_ones
        
        net.logit_loss_fn = nn.CrossEntropyLoss(reduce = True,
                        weight = torch.FloatTensor([1, w]))
        
        lloss,closs, ylogits, ycount = net.run(*inputs, ytrue)
        cacc = np.mean(ycount.round() == ytrue.sum(1))
        tp, tn, fp, fn = confusion(ylogits, ytrue, lens)
        tpr = tp/(tp+fn)
        tnr = tn/(tn+fp)
        lacc = (tp + tn)/(tp+fp+tn+fn)
        
        if tpr < .1:
            c *= 1.05
        if tnr < .1:
            c *= .95
        
        msg = "{:1.4f},{:1.4f},{:1.4f},"\
                "{:1.4f},{:1.4f},{:1.4f},{:1.4f}"\
                .format(lloss,
                        closs,
                    tpr, # True positive rate
                    tnr, # True negative rate
                    lacc, # Logits accuracy
                    cacc, # Count accuracy
                    w) 
                
        if i % log_every == 0:
            print(msg)
            if platform == "linux":
                with open("results/" + name + ".txt", "a") as f:
                    print(msg, file = f)
        
        if platform == "linux" and i % save_every == 0:
            torch.save(net, "results/" + name)
